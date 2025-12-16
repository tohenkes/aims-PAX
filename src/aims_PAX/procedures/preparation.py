import os
import random
from pathlib import Path
import torch
from typing import Union, List
import numpy as np
from mace import tools
from mace.calculators import MACECalculator
from so3krates_torch.calculator.so3 import TorchkratesCalculator, MultiHeadSO3LRCalculator
from aims_PAX.tools.uncertainty import (
    HandleUncertainty,
    MolForceUncertainty,
)
from aims_PAX.tools.utilities.data_handling import (
    load_ensemble_sets_from_folder,
    ase_to_model_ensemble_sets,
    create_dataloader,
)
from aims_PAX.tools.utilities.utilities import (
    get_ensemble_training_setups,
    ensemble_from_folder,
    Z_from_geometry,
    list_files_in_directory,
    get_atomic_energies_from_ensemble,
    create_ztable,
    get_atomic_energies_from_pt,
    dtype_mapping,
    setup_ensemble_dicts,
    create_seeds_tags_dict,
    setup_logger,
    update_model_auxiliaries,
    save_checkpoint,
    create_keyspec,
    log_yaml_block,
    normalize_md_settings,
    apply_model_settings,
    apply_finetuning_settings,
    AIMSControlParser,
    ModifyMD,
)
from aims_PAX.tools.utilities.input_utils import read_geometry
from aims_PAX.tools.model_tools.train_epoch import (
    train_epoch,
    validate_epoch_ensemble,
)
from aims_PAX.tools.model_tools.training_tools import (
    setup_model_training
)
from aims_PAX.tools.utilities.mpi_utils import CommHandler
import ase
import logging
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.npt import NPT
from ase.md.nose_hoover_chain import MTKNPT
from ase import units
from contextlib import nullcontext
from mace.calculators import mace_mp

try:
    import asi4py
except Exception as e:
    asi4py = None


class PrepareInitialDatasetProcedure:
    """
    Class to prepare the inital dataset generation procedure for
    active learning. It handles all the input files, prepares the
    calculators, models, directories etc.
    TODO: change training so it uses the code from AL
    """

    def __init__(
        self,
        model_settings: dict,
        aimsPAX_settings: dict,
        path_to_control: str = "./control.in", #TODO: Rename
        path_to_geometry: str = "./geometry.in",
        use_mpi: bool = True,
    ) -> None:
        """
        Args:
            model_settings (dict): Settings for the model and its training.
            al_settings (dict): Settings for the active learning procedure.
            path_to_aims_lib (str): Path to the compiled AIMS library.
            atomic_energies_dict (dict, optional): Dictionary containing the
                                            atomic energies. Defaults to None.
            species_dir (str, optional): Path to the basis set settings
                                                of AIMS. Defaults to None.
            path_to_control (str, optional): Path to the AIMS control file.
                                                Defaults to "./control.in".
            path_to_geometry (str, optional): Path to the initial geometry.
                                                Defaults to "./geometry.in".
            ensemble_seeds (np.array, optional): Seeds for the individual
                                        ensemble members. Defaults to None.
        """

        self.comm_handler = CommHandler(use_mpi=use_mpi)
        self.rank = self.comm_handler.get_rank()
        self.world_size = self.comm_handler.get_size()

        self.log_dir = Path(aimsPAX_settings["MISC"]["log_dir"])
        logger_level = (
            logging.DEBUG
            if model_settings["MISC"]["log_level"].lower() == "debug"
            else logging.INFO
        )

        self.logger = setup_logger(
            level=logger_level,
            tag="initial_dataset",
            directory=self.log_dir,
        )
        if self.rank == 0:
            logging.info("Initializing initial dataset procedure.")
            logging.info(f"Procedure runs on {self.world_size} workers.")
            logging.info(
                "Using followng settings for the initial dataset procedure:"
            )
            log_yaml_block(
                "INITIAL_DATASET_GENERATION",
                aimsPAX_settings["INITIAL_DATASET_GENERATION"],
            )

        self.control_parser = AIMSControlParser()
        self._handle_model_settings(model_settings)
        if self.rank == 0:
            logging.info(f"Using following settings for {self.model_choice.upper()}:")
            log_yaml_block(self.model_choice.upper(), model_settings)

        self._handle_settings(aimsPAX_settings)
        self._handle_aims_settings(path_to_control)
        self._create_folders()

        if self.restart:
            if self.rank == 0:
                logging.info(
                    "Restarting initial dataset acquisition from checkpoint."
                )
            try:
                self.init_ds_restart_dict = np.load(
                    "restart/initial_ds/initial_ds_restart.npy",
                    allow_pickle=True,
                ).item()
            except FileNotFoundError:
                logging.error(
                    "Restart file under 'restart/initial_ds/"
                    "initial_ds_restart.npy' not found."
                )
                raise
            self.trajectories = self.init_ds_restart_dict["trajectories"]
            self.atoms = list(self.trajectories.values())
            self.step = self.init_ds_restart_dict["step"]
        else:
            self.trajectories = read_geometry(path_to_geometry, log=True)
            self.atoms = list(self.trajectories.values())
            self.step = 0
            # TODO: multiple trajectories per atoms
            # TODO: make this compatible with the other methods in case the
            # geometries are not different systems

        self.num_trajectories = len(self.trajectories)
        self.md_settings, _ = normalize_md_settings(
            md_settings=self.md_settings_raw,
            num_trajectories=self.num_trajectories
        )
        
        if self.rank == 0:
            logging.info(
                "Running Initial Dataset Procedure with "
                f"{len(self.trajectories)} geometries."
            )
            
        if self.model_choice == "mace":
            self.z = Z_from_geometry(self.trajectories)
        elif self.model_choice in ["so3lr", "so3krates"]:
            self.z = np.array([i for i in range(1, 119)])
        self.z_table = create_ztable(self.z)

        self.md_drivers = {idx: None for idx in self.trajectories.keys()}

        self._setup_seeds()
        
        self.seeds_tags_dict = create_seeds_tags_dict(
            seeds=self.ensemble_seeds,
            model_settings=self.model_settings,
            misc_settings=self.misc,
        )

        self._handle_atomic_energies()
        self.epoch = 0

        # the ensemble dictionary contains the models and their tags as values
        # and keys the seeds_tags_dict connects the seeds to the tags of each
        # ensemble member the training_setups dictionary contains the training
        # setups (optimizer, scheduler etc.) for each ensemble member
        if self.rank == 0:
            self.ensemble = setup_ensemble_dicts(
                seeds_tags_dict=self.seeds_tags_dict,
                model_settings=self.model_settings,
                z_table=self.z_table,
                ensemble_atomic_energies_dict=self.ensemble_atomic_energies_dict,
                device=self.device,
            )
            self.training_setups = get_ensemble_training_setups(
                ensemble=self.ensemble,
                model_settings=self.model_settings,
                restart=self.restart,
                checkpoints_dir=self.checkpoints_dir,
                mol_idxs=self.mol_idxs,
            )
            if self.restart:
                self.epoch = (
                    self.training_setups[list(self.ensemble.keys())[0]][
                        "epoch"
                    ]
                    + 1
                )

        self.comm_handler.barrier()
        self.epoch = self.comm_handler.bcast(self.epoch, root=0)
        self.comm_handler.barrier()

        # each ensemble member has their own initial dataset.
        # we create a ASE and model dataset because it makes
        # conversion and saving easier
        if self.rank == 0:
            if self.restart:
                self.ensemble_ase_sets = load_ensemble_sets_from_folder(
                    ensemble=self.ensemble,
                    path_to_folder=self.dataset_dir / "initial",
                )
                self.ensemble_model_sets = ase_to_model_ensemble_sets(
                    ensemble_ase_sets=self.ensemble_ase_sets,
                    z_table=self.z_table,
                    r_max=self.r_max,
                    seed=self.seed,
                    key_specification=self.key_specification,
                    r_max_lr=self.r_max_lr,
                    all_heads=self.all_heads,
                )

            else:
                self.ensemble_ase_sets = {
                        tag: {"train": [], "valid": []}
                        for tag in self.ensemble.keys()
                    }

                if self.use_multihead_model:
                    self.ensemble_model_sets = {
                            tag: {"train": [], "valid": {}}
                            for tag in self.ensemble.keys()
                        }
                    for head in self.all_heads:
                        for tag in self.ensemble.keys():
                            self.ensemble_model_sets[tag]["valid"][
                                head
                            ] = []
                else:
                    self.ensemble_model_sets = {
                            tag: {"train": [], "valid": {
                                "Default": []
                            }}
                            for tag in self.ensemble.keys()
                        }
                

                        
        if self.analysis:
            if self.restart:
                self.collect_losses = self.init_ds_restart_dict[
                    "last_initial_losses"
                ]
            else:
                self.collect_losses = {
                    "epoch": [],
                    "avg_losses": [],
                    "ensemble_losses": [],
                }

    def _setup_seeds(self):
        """
        Sets up the random seeds for reproducibility.
        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.ensemble_seeds = np.random.randint(
            0, 1000, size=self.ensemble_size
        )

    def _handle_model_settings(self, model_settings: dict) -> None:
        """
        Saves the model settings to class attributes.
        and fall back to defaults if not.

        Args:
            model_settings (dict): Dictionary containing the model settings.
        """

        apply_model_settings(
            target=self,
            model_settings=model_settings
        )
        # No CuEQ training during initial dataset generation
        # (because avg_num_neighbors, mean, std, atomic energies etc
        # are changing all the time; not possible to modify with CuEQ)
        self.enable_cueq_train = False
        model_settings["MISC"]["enable_cueq_train"] = False

    def _handle_settings(self, aimsPAX_settings: dict) -> None:
        """
        Saves the active learning settings to class attributes.
        TODO: Create function to check if all necessary settings are present
        and fall back to defaults if not.

        Args:
            al_settings (dict): Dictionary containing the active
                                learning settings.
        """

        self.idg_settings = aimsPAX_settings["INITIAL_DATASET_GENERATION"]
        self.misc = aimsPAX_settings["MISC"]
        self.md_settings_raw = aimsPAX_settings["MD"]
        self.cluster_settings = aimsPAX_settings.get("CLUSTER", None)

        self.ensemble_size = self.idg_settings["ensemble_size"]
        self.desired_acc = self.idg_settings["desired_acc"]
        self.desired_acc_scale_idg = self.idg_settings["desired_acc_scale_idg"]
        self.n_points_per_sampling_step_idg = self.idg_settings[
            "n_points_per_sampling_step_idg"
        ]
        self.max_initial_epochs = self.idg_settings["max_initial_epochs"]
        self.converge_initial = self.idg_settings["converge_initial"]
        self.max_convergence_epochs = self.idg_settings[
            "max_convergence_epochs"
        ]
        self.valid_skip = self.idg_settings["valid_skip"]
        self.skip_step = self.idg_settings["skip_step_initial"]
        self.intermediate_epochs = self.idg_settings["intermediate_epochs_idg"]
        self.valid_ratio = self.idg_settings["valid_ratio"]
        self.ASI_path = self.idg_settings["aims_lib_path"]
        self.species_dir = self.idg_settings["species_dir"]
        self.analysis = self.idg_settings["analysis"]
        self.margin = self.idg_settings["margin"]
        self.mol_idxs = self.misc["mol_idxs"]
        self.key_specification = create_keyspec(
            energy_key=self.misc['energy_key'],
            forces_key=self.misc['forces_key'],
            stress_key=self.misc['stress_key'],
            dipole_key=self.misc['dipole_key'],
            polarizability_key=self.misc['polarizability_key'],
            head_key=self.misc['head_key'],
            charges_key=self.misc['charges_key'],
            total_charge_key=self.misc['total_charge_key'],
            total_spin_key=self.misc['total_spin_key'],
        )
        self.idg_progress_dft_update = self.idg_settings["progress_dft_update"]
        if not self.idg_settings["scheduler_initial"]:
            self.model_settings["lr_scheduler"] = None

        self.initial_sampling = self.idg_settings["initial_sampling"]
        self.foundational_model = self.idg_settings["foundational_model"]
        self.foundational_model_settings = self.idg_settings["foundational_model_settings"]
        
        self.restart = os.path.exists(
            "restart/initial_ds/initial_ds_restart.npy"
        )
        self.create_restart = self.misc["create_restart"]
        if self.create_restart:
            self.init_ds_restart_dict = {
                "trajectories": None,
                "last_initial_losses": None,
                "initial_ds_done": False,
            }
        self.distinct_model_sets = self.idg_settings["distinct_model_sets"]

    def _update_restart_dict(self):
        self._collect_restart_points(self.trajectories)
        self.init_ds_restart_dict["trajectories"] = self.last_points
        self.init_ds_restart_dict["step"] = self.step
        if self.analysis:
            self.init_ds_restart_dict["last_initial_losses"] = (
                self.collect_losses
            )

    def _create_folders(self):
        """
        Creates the necessary directories for saving the datasets.
        """
        self.dataset_dir = Path(self.misc["dataset_dir"])
        (self.dataset_dir / "initial" / "training").mkdir(
            parents=True, exist_ok=True
        )
        (self.dataset_dir / "initial" / "validation").mkdir(
            parents=True, exist_ok=True
        )
        os.makedirs("model", exist_ok=True)
        if self.analysis:
            os.makedirs("analysis", exist_ok=True)
        if self.create_restart:
            os.makedirs("restart/initial_ds", exist_ok=True)

    def _setup_aims_calculator(
        self,
        atoms: ase.Atoms,
    ) -> ase.Atoms:
        """
        Attaches the AIMS calculator to the atoms object.
        Uses the AIMS settings from the control.in to set up the calculator.

        Args:
            atoms (ase.Atoms): Atoms object to attach the calculator to.
            pbc (bool, optional): If periodic boundry conditions are required
                                    or not.
            Defaults to False.

        Returns:
            ase.Atoms: Atoms object with the calculator attached.
        """
        aims_settings = self.aims_settings.copy()

        def init_via_ase(asi):
            from ase.calculators.aims import Aims, AimsProfile

            aims_settings["profile"] = AimsProfile(
                command="asi-doesnt-need-command"
            )
            calc = Aims(**aims_settings)
            calc.write_inputfiles(asi.atoms, properties=self.properties)

        if asi4py is None:
            raise ImportError(
                "asi4py is not properly installed. "
                "Please install it to use the AIMS calculator."
            )
        calc = asi4py.asecalc.ASI_ASE_calculator(
            self.ASI_path, init_via_ase, self.comm_handler.comm, atoms
        )
        return calc

    def _handle_aims_settings(
            self,
            control_source: Union[str, dict[int, str]],
            log: bool = False
            ):
        """
        Parses the AIMS control file to get the settings for the AIMS
        calculator.

        Args:
            path_to_control (str): Path to the AIMS control file.
            species_dir (str): Path to the species directory of AIMS.
        """
        if isinstance(control_source, str):
            aims_settings = self.control_parser(control_source)
            aims_settings["compute_forces"] = True
            aims_settings["species_dir"] = self.species_dir
            aims_settings["postprocess_anyway"] = (
                True  # this is necesssary to check for convergence in ASI
            )
            self.aims_settings = {0: aims_settings}
        elif isinstance(control_source, dict):
            self.aims_settings = {}
            for key, value in control_source.items():
                aims_settings = self.control_parser(value)
                aims_settings["compute_forces"] = True
                aims_settings["species_dir"] = self.species_dir
                aims_settings["postprocess_anyway"] = (
                    True  # this is necesssary to check for convergence in ASI
                )
                if log:
                    logging.info(
                        f"Control file for geometry {key}: {value}."
                    )
                self.aims_settings[key] = aims_settings
        
    def setup_md(self, atoms: ase.Atoms, md_settings: dict):
        """
        Sets up the ASE molecular dynamics object for the atoms object.
        TODO: Add more flexibility and support for other settings

        Args:
            atoms (ase.Atoms): Atoms to be propagated.
            md_settings (dict): Dictionary containing the MD settings.

        Returns:
            ase.md.MolecularDynamics: ASE MD engine.
        """

        if not self.restart:
            MaxwellBoltzmannDistribution(
                atoms, temperature_K=md_settings["temperature"]
            )

        if md_settings["stat_ensemble"].lower() == "nvt":
            if md_settings["thermostat"].lower() == "langevin":
                dyn = Langevin(
                    atoms,
                    timestep=md_settings["timestep"] * units.fs,
                    friction=md_settings["friction"] / units.fs,
                    temperature_K=md_settings["temperature"],
                    rng=np.random.RandomState(md_settings["MD_seed"]),
                )
        elif md_settings["stat_ensemble"].lower() == "npt":
            if md_settings["barostat"].lower() == "berendsen":
                npt_settings = {
                    "atoms": atoms,
                    "timestep": md_settings["timestep"] * units.fs,
                    "temperature": md_settings["temperature"],
                    "pressure_au": md_settings["pressure"] * units.Pascal,
                }

                if md_settings.get("taup", False):
                    npt_settings["taup"] = md_settings["taup"] * units.fs
                if md_settings.get("taut", False):
                    npt_settings["taut"] = md_settings["taut"] * units.fs
                if md_settings.get("compressibility_au", False):
                    npt_settings["compressibility_au"] = md_settings[
                        "compressibility_au"
                    ]
                if md_settings.get("fixcm", False):
                    npt_settings["fixcm"] = md_settings["fixcm"]

                dyn = NPTBerendsen(**npt_settings)
            if md_settings["barostat"].lower() == "npt":
                npt_settings = {
                    "atoms": atoms,
                    "timestep": md_settings["timestep"] * units.fs,
                    "temperature_K": md_settings["temperature"],
                    "externalstress": md_settings["pressure"] * units.Pascal
                    * units.bar,
                    "ttime": md_settings["ttime"] * units.fs,
                    "pfactor": md_settings["pfactor"] * units.fs,
                }

                if md_settings.get("mask", False):
                    npt_settings["mask"] = md_settings["mask"]

                dyn = NPT(**npt_settings)
            if md_settings["barostat"].lower() == "mtk":
                npt_settings = {
                    "atoms": atoms,
                    "timestep": md_settings["timestep"] * units.fs,
                    "temperature_K": md_settings["temperature"],
                    "pressure_au": md_settings["pressure"] * units.Pascal,
                    "tdamp": md_settings["tdamp"] * units.fs,
                    "pdamp": md_settings["pdamp"] * units.fs,
                    "tchain": md_settings["tchain"],
                    "pchain": md_settings["pchain"],
                    "tloop": md_settings["tloop"],
                    "ploop": md_settings["ploop"],
                }
                
                dyn = MTKNPT(**npt_settings)

        return dyn

    def _handle_atomic_energies(
        self,
    ):
        """
        Handles the atomic energies for the initial dataset generation.
        Either they are loaded from a model checkpoint, initialized to zero
        or specified by the user in the model settings.
        """
        self.ensemble_atomic_energies = None
        self.ensemble_atomic_energies_dict = None
        self.update_atomic_energies = False
        if self.rank == 0:
            if self.atomic_energies_dict is None:
                if self.restart:
                    logging.info("Loading atomic energies from checkpoint.")
                    (
                        self.ensemble_atomic_energies,
                        self.ensemble_atomic_energies_dict,
                    ) = get_atomic_energies_from_pt(
                        path_to_checkpoints=self.checkpoints_dir,
                        z=self.z,
                        seeds_tags_dict=self.seeds_tags_dict,
                        dtype=self.dtype,
                        model_choice=self.model_choice
                    )
                else:

                    logging.info(
                        "No atomic energies specified. "
                        " Fitting to training data."
                    )
                    self.ensemble_atomic_energies_dict = {
                        tag: {z: 0 for z in np.sort(np.unique(self.z))}
                        for tag in self.seeds_tags_dict.keys()
                    }
                    self.ensemble_atomic_energies = {
                        tag: np.array(
                            [
                                self.ensemble_atomic_energies_dict[tag][z]
                                for z in self.ensemble_atomic_energies_dict[
                                    tag
                                ].keys()
                            ]
                        )
                        for tag in self.seeds_tags_dict.keys()
                    }

                self.update_atomic_energies = True

            else:

                logging.info("Using specified atomic energies.")
                self.ensemble_atomic_energies_dict = {
                    tag: self.atomic_energies_dict
                    for tag in self.seeds_tags_dict.keys()
                }

                self.ensemble_atomic_energies = {
                    tag: np.array(
                        [
                            self.ensemble_atomic_energies_dict[tag][z]
                            for z in self.ensemble_atomic_energies_dict[
                                tag
                            ].keys()
                        ]
                    )
                    for tag in self.seeds_tags_dict.keys()
                }

            logging.info(
                f"{self.ensemble_atomic_energies_dict[list(self.seeds_tags_dict.keys())[0]]}"
            )

    def check_initial_ds_done(self) -> bool:
        """
        Checks if the initial dataset generation is already done.
        This is mostly relevant when one is restarting using the
        aims_PAX command which runs both the initial dataset generation
        and active learning.

        Returns:
            bool: Whether the initial dataset generation is done or not.
        """
        if self.create_restart:
            check = self.init_ds_restart_dict.get("initial_ds_done", False)
            if check:
                if self.rank == 0:
                    logging.info(
                        "Initial dataset generation is already done. Closing"
                    )
                    self.logger.handlers.clear()
            return check
        else:
            return False

    def _run_MD(self, atoms: ase.Atoms, dyn) -> ase.Atoms:
        """
        Runs molecular dynamics simulation for the atoms object. Saves
        energy and forces in a model readable format.

        Args:
            atoms (ase.Atoms): Atoms object to be propagated.
            dyn (ase.md.MolecularDynamics): ASE MD engine.
        Returns:
            ase.Atoms: Atoms object with the energy and forces saved in the
                        model readable format.
        """

        dyn.run(self.skip_step)
        current_energy = np.array(atoms.get_potential_energy())
        current_forces = np.array(atoms.get_forces())
        current_point = atoms.copy()
        # model reads energies and forces from the info & arrays dictionary
        current_point.info["REF_energy"] = current_energy
        current_point.arrays["REF_forces"] = current_forces
        return current_point

    def _collect_restart_points(
        self,
        trajectories: dict[int, ase.Atoms],
    ):
        self.last_points = {}
        for idx, atoms in trajectories.items():
            current_point = atoms.copy()
            current_point.set_velocities(atoms.get_velocities())
            current_point.set_masses(atoms.get_masses())
            self.last_points[idx] = current_point

    def converge(self):
        """
        Converges the ensemble on the acquired initial dataset.
        Stops when the validation loss does not improve for a certain number
        of epochs (defined in patience) or when the maximum number of epochs
        is reached.

        """
        if self.rank == 0:
            logging.info("Converging.")
            for _, (tag, model) in enumerate(self.ensemble.items()):

                (
                    self.ensemble_model_sets[tag]["train_loader"],
                    self.ensemble_model_sets[tag]["valid_loader"],
                ) = create_dataloader(
                    self.ensemble_model_sets[tag]["train"],
                    self.ensemble_model_sets[tag]["valid"],
                    self.set_batch_size,
                    self.set_valid_batch_size,
                )

                update_model_auxiliaries(
                    model=model,
                    model_choice=self.model_choice,
                    model_sets=self.ensemble_model_sets[tag],
                    atomic_energies_list=self.ensemble_atomic_energies[tag],
                    scaling=self.scaling,
                    update_atomic_energies=self.update_atomic_energies,
                    z_table=self.z_table,
                    atomic_energies_dict=self.ensemble_atomic_energies_dict[
                        tag
                    ],
                    update_avg_num_neighbors=self.config.update_avg_num_neighbors,
                    dtype=self.dtype,
                    device=self.device,
                )

            self.training_setups_convergence = {}
            for tag in self.ensemble.keys():
                self.training_setups_convergence[tag] = setup_model_training(
                    settings=self.model_settings,
                    model=self.ensemble[tag],
                    model_choice=self.model_choice,
                    tag=tag,
                    restart=self.restart,
                    convergence=True,
                    checkpoints_dir=self.checkpoints_dir,
                    mol_idxs=self.mol_idxs,
                )
            best_valid_loss = np.inf
            epoch = 0
            if self.restart:
                epoch = self.training_setups_convergence[
                    list(self.ensemble.keys())[0]
                ]["epoch"]

            convergence_patience = self.idg_settings["convergence_patience"]
            no_improvement = 0
            ensemble_valid_losses = {
                tag: np.inf for tag in self.ensemble.keys()
            }
            for j in range(self.max_convergence_epochs):
                for tag, model in self.ensemble.items():
                    logger = tools.MetricsLogger(
                        directory=self.model_settings["GENERAL"]["loss_dir"],
                        tag=tag + "_train",
                    )
                    train_epoch(
                        model=model,
                        train_loader=self.ensemble_model_sets[tag][
                            "train_loader"
                        ],
                        loss_fn=self.training_setups_convergence[tag][
                            "loss_fn"
                        ],
                        optimizer=self.training_setups_convergence[tag][
                            "optimizer"
                        ],
                        lr_scheduler=self.training_setups_convergence[tag][
                            "lr_scheduler"
                        ],
                        valid_loss=ensemble_valid_losses[tag],
                        epoch=epoch,
                        start_epoch=epoch,
                        logger=logger,
                        device=self.training_setups_convergence[tag]["device"],
                        max_grad_norm=self.training_setups_convergence[tag][
                            "max_grad_norm"
                        ],
                        output_args=self.training_setups_convergence[tag][
                            "output_args"
                        ],
                        ema=self.training_setups_convergence[tag]["ema"],
                    )

                if (
                    epoch % self.valid_skip == 0
                    or epoch == self.max_convergence_epochs - 1
                ):
                    ensemble_valid_losses, valid_loss, _ = (
                        validate_epoch_ensemble(
                            ensemble=self.ensemble,
                            valid_loader=self.ensemble_model_sets[tag][
                                "valid_loader"
                            ],
                            training_setups=self.training_setups_convergence,
                            logger=logger,
                            log_errors=self.model_settings["MISC"][
                                "error_table"
                            ],
                            epoch=epoch,
                        )
                    )
                    if (
                        best_valid_loss > valid_loss
                        and (best_valid_loss - valid_loss) > self.margin
                    ):
                        best_valid_loss = valid_loss
                        best_epoch = epoch
                        no_improvement = 0
                        for tag, model in self.ensemble.items():
                            param_context = (
                                self.training_setups_convergence[tag][
                                    "ema"
                                ].average_parameters()
                                if self.training_setups_convergence[tag]["ema"]
                                is not None
                                else nullcontext()
                            )
                            with param_context:
                                torch.save(
                                    model,
                                    Path(
                                        self.model_settings["GENERAL"][
                                            "model_dir"
                                        ]
                                    )
                                    / (tag + ".model"),
                                )
                            save_checkpoint(
                                checkpoint_handler=self.training_setups_convergence[
                                    tag
                                ][
                                    "checkpoint_handler"
                                ],
                                training_setup=self.training_setups_convergence[
                                    tag
                                ],
                                model=model,
                                epoch=epoch,
                                keep_last=False,
                            )
                    else:
                        no_improvement += 1

                epoch += 1
                if no_improvement > convergence_patience:
                    logging.info(
                        f"No improvements for {convergence_patience} epochs. "
                        "Training converged. Best model based on validation"
                        " loss saved"
                    )
                    break
                if j == self.max_convergence_epochs - 1:
                    logging.info(
                        f"Maximum number of epochs reached. Best model "
                        f"(Epoch {best_epoch}) based on validation loss saved."
                    )
                    break


class ALConfiguration:
    """Handles all configuration setup for active learning."""

    def __init__(
        self,
        model_settings: dict,
        aimsPAX_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
    ):
        self.model_settings = model_settings
        self.path_to_control = path_to_control
        self.path_to_geometry = path_to_geometry
        self.aimsPAX_settings = aimsPAX_settings
        self.al_settings = aimsPAX_settings["ACTIVE_LEARNING"]
        self.md_settings_raw = aimsPAX_settings["MD"]
        self.cluster_settings = aimsPAX_settings.get("CLUSTER", None)
        self.misc = aimsPAX_settings.get("MISC", {})

        self._setup_model_configuration()
        self._setup_aimsPAX_configuration()

    def _setup_model_configuration(self):
        """Setup model-specific configuration."""
        
        apply_model_settings(
            target=self,
            model_settings=self.model_settings
        )
        self.checkpoints_dir += "/al"
        
    def _setup_aimsPAX_configuration(self):
        """Setup active learning configuration."""
        # Training parameters
        self.max_MD_steps = self.al_settings["max_MD_steps"]
        self.epochs_per_worker = self.al_settings["epochs_per_worker"]
        self.max_convergence_epochs = self.al_settings[
            "max_convergence_epochs"
        ]
        self.intermediate_epochs_al = self.al_settings[
            "intermediate_epochs_al"
        ]
        self.convergence_patience = self.al_settings[
            "convergence_patience"
        ]  # for convergence only
        self.desired_accuracy = self.al_settings["desired_acc"]
        self.margin = self.al_settings["margin"]  # for convergence only

        # Data parameters
        self.num_trajectories = self.al_settings["num_trajectories"]
        self.skip_step = self.al_settings["skip_step_mlff"]
        self.valid_skip = self.al_settings["valid_skip"]
        self.analysis_skip = self.al_settings["analysis_skip"]
        self.valid_ratio = self.al_settings["valid_ratio"]
        self.max_train_set_size = self.al_settings["max_train_set_size"]
        self.c_x = self.al_settings["c_x"]
        self.extend_existing_final_ds = self.al_settings[
            'extend_existing_final_ds'
        ]
        
        # Training procedures (TODO: Move training methods from model file here)
        self.replay_strategy = self.al_settings["replay_strategy"]
        self.train_subset_size = self.al_settings["train_subset_size"]
        self.valid_subset_size = self.al_settings["valid_subset_size"]
        if self.replay_strategy == "random_subset":
            assert self.train_subset_size is not None, (
                "train_subset_size must be specified for random_subset "
                "replay strategy."
            )
        
        # Paths
        self.dataset_dir = Path(self.misc["dataset_dir"])
        self.log_dir = self.misc["log_dir"]
        self.species_dir = self.al_settings["species_dir"]
        self.ASI_path = self.al_settings["aims_lib_path"]

        # Optional settings
        self.analysis = self.al_settings["analysis"]
        self.seeds_tags_dict = self.al_settings.get("seeds_tags_dict", None)

        # Uncertainty parameters
        self.converge_al = self.al_settings["converge_al"]
        self.converge_best = self.al_settings["converge_best"]
        self.uncertainty_type = self.al_settings["uncertainty_type"]

        self.uncert_not_crossed_limit = self.al_settings[
            "uncert_not_crossed_limit"
        ]
        self.freeze_threshold_dataset = self.al_settings[
            "freeze_threshold_dataset"
        ]
        self.freeze_threshold = False

        self.key_specification = create_keyspec(
            energy_key=self.misc['energy_key'],
            forces_key=self.misc['forces_key'],
            stress_key=self.misc['stress_key'],
            dipole_key=self.misc['dipole_key'],
            polarizability_key=self.misc['polarizability_key'],
            head_key=self.misc['head_key'],
            charges_key=self.misc['charges_key'],
            total_charge_key=self.misc['total_charge_key'],
            total_spin_key=self.misc['total_spin_key'],
        )

        # Molecular indices
        self._setup_molecular_indices()

        # Restart handling
        self.restart = os.path.exists("restart/al/al_restart.npy")
        self.create_restart = self.misc["create_restart"]

        # foundational model usage during AL
        self.use_foundational = self.al_settings["use_foundational"]
        self.foundational_model_settings = self.al_settings["foundational_model_settings"]

    def _setup_molecular_indices(self):
        """
        Setup molecular indices configuration.
        Only needed if intermolecular uncertainty is used.
        """
        mol_idxs_path = self.misc["mol_idxs"]
        self.mol_idxs = (
            np.load(mol_idxs_path, allow_pickle=True)["arr_0"].tolist()
            if mol_idxs_path is not None
            else None
        )

        if self.mol_idxs is not None:
            self.intermol_crossed = 0
            self.intermol_crossed_limit = self.al_settings[
                "intermol_crossed_limit"
            ]
            self.intermol_forces_weight = self.al_settings[
                "intermol_forces_weight"
            ]
            self.switched_on_intermol = False

            # Check if using intermolecular loss
            loss_type = self.model_settings["TRAINING"]["loss"].lower()
            self.using_intermol_loss = loss_type == "intermol"


class ALStateManager:
    """Manages the state of trajectories, ensembles, and analysis data."""

    def __init__(self, config: ALConfiguration, comm_handler):
        self.config = config
        self.comm_handler = comm_handler
        self.rank = comm_handler.get_rank()

        # Initialize state dictionaries
        self.trajectories = {}
        self.MD_checkpoints = {}
        self.trajectory_status = {}
        self.trajectory_MD_steps = {}
        self.trajectory_total_epochs = {}
        self.trajectory_intermediate_epochs = {}
        self.uncert_not_crossed = {}
        self.last_point_added = {}

        # Ensemble state
        self.seeds_tags_dict = {}
        self.ensemble_reset_opt = {}
        self.ensemble_no_improvement = {}
        self.ensemble_best_valid = {}

        # Metrics and counters
        self.current_valid_error = np.inf
        self.threshold = np.inf
        self.total_points_added = 0
        self.train_points_added = 0
        self.valid_points_added = 0
        self.num_MD_limits_reached = 0
        self.num_workers_training = 0
        self.num_workers_waiting = 0
        self.total_epoch = 0
        self.check = 0
        self.uncertainties = []

        # Multihead states
        self.head_data_counter = 0

        # Analysis state
        if self.config.analysis:
            self._initialize_analysis_state()

    def initialize_fresh_state(self, path_to_geometry: str):
        """Initialize state for a fresh run."""
        atoms = read_geometry(path_to_geometry, log=True)
        if self.rank == 0:
            logging.info(
                "Running Active Learning Procedure with "
                f"{len(atoms)} geometries."
            )
        # Initialize trajectory data
        self.trajectories = self._create_trajectories(
            atoms, self.config.num_trajectories
        )
        num_traj = len(self.trajectories)
        self.MD_checkpoints = {
            i: self.trajectories[i].copy() for i in range(num_traj)
        }
        self.trajectory_status = {i: "running" for i in range(num_traj)}
        self.trajectory_MD_steps = {i: 0 for i in range(num_traj)}
        self.trajectory_total_epochs = {i: 0 for i in range(num_traj)}
        self.trajectory_intermediate_epochs = {i: 0 for i in range(num_traj)}
        self.uncert_not_crossed = {i: 0 for i in range(num_traj)}

        # Initialize ensemble data
        if hasattr(self, "seeds_tags_dict"):
            tags = self.config.seeds_tags_dict.keys()
            self.ensemble_reset_opt = {tag: False for tag in tags}
            self.ensemble_no_improvement = {tag: 0 for tag in tags}
            self.ensemble_best_valid = {tag: np.inf for tag in tags}

    def _create_trajectories(
        self, atoms: List[ase.Atoms], num_trajectories: int
    ) -> dict:
        """
        Creates a dictionary of trajectories with given starting
        geometries.

        If `len(atoms)` > 1, each entry in the list
        is used for a different trajectory. If the number of
        user-specified trajectories, `num_trajectories`, is
        smaller than the length of `atoms`, a warning is issued and
        and `num_trajectories` is overwritten. If the number of
        user-specified trajectories is larger than the length of
        `atoms`, a warning is issued and it just continues to loop
        through atoms again until num_trajectory is met.

        If `len(atoms)` == 1, the same geometry is used for all
        trajectories.

        Args:
            atoms (Union[ase.Atoms, List]): Starting geometries for
                                                trajectories.
            num_trajectories (int): Number of trajectories to create.

        Returns:
            dict: Dictionary of trajectories indexed by trajectory number.
        """
        trajectories = {}
        if len(atoms) == 1:
            for i in range(num_trajectories):
                trajectories[i] = atoms[0].copy()
            return trajectories

        elif len(atoms) > 1:
            if num_trajectories > len(atoms):
                logging.warning(
                    f"Number of trajectories ({num_trajectories}) "
                    f"is larger than the number of provided geometries "
                    f"({len(atoms)}). Looping through geometries until "
                    f"the number of trajectories is met."
                )
            elif num_trajectories < len(atoms):
                logging.warning(
                    f"Number of trajectories ({num_trajectories}) "
                    f"is smaller than the number of provided geometries "
                    f"({len(atoms)}). Overwriting to match the number of "
                    f"provided geometries."
                )
                num_trajectories = len(atoms)
                self.config.num_trajectories = num_trajectories

            for i in range(num_trajectories):
                trajectories[i] = atoms[i % len(atoms)].copy()

            return trajectories

    def _initialize_analysis_state(self):
        """Initialize analysis-specific state."""
        num_traj = self.config.num_trajectories

        self.trajectories_analysis_prediction = {
            i: None for i in range(num_traj)
        }
        self.t_intervals = {i: [] for i in range(num_traj)}
        self.analysis_checks = self._create_analysis_checks_dict(num_traj)

        if self.config.mol_idxs is not None:
            self._add_molecular_analysis_fields()
            self.uncertainty_checks = []

        self.collect_losses = {
            "epoch": [],
            "avg_losses": [],
            "ensemble_losses": [],
        }
        self.collect_thresholds = {i: [] for i in range(num_traj)}

    def _create_analysis_checks_dict(self, num_trajectories: int) -> dict:
        """Create analysis checks dictionary structure."""
        base_fields = [
            "prediction",
            "true_forces",
            "atom_wise_uncertainty",
            "uncertainty_via_max",
            "uncertainty_via_mean",
            "max_error",
            "mean_error",
            "atom_wise_error",
            "threshold",
            "train_set_length",
            "current_md_step",
        ]

        return {
            trajectory: {field: [] for field in base_fields}
            for trajectory in range(num_trajectories)
        }

    def _add_molecular_analysis_fields(self):
        """Add molecular-specific analysis fields."""
        mol_fields = [
            "mol_forces_prediction",
            "mol_forces_true",
            "total_uncertainty",
            "mol_forces_uncertainty",
            "mol_wise_error",
            "max_mol_error",
            "mean_mol_error",
        ]

        for trajectory in range(self.config.num_trajectories):
            for field in mol_fields:
                self.analysis_checks[trajectory][field] = []


class ALEnsemble:
    """Manages ensemble setup and dataset handling for active learning."""

    def __init__(self, config: ALConfiguration, comm_handler):
        self.config = config
        self.comm_handler = comm_handler
        self.rank = comm_handler.get_rank()

        # System properties
        self.z = None
        self.z_table = None

        # Ensemble and datasets (rank 0 only)
        self.ensemble = None
        self.training_setups = None
        self.ensemble_ase_sets = None
        self.ensemble_model_sets = None
        self.train_dataset_len = None

        # Setup system properties
        self._setup_system_properties(self.config.path_to_geometry)

        # Load seeds_tags_dict if not provided
        self._load_seeds_tags_dict()

    def _setup_system_properties(self, path_to_geometry: str):
        """Setup system-specific properties."""
        atoms = read_geometry(path_to_geometry)
        if self.config.model_choice == "mace":
            self.z = Z_from_geometry(atoms)
        elif self.config.model_choice in ["so3lr", "so3krates"]:
            self.z = np.array([i for i in range(1, 119)])
            
        self.z_table = create_ztable(self.z)

    def _load_seeds_tags_dict(self):
        """Load seeds_tags_dict from file if not provided."""
        if self.config.seeds_tags_dict is None:
            try:
                self.config.seeds_tags_dict = dict(
                    np.load(
                        self.config.dataset_dir / "seeds_tags_dict.npz",
                        allow_pickle=True,
                    )
                )
            except Exception:
                logging.error(
                    "Could not load seeds_tags_dict.npz."
                    "Either specify it in the active "
                    "learning settings or put it as "
                    f"seeds_tags_dict.npz in {self.config.dataset_dir}."
                )

    def setup_ensemble_and_datasets(self):
        """Setup ensemble and datasets."""
        if self.rank == 0:
            self._setup_ensemble()
            self._setup_datasets()
        else:
            self.train_dataset_len = None

        self._broadcast_dataset_info()

    def _setup_ensemble(self):
        """Setup the ensemble (rank 0 only)."""
        if self.config.enable_cueq_train:
            assert self.config.model_choice not in [
                "so3lr",
                "so3krates",
            ], (
                "CuEQ is not supported for So3krates/SO3LR."
            )
        self.ensemble = ensemble_from_folder(
            path_to_models=self.config.model_dir,
            device=self.config.device,
            dtype=dtype_mapping[self.config.dtype],
            convert_to_cueq=self.config.enable_cueq_train,
        )
        if self.config.model_settings["TRAINING"]["perform_finetuning"]:
            for tag, model in self.ensemble.items():
                self.ensemble[tag] = apply_finetuning_settings(
                    model=model,
                    model_settings=self.config.model_settings,
                )

        self.training_setups = get_ensemble_training_setups(
            ensemble=self.ensemble,
            model_settings=self.config.model_settings,
            restart=self.config.restart,
            checkpoints_dir=self.config.checkpoints_dir,
            mol_idxs=self.config.mol_idxs,
        )

    def _setup_datasets(self):
        """Setup initial datasets (rank 0 only)."""
        dataset_subdir = "final" if (
            self.config.restart or self.config.extend_existing_final_ds
        ) else "initial"
        
        log_message = (
            "Loading datasets from checkpoint."
            if self.config.restart
            else "Loading datasets."
        )

        logging.info(log_message)

        self.ensemble_ase_sets = load_ensemble_sets_from_folder(
            ensemble=self.ensemble,
            path_to_folder=Path(
                self.config.misc["dataset_dir"] + f"/{dataset_subdir}"
            ),
        )

        self.ensemble_model_sets = ase_to_model_ensemble_sets(
            ensemble_ase_sets=self.ensemble_ase_sets,
            z_table=self.z_table,
            r_max=self.config.r_max,
            r_max_lr=self.config.r_max_lr,
            seed=self.config.seed,
            key_specification=self.config.key_specification,
            all_heads=self.config.all_heads,
        )

        self.train_dataset_len = len(
            self.ensemble_ase_sets[list(self.ensemble.keys())[0]]["train"]
        )
        logging.info(
            f'Length of training set: {self.train_dataset_len}'
        )

    def _broadcast_dataset_info(self):
        """Broadcast dataset information to all ranks."""
        self.comm_handler.barrier()
        self.train_dataset_len = self.comm_handler.bcast(
            self.train_dataset_len, root=0
        )
        self.comm_handler.barrier()

    def _check_subset_size(self, data_set, set_subset_size):
        
        subset_size = (
                len(data_set)
                if len(data_set)
                < set_subset_size
                else set_subset_size
            )
        return subset_size

    def create_training_subset(
            self, 
            model_point,
            idx: int, 
            ):
        """
        Creates a single batch of specified size. It includes the newly sampled
        point and a random selection of points from the current training set.
        """
        
        assert self.config.train_subset_size is not None, (
            f"train_subset_size must be specified when using replay strategy "
            f"{self.config.replay_strategy}."
        )
        set_valid_size = (
            self.config.valid_subset_size if self.config.valid_subset_size is not None else np.inf
        )

        for tag in self.ensemble_ase_sets.keys():
            self.ensemble_model_sets[tag]["train_subset"] = {}
            self.ensemble_model_sets[tag]["valid_subset"] = {}

            if self.config.replay_strategy == "random_subset":
                train_subset_size = self._check_subset_size(
                    data_set=self.ensemble_model_sets[tag]["train"],
                    set_subset_size=self.config.train_subset_size
                )
                valid_subset_sizes = {}
                for head_name, head_data in self.ensemble_model_sets[
                    tag
                ]["valid"].items():
                    valid_subset_sizes[head_name] = self._check_subset_size(
                        data_set=head_data,
                        set_subset_size=set_valid_size
                        )
                   
                train_batch_size = min(
                    self.config.set_batch_size, train_subset_size
                )
                valid_batch_sizes = [
                    min(
                        self.config.set_valid_batch_size,
                        valid_subset_sizes[head_name],
                    )
                    for head_name in valid_subset_sizes.keys()
                ]
                valid_batch_size = min(valid_batch_sizes)
                
            random_sample_train = random.sample(
                self.ensemble_model_sets[tag]["train"],
                train_subset_size - 1,
            )
            
            random_samples_valid = {}
            for head_name, head_data in self.ensemble_model_sets[
                tag
            ]["valid"].items():
                random_samples_valid[head_name] = random.sample(
                    head_data,
                    valid_subset_sizes[head_name],
                )
                
            train_set = random_sample_train + model_point
            valid_set = random_samples_valid
            (
                self.ensemble_model_sets[tag]["train_subset"][idx],
                self.ensemble_model_sets[tag]["valid_subset"][idx],
            ) = create_dataloader(
                train_set,
                valid_set,
                train_batch_size,
                valid_batch_size,
            )
            
        logging.info(f"Using replay strategy: \"{self.config.replay_strategy}\". Sample sizes:")
        logging.info(f'Training set has {train_subset_size} point(s) with {len(self.ensemble_model_sets[tag]["train_subset"][idx])} batch(es).')
        
        for head_name, head_data in self.ensemble_model_sets[tag]["valid_subset"][idx].items():
            logging.info(f'Validation set for head "{head_name}" has {valid_subset_sizes[head_name]} point(s) with {len(head_data)} batch(es).')


class ALCalculatorMLFF:
    """Manages calculators for active learning procedure."""

    def __init__(
        self,
        config: ALConfiguration,
        ensemble_manager: ALEnsemble,
        comm_handler,
    ):
        self.config = config
        self.ensemble_manager = ensemble_manager
        self.comm_handler = comm_handler
        self.rank = comm_handler.get_rank()

        # model calculator
        self.models = None
        self.mlff_calc = None

        # Atomic energies handling
        self.update_atomic_energies = False
        self.ensemble_atomic_energies = None
        self.ensemble_atomic_energies_dict = None

    def setup_ml_calculators(
        self,
    ):
        """Setup all required calculators."""
        self.handle_atomic_energies()
        if self.rank == 0:
            self._setup_mlff_calculator()

    def handle_atomic_energies(self):
        """Handle atomic energies initialization."""
        self.update_atomic_energies = False
        if self.rank != 0:
            return

        if self.config.atomic_energies_dict is None:
            self._load_atomic_energies_from_source()
            self.update_atomic_energies = True
        else:
            self._use_specified_atomic_energies()

        self._log_atomic_energies()

    def _load_atomic_energies_from_source(self):
        """Load atomic energies from checkpoint or ensemble."""
        if self.config.restart:
            logging.info("Loading atomic energies from checkpoint.")
            self._load_from_checkpoint()
        else:
            logging.info("Loading atomic energies from existing ensemble.")
            self._load_from_ensemble()

    def _load_from_checkpoint(self):
        """Load atomic energies from checkpoint files."""
        (
            self.ensemble_atomic_energies,
            self.ensemble_atomic_energies_dict,
        ) = get_atomic_energies_from_pt(
            path_to_checkpoints=self.config.checkpoints_dir,
            z=self.ensemble_manager.z,
            seeds_tags_dict=self.config.seeds_tags_dict,
            dtype=self.config.dtype,
            model_choice=self.config.model_choice,
        )

    def _load_from_ensemble(self):
        """Load atomic energies from existing ensemble."""
        (
            self.ensemble_atomic_energies,
            self.ensemble_atomic_energies_dict,
        ) = get_atomic_energies_from_ensemble(
            ensemble=self.ensemble_manager.ensemble,
            z=self.ensemble_manager.z,
            model_choice=self.config.model_choice,
            dtype=self.config.dtype,
        )

    def _use_specified_atomic_energies(self):
        """Use user-specified atomic energies."""
        logging.info("Using specified atomic energies.")

        self.ensemble_atomic_energies_dict = {
            tag: self.config.atomic_energies_dict
            for tag in self.config.seeds_tags_dict.keys()
        }

        self.ensemble_atomic_energies = {
            tag: self._convert_atomic_energies_to_array(tag)
            for tag in self.config.seeds_tags_dict.keys()
        }

    def _convert_atomic_energies_to_array(self, tag: str) -> np.ndarray:
        """Convert atomic energies dict to numpy array."""
        energies_dict = self.ensemble_atomic_energies_dict[tag]
        return np.array(
            [energies_dict[z] for z in energies_dict.keys()],
            dtype=self.config.dtype,
        )

    def _log_atomic_energies(self):
        """Log atomic energies for first tag."""
        first_tag = list(self.config.seeds_tags_dict.keys())[0]
        first_energies = self.ensemble_atomic_energies_dict[first_tag]
        logging.info(f"Atomic energies: {first_energies}")

    def _setup_mlff_calculator(self):
        """Setup model calculator with ensemble models."""

        if self.config.use_foundational:
            logging.info("Using foundational model for MD.")
            foundational_model_settings = self.config.foundational_model_settings
            mace_model = foundational_model_settings['mace_model']
            # for propagation
            self.mlff_calc = mace_mp(
                model=mace_model,
                dispersion=False,
                default_dtype=self.config.dtype,
                device=self.config.device,
                enable_cueq=self.config.enable_cueq,
            )
            # for uncertainty estimation
            model_paths = list_files_in_directory(self.config.model_dir)
            self.models = [
                torch.load(
                    f=model_path,
                    map_location=self.config.device,
                    weights_only=False,
                )
                for model_path in model_paths
            ]
            self.mlff_calc_ensemble = MACECalculator(
                models=self.models,
                device=self.config.device,
                default_dtype=self.config.dtype,
                enable_cueq=self.config.enable_cueq,
            )
            
        else:
            logging.info("Using custom model for MD.")
            model_paths = list_files_in_directory(self.config.model_dir)
            self.models = [
                torch.load(
                    f=model_path,
                    map_location=self.config.device,
                    weights_only=False,
                )
                for model_path in model_paths if model_path.endswith('.model')
            ]
            if self.config.model_choice == "mace":
                self.mlff_calc = MACECalculator(
                    models=self.models,
                    device=self.config.device,
                    default_dtype=self.config.dtype,
                    enable_cueq=self.config.enable_cueq,
                )
            elif self.config.model_choice in ["so3lr", "so3krates"]:
                if self.config.use_multihead_model:
                    self.mlff_calc = MultiHeadSO3LRCalculator(
                        model=self.models,
                        device=self.config.device,
                        default_dtype=self.config.dtype,
                        r_max_lr=self.config.r_max_lr,
                        dispersion_energy_cutoff_lr_damping=self.config.dispersion_energy_cutoff_lr_damping,
                        compute_stress=self.config.compute_stress,
                        
                    )
                else:
                    self.mlff_calc = TorchkratesCalculator(
                        models=self.models,
                        device=self.config.device,
                        default_dtype=self.config.dtype,
                        r_max_lr=self.config.r_max_lr,
                        dispersion_energy_cutoff_lr_damping=self.config.dispersion_energy_cutoff_lr_damping,
                        compute_stress=self.config.compute_stress,
                    )


class ALMD:
    """Manages molecular dynamics setup and execution for active learning."""

    def __init__(
        self,
        config: ALConfiguration,
        state_manager: ALStateManager,
        comm_handler,
    ):
        self.config = config
        self.state_manager = state_manager
        self.comm_handler = comm_handler
        self.rank = comm_handler.get_rank()
        self.train_dataset_len = 0

        self.md_settings, _ = normalize_md_settings(
            md_settings=self.config.md_settings_raw,
            num_trajectories=self.config.num_trajectories,
        )
        if self.rank == 0:
            logging.info(
                f'Using following settings for MDs:'
            )
            log_yaml_block(
                "MD_SETTINGS",
                self.md_settings
            )
            
        self.md_drivers = {}

        # Setup MD modification if configured
        self._setup_md_modify()

        self._setup_uncertainty()

    def setup_md_drivers(self, trajectories: dict, mlff_calculator):
        """Setup MD drivers for all trajectories."""
        if self.rank != 0:
            return

        for trajectory in trajectories.values():
            trajectory.calc = mlff_calculator

        self.md_drivers = {
            trajectory_idx: self._setup_md_dynamics(
                atoms=trajectories[trajectory_idx],
                md_settings=self.md_settings[trajectory_idx],
                idx=trajectory_idx,
            )
            for trajectory_idx in self.state_manager.trajectories.keys()
        }

    def _setup_md_dynamics(
        self, atoms: ase.Atoms, md_settings: dict, idx: int
    ):
        """Setup ASE molecular dynamics object for given atoms."""
        stat_ensemble = md_settings["stat_ensemble"].lower()
        self._initialize_velocities(atoms, md_settings)
        dyn = self._create_dynamics_engine(atoms, md_settings, stat_ensemble, idx)
        return dyn

    def _create_dynamics_engine(
        self, atoms: ase.Atoms, md_settings: dict, stat_ensemble: str, idx: int
    ):
        """Create appropriate dynamics engine based on ensemble type."""
        if stat_ensemble == "nvt":
            return self._create_nvt_dynamics(atoms, md_settings, idx)
        elif stat_ensemble == "npt":
            return self._create_npt_dynamics(atoms, md_settings, idx)
        else:
            raise ValueError(f"Unsupported ensemble type: {stat_ensemble}")

    def _create_nvt_dynamics(
        self, atoms: ase.Atoms, md_settings: dict, idx: int
    ):
        """Create NVT dynamics engine."""
        thermostat = md_settings["thermostat"].lower()

        if thermostat == "langevin":
            return Langevin(
                atoms,
                timestep=md_settings["timestep"] * units.fs,
                friction=md_settings["friction"] / units.fs,
                temperature_K=md_settings["temperature"],
                rng=np.random.RandomState(md_settings["MD_seed"]),
            )
        else:
            raise ValueError(f"Unsupported thermostat: {thermostat}")

    def _create_npt_dynamics(
        self, atoms: ase.Atoms, md_settings: dict, idx: int
    ):
        """Create NPT dynamics engine."""
        barostat = md_settings["barostat"].lower()

        if barostat == "berendsen":
            return self._create_berendsen_npt(atoms, md_settings, idx)
        elif barostat == "mtk":
            return self._create_mkt_npt(atoms, md_settings, idx)
        else:
            raise ValueError(f"Unsupported barostat: {barostat}")

    def _create_berendsen_npt(
        self, atoms: ase.Atoms, md_settings: dict, idx: int
    ) -> NPTBerendsen:
        """Create Berendsen NPT dynamics engine."""
        npt_settings = {
            "atoms": atoms,
            "timestep": md_settings["timestep"] * units.fs,
            "temperature": md_settings["temperature"],
            "pressure_au": md_settings["pressure"] * units.Pascal,
        }

        # Add optional parameters
        optional_params = {
            "taup": ("taup", lambda x: x * units.fs),
            "taut": ("taut", lambda x: x * units.fs),
            "compressibility_au": ("compressibility_au", lambda x: x),
            "fixcm": ("fixcm", lambda x: x),
        }

        for param, (key, converter) in optional_params.items():
            if md_settings.get(param):
                npt_settings[key] = converter(md_settings[param])

        return NPTBerendsen(**npt_settings)

    def _create_mkt_npt(
        self,
        atoms: ase.Atoms,
        md_settings: dict,
        idx: int,
    ) -> MTKNPT:
        """Create MTK NPT dynamics engine."""
        npt_settings = {
            "atoms": atoms,
            "timestep": md_settings["timestep"] * units.fs,
            "temperature_K": md_settings["temperature"],
            "pressure_au": md_settings["pressure"] * units.Pascal,
            "tdamp": md_settings["tdamp"] * units.fs,
            "pdamp": md_settings["pdamp"] * units.fs,
            "tchain": md_settings["tchain"],
            "pchain": md_settings["pchain"],
            "tloop": md_settings["tloop"],
            "ploop": md_settings["ploop"],
        }
        return MTKNPT(**npt_settings)
    
    def _initialize_velocities(self, atoms: ase.Atoms, md_settings: dict):
        """
        Initialize Maxwell-Boltzmann velocity distribution
        if not restarting.
        """
        if not self.config.restart:
            MaxwellBoltzmannDistribution(
                atoms, temperature_K=md_settings["temperature"]
            )

    def _setup_md_modify(self):
        """Setup MD modification if configured."""
        self.md_mod_settings = self.md_settings.get("MODIFY", None)
        if self.md_mod_settings is not None:
            self.md_mod_metric = self.md_mod_settings.get("metric", None)
            assert (
                self.md_mod_metric is not None
            ), "No metric specified for MD modification."

            if self.md_mod_metric == "train_size":
                # This will need to access train_dataset_len from somewhere
                self.get_md_mod_metric = lambda: self.train_dataset_len

            self.md_modifier = ModifyMD(settings=self.md_mod_settings)
            self.mod_md = True
        else:
            self.mod_md = False

    def _setup_uncertainty(self):
        """Setup uncertainty calculation."""
        if self.config.mol_idxs is not None:
            self.get_uncertainty = MolForceUncertainty(
                mol_idxs=self.config.mol_idxs,
                uncertainty_type=self.config.uncertainty_type,
            )
        else:
            self.get_uncertainty = HandleUncertainty(
                uncertainty_type=self.config.uncertainty_type
            )

    def get_temperature_function(self, ensemble_type: str):
        """
        Get appropriate temperature extraction
        function for ensemble type.
        """
        temperature_functions = {
            "nvt": lambda driver: driver.temp,
            "npt": lambda driver: driver.temperature,
        }
        return temperature_functions.get(
            ensemble_type, lambda driver: driver.temp
        )


class ALRestart:
    """Handles active learning restart functionality."""

    def __init__(
        self,
        config: ALConfiguration,
        state_manager: ALStateManager,
        comm_handler,
        md_manager: ALMD,
        ensemble_manager: ALEnsemble,
    ):
        self.config = config
        self.state_manager = state_manager
        self.md_manager = md_manager
        self.comm_handler = comm_handler
        self.ensemble_manager = ensemble_manager
        self.rank = comm_handler.get_rank()

        if config.create_restart:
            self._initialize_restart_dict()

    def _initialize_restart_dict(self):
        """Initialize restart dictionary with default values."""
        self.al_restart_dict = {
            "trajectories": None,
            "MD_checkppoints": None,
            "trajectory_status": None,
            "trajectory_MD_steps": None,
            "trajectory_epochs": None,
            "ensemble_reset_opt": None,
            "ensemble_no_improvement": None,
            "ensemble_best_valid": None,
            "uncert_not_crossed": None,
            "current_valid_error": None,
            "threshold": None,
            "total_points_added": None,
            "train_points_added": None,
            "valid_points_added": None,
            "num_MD_limits_reached": None,
            "num_workers_training": None,
            "num_workers_waiting": None,
            "total_epoch": None,
            "check": None,
            "uncertainties": None,
            "al_done": False,
            "best_member": None,
            "last_point_added": None,
        }

        self.save_restart = False
        self._add_conditional_restart_keys()

    def _add_conditional_restart_keys(self):
        """Add conditional restart keys based on configuration."""

        if self.config.analysis:
            self.al_restart_dict.update(
                {
                    "t_intervals": None,
                    "analysis_checks": None,
                    "collect_losses": None,
                    "collect_thresholds": None,
                }
            )

    def handle_restart(self):
        """Handle active learning restart."""
        self._initialize_restart_attributes()

        if self.rank == 0:
            self._load_restart_checkpoint()

        self._broadcast_restart_state()

    def _initialize_restart_attributes(self):
        """Initialize all restart attributes to None."""
        base_attributes = [
            "trajectories",
            "MD_checkpoints",
            "trajectory_intermediate_epochs",
            "ensemble_reset_opt",
            "ensemble_no_improvement",
            "ensemble_best_valid",
            "threshold",
            "trajectory_status",
            "trajectory_MD_steps",
            "trajectory_total_epochs",
            "current_valid_error",
            "total_points_added",
            "train_points_added",
            "valid_points_added",
            "num_MD_limits_reached",
            "num_workers_training",
            "num_workers_waiting",
            "total_epoch",
            "check",
            "uncertainties",
            "uncert_not_crossed",
            "last_point_added",
        ]

        if self.config.analysis:
            analysis_attributes = [
                "t_intervals",
                "analysis_checks",
                "collect_losses",
                "collect_thresholds",
            ]
            if self.config.mol_idxs is not None:
                analysis_attributes.append("uncertainty_checks")
            base_attributes.extend(analysis_attributes)

        # Set all attributes to None in state manager
        for attr in base_attributes:
            setattr(self.state_manager, attr, None)

    def _load_restart_checkpoint(self):
        """Load restart data from checkpoint file."""
        logging.info("Restarting active learning procedure from checkpoint.")

        self.al_restart_dict = np.load(
            "restart/al/al_restart.npy", allow_pickle=True
        ).item()

        # Load all available keys from restart dict
        for key, value in self.al_restart_dict.items():
            if hasattr(self.state_manager, key):
                setattr(self.state_manager, key, value)
            if hasattr(self.md_manager, key):
                setattr(self.md_manager, key, value)

        # Special handling for uncertainty_checks
        if self.config.analysis and self.config.mol_idxs is not None:
            self.state_manager.uncertainty_checks = []
            
        # Special handling for subset data sets if applies
        if self.config.replay_strategy in [
            "random_subset"
            ]:
            for idx in range(self.config.num_trajectories):
                self.ensemble_manager.create_training_subset(
                    model_point=self.state_manager.last_point_added[idx],
                    idx=idx,
                )

    def _broadcast_restart_state(self):
        """Broadcast restart state from rank 0 to all processes."""
        self.comm_handler.barrier()

        # Define attributes to broadcast
        broadcast_attributes = [
            "trajectory_status",
            "trajectory_MD_steps",
            "trajectory_total_epochs",
            "current_valid_error",
            "total_points_added",
            "train_points_added",
            "valid_points_added",
            "num_MD_limits_reached",
            "num_workers_training",
            "num_workers_waiting",
            "total_epoch",
            "check",
            "uncertainties",
            "uncert_not_crossed",
        ]

        if self.config.analysis:
            analysis_broadcast = [
                "t_intervals",
                "analysis_checks",
                "collect_losses",
                "collect_thresholds",
            ]
            if self.config.mol_idxs is not None:
                analysis_broadcast.append("uncertainty_checks")
            broadcast_attributes.extend(analysis_broadcast)

        # Broadcast all attributes
        for attr in broadcast_attributes:
            value = getattr(self.state_manager, attr)
            broadcasted_value = self.comm_handler.bcast(value, root=0)
            setattr(self.state_manager, attr, broadcasted_value)

        self.comm_handler.barrier()

    def _is_nvt_or_npt_ensemble(self) -> bool:
        """Check if ensemble type is NVT or NPT."""
        return self.md_manager.md_settings["stat_ensemble"].lower() in [
            "nvt",
            "npt",
        ]

    def update_restart_dict(
        self, trajectories_keys, md_drivers, save_restart: str = None
    ):
        """Update restart dictionary with current state."""
        self._update_base_restart_attributes()
        self._update_analysis_attributes()

        if save_restart is not None:
            np.save(save_restart, self.al_restart_dict)

    def _update_base_restart_attributes(self):
        """Update base restart attributes."""
        base_attributes = [
            "trajectories",
            "MD_checkpoints",
            "trajectory_status",
            "trajectory_MD_steps",
            "trajectory_total_epochs",
            "trajectory_intermediate_epochs",
            "ensemble_reset_opt",
            "ensemble_no_improvement",
            "ensemble_best_valid",
            "current_valid_error",
            "threshold",
            "total_points_added",
            "train_points_added",
            "valid_points_added",
            "num_MD_limits_reached",
            "num_workers_training",
            "num_workers_waiting",
            "total_epoch",
            "check",
            "uncertainties",
            "uncert_not_crossed",
            "last_point_added",
        ]

        update_dict = {
            attr: getattr(self.state_manager, attr) for attr in base_attributes
        }
        self.al_restart_dict.update(update_dict)

    def _get_temperature_function(self, ensemble_type: str):
        """
        Get the appropriate temperature extraction
        function for the ensemble type.
        """
        temperature_functions = {
            "nvt": lambda driver: driver.temp,
            "npt": lambda driver: driver.temperature,
        }

        return temperature_functions.get(
            ensemble_type, lambda driver: driver.temp
        )

    def _update_analysis_attributes(self):
        """Update analysis-related restart attributes."""
        if not self.config.analysis:
            return

        analysis_update = {
            "t_intervals": self.state_manager.t_intervals,
            "analysis_checks": self.state_manager.analysis_checks,
            "collect_losses": self.state_manager.collect_losses,
            "collect_thresholds": self.state_manager.collect_thresholds,
        }

        if self.config.mol_idxs is not None:
            analysis_update["uncertainty_checks"] = (
                self.state_manager.uncertainty_checks
            )

        self.al_restart_dict.update(analysis_update)


class PrepareALProcedure:
    """
    Refactored active learning preparation
    class with clear separation of concerns.
    """

    def __init__(
        self,
        model_settings: dict,
        aimsPAX_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
        use_mpi: bool = True,
        comm_handler: CommHandler = None,
    ):
        """Initialize the active learning procedure."""
        # Setup communication first
        self._setup_communication(comm_handler, use_mpi)

        # Initialize configuration
        self.config = ALConfiguration(
            model_settings,
            aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )

        # Setup logging and folders
        self._setup_logging()
        self._create_folders()

        if self.rank == 0:
            logging.info("Initializing active learning procedure.")
            logging.info(f"Procedure runs on {self.world_size} workers.")
            logging.info(
                "Using followng settings for the active learning procedure:"
            )
            log_yaml_block(
                "ACTIVE_LEARNING", aimsPAX_settings["ACTIVE_LEARNING"]
            )
            logging.info(f"Using following settings for {self.config.model_choice}:")
            log_yaml_block(self.config.model_choice, model_settings)
        # Initialize all managers
        self.state_manager = ALStateManager(self.config, self.comm_handler)
        self.ensemble_manager = ALEnsemble(
            self.config, self.comm_handler
        )
        self.mlff_manager = ALCalculatorMLFF(
            self.config, self.ensemble_manager, self.comm_handler
        )
        self.md_manager = ALMD(
            self.config, self.state_manager, self.comm_handler
        )
        self.restart_manager = ALRestart(
            self.config,
            self.state_manager,
            self.comm_handler,
            self.md_manager,
            self.ensemble_manager
        )

        # Set random seed
        np.random.seed(self.config.seed)
        # Setup calculators and datasets
        self.ensemble_manager.setup_ensemble_and_datasets()
        self.mlff_manager.setup_ml_calculators()

        # Pass train_dataset_len to MD manager for modification
        self.md_manager.train_dataset_len = (
            self.ensemble_manager.train_dataset_len
        )

        if self.config.restart:
            self.restart_manager.handle_restart()

        else:
            self.state_manager.initialize_fresh_state(
                self.config.path_to_geometry
            )
        # assign mlff to checkpoints
        if self.rank == 0:
            for ckpt in self.state_manager.MD_checkpoints.values():
                ckpt.calc = self.mlff_manager.mlff_calc
                # compute forces for checkpoint
                ckpt.calc.calculate(ckpt)

            # Make sure state manager has access to seeds_tags_dict
            self.state_manager.seeds_tags_dict = self.config.seeds_tags_dict
        self.first_wait_after_restart = {
            idx: True for idx in range(self.config.num_trajectories)
        }
        if self.rank == 0:
            self.md_manager.setup_md_drivers(
                self.state_manager.trajectories, self.mlff_manager.mlff_calc
            )

        # TODO: Remove hardcode
        self.use_scheduler = False

    def _setup_communication(self, comm_handler: CommHandler, use_mpi: bool):
        """Setup MPI communication."""
        if comm_handler is not None:
            self.comm_handler = comm_handler
        else:
            self.comm_handler = CommHandler(use_mpi=use_mpi)

        self.rank = self.comm_handler.get_rank()
        self.world_size = self.comm_handler.get_size()

    def _setup_logging(self):
        """Setup logging configuration."""
        # Clear existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logger_level = (
            logging.DEBUG
            if self.config.model_settings["MISC"]["log_level"].lower()
            == "debug"
            else logging.INFO
        )

        self.log_dir = Path(self.config.log_dir)
        tools.setup_logger(
            level=logger_level,
            tag="active_learning",
            directory=self.log_dir,
        )

    def _create_folders(self):
        """Create necessary directories."""
        (self.config.dataset_dir / "final" / "training").mkdir(
            parents=True, exist_ok=True
        )
        (self.config.dataset_dir / "final" / "validation").mkdir(
            parents=True, exist_ok=True
        )

        if self.config.analysis:
            os.makedirs("analysis", exist_ok=True)

        if self.config.create_restart:
            os.makedirs("restart/al", exist_ok=True)

    def check_al_done(self) -> bool:
        """Check if active learning is done."""
        if self.config.create_restart:
            check = self.restart_manager.al_restart_dict.get("al_done", False)
            if check and self.rank == 0:
                logging.info(
                    "Active learning procedure is already done. Closing"
                )
            return check
        return False

    # TODO: check which to add or remove for convenience and readability
    @property
    def trajectories(self):
        return self.state_manager.trajectories

    @property
    def mlff_calc(self):
        return self.mlff_manager.mlff_calc

    @property
    def ensemble(self):
        return self.ensemble_manager.ensemble

    @property
    def get_uncertainty(self):
        return self.md_manager.get_uncertainty
