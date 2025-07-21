import os
from pathlib import Path
import torch
import numpy as np
import time
import shutil
from mace import tools
from mace.calculators import mace_mp
from aims_PAX.tools.utilities.data_handling import (
    create_dataloader,
    update_datasets,
    save_datasets,
    load_ensemble_sets_from_folder,
    ase_to_mace_ensemble_sets,
)
from aims_PAX.tools.utilities.utilities import (
    ensemble_training_setups,
    setup_ensemble_dicts,
    update_model_auxiliaries,
    save_checkpoint,
    setup_mace_training,
    Z_from_geometry_in,
    get_atomic_energies_from_pt,
    create_seeds_tags_dict,
    create_ztable,
    save_ensemble,
    setup_logger,
    CommHandler,
    AIMSControlParser,
    dtype_mapping,
)
from aims_PAX.tools.utilities.parsl_tools import (
    recalc_aims_parsl,
    handle_parsl_logger,
    prepare_parsl,
)
from aims_PAX.tools.train_epoch_mace import (
    train_epoch,
    validate_epoch_ensemble,
)
import ase
from ase.io import read
import logging
import random
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.npt import NPT
from ase import units
from contextlib import nullcontext
import sys
try:
    import asi4py
except Exception as e:
    asi4py = None
try:
    import parsl
except ImportError:
    parsl = None


sys.stdout.flush()


class PrepareInitialDatasetProcedure:
    """
    Class to prepare the inital dataset generation procedure for
    active learning. It handles all the input files, prepares the
    calculators, models, directories etc.
    """

    def __init__(
        self,
        mace_settings: dict,
        al_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
        use_mpi: bool = True,
    ) -> None:
        """
        Args:
            mace_settings (dict): Settings for the MACE model and its training.
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

        self.log_dir = Path(mace_settings["GENERAL"]["log_dir"])
        logger_level = (
            logging.DEBUG
            if mace_settings["MISC"]["log_level"].lower() == "debug"
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

        self.control_parser = AIMSControlParser()
        self._handle_mace_settings(mace_settings)
        self._handle_al_settings(al_settings)
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
                    allow_pickle=True
                ).item()
            except FileNotFoundError:
                logging.error(
                    "Restart file under 'restart/initial_ds/"
                    "initial_ds_restart.npy' not found."
                )
                raise
            self.atoms = self.init_ds_restart_dict["last_geometry"]
            self.step = self.init_ds_restart_dict["step"]
        else:
            self.atoms = read(path_to_geometry)
            self.step = 0

        self.z = Z_from_geometry_in()
        self.n_atoms = len(self.z)
        self.z_table = create_ztable(self.z)

        np.random.seed(self.seed)
        random.seed(self.seed)
        self.ensemble_seeds = np.random.randint(
            0, 10000, size=self.ensemble_size
        )
        self.seeds_tags_dict = create_seeds_tags_dict(
            seeds=self.ensemble_seeds,
            mace_settings=self.mace_settings,
            al_settings=self.al,
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
                mace_settings=self.mace_settings,
                z_table=self.z_table,
                ensemble_atomic_energies_dict=self.ensemble_atomic_energies_dict,
            )
            self.training_setups = ensemble_training_setups(
                ensemble=self.ensemble,
                mace_settings=self.mace_settings,
                restart=self.restart,
                checkpoints_dir=self.checkpoints_dir,
                al_settings=self.al,
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
        # we create a ASE and MACE dataset because it makes 
        # conversion and saving easier
        if self.rank == 0:
            if self.restart:
                self.ensemble_ase_sets = load_ensemble_sets_from_folder(
                    ensemble=self.ensemble,
                    path_to_folder=self.dataset_dir / "initial",
                )
                self.ensemble_mace_sets = ase_to_mace_ensemble_sets(
                    ensemble_ase_sets=self.ensemble_ase_sets,
                    z_table=self.z_table,
                    r_max=self.r_max,
                    seed=self.seed,
                )

            else:
                self.ensemble_mace_sets, self.ensemble_ase_sets = (
                    {
                        tag: {"train": [], "valid": []}
                        for tag in self.ensemble.keys()
                    },
                    {
                        tag: {"train": [], "valid": []}
                        for tag in self.ensemble.keys()
                    },
                )
        # analysis means that at certain MD steps we
        # compute ab initio data, ML predictions and
        # other metrics for analysis of the procedure
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

    def _handle_mace_settings(self, mace_settings: dict) -> None:
        """
        Saves the MACE settings to class attributes.
        TODO: Create function to check if all necessary settings are present
        and fall back to defaults if not.

        Args:
            mace_settings (dict): Dictionary containing the MACE settings.
        """

        self.mace_settings = mace_settings
        self.seed = self.mace_settings["GENERAL"]["seed"]
        self.r_max = self.mace_settings["ARCHITECTURE"]["r_max"]
        self.set_batch_size = self.mace_settings["TRAINING"]["batch_size"]
        self.set_valid_batch_size = self.mace_settings["TRAINING"][
            "valid_batch_size"
        ]
        self.checkpoints_dir = (
            self.mace_settings["GENERAL"]["checkpoints_dir"] + "/initial"
        )
        self.scaling = self.mace_settings["TRAINING"]["scaling"]
        self.dtype = self.mace_settings["GENERAL"]["default_dtype"]
        torch.set_default_dtype(dtype_mapping[self.dtype])
        self.device = self.mace_settings["MISC"]["device"]
        self.atomic_energies_dict = self.mace_settings["ARCHITECTURE"].get(
            "atomic_energies", None
        )
        self.compute_stress = self.mace_settings.get("compute_stress", False)
        self.properties = ["energy", "forces"]
        if self.compute_stress:
            self.properties.append("stress")

    def _handle_al_settings(self, al_settings: dict) -> None:
        """
        Saves the active learning settings to class attributes.
        TODO: Create function to check if all necessary settings are present
        and fall back to defaults if not.

        Args:
            al_settings (dict): Dictionary containing the active 
                                learning settings.
        """

        self.al = al_settings["ACTIVE_LEARNING"]
        self.misc = al_settings["MISC"]
        self.md_settings = al_settings["MD"]
        self.cluster_settings = al_settings.get("CLUSTER", None)
        self.ensemble_size = self.al["ensemble_size"]
        self.desired_acc = self.al["desired_acc"]
        self.lamb = self.al["lambda"]
        self.n_samples = self.al["n_samples"]
        self.max_initial_epochs = self.al["max_initial_epochs"]
        self.max_final_epochs = self.al["max_final_epochs"]
        self.valid_skip = self.al["valid_skip"]
        self.skip_step = self.al["skip_step_initial"]
        self.intermediate_epochs = self.al["intermediate_epochs"]
        self.valid_ratio = self.al["valid_ratio"]
        self.ASI_path = self.al["aims_lib_path"]
        self.species_dir = self.al["species_dir"]
        self.analysis = self.al.get("analysis", False)
        self.margin = self.al.get("margin", 0.001)
        if not self.al["scheduler_initial"]:
            self.mace_settings["lr_scheduler"] = None

        self.initial_foundational_size = self.al.get(
            "initial_foundational_size", None
        )
        if self.initial_foundational_size is not None:
            assert self.al["initial_foundational_size"] in (
                "small",
                "medium",
                "large",
            ), "Initial foundational size not recognized."

        self.restart = os.path.exists(
            "restart/initial_ds/initial_ds_restart.npy"
        )
        self.create_restart = self.misc.get("create_restart", False)
        if self.create_restart:
            self.init_ds_restart_dict = {
                "last_geometry": None,
                "last_initial_losses": None,
                "initial_ds_done": False,
            }

    def _update_restart_dict(self):
        self.init_ds_restart_dict["last_geometry"] = self.last_point
        self.init_ds_restart_dict["step"] = self.step
        if self.analysis:
            self.init_ds_restart_dict["last_initial_losses"] = (
                self.collect_losses
            )

    def _create_folders(self):
        """
        Creates the necessary directories for saving the datasets.
        """
        self.dataset_dir = Path(self.al["dataset_dir"])
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

    def _handle_aims_settings(self, path_to_control: str):
        """
        Parses the AIMS control file to get the settings for the AIMS 
        calculator.

        Args:
            path_to_control (str): Path to the AIMS control file.
            species_dir (str): Path to the species directory of AIMS.
        """

        self.aims_settings = self.control_parser(path_to_control)
        self.aims_settings["compute_forces"] = True
        self.aims_settings["species_dir"] = self.species_dir
        self.aims_settings["postprocess_anyway"] = (
            True  # this is necesssary to check for convergence in ASI
        )

    def _setup_md(self, atoms: ase.Atoms, md_settings: dict):
        """
        Sets up the ASE molecular dynamics object for the atoms object.
        TODO: Add more flexibility and support for other settings

        Args:
            atoms (ase.Atoms): Atoms to be propagated.
            md_settings (dict): Dictionary containing the MD settings.

        Returns:
            ase.md.MolecularDynamics: ASE MD engine.
        """

        if md_settings["stat_ensemble"].lower() == "nvt":
            if md_settings["thermostat"].lower() == "langevin":
                dyn = Langevin(
                    atoms,
                    timestep=md_settings["timestep"] * units.fs,
                    friction=md_settings["friction"] / units.fs,
                    temperature_K=md_settings["temperature"],
                    rng=np.random.RandomState(md_settings["seed"]),
                )
        elif md_settings["stat_ensemble"].lower() == "npt":
            if md_settings["barostat"].lower() == "berendsen":
                npt_settings = {
                    "atoms": atoms,
                    "timestep": md_settings["timestep"] * units.fs,
                    "temperature": md_settings["temperature"],
                    "pressure_au": md_settings["pressure_au"],
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
                    "externalstress": md_settings["externalstress"]
                    * units.bar,
                    "ttime": md_settings["ttime"] * units.fs,
                    "pfactor": md_settings["pfactor"] * units.fs,
                }

                if md_settings.get("mask", False):
                    npt_settings["mask"] = md_settings["mask"]

                dyn = NPT(**npt_settings)

        if not self.restart:
            MaxwellBoltzmannDistribution(
                atoms, temperature_K=md_settings["temperature"]
            )

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
                    )
                else:

                    logging.info(
                        "No atomic specified. Initializing with 0 and fit to training data."
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
        energy and forces in a MACE readable format.

        Args:
            atoms (ase.Atoms): Atoms object to be propagated.
            dyn (ase.md.MolecularDynamics): ASE MD engine.
        Returns:
            ase.Atoms: Atoms object with the energy and forces saved in the
                        MACE readable format.
        """

        dyn.run(self.skip_step)
        current_energy = np.array(atoms.get_potential_energy())
        current_forces = np.array(atoms.get_forces())
        current_point = atoms.copy()
        # MACE reads energies and forces from the info & arrays dictionary
        current_point.info["REF_energy"] = current_energy
        current_point.arrays["REF_forces"] = current_forces

        if self.create_restart:
            current_point.set_velocities(atoms.get_velocities())
            current_point.set_masses(atoms.get_masses())
            self.last_point = current_point

        return current_point

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
                    self.ensemble_mace_sets[tag]["train_loader"],
                    self.ensemble_mace_sets[tag]["valid_loader"],
                ) = create_dataloader(
                    self.ensemble_mace_sets[tag]["train"],
                    self.ensemble_mace_sets[tag]["valid"],
                    self.set_batch_size,
                    self.set_valid_batch_size,
                )

                update_model_auxiliaries(
                    model=model,
                    mace_sets=self.ensemble_mace_sets[tag],
                    atomic_energies_list=self.ensemble_atomic_energies[tag],
                    scaling=self.scaling,
                    update_atomic_energies=self.update_atomic_energies,
                    z_table=self.z_table,
                    atomic_energies_dict=self.ensemble_atomic_energies_dict[
                        tag
                    ],
                    dtype=self.dtype,
                    device=self.device,
                )

            self.training_setups_convergence = {}
            for tag in self.ensemble.keys():
                self.training_setups_convergence[tag] = setup_mace_training(
                    settings=self.mace_settings,
                    model=self.ensemble[tag],
                    tag=tag,
                    restart=self.restart,
                    convergence=True,
                    checkpoints_dir=self.checkpoints_dir,
                    al_settings=self.al,
                )
            best_valid_loss = np.inf
            epoch = 0
            if self.restart:
                epoch = self.training_setups_convergence[
                    list(self.ensemble.keys())[0]
                ]["epoch"]

            patience = self.al["patience"]
            no_improvement = 0
            ensemble_valid_losses = {
                tag: np.inf for tag in self.ensemble.keys()
            }
            for j in range(self.max_final_epochs):
                for tag, model in self.ensemble.items():
                    logger = tools.MetricsLogger(
                        directory=self.mace_settings["GENERAL"]["results_dir"],
                        tag=tag + "_train",
                    )
                    train_epoch(
                        model=model,
                        train_loader=self.ensemble_mace_sets[tag][
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
                    or epoch == self.max_final_epochs - 1
                ):
                    ensemble_valid_losses, valid_loss, _ = (
                        validate_epoch_ensemble(
                            ensemble=self.ensemble,
                            training_setups=self.training_setups_convergence,
                            ensemble_set=self.ensemble_mace_sets,
                            logger=logger,
                            log_errors=self.mace_settings["MISC"][
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
                                        self.mace_settings["GENERAL"][
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
                if no_improvement > patience:
                    logging.info(
                        f"No improvements for {patience} epochs. "
                        "Training converged. Best model based on validation"
                        " loss saved"
                    )
                    break
                if j == self.max_final_epochs - 1:
                    logging.info(
                        f"Maximum number of epochs reached. Best model "
                        f"(Epoch {best_epoch}) based on validation loss saved."
                    )
                    break


class InitialDatasetProcedure(PrepareInitialDatasetProcedure):
    """
    Class to generate the initial dataset for the active learning procedure.
    Handles the molecular dynamics simulations, the sampling of points, the 
    training of the ensemble members and the saving of the datasets.

    This is the base class for the serial, parallel, and PARSL version of 
    this workflow.
    """

    def _sample_points(self):
        """
        Dummy function to sample points. This function should be overwritten in the
        derived classes.
        """
        raise NotImplementedError

    def _train(self) -> bool:
        """
        Trains the model(s) on the sampled points and updates the
        average number of neighbors, shifts and the scaling factor
        for each ensemble member.

        Returns:
            bool: Returns True if the maximum number of epochs is reached.
        """
        if self.rank == 0:
            random.shuffle(self.sampled_points)
            # each ensemble member collects their respective points
            for number, (tag, model) in enumerate(self.ensemble.items()):

                member_points = self.sampled_points[
                    self.n_samples * number: self.n_samples * (number + 1)
                ]

                (
                    self.ensemble_ase_sets[tag],
                    self.ensemble_mace_sets[tag],
                ) = update_datasets(
                    new_points=member_points,
                    mace_set=self.ensemble_mace_sets[tag],
                    ase_set=self.ensemble_ase_sets[tag],
                    valid_split=self.valid_ratio,
                    z_table=self.z_table,
                    seed=self.seed,
                    r_max=self.r_max,
                )

                batch_size = (
                    1
                    if len(self.ensemble_mace_sets[tag]["train"])
                    < self.set_batch_size
                    else self.set_batch_size
                )
                valid_batch_size = (
                    1
                    if len(self.ensemble_mace_sets[tag]["valid"])
                    < self.set_valid_batch_size
                    else self.set_valid_batch_size
                )

                (
                    self.ensemble_mace_sets[tag]["train_loader"],
                    self.ensemble_mace_sets[tag]["valid_loader"],
                ) = create_dataloader(
                    self.ensemble_mace_sets[tag]["train"],
                    self.ensemble_mace_sets[tag]["valid"],
                    batch_size,
                    valid_batch_size,
                )
                # because we are continously training the model we
                # have to update the average number of neighbors, shifts
                # and the scaling factor continously as well
                update_model_auxiliaries(
                    model=model,
                    mace_sets=self.ensemble_mace_sets[tag],
                    atomic_energies_list=self.ensemble_atomic_energies[tag],
                    scaling=self.scaling,
                    update_atomic_energies=self.update_atomic_energies,
                    z_table=self.z_table,
                    atomic_energies_dict=self.ensemble_atomic_energies_dict[
                        tag
                    ],
                    dtype=self.dtype,
                    device=self.device,
                )
                logging.info(
                    f"Training set size for '{tag}': "
                    f"{len(self.ensemble_mace_sets[tag]['train'])}; Validation"
                    f" set size: {len(self.ensemble_mace_sets[tag]['valid'])}."
                )

            logging.info("Training.")
            ensemble_valid_losses = {
                tag: np.inf for tag in self.ensemble.keys()
            }
            for _ in range(self.intermediate_epochs):
                # each member gets trained individually
                for tag, model in self.ensemble.items():

                    logger = tools.MetricsLogger(
                        directory=self.mace_settings["GENERAL"]["results_dir"],
                        tag=tag + "_train",
                    )
                    train_epoch(
                        model=model,
                        train_loader=self.ensemble_mace_sets[tag][
                            "train_loader"
                        ],
                        loss_fn=self.training_setups[tag]["loss_fn"],
                        optimizer=self.training_setups[tag]["optimizer"],
                        lr_scheduler=self.training_setups[tag]["lr_scheduler"],
                        epoch=self.epoch,
                        start_epoch=0,
                        valid_loss=ensemble_valid_losses[tag],
                        logger=logger,
                        device=self.training_setups[tag]["device"],
                        max_grad_norm=self.training_setups[tag][
                            "max_grad_norm"
                        ],
                        output_args=self.training_setups[tag]["output_args"],
                        ema=self.training_setups[tag]["ema"],
                    )
                # the validation errors are averages over the ensemble members
                if (
                    self.epoch % self.valid_skip == 0
                    or (self.epoch + 1) % self.valid_skip == 0
                ):
                    (
                        ensemble_valid_losses,
                        valid_loss,
                        metrics,
                    ) = validate_epoch_ensemble(
                        ensemble=self.ensemble,
                        training_setups=self.training_setups,
                        ensemble_set=self.ensemble_mace_sets,
                        logger=logger,
                        log_errors=self.mace_settings["MISC"]["error_table"],
                        epoch=self.epoch,
                    )
                    self.current_valid = metrics["mae_f"]

                    if self.analysis:
                        self._handle_analysis(
                            valid_loss=valid_loss,
                            ensemble_valid_losses=ensemble_valid_losses,
                        )

                    for tag, model in self.ensemble.items():
                        save_checkpoint(
                            checkpoint_handler=self.training_setups[tag][
                                "checkpoint_handler"
                            ],
                            training_setup=self.training_setups[tag],
                            model=model,
                            epoch=self.epoch,
                            keep_last=False,
                        )
                        save_datasets(
                            self.ensemble,
                            self.ensemble_ase_sets,
                            path=self.dataset_dir / "initial",
                            initial=True,
                        )
                    if self.create_restart:
                        self._update_restart_dict()
                        np.save(
                            "restart/initial_ds/initial_ds_restart.npy",
                            self.init_ds_restart_dict,
                        )
                    if self.desired_acc * self.lamb >= self.current_valid:
                        logging.info(
                            f"Accuracy criterion reached at step {self.step}."
                        )
                        logging.info(
                            f"Criterion: {self.desired_acc * self.lamb}; Current accuracy: {self.current_valid}."
                        )

                        break

                self.epoch += 1

            if (
                self.epoch == self.max_initial_epochs
            ):  # TODO: change to a different variable (shares with al-algo right now)
                logging.info(f"Maximum number of epochs reached.")
                return True

    def _sample_and_train(self):
        """
        Combines the sampling of points and the training of the ensemble members
        in one method (easier for overwritting for derived classes).
        """
        if self.rank == 0:
            logging.info(f"Sampling new points at step {self.step}.")
        self.sampled_points = []
        # in case SCF fails to converge no point is returned
        while len(self.sampled_points) == 0:
            self.sampled_points = self._sample_points()

        self.step += 1
        self._train()

    def _setup_calcs(self):
        """
        Dummy function to set up the calculators. This function should be overwritten in the
        derived classes.
        """
        raise NotImplementedError

    def _close_aims(self):
        """
        Dummy function to close the AIMS calculators. This function should be overwritten in the
        derived classes.
        """
        raise NotImplementedError

    def _handle_analysis(
        self,
        valid_loss: float,
        ensemble_valid_losses: dict,
        save_path: str = "analysis/initial_losses.npz",
    ):
        """
        Collects number of epochs, average validation loss and
        per ensemble member validation losses and saves in a
        npz file.

        Args:
            valid_loss (float): Averaged validation loss over the ensemble.
            ensemble_valid_losses (dict): Per ensemble member 
                                                    validation losses.
            save_path (str, optional): Path to save the analysis data. 
                    Defaults to "analysis/initial_losses.npz".
        """
        self.collect_losses["epoch"].append(self.epoch)
        self.collect_losses["avg_losses"].append(valid_loss)
        self.collect_losses["ensemble_losses"].append(ensemble_valid_losses)
        np.savez(save_path, **self.collect_losses)

    def run(self):
        """
        Main function to run the initial dataset generation procedure.
        It samples points and trains the ensemble members until the 
        stopping criterion is met.

        """

        # initializing md and FHI aims
        self.dyn = self._setup_md(self.atoms, md_settings=self.md_settings)

        self._setup_calcs()

        self.current_valid = np.inf
        # criterion for initial dataset is multiple of the desired accuracy
        # TODO: add maximum initial dataset len criterion
        while (
            self.desired_acc * self.lamb <= self.current_valid
            and self.epoch < self.max_initial_epochs
        ):

            self._sample_and_train()
            # only one worker is doing the training right now,
            # so we have to broadcast the criterion so they
            # don't get stuck in the while loop
            self.comm_handler.barrier()
            self.current_valid = self.comm_handler.bcast(
                self.current_valid, root=0
            )
            self.epoch = self.comm_handler.bcast(self.epoch, root=0)
            self.comm_handler.barrier()

        if self.rank == 0:

            save_ensemble(
                ensemble=self.ensemble,
                training_setups=self.training_setups,
                mace_settings=self.mace_settings,
            )

            if self.create_restart:
                self._update_restart_dict()
                self.init_ds_restart_dict["initial_ds_done"] = True
                np.save(
                    "restart/initial_ds/initial_ds_restart.npy",
                    self.init_ds_restart_dict,
                )
        self.logger.handlers.clear()
        self._close_aims()
        return 0


class InitialDatasetAIMD(InitialDatasetProcedure):
    """
    Class to generate the initial dataset for the active learning procedure.
    Handles the molecular dynamics simulations, the sampling of points, the
    training of the ensemble members and the saving of the datasets.

    Uses ab initio MD to sample points. Runs serially.

    """

    def _sample_points(self) -> list:
        """
        Samples geometries solely using AIMD.

        Returns:
            list: List of ASE Atoms objects.
        """
        return [
            self._run_MD(atoms=self.atoms, dyn=self.dyn)
            for _ in range(self.ensemble_size * self.n_samples)
        ]

    def _setup_calcs(self):
        """
        Sets up the calculators for the initial dataset generation.
        In this case it sets up the AIMS calculators for AIMD.
        """
        self.atoms.calc = self._setup_aims_calculator(self.atoms)

    def _close_aims(self):
        """
        Kills the AIMS calculators.
        """
        self.atoms.calc.close()


class InitialDatasetFoundational(InitialDatasetProcedure):
    """
    Class to generate the initial dataset for the active learning procedure.
    Handles the molecular dynamics simulations, the sampling of points, the
    training of the ensemble members and the saving of the datasets.

    Uses a "foundational" model to sample points. These are then recomputed
    using DFT. Runs serially.
    """

    def _setup_foundational(self):
        """
        Creates the foundational model for sampling.

        Returns:
            ase.Calculator: ASE calculator object.
        """
        return mace_mp(
            model=self.initial_foundational_size,
            dispersion=False,
            default_dtype=self.dtype,
            device=self.device,
        )

    def _recalc_aims(
            self,
            current_point: ase.Atoms
            ) -> ase.Atoms:
        """
        Recalculates the energies and forces of the current point using
        the AIMS calculator. If the SCF is converged, it saves the energy
        and forces (and stress) in the MACE readable format.
        If not, it returns None.

        Args:
            current_point (ase.Atoms): System to recompute.

        Returns:
            ase.Atoms: Atoms object containing generated DFT data.
        """
        self.aims_calc.calculate(current_point, properties=self.properties)
        if self.aims_calc.asi.is_scf_converged:
            current_point.info["REF_energy"] = self.aims_calc.results["energy"]
            current_point.arrays["REF_forces"] = self.aims_calc.results[
                "forces"
            ]
            if self.compute_stress:
                current_point.info["REF_stress"] = self.aims_calc.results[
                    "stress"
                ]
            return current_point
        else:
            if self.rank == 0:
                logging.info("SCF not converged.")
            return None

    def _md_w_foundational(
        self,
    ):
        """
        Samples points using the foundational model.
        """

        self.comm_handler.barrier()
        self.sampled_points = []
        if self.rank == 0:
            for _ in range(self.ensemble_size * self.n_samples):
                current_point = self._run_MD(self.atoms, self.dyn)
                self.sampled_points.append(current_point)
            logging.info(
                f"Sampled {len(self.sampled_points)} points using foundational model."
            )
        self.comm_handler.barrier()
        self.sampled_points = self.comm_handler.bcast(
            self.sampled_points, root=0
        )
        self.comm_handler.barrier()

    def _sample_points(self) -> list:
        """
        Samples geometries using foundational model and recalculates
        the energies and forces with DFT.

        Returns:
            list: List of ASE Atoms objects.
        """
        self._md_w_foundational()
        if self.rank == 0:
            logging.info("Recalculating energies and forces with DFT.")
        recalculated_points = []
        for atoms in self.sampled_points:
            temp = self._recalc_aims(atoms)
            if temp is not None:
                recalculated_points.append(temp)
        return recalculated_points

    def _setup_calcs(self):
        """
        Sets up the calculators for the initial dataset generation.
        In this case it sets up the AIMS calculators for recalculating
        the energies and forces and the foundational model for MD.
        """
        self.aims_calc = self._setup_aims_calculator(self.atoms)
        if self.rank == 0:
            logging.info(
                f"Initial dataset generation with foundational model of size: {self.initial_foundational_size}."
            )
            self.atoms.calc = self._setup_foundational()

    def _close_aims(self):
        """
        Kills the AIMS calculator.
        """
        self.aims_calc.close()


class InitialDatasetFoundationalParallel(InitialDatasetFoundational):
    """
    Class to generate the initial dataset for the active learning procedure.
    Handles the molecular dynamics simulations, the sampling of points, the
    training of the ensemble members and the saving of the datasets.

    Uses a "foundational" model to sample points. These are then recomputed
    using DFT. Runs in parallel using MPI. The MD using the foundational 
    model is propagted while DFT is being run.

    !!! WARNING: Not recommended (especially for large systems) as the
        model can generate strange geometries when run to long, which 
        can happen when DFT calculations take too much time. Speedup is
        modest as DFT calculations are processed one at a time. Ideally
        use the PARSL version. !!!
    """

    def __init__(
        self,
        mace_settings: dict,
        al_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
    ):

        # this is necessary because of the way the MPI communicator is split
        super().__init__(
            mace_settings=mace_settings,
            al_settings=al_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )
        if self.rank == 0:
            logging.warning(
                "Not recommended (especially for large systems) as the "
                "model can generate strange geometries when run to long, which"
                "can happen when DFT calculations take too much time. Speedup "
                "is modest as DFT calculations are processed one at a time. "
                "Ideally use the PARSL version."
            )
        # one for ML and one for DFT
        if self.rank == 0:
            self.color = 0
        else:
            self.color = 1

        self.comm = self.comm_handler.comm.Split(
            color=self.color, key=self.rank
        )

    def _close_aims(self):
        # this is just to overwrite the function in the parent class
        # due to the communicators we are closing it inside the
        # sample_and_train function
        return None

    def _setup_aims_calculator(
        self,
        atoms: ase.Atoms,
    ) -> ase.Atoms:
        """
        Attaches the AIMS calculator to the atoms object. Uses the AIMS
        settings from the control.in to set up the calculator.

        Args:
            atoms (ase.Atoms): Atoms object to attach the calculator to.
            pbc (bool, optional): If periodic boundry conditions are required
            or not. Defaults to False.

        Returns:
            ase.Atoms: Atoms object with the calculator attached.
        """
        aims_settings = self.aims_settings.copy()
        # only one communictor initializes aims
        if self.color == 1:
            self.properties = ["energy", "forces"]
            if self.compute_stress:
                self.properties.append("stress")

            def init_via_ase(asi):
                from ase.calculators.aims import Aims, AimsProfile

                aims_settings["profile"] = AimsProfile(
                    command="asi-doesnt-need-command"
                )
                calc = Aims(**aims_settings)
                calc.write_inputfiles(asi.atoms, properties=self.properties)

            calc = ASI_ASE_calculator(
                self.ASI_path, init_via_ase, self.comm, atoms
            )
            return calc
        else:
            return None

    def _sample_and_train(self) -> list:
        """
        Samples points using the foundational models, computes
        DFT data in parallel and trains the ensemble members.
        Contains all the MPI communications.

        Returns:
            list: List of ASE Atoms objects with the sampled points
                and their DFT data.
        """
        self.sampled_points = []

        # TODO: add stress
        temp_sampled_geometries = []
        temp_sampled_forces = []
        temp_sampled_energies = []
        self.req_geometries, self.req_energies, self.req_forces = (
            None,
            None,
            None,
        )

        self.req = None  # handling data communication
        self.criterion_req = (
            None  # handling the communication regarding stopping
        )
        current_point = None
        recieved_points = None
        criterion_met = False
        self.atoms_dummy = self.atoms.copy()

        if self.rank == 0:
            logging.info("Starting sampling and training using parallel mode.")

        while not criterion_met:
            if self.color == 0:
                current_point = self._run_MD(self.atoms, self.dyn)
                # TODO: also send cell and pbc
                geometry = current_point.get_positions()
                # using isend to create a queue of messages
                sample_send = self.comm_handler.comm.Isend(
                    geometry, dest=1, tag=96
                )
                sample_send.Wait()

                # creating requests if there are none
                if (
                    self.req_geometries is None
                    and self.req_energies is None
                    and self.req_forces is None
                ):
                    # creates buffers in memory for receiving data
                    buf_geometries, buf_energies, buf_forces = (
                        np.zeros(
                            (
                                self.n_samples * self.ensemble_size,
                                len(self.atoms),
                                3,
                            ),
                            dtype=float,
                        ),
                        np.zeros(
                            self.n_samples * self.ensemble_size, dtype=float
                        ),
                        np.zeros(
                            (
                                self.n_samples * self.ensemble_size,
                                len(self.atoms),
                                3,
                            ),
                            dtype=float,
                        ),
                    )
                    # non-blocking recieve for data
                    self.req_geometries = self.comm_handler.comm.Irecv(
                        buf=buf_geometries, source=1, tag=2210
                    )
                    self.req_energies = self.comm_handler.comm.Irecv(
                        buf=buf_energies, source=1, tag=2211
                    )
                    self.req_forces = self.comm_handler.comm.Irecv(
                        buf=buf_forces, source=1, tag=2212
                    )

                else:
                    # listening for data
                    status_geometries = (
                        self.req_geometries.Test()
                    )  # non-blocking recieve
                    status_energies = (
                        self.req_energies.Test()
                    )  # non-blocking recieve
                    status_forces = (
                        self.req_forces.Test()
                    )  # non-blocking recieve

                    if status_energies and status_forces and status_geometries:
                        recieved_points = []

                        for i in range(self.n_samples * self.ensemble_size):
                            temp = self.atoms_dummy.copy()
                            temp.set_positions(buf_geometries[i])
                            temp.info["REF_energy"] = buf_energies[i]
                            temp.arrays["REF_forces"] = buf_forces[i]
                            recieved_points.append(temp)

                        self.req_geometries = None
                        self.req_energies = None
                        self.req_forces = None

                        # checking if the criterion is met
                        criterion_met = (
                            self.desired_acc * self.lamb >= self.current_valid
                            or self.epoch >= self.max_initial_epochs
                        )
                        if criterion_met:
                            # instructs the DFT worker to stop when
                            # the criterion is met
                            for dest in range(
                                1, self.comm_handler.comm.Get_size()
                            ):
                                self.criterion_send = (
                                    self.comm_handler.comm.isend(
                                        None, dest=dest, tag=2305
                                    )
                                )
                                self.criterion_send.Wait()
                            break
                        logging.info(
                            "Recieved points from DFT worker; training."
                        )
                        self.sampled_points.extend(recieved_points)
                        recieved_points = None
                        self._train()
                        logging.info("Training done, going back to sampling.")
            # DFT workers
            if self.color == 1:                
                current_geometry = None
                # recieving the criterion
                if self.criterion_req is None:
                    self.criterion_req = self.comm_handler.comm.irecv(
                        source=0, tag=2305
                    )
                criterion_met = self.criterion_req.Test()
                if criterion_met:
                    break

                # recieving sampled point to recompute
                if self.rank == 1:
                    if self.req is None:
                        buffer = np.zeros(
                            self.atoms_dummy.get_positions().shape, dtype=float
                        )
                        self.req = self.comm_handler.comm.Irecv(
                            buf=buffer, source=0, tag=96
                        )
                    self.req.wait()  # blocking recieve
                    current_geometry = buffer.copy()

                self.req = None
                self.comm.Barrier()
                current_geometry = self.comm.bcast(current_geometry, root=0)
                self.comm.Barrier()

                current_point = self.atoms_dummy.copy()
                current_point.set_positions(current_geometry)

                dft_result = self._recalc_aims(current_point)

                # one rank sends data back
                if self.rank == 1:
                    energies, forces = (
                        dft_result.info["REF_energy"],
                        dft_result.arrays["REF_forces"],
                    )
                    # TODO: add stress
                    if dft_result is not None:
                        temp_sampled_geometries.append(current_geometry)
                        temp_sampled_energies.append(energies)
                        temp_sampled_forces.append(forces)

                        # if enough are computed send them to training worker
                    if (
                        len(temp_sampled_energies)
                        % (self.n_samples * self.ensemble_size)
                        == 0
                        and len(temp_sampled_energies) != 0
                    ):
                        logging.info(
                            f"Computed {len(temp_sampled_energies)} points "
                            "with DFT and sending them to training worker."
                        )

                        # TODO: create loop or package data in one
                        self.req_send = self.comm_handler.comm.Isend(
                            np.array(temp_sampled_geometries), dest=0, tag=2210
                        )
                        self.req_send.Wait()

                        self.req_send = self.comm_handler.comm.Isend(
                            np.array(temp_sampled_energies), dest=0, tag=2211
                        )
                        self.req_send.Wait()

                        self.req_send = self.comm_handler.comm.Isend(
                            np.array(temp_sampled_forces), dest=0, tag=2212
                        )
                        self.req_send.Wait()

                        temp_sampled_geometries = []
                        temp_sampled_energies = []
                        temp_sampled_forces = []

        self.comm_handler.barrier()
        self.current_valid = self.comm_handler.bcast(
            self.current_valid, root=0
        )
        self.epoch = self.comm_handler.bcast(self.epoch, root=0)
        self.comm_handler.barrier()

        if self.color == 1:
            self.aims_calc.close()
        self.comm.Free()


class InitialDatasetPARSL(InitialDatasetFoundational):
    """
    Class to generate the initial dataset for the active learning procedure.
    Handles the molecular dynamics simulations, the sampling of points, the
    training of the ensemble members and the saving of the datasets.

    Uses a "foundational" model to sample points. These are then recomputed
    using DFT. Uses PARSL and can run DFT in parallel on multiple nodes.
    """

    def __init__(
        self,
        mace_settings: dict,
        al_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
        close_parsl: bool = True,
    ):

        super().__init__(
            mace_settings=mace_settings,
            al_settings=al_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
            use_mpi=False,
        )
        self.close_parsl = close_parsl

        if parsl is None:
            raise ImportError(
                "Parsl is not installed. Please install parsl" 
                " to use this feature."
            )

        if self.rank == 0:
            logging.info("Setting up PARSL for initial dataset generation.")
            # TODO: create function to check if all
            # necessary settings are provided and fall back to
            # defaults if not
            parsl_setup_dict = prepare_parsl(
                cluster_settings=self.cluster_settings
            )
            self.config = parsl_setup_dict["config"]
            self.calc_dir = parsl_setup_dict["calc_dir"]
            self.clean_dirs = parsl_setup_dict["clean_dirs"]
            self.launch_str = parsl_setup_dict["launch_str"]
            self.calc_idx = parsl_setup_dict["calc_idx"]
            handle_parsl_logger(
                log_dir=self.log_dir / "parsl_initial_dataset.log",
            )

        self.comm_handler.barrier()

    def _sample_points(self) -> list:
        """
        Samples geometries using foundational model and recalculates
        the energies and forces with DFT using PARSL.

        Returns:
            list: List of ASE Atoms objects with the sampled points
                and their DFT data.
        """
        self._md_w_foundational()
        recalculated_points = []
        if self.rank == 0:
            logging.info("Recalculating energies and forces with DFT.")
            job_results = {}
            for i, atoms in enumerate(self.sampled_points):
                self.calc_idx += 1
                # launches a parsl app and returns a future
                # that can be used to get the result later
                temp_result = recalc_aims_parsl(
                    positions=atoms.get_positions(),
                    species=atoms.get_chemical_symbols(),
                    cell=atoms.get_cell(),
                    pbc=atoms.pbc,
                    aims_settings=self.aims_settings,
                    directory=self.calc_dir / f"calc_{self.calc_idx}",
                    properties=self.properties,
                    ase_aims_command=self.launch_str,
                )
                job_results[i] = temp_result

            while len(job_results) > 0:
                for i in list(job_results.keys()):
                    result = job_results[i]
                    if result.done():
                        temp = result.result()
                        if temp is None:
                            logging.warning(
                                f"SCF not converged for point {i}. Skipping."
                            )
                            del job_results[i]
                            continue
                        current_point = self.sampled_points[i]
                        current_point.info["REF_energy"] = temp["energy"]
                        current_point.arrays["REF_forces"] = temp["forces"]
                        if self.compute_stress:
                            current_point.info["REF_stress"] = temp["stress"]
                        recalculated_points.append(current_point)

                        del job_results[i]
                time.sleep(0.5)

            if self.clean_dirs:
                try:
                    for calc_dir in self.calc_dir.glob("calc_*"):
                        shutil.rmtree(calc_dir)
                except Exception as e:
                    logging.error(
                        f"Error while cleaning directories: {e}. "
                        "Please check the directories manually."
                    )

        return recalculated_points

    def run(self):
        if self.rank == 0:
            parsl.load(self.config)
        super().run()
        if self.rank == 0:
            if self.clean_dirs:
                try:
                    shutil.rmtree(self.calc_dir)
                except Exception as e:
                    logging.error(
                        f"Error while cleaning directories: {e}. "
                        "Please check the directories manually."
                    )

    def _setup_calcs(
        self,
    ) -> None:
        if self.rank == 0:
            logging.info(
                "Initial dataset generation with foundational "
                 f"model of size: {self.initial_foundational_size}."
            )
            self.atoms.calc = self._setup_foundational()

    def _close_aims(self):
        if self.close_parsl:
            logging.info("Closing PARSL.")
            parsl.dfk().cleanup()
        else:
            logging.info(
                "Not closing PARSL. Please close it manually if needed."
            )
