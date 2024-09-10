import os
from pathlib import Path
import ase.build
import torch
import numpy as np
from mace import tools
from mace.calculators import mace_mp
from FHI_AL.tools.utilities import (
    create_dataloader,
    ensemble_training_setups,
    setup_ensemble_dicts,
    update_datasets,
    update_model_auxiliaries,
    save_checkpoint,
    save_datasets,
    load_ensemble_sets_from_folder,
    ase_to_mace_ensemble_sets,
    setup_mace_training,
    Z_from_geometry_in,
    get_atomic_energies_from_pt,
    create_seeds_tags_dict,
    create_ztable,
    AIMSControlParser,
)
from FHI_AL.tools.train_epoch_mace import train_epoch, validate_epoch_ensemble
import ase
from ase.io import read, write
import logging
import random
from mpi4py import MPI
from asi4py.asecalc import ASI_ASE_calculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units

WORLD_COMM = MPI.COMM_WORLD
WORLD_SIZE = WORLD_COMM.Get_size()
RANK = WORLD_COMM.Get_rank()


class PrepareInitialDatasetProcedure:
    """
    Class to prepare the inital dataset generation procedure for 
    active learning. It handles all the input files, prepars the
    calculators, models, directories etc.
    """
    def __init__(
        self,
        mace_settings: dict,
        al_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
    ) -> None:
        """
        Args:
            mace_settings (dict): Settings for the MACE model and its training.
            al_settings (dict): Settings for the active learning procedure.
            path_to_aims_lib (str): Path to the compiled AIMS library.
            atomic_energies_dict (dict, optional): Dictionary containing the atomic energies. Defaults to None.
            species_dir (str, optional): Path to the basis set settings of AIMS. Defaults to None.
            path_to_control (str, optional): Path to the AIMS control file. Defaults to "./control.in".
            path_to_geometry (str, optional): Path to the initial geometry. Defaults to "./geometry.in".
            ensemble_seeds (np.array, optional): Seeds for the individual ensemble members. Defaults to None.
        """

        # basic logger is being set up here
        logging.basicConfig(
            filename="initial_dataset.log",
            encoding="utf-8",
            level=logging.DEBUG,
            force=True,
        )
        tools.setup_logger(
            level=mace_settings["MISC"]["log_level"],
            #    tag=tag,
            directory=mace_settings["GENERAL"]["log_dir"],
        )
        if RANK == 0:
            logging.info('Initializing initial dataset procedure.')
            
        self.control_parser = AIMSControlParser()
        self.handle_mace_settings(mace_settings)
        self.handle_al_settings(al_settings)
        self.handle_aims_settings(path_to_control, self.species_dir)
        self.create_folders()
            
        if self.restart:
            if RANK == 0:
                logging.info('Restarting initial dataset acquisition from checkpoint.')
            self.atoms = read("restart/last_geometry.in")
            self.last_velocities = np.load("restart/last_velocities.npz")["arr_0"]
            self.atoms.set_velocities(self.last_velocities)
        else:
            self.atoms = read(path_to_geometry)
        
        self.z = Z_from_geometry_in()
        self.n_atoms = len(self.z)
        self.z_table = create_ztable(self.z)

        np.random.seed(self.seed)
        #random.seed(self.seed) # this only influences the shuffling of the sampled geometries
        self.ensemble_seeds = np.random.randint(
            0, 1000, size=self.ensemble_size
        )
        self.seeds_tags_dict = create_seeds_tags_dict(
            seeds=self.ensemble_seeds,
            mace_settings=self.mace_settings,
            al_settings=self.al,

        )


        self.handle_atomic_energies()
        self.epoch = 0
        
        # the ensemble dictionary contains the models and their tags as values and keys
        # the seeds_tags_dict connects the seeds to the tags of each ensemble member
        # the training_setups dictionary contains the training setups for each ensemble member
        if RANK == 0:
            self.ensemble = setup_ensemble_dicts(
                seeds_tags_dict=self.seeds_tags_dict,
                mace_settings=self.mace_settings,
                z_table=self.z_table,
                ensemble_atomic_energies_dict=self.ensemble_atomic_energies_dict,
            )
            self.training_setups = ensemble_training_setups(
                ensemble=self.ensemble,
                mace_settings=self.mace_settings,
                restart=self.restart
            )
            if self.restart:
                self.epoch = self.training_setups[list(self.ensemble.keys())[0]]["epoch"]
        
        MPI.COMM_WORLD.Barrier()
        self.epoch = MPI.COMM_WORLD.bcast(self.epoch, root=0)
        MPI.COMM_WORLD.Barrier()
        
        # each ensemble member has their own initial dataset.
        # we create a ASE and MACE dataset because it makes the conversion and
        # saving easier
        if RANK == 0:
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
                    {tag: {"train": [], "valid": []} for tag in self.ensemble.keys()},
                    {tag: {"train": [], "valid": []} for tag in self.ensemble.keys()},
                )
                
        if self.analysis:
            self.collect_losses = {
                "epoch": [],
                "avg_losses": [],
                "ensemble_losses": [],
            }

    #TODO: path to settings could maybe be better
    def handle_mace_settings(self, mace_settings: dict) -> None:
        """
        Saves the MACE settings to class attributes.

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
        self.checkpoints_dir = self.mace_settings["GENERAL"]["checkpoints_dir"]
        self.scaling = self.mace_settings["TRAINING"]["scaling"]
        self.dtype = self.mace_settings["GENERAL"]["default_dtype"]
        self.device = self.mace_settings["MISC"]["device"]
        self.atomic_energies_dict = self.mace_settings[
            "ARCHITECTURE"].get("atomic_energies", None)       

    def handle_al_settings(self, al_settings: dict) -> None:
        """
        Saves the active learning settings to class attributes.

        Args:
            al_settings (dict): Dictionary containing the active learning settings.
        """

        self.al = al_settings["ACTIVE_LEARNING"]
        self.misc = al_settings["MISC"]
        self.md_settings = al_settings["MD"]
        self.ensemble_size = self.al["ensemble_size"]
        self.desired_acc = self.al["desired_acc"]
        self.lamb = self.al["lambda"]
        self.n_samples = self.al["n_samples"]
        self.max_initial_epochs = self.al["max_initial_epochs"]
        self.max_final_epochs = self.al["max_final_epochs"]
        self.valid_skip = self.al["valid_skip"]
        self.skip_step = self.al["skip_step_initial"]
        self.intermediate_epochs = self.al["intermediate_epochs"]
        self.initial_valid_ratio = self.al["initial_valid_ratio"]
        self.ASI_path = self.al["aims_lib_path"]
        self.species_dir = self.al["species_dir"]
        self.analysis = self.al.get("analysis", False)
        self.margin = self.al.get("margin", 0.001)
        if not self.al["scheduler_initial"]:
            self.mace_settings["lr_scheduler"] = None
        self.create_restart = self.misc.get("create_restart", False)
        self.restart = os.path.exists("restart")
        
    def create_folders(self):
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
            os.makedirs("restart", exist_ok=True)
        
    def setup_aims_calculator(
            self,
            atoms: ase.Atoms,
            ) -> ase.Atoms:
        """
        Attaches the AIMS calculator to the atoms object. Uses the AIMS settings
        from the control.in to set up the calculator.

        Args:
            atoms (ase.Atoms): Atoms object to attach the calculator to.
            pbc (bool, optional): If periodic boundry conditions are required or not.
            Defaults to False.

        Returns:
            ase.Atoms: Atoms object with the calculator attached.
        """
        aims_settings = self.aims_settings.copy()

        def init_via_ase(asi):
            
            from ase.calculators.aims import Aims
            calc = Aims(**aims_settings)
            calc.write_input(asi.atoms)

        calc = ASI_ASE_calculator(
            self.ASI_path,
            init_via_ase,
            MPI.COMM_WORLD,
            atoms
            )
        return calc
    
    def handle_aims_settings(
        self,
        path_to_control: str,
        species_dir: str,
        ):
        """
        Parses the AIMS control file to get the settings for the AIMS calculator.

        Args:
            path_to_control (str): Path to the AIMS control file.
            species_dir (str): Path to the species directory of AIMS.
        """
        
        self.aims_settings = self.control_parser(path_to_control)
        self.aims_settings['compute_forces'] = True
        self.aims_settings['species_dir'] = species_dir

    def setup_md(
        self,
        atoms: ase.Atoms,
        md_settings: dict
        ):
        """
        Sets up the ASE molecular dynamics object for the atoms object.

        Args:
            atoms (ase.Atoms): Atoms to be propagated.
            md_settings (dict): Dictionary containing the MD settings.

        Returns:
            ase.md.MolecularDynamics: ASE MD engine.
        """
        #TODO: make this more flexible
        if md_settings["stat_ensemble"].lower() == 'nvt':
            if md_settings['thermostat'].lower() == 'langevin':
                dyn = Langevin(
                    atoms,
                    timestep=md_settings['timestep'] * units.fs,
                    friction=md_settings['friction'] / units.fs,
                    temperature_K=md_settings['temperature'],
                    rng=np.random.RandomState(md_settings['seed'])
                )
        # make this optional and have the possibility for different initial temperature
        MaxwellBoltzmannDistribution(atoms, temperature_K=md_settings['temperature'])
    
        return dyn

    def handle_atomic_energies(
        self,
    ):
        self.update_atomic_energies = False

        if self.atomic_energies_dict is None:
            if self.restart:
                (
                    self.ensemble_atomic_energies,
                    self.ensemble_atomic_energies_dict
                ) = get_atomic_energies_from_pt(
                    path_to_checkpoints=self.checkpoints_dir,
                    z=self.z,
                    seeds_tags_dict=self.seeds_tags_dict,
                )
            else:
                self.ensemble_atomic_energies_dict = {
                    tag:{
                    z: 0 for z in np.sort(np.unique(self.z))
                    } for tag in self.seeds_tags_dict.keys()
                    }
                self.ensemble_atomic_energies = {tag: np.array(
                        [
                            self.ensemble_atomic_energies_dict[tag][z]
                            for z in self.ensemble_atomic_energies_dict[tag].keys()
                        ]
                    )
                for tag in self.seeds_tags_dict.keys()}
                
            self.update_atomic_energies = True

        else:
            self.ensemble_atomic_energies_dict = {
                tag: self.atomic_energies_dict for tag in self.seeds_tags_dict.keys()
            }

            self.ensemble_atomic_energies = {tag: np.array(
                [
                    self.ensemble_atomic_energies_dict[tag][z]
                    for z in self.ensemble_atomic_energies_dict[tag].keys()
                ]
            ) for tag in self.seeds_tags_dict.keys()}

        if RANK == 0:
            logging.info(f'{self.ensemble_atomic_energies_dict[list(self.seeds_tags_dict.keys())[0]]}')
    
    def check_initial_ds_done(self):
        check = os.path.exists("./restart/initial_ds_done.txt")
        if check:
            if RANK == 0:
                logging.info('Initial dataset generation is already done. Closing')
        return check

    def run_MD(
            self,
            atoms: ase.Atoms,
            dyn
            ):
        """
        Runs the molecular dynamics simulation for the atoms object. Saves
        energy and forces in a MACE readable format.

        Args:
            atoms (ase.Atoms): Atoms object to be propagated.
            dyn (ase.md.MolecularDynamics): ASE MD engine.
        """
        self.sampled_points = []
        for i in range(self.n_samples * self.ensemble_size * self.skip_step):
            dyn.step()
            if RANK == 0:
                if i % self.skip_step == 0:
                    current_energy = np.array(atoms.get_potential_energy())
                    current_forces = np.array(atoms.get_forces())
                    current_point = atoms.copy()
                    # MACE reads energies and forces from the info & arrays dictionary
                    current_point.info['energy'] = current_energy
                    current_point.arrays['forces'] = current_forces 
                    self.sampled_points.append(current_point)

                if self.create_restart:
                    self.last_point = current_point
                    self.last_velocities = dyn.v
                    
    def converge(self):
        """
        Function to converge the ensemble on the acquired initial dataset.
        """
        if RANK == 0:
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
                    atomic_energies=self.ensemble_atomic_energies[tag],
                    scaling=self.scaling,
                    update_atomic_energies=self.update_atomic_energies,
                    z_table=self.z_table,
                    atomic_energies_dict=self.ensemble_atomic_energies_dict[tag],
                    dtype=self.dtype,
                    device=self.device,
                )

            # TODO: reset or not?
            self.training_setups_convergence = {}
            for tag in self.ensemble.keys():
                self.training_setups_convergence[tag] = setup_mace_training(
                    settings=self.mace_settings,
                    model=self.ensemble[tag],
                    tag=tag,
                    restart=self.restart,
                    convergence=True
                )
            best_valid_loss = np.inf
            epoch = 0
            if self.restart:
                epoch = self.training_setups_convergence[list(self.ensemble.keys())[0]]["epoch"]
                
            patience = self.al["patience"]
            no_improvement = 0
            ensemble_valid_losses = {tag: np.inf for tag in self.ensemble.keys()}
            for j in range(self.max_final_epochs):
                # ensemble_loss = 0
                for tag, model in self.ensemble.items():
                    logger = tools.MetricsLogger(
                        directory=self.mace_settings["GENERAL"]["results_dir"],
                        tag=tag + "_train",
                    )
                    train_epoch(
                        model=model,
                        train_loader=self.ensemble_mace_sets[tag]["train_loader"],
                        loss_fn=self.training_setups_convergence[tag]["loss_fn"],
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
                    # ensemble_loss += loss
                # ensemble_loss /= len(ensemble)

                if (
                    epoch % self.valid_skip == 0
                    or epoch == self.max_final_epochs - 1
                ):
                    ensemble_valid_losses, valid_loss, _ = validate_epoch_ensemble(
                        ensemble=self.ensemble,
                        training_setups=self.training_setups_convergence,
                        ensemble_set=self.ensemble_mace_sets,
                        logger=logger,
                        log_errors=self.mace_settings["MISC"]["error_table"],
                        epoch=epoch,
                    )
                    if best_valid_loss > valid_loss and (best_valid_loss - valid_loss) > self.margin:
                        best_valid_loss = valid_loss
                        no_improvement = 0
                        for tag, model in self.ensemble.items():
                            torch.save(
                                model,
                                Path(self.mace_settings["GENERAL"]["model_dir"])
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
                        f"No improvements for {patience} epochs. Training converged. Best model based on validation loss saved"
                    )
                    break
                if j == self.max_final_epochs - 1:
                    logging.info(
                        "Maximum number of epochs reached. Best model based on validation loss saved"
                    )

class InitalDatasetProcedure(PrepareInitialDatasetProcedure):
    """
    Class to generate the initial dataset for the active learning procedure. Handles the
    molecular dynamics simulations, the sampling of points, the training of the ensemble
    members and the saving of the datasets.

    """
    
    def run(self):
        """
        Main function to run the initial dataset generation procedure.
        It samples points and trains the ensemble members until the desired accuracy
        is reached or the maximum number of epochs is reached.

        """
        
        # initializing md and FHI aims
        self.dyn = self.setup_md(self.atoms, md_settings=self.md_settings)
        self.atoms.calc = self.setup_aims_calculator(
            self.atoms
            )
        
        current_valid = np.inf
        step = 0
        # criterion for initial dataset is multiple of the desired accuracy
        # TODO: add maximum initial dataset len criterion
        while (
            self.desired_acc * self.lamb <= current_valid 
            and self.epoch < self.max_initial_epochs
        ):
            
            if RANK == 0:
                logging.info(f"Sampling new points at step {step}.")
            self.run_MD(self.atoms, self.dyn)
            
            if RANK == 0:
                random.shuffle(self.sampled_points)
                # each ensemble member collects their respective points
                for number, (tag, model) in enumerate(self.ensemble.items()):

                    member_points = self.sampled_points[
                        self.n_samples * number : self.n_samples * (number + 1)
                    ]
                    
                    (
                        self.ensemble_ase_sets[tag],
                        self.ensemble_mace_sets[tag],
                    ) = update_datasets(
                        new_points=member_points,
                        mace_set=self.ensemble_mace_sets[tag],
                        ase_set=self.ensemble_ase_sets[tag],
                        valid_split=self.initial_valid_ratio,
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
                    # and the scaling factor also continously
                    update_model_auxiliaries(
                        model=model,
                        mace_sets=self.ensemble_mace_sets[tag],
                        atomic_energies=self.ensemble_atomic_energies[tag],
                        scaling=self.scaling,
                        update_atomic_energies=self.update_atomic_energies,
                        z_table=self.z_table,
                        atomic_energies_dict=self.ensemble_atomic_energies_dict[tag],
                        dtype=self.dtype,
                        device=self.device,
                    )
                    logging.info(
                        f"Training set size for '{tag}': {len(self.ensemble_mace_sets[tag]['train'])}; Validation set size: {len(self.ensemble_mace_sets[tag]['valid'])}."
                    )
                    #logging.info(
                    #    f"Updated model auxiliaries for '{tag}'."
                    #)

                    step += 1
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
                        current_valid = metrics["mae_f"]
                        
                        if self.analysis:
                            self.collect_losses["epoch"].append(
                                self.epoch
                            )
                            self.collect_losses["avg_losses"].append(
                                valid_loss
                            )
                            self.collect_losses["ensemble_losses"].append(
                                ensemble_valid_losses
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
                            write(
                                f"restart/last_geometry.in",
                                self.last_point,
                            )
                            np.savez(
                                f"restart/last_velocities.npz",
                                self.last_velocities
                            )
                        if self.desired_acc * self.lamb >= current_valid:
                            logging.info(
                                f"Accuracy criterion reached at step {step}."
                            )
                            logging.info(
                                f"Criterion: {self.desired_acc * self.lamb}; Current accuracy: {current_valid}."
                            )
                                
                            break

                    self.epoch += 1


                if (
                    self.epoch == self.max_initial_epochs
                ):  # TODO: change to a different variable (shares with al-algo right now)
                    logging.info(f"Maximum number of epochs reached.")

            # only one worker is doing the training right now,
            # so we have to broadcast the criterion so they
            # don't get stuck in the while loop
            MPI.COMM_WORLD.Barrier()
            current_valid = MPI.COMM_WORLD.bcast(current_valid, root=0)
            self.epoch = MPI.COMM_WORLD.bcast(self.epoch, root=0)
            MPI.COMM_WORLD.Barrier()
            if RANK == 0:
                if self.analysis:
                    np.savez(
                        "analysis/initial_losses.npz",
                        **self.collect_losses
                    )
        
        self.atoms.calc.close()
        
        if RANK == 0:
            for tag, model in self.ensemble.items():
                torch.save(
                    model,
                    Path(
                        self.mace_settings["GENERAL"]["model_dir"]
                    )
                    / (tag + ".model"),
                )
            # save a simple text file claryfing that the initial dataset generation is done
            # (used if the procedure is restarted and the user wants to converge)
            with open("restart/initial_ds_done.txt", "w") as f:
                f.write("Initial dataset generation is done.")
        return 0

class InitialDSFoundational(PrepareInitialDatasetProcedure):
    
    def setup_foundational(self):
        assert self.al["initial_foundational_size"] in ("small", "medium", "large"), "Initial foundational size not recognized."
        calc = mace_mp(
            model=self.al["initial_foundational_size"],
            dispersion=False,
            default_dtype=self.dtype,
            device=self.device)
        return calc
    
    def run(self):
        """
        Main function to run the initial dataset generation procedure.
        It samples points and trains the ensemble members until the desired accuracy
        is reached or the maximum number of epochs is reached.

        """
        
        
        # initializing md and FHI aims
        self.dyn = self.setup_md(self.atoms, md_settings=self.md_settings)
        #self.aims_calc = self.setup_aims_calculator(
        #    self.atoms
        #    )
        
        current_valid = np.inf
        step = 0
        # criterion for initial dataset is multiple of the desired accuracy
        # TODO: add maximum initial dataset len criterion
        while (
            self.desired_acc * self.lamb <= current_valid 
            and self.epoch < self.max_initial_epochs
        ):
            
            if RANK == 0:
                logging.info(f"Sampling new points at step {step}.")
            self.run_MD(self.atoms, self.dyn)
            
            if RANK == 0:
                random.shuffle(self.sampled_points)
                # each ensemble member collects their respective points
                for number, (tag, model) in enumerate(self.ensemble.items()):

                    member_points = self.sampled_points[
                        self.n_samples * number : self.n_samples * (number + 1)
                    ]
                    
                    (
                        self.ensemble_ase_sets[tag],
                        self.ensemble_mace_sets[tag],
                    ) = update_datasets(
                        new_points=member_points,
                        mace_set=self.ensemble_mace_sets[tag],
                        ase_set=self.ensemble_ase_sets[tag],
                        valid_split=self.initial_valid_ratio,
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
                    # and the scaling factor also continously
                    update_model_auxiliaries(
                        model=model,
                        mace_sets=self.ensemble_mace_sets[tag],
                        atomic_energies=self.ensemble_atomic_energies[tag],
                        scaling=self.scaling,
                        update_atomic_energies=self.update_atomic_energies,
                        z_table=self.z_table,
                        atomic_energies_dict=self.ensemble_atomic_energies_dict[tag],
                        dtype=self.dtype,
                        device=self.device,
                    )
                    logging.info(
                        f"Training set size for '{tag}': {len(self.ensemble_mace_sets[tag]['train'])}; Validation set size: {len(self.ensemble_mace_sets[tag]['valid'])}."
                    )
                    #logging.info(
                    #    f"Updated model auxiliaries for '{tag}'."
                    #)

                    step += 1
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
                        current_valid = metrics["mae_f"]
                        
                        if self.analysis:
                            self.collect_losses["epoch"].append(
                                self.epoch
                            )
                            self.collect_losses["avg_losses"].append(
                                valid_loss
                            )
                            self.collect_losses["ensemble_losses"].append(
                                ensemble_valid_losses
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
                            write(
                                f"restart/last_geometry.in",
                                self.last_point,
                            )
                            np.savez(
                                f"restart/last_velocities.npz",
                                self.last_velocities
                            )
                        if self.desired_acc * self.lamb >= current_valid:
                            logging.info(
                                f"Accuracy criterion reached at step {step}."
                            )
                            logging.info(
                                f"Criterion: {self.desired_acc * self.lamb}; Current accuracy: {current_valid}."
                            )
                                
                            break

                    self.epoch += 1


                if (
                    self.epoch == self.max_initial_epochs
                ):  # TODO: change to a different variable (shares with al-algo right now)
                    logging.info(f"Maximum number of epochs reached.")

            # only one worker is doing the training right now,
            # so we have to broadcast the criterion so they
            # don't get stuck in the while loop
            MPI.COMM_WORLD.Barrier()
            current_valid = MPI.COMM_WORLD.bcast(current_valid, root=0)
            self.epoch = MPI.COMM_WORLD.bcast(self.epoch, root=0)
            MPI.COMM_WORLD.Barrier()
            if RANK == 0:
                if self.analysis:
                    np.savez(
                        "analysis/initial_losses.npz",
                        **self.collect_losses
                    )
        
        for tag, model in self.ensemble.items():
            torch.save(
                model,
                Path(
                    self.mace_settings["GENERAL"]["model_dir"]
                )
                / (tag + ".model"),
            )
        self.atoms.calc.close()
        if RANK == 0:
            # save a simple text file claryfing that the initial dataset generation is done
            # (used if the procedure is restarted and the user wants to converge)
            with open("restart/initial_ds_done.txt", "w") as f:
                f.write("Initial dataset generation is done.")
        return 0