import sys
import os
from pathlib import Path
import ase.build
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
from mace import tools
from FHI_AL.custom_MACECalculator import MACECalculator
from FHI_AL.utilities import (
    create_dataloader,
    ensemble_training_setups,
    ensemble_from_folder,
    setup_ensemble_dicts,
    update_datasets,
    update_model_auxiliaries,
    save_checkpoint,
    save_datasets,
    pre_trajectories_from_folder,
    load_ensemble_sets_from_folder,
    ase_to_mace_ensemble_sets,
    create_mace_dataset,
    ensemble_prediction,
    setup_mace_training,
    max_sd_2,
    Z_from_geometry_in,
    list_files_in_directory,
    BOHR,
    BOHR_INV,
    HARTREE,
    HARTREE_INV,
    AIMSControlParser,
)
from FHI_AL.train_epoch_mace import train_epoch, validate_epoch_ensemble
import ase
from ase.io import read
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
        path_to_aims_lib: str,
        atomic_energies_dict: dict = None,
        species_dir: str = None,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
        ensemble_seeds: np.array = None
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
        self.ASI_path = path_to_aims_lib
        self.handle_mace_settings(mace_settings)
        self.handle_al_settings(al_settings)
        self.handle_aims_settings(path_to_control, species_dir)
        self.create_folders()
        self.z = Z_from_geometry_in()
        self.update_atomic_energies = False

        if atomic_energies_dict is None:
            self.atomic_energies_dict = {
                z: 0 for z in np.unique(self.z)
            }
            self.update_atomic_energies = True
        else:
            self.atomic_energies_dict = atomic_energies_dict

        self.atomic_energies = np.array(
            [
                self.atomic_energies_dict[z]
                for z in self.atomic_energies_dict.keys()
            ]
        )
        self.z_table = tools.get_atomic_number_table_from_zs(
            z for z in self.atomic_energies_dict.keys()
        )
        if RANK == 0:
            logging.info(f'{self.atomic_energies_dict}')
        self.atoms = read(path_to_geometry)
        self.n_atoms = len(self.z)
        #self.handle_MD_settings(al_settings)
        self.epoch = 0
        
        np.random.seed(self.mace_settings["GENERAL"]["seed"])
        random.seed(self.mace_settings["GENERAL"]["seed"])
        
        if ensemble_seeds is not None:
            self.ensemble_size = len(ensemble_seeds)
            self.ensemble_seeds = ensemble_seeds
        else:
            self.ensemble_seeds = np.random.randint(
                0, 1000, size=self.ensemble_size
            )
        # the ensemble dictionary contains the models and their tags as values and keys
        # the seeds_tags_dict connects the seeds to the tags of each ensemble member
        # the training_setups dictionary contains the training setups for each ensemble member
        (
            self.seeds_tags_dict,
            self.ensemble,
            self.training_setups,
        ) = setup_ensemble_dicts(
            seeds=self.ensemble_seeds,
            mace_settings=self.mace_settings,
            al_settings=self.al_settings,
            atomic_energies_dict=self.atomic_energies_dict,
        )
        # each ensemble member has their own initial dataset.
        # we create a ASE and MACE dataset because it makes the conversion and
        # saving easier
        self.ensemble_mace_sets, self.ensemble_ase_sets = (
            {tag: {"train": [], "valid": []} for tag in self.ensemble.keys()},
            {tag: {"train": [], "valid": []} for tag in self.ensemble.keys()},
        )

        # initializing the md and FHI aims
        self.dyn = self.setup_md(self.atoms, md_settings=self.md_settings)
        self.setup_calculator(
            self.atoms,
            pbc=self.atoms.pbc.any()
            )

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
        self.scaling = self.mace_settings["TRAINING"]["scaling"]
        self.dtype = self.mace_settings["GENERAL"]["default_dtype"]

    def handle_al_settings(self, al_settings: dict) -> None:
        """
        Saves the active learning settings to class attributes.

        Args:
            al_settings (dict): Dictionary containing the active learning settings.
        """

        self.al_settings = al_settings["ACTIVE_LEARNING"]
        self.md_settings = al_settings["MD"]
        self.ensemble_size = self.al_settings["ensemble_size"]
        self.desired_acc = self.al_settings["desired_acc"]
        self.lamb = self.al_settings["lambda"]
        self.n_samples = self.al_settings["n_samples"]
        self.max_initial_epochs = self.al_settings["max_initial_epochs"]
        self.max_final_epochs = self.al_settings["max_final_epochs"]
        self.valid_skip = self.al_settings["valid_skip"]
        self.skip_step = self.al_settings["skip_step"]
        self.intermediate_epochs = self.al_settings["intermediate_epochs"]
        self.initial_valid_ratio = self.al_settings["initial_valid_ratio"]

    def create_folders(self):
        """
        Creates the necessary directories for saving the datasets.
        """
        self.dataset_dir = Path(self.al_settings["dataset_dir"])
        (self.dataset_dir / "initial" / "training").mkdir(
            parents=True, exist_ok=True
        )
        (self.dataset_dir / "initial" / "validation").mkdir(
            parents=True, exist_ok=True
        )
        os.makedirs("model", exist_ok=True)

    def get_atomic_energies(self):
        #!!!!!!!!!!!!!!!!!!
        # TH:
        # Was a nice idea to have it calculate the isolated atomic energies for the user
        # automatically. Turns out that if you have scalapack enabled it does not work 
        # with very small systems (1-2 atoms). If you switch the KS_method it works but
        # the next time FHI aims is initialized it does not use scalapack for parallelism
        # for some reasons. This means quite the performance hit. So we'll leave it out for
        # now and let the user provide the atomic energies or use the average atomic energies.
        #!!!!!!!!!!!!!!!!!!
        """
        Calculates the isolated atomic energies for the elements in the geometry using AIMS.
        """     
        if RANK == 0:
            logging.info('Calculating the isolated atomic energies.')
        self.atomic_energies_dict = {}        
        unique_atoms = np.unique(self.z)
        for element in unique_atoms:
            atom = ase.Atoms([int(element)],positions=[[0,0,0]])
            if RANK == 0:
                logging.info(f'Calculating the atomic energy energy of {atom.symbols[0]}.')
            self.setup_calculator(atom,pbc=False,serial=True)
            self.atomic_energies_dict[element] = atom.get_potential_energy() 
            atom.calc.close() # kills AIMS process so we can start a new one later
 
        self.atomic_energies = np.array(
            [
                self.atomic_energies_dict[z]
                for z in self.atomic_energies_dict.keys()
            ]
        )
        self.z_table = tools.get_atomic_number_table_from_zs(
            z for z in self.atomic_energies_dict.keys()
        )
        if RANK == 0:
            logging.info(f'{self.atomic_energies_dict}')
        
    def setup_calculator(
            self,
            atoms: ase.Atoms,
            pbc: bool = False,
            serial: bool = False 
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
        # we only calculate the free atoms here so k_grid is not needed
        if not pbc:
            aims_settings = self.aims_settings.copy()
            if aims_settings.get('k_grid') is not None:
                aims_settings.pop('k_grid')
        else:
            aims_settings = self.aims_settings.copy()
            
        # 1-2 atoms are too small to use scalapack KS solver
        # so we fall back to the serial one for calculating single atoms
        if serial:
            aims_settings['KS_method'] = 'serial'
        else:
            aims_settings['KS_method'] = 'scalapack_fast'
        def init_via_ase(asi):
            
            from ase.calculators.aims import Aims
            calc = Aims(**aims_settings)
            calc.write_input(asi.atoms)

        atoms.calc = ASI_ASE_calculator(
            self.ASI_path,
            init_via_ase,
            MPI.COMM_WORLD,
            atoms
            )
        return atoms
    
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
        
    # def handle_MD_settings(
    #         self,
    #         al_settings: dict
    #         ):
    #     """
    #     Parses the MD settings from the active learning settings.

    #     Args:
    #         al_settings (dict): Dictionary containing the active learning settings.
    #     """
    #     #TODO: make this more flexible
    #     self.md_settings = al_settings["MD"]
    #     self.stat_ensemble = self.md_settings["stat_ensemble"].lower()
    #     if self.stat_ensemble == 'nvt':
    #         self.thermostat = self.md_settings['thermostat'].lower()
    #         if self.thermostat == 'langevin':
    #             self.friction = self.md_settings['friction']
    #             self.md_seed = self.md_settings['seed']
                
    #     self.timestep = self.md_settings['timestep']
    #     self.temperature = self.md_settings['temperature']

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


class InitalDatasetProcedure(PrepareInitialDatasetProcedure):
    """
    Class to generate the initial dataset for the active learning procedure. Handles the
    molecular dynamics simulations, the sampling of points, the training of the ensemble
    members and the saving of the datasets.

    """
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
                    self.sampled_points.append(
                        current_point
                    )
    
    def run(self):
        """
        Main function to run the initial dataset generation procedure.
        It samples points and trains the ensemble members until the desired accuracy
        is reached or the maximum number of epochs is reached.

        """

        
        current_valid = np.inf
        step = 0
        # criterion for initial dataset is multiple of the desired accuracy
        while (
            self.desired_acc * self.lamb <= current_valid 
            and self.epoch < self.max_initial_epochs
        ):
            
            if RANK == 0:
                logging.info(f"Sampling new points at step {step}.")
            self.run_MD(self.atoms, self.dyn)
            
            # Currently there is only one process doing the training.
            # This does not really matter when using GPUs but with
            # CPUs we are not using the full potential
            # TODO: look into pyTorch distributed
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
                        atomic_energies=self.atomic_energies,
                        scaling=self.scaling,
                        update_atomic_energies=self.update_atomic_energies,
                        z_table=self.z_table,
                        atomic_energies_dict=self.atomic_energies_dict,
                        dtype=self.dtype,
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
                            _,
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

                        # if the desired accuracy is reached we save the models
                        if self.desired_acc * self.lamb >= current_valid:
                            logging.info(
                                f"Accuracy criterion reached at step {step}."
                            )
                            logging.info(
                                f"Criterion: {self.desired_acc * self.lamb}; Current accuracy: {current_valid}."
                            )
                            for tag, model in self.ensemble.items():
                                torch.save(
                                    model,
                                    Path(
                                        self.mace_settings["GENERAL"]["model_dir"]
                                    )
                                    / (tag + ".model"),
                                )

                                save_checkpoint(
                                    checkpoint_handler=self.training_setups[tag][
                                        "checkpoint_handler"
                                    ],
                                    training_setup=self.training_setups[tag],
                                    model=model,
                                    epoch=self.epoch,
                                    keep_last=True,
                                )
                            
                            break
                        # if the desired accuracy is not reached we save a checkpoint
                        else:
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
                    self.epoch += 1


                if (
                    self.epoch == self.max_initial_epochs
                ):  # TODO: change to a different variable (shares with al-algo right now)
                    logging.info(f"Maximum number of epochs reached.")

                save_datasets(
                    self.ensemble,
                    self.ensemble_ase_sets,
                    path=self.dataset_dir / "initial",
                    initial=True,
                )
            # only one worker is doing the training right now,
            # so we have to broadcast the criterion so they
            # don't get stuck in the while loop
            MPI.COMM_WORLD.Barrier()
            current_valid = MPI.COMM_WORLD.bcast(current_valid, root=0)
            self.epoch = MPI.COMM_WORLD.bcast(self.epoch, root=0)
            MPI.COMM_WORLD.Barrier()
        self.atoms.calc.close()
        return 0


    def converge(self):
        """
        Function to converge the ensemble on the acquired initial dataset.
        """
        if RANK == 0:
            logging.info("Converging.")
            for _, (tag, model) in enumerate(self.ensemble.items()):

                (
                    self.ensemble_ase_sets[tag]["train_loader"],
                    self.ensemble_ase_sets[tag]["valid_loader"],
                ) = create_dataloader(
                    self.ensemble_mace_sets[tag]["train"],
                    self.ensemble_mace_sets[tag]["valid"],
                    self.set_batch_size,
                    self.set_valid_batch_size,
                )

                update_model_auxiliaries(
                    model=model,
                    mace_sets=self.ensemble_mace_sets[tag],
                    atomic_energies=self.atomic_energies,
                    scaling=self.scaling,
                    update_atomic_energies=self.update_atomic_energies,
                    z_table=self.z_table,
                    atomic_energies_dict=self.atomic_energies_dict,
                    dtype=self.dtype,
                )

            # TODO: reset or not?
            self.training_setups_convergence = {}
            for tag in self.ensemble.keys():
                self.training_setups_convergence[tag] = setup_mace_training(
                    settings=self.mace_settings,
                    model=self.ensemble[tag],
                    tag=tag,
                )
            best_valid_loss = np.inf
            epoch = 0
            patience = self.al_settings["patience"]
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
                        train_loader=self.ensemble_ase_sets[tag]["train_loader"],
                        loss_fn=self.training_setups_convergence[tag]["loss_fn"],
                        optimizer=self.training_setups_convergence[tag][
                            "optimizer"
                        ],
                        lr_scheduler=self.training_setups_convergence[tag][
                            "lr_scheduler"
                        ],
                        valid_loss=ensemble_valid_losses[tag],
                        epoch=epoch,
                        start_epoch=0,
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
                    if best_valid_loss > valid_loss and (best_valid_loss - valid_loss) > 0.01:
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
                                    "checkpoint_handler_convergence"
                                ],
                                training_setup=self.training_setups_convergence[
                                    tag
                                ],
                                model=model,
                                epoch=epoch,
                                keep_last=True,
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

class PrepareALProcedure:
    """
    Class to prepare the active learning procedure. It handles all the input files,
    prepares the calculators, models, directories etc.
    """
    def __init__(
        self,
        mace_settings,
        al_settings,
        species_dir: str,
        path_to_aims_lib: str,
        atomic_energies_dict: dict = None,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
        seeds_tags_dict: dict = None,
        path_to_trajectories: str = None,
        analysis: bool = False,
    ) -> None:
        
        tools.setup_logger(
            level=mace_settings["MISC"]["log_level"],
            #    tag=tag,
            directory=mace_settings["GENERAL"]["log_dir"],
        )
        logging.basicConfig(
            filename="AL.log",
            encoding="utf-8",
            level=logging.INFO,
            force=True,
        )
        if RANK == 0:
            logging.info("Initializing active learning procedure.")
        
        self.control_parser = AIMSControlParser()
        self.handle_al_settings(al_settings['ACTIVE_LEARNING'])
        self.handle_mace_settings(mace_settings)
        self.handle_aims_settings(path_to_control, species_dir)
        np.random.seed(self.mace_settings["GENERAL"]["seed"])
        self.ASI_path = path_to_aims_lib
        #TODO: will break when using multiple settings for different trajectories and 
        # it should be adapted to the way the initial dataset procecure treats md settings
        self.md_settings = al_settings["MD"]
        
        self.create_folders()
        
        #TODO: this would change with multiple species
        self.z = Z_from_geometry_in()
        
        self.n_atoms = len(self.z)
        
        if seeds_tags_dict is None:
            self.seeds = dict(
                np.load(
                    self.dataset_dir / "seeds_tags_dict.npz", allow_pickle=True
                )
            )
        # TODO: remove hardcode
        self.use_scheduler = False


        self.ensemble = ensemble_from_folder(
            path_to_models="./model",
            device=self.device,
        )

        

        if atomic_energies_dict is not None:
            # if they have been explicitly specified the model should not update them
            # anymore
            self.update_atomic_energies = False
            self.ensemble_atomic_energies_dict = {}
            self.ensemble_atomic_energies = {}
            for tag in self.ensemble.keys():
                self.ensemble_atomic_energies_dict[tag] = atomic_energies_dict
                self.ensemble_atomic_energies[tag] = np.array(
                    [
                        self.ensemble_atomic_energies_dict[tag][z]
                        for z in self.ensemble_atomic_energies_dict.keys()
                    ]
                )
        else:
            try:
                self.get_atomic_energies_from_ensemble()
            except:
                if RANK == 0:
                    logging.info('Could not load atomic energies from ensemble members.')
            self.update_atomic_energies = True



        self.training_setups = ensemble_training_setups(
            ensemble=self.ensemble,
            mace_settings=self.mace_settings,
        )
        if path_to_trajectories is not None:
            self.simulate_trajectories(path_to_trajectories)

        # TODO: make option to directly load the dataset from the
        # initial dataset object instead of reading from disk
        if RANK == 0:
            logging.info("Loading initial datasets.")
        self.ensemble_ase_sets = load_ensemble_sets_from_folder(
            ensemble=self.ensemble,
            path_to_folder=self.al_settings["dataset_dir"] + "/initial",
        )
        self.ensemble_mace_sets = ase_to_mace_ensemble_sets(
            ensemble_ase_sets=self.ensemble_ase_sets,
            z_table=self.z_table,
            r_max=self.r_max,
            seed=self.seeds,
        )
        # this initializes the FHI aims process
        self.aims_calculator = self.setup_aims_calc()
        self.setup_mace_calc()

        self.trajectories = {
            trajectory: read(path_to_geometry) for trajectory in range(self.num_trajectories)
        }
        for trajectory in self.trajectories.values():
            trajectory.calc = self.mace_calc

        #TODO: make this more flexible, to allow different drivers per trajectory
        # just make a dictionary of different md settings and pass them here
        self.md_drivers = {
            trajectory: self.setup_md_al(
                atoms=self.trajectories[trajectory], md_settings=self.md_settings
            ) for trajectory in range(self.num_trajectories)
        }
        # this saves the state of the trajectories
        self.trajectory_training = {
            trajectory: "running"
            for trajectory in range(self.num_trajectories)
        }
        self.trajectory_MD_steps = {
            trajectory: 0 for trajectory in range(self.num_trajectories)
        }
        self.trajectory_epochs = {
            trajectory: 0 for trajectory in range(self.num_trajectories)
        }
        self.ensemble_reset_opt = {
            tag: False for tag in self.ensemble.keys()
        }
        # this saves the intervals between points that cross the uncertainty threshold
        self.t_intervals = {
            trajectory: [] for trajectory in range(self.num_trajectories)
        }
        # this saves uncertainty and true errors for each trajectory
        self.sanity_checks = {
            trajectory: [] for trajectory in range(self.num_trajectories)
        }

        self.uncertainties = []  # for moving average   
        self.sanity_checks_valid = {}
        self.analysis = analysis # whether to do save sanity checks, intervals etc.
        
        
    def get_atomic_energies_from_ensemble(self):
        """
        Loads the atomic energies from existing ensemble members.
        """
        if RANK == 0:
            logging.info("Loading atomic energies from existing ensemble.")
        self.ensemble_atomic_energies_dict = {}
        self.ensemble_atomic_energies = {}
        for tag, model in self.ensemble.items():
            self.ensemble_atomic_energies[tag] = np.array(model.atomic_energies_fn.atomic_energies.cpu())
            self.ensemble_atomic_energies_dict[tag] = {}
            for i, atomic_energy in enumerate(self.ensemble_atomic_energies[tag]):
                # TH: i don't know if the atomic energies are really sorted by atomic number
                # inside the models. TODO: check that
                self.ensemble_atomic_energies_dict[tag][np.sort(np.unique(self.z))[i]] = atomic_energy
            
        self.z_table = tools.get_atomic_number_table_from_zs(
            z for z in self.ensemble_atomic_energies_dict[tag].keys()
        )
        
            
    def handle_mace_settings(self, mace_settings: dict):
        self.mace_settings = mace_settings
        self.seed = self.mace_settings["GENERAL"]["seed"]
        self.r_max = self.mace_settings["ARCHITECTURE"]["r_max"]
        self.set_batch_size = self.mace_settings["TRAINING"]["batch_size"]
        self.set_valid_batch_size = self.mace_settings["TRAINING"][
            "valid_batch_size"
        ]
        self.scaling = self.mace_settings["TRAINING"]["scaling"]
        self.device = self.mace_settings["MISC"]["device"]
        self.model_dir = self.mace_settings["GENERAL"]["model_dir"]
        self.dtype = self.mace_settings["GENERAL"]["default_dtype"]
    
    def handle_al_settings(self, al_settings):
        self.al_settings = al_settings
        self.max_MD_steps = al_settings["max_MD_steps"]
        self.max_epochs_worker = al_settings["max_epochs_worker"]
        self.max_final_epochs = al_settings["max_final_epochs"]
        self.desired_accuracy = al_settings["desired_acc"]
        self.num_trajectories = al_settings["num_trajectories"]
        self.skip_step = al_settings["skip_step"]
        self.valid_skip = al_settings["valid_skip"]
        self.sanity_skip = al_settings["sanity_skip"]
        self.valid_ratio = al_settings["valid_ratio"]
        self.max_set_size = al_settings["max_set_size"]
        self.num_trajectories = al_settings["num_trajectories"]
        self.c_x = al_settings["c_x"]
        self.intermediate_epochs = al_settings["intermediate_epochs"]
        self.dataset_dir = Path(al_settings["dataset_dir"])
        self.patience = al_settings["patience"]
    
    def handle_aims_settings(
        self,
        path_to_control: str,
        species_dir: str
        ):
        """
        Loads and parses the AIMS control file to get the settings for the AIMS calculator.

        Args:
            path_to_control (str): Path to the AIMS control file.
            species_dir (str): Path to the species directory of AIMS.

        """
        
        self.aims_settings = self.control_parser(path_to_control)
        self.aims_settings['compute_forces'] = True
        self.aims_settings['species_dir'] = species_dir
    
    def create_folders(self):
        """
        Create the folders for the final datasets.
        """
        (self.dataset_dir / "final" / "training").mkdir(
            parents=True, exist_ok=True
        )
        (self.dataset_dir / "final" / "validation").mkdir(
            parents=True, exist_ok=True
        )

    def setup_aims_calc(
            self,
            path_to_geometry: str = "./geometry.in"
            ):
        """
        Creates and returns the AIMS calculator for a given atoms object.

        Args:
            path_to_geometry (str, optional): Path to geometry file. Defaults to "./geometry.in".
        """
        
        def init_via_ase(asi):
            
            from ase.calculators.aims import Aims
            calc = Aims(**self.aims_settings)
            calc.write_input(asi.atoms)
            
        calculator = ASI_ASE_calculator(
            self.ASI_path,
            init_via_ase,
            MPI.COMM_WORLD,
            read(path_to_geometry) # TODO: must be changed when we have multiple species and then we need multiaims
            )
        return calculator

    def setup_mace_calc(self):
        """
        Loads the models of the existing ensemble and creates the MACE calculator.
        """
        model_paths = list_files_in_directory(self.model_dir)
        self.models = [
            torch.load(f=model_path, map_location=self.device) for model_path in model_paths
        ]
        # the calculator needs to be updated consistently see below
        self.mace_calc = MACECalculator(
            models=self.models,
            device=self.device,
            default_dtype=self.dtype)
        
    def setup_md_al(
            self,
            atoms: ase.Atoms,
            md_settings: dict
            ):
        """
        Sets up the ASE molecular dynamics object for the atoms object using
        the MD settings.

        Args:
            atoms (ase.Atoms): Atoms object to be propagated.
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

    def simulate_trajectories(
            self,
            path_to_trajectories: str
            ):
        """
        Loads existing trajectories from a folder. 

        Args:
            path_to_trajectories (str): Path to the folder containing the 
                                        trajectories in ASE readable format.
        """
        self.trajectories = pre_trajectories_from_folder(
            path=path_to_trajectories,
            num_trajectories=self.num_trajectories,
        )

class ALProcedure(PrepareALProcedure):
    """
    Class for the active learning procedure. It handles the training of the ensemble
    members, the molecular dynamics simulations, the sampling of points and the saving
    of the datasets.
    """
    
    def sanity_check(
            self,
            sanity_prediction: np.ndarray,
            true_forces: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the force uncertainty and the maximum 
        force error for a given prediction and true forces.

        Args:
            sanity_prediction (np.ndarray): Ensemble prediction. [n_members, n_points, n_atoms, 3]
            true_forces (np.ndarray): True forces. [n_points, n_atoms, 3]

        Returns:
            tuple[np.ndarray, np.ndarray]: force uncertainty, true force error
        """
        sanity_uncertainty = max_sd_2(sanity_prediction)
        mean_sanity_prediction = sanity_prediction.mean(0).squeeze()
        difference = true_forces - mean_sanity_prediction
        diff_sq = difference**2
        diff_sq_mean = np.mean(diff_sq, axis=-1)
        max_error = np.max(np.sqrt(diff_sq_mean), axis=-1)
        return sanity_uncertainty, max_error

    def waiting_task(
            self,
            idx: int
            ):
        """
        Currently only adds the current point to the training or validation set.

        Args:
            idx (int): Index of the trajectory worker.

        """

        # there is no waiting time here and if we do it sequentially there is not waiting either
        # thus the models directly continue training with the new point which could make quite the difference
        # same with adding a training point to each of the ensemble members which slows down things considerably
        # as we have to wait for enough training points to be acquired
        # if calculation is finished:

        if self.point_added % self.valid_ratio == 0:
            self.trajectory_training[idx] = "running"
            self.num_workers_waiting -= 1
            if RANK == 0:
                logging.info(
                    f"Trajectory worker {idx} is adding a point to the validation set."
                )
            # while the initial datasets are different for each ensemble member we add the new points to
            # all ensemble member datasets
            for tag in self.ensemble_ase_sets.keys():
                self.ensemble_ase_sets[tag]["valid"] += [self.point]
                self.ensemble_mace_sets[tag]["valid"] += self.mace_point

            if self.analysis:
                if RANK == 0:    
                    sanity_prediction = ensemble_prediction(
                        models=list(self.ensemble.values()),
                        atoms_list=self.ensemble_ase_sets[tag]["valid"],
                        device=self.device,
                        dtype=self.mace_settings["GENERAL"]["default_dtype"],
                    )
                MPI.COMM_WORLD.Barrier()
                sanity_prediction = MPI.COMM_WORLD.bcast(sanity_prediction, root=0)
                MPI.COMM_WORLD.Barrier()
                
                self.sanity_check(sanity_prediction, self.point.arrays["forces"])

        else:
            self.trajectory_training[idx] = "training"
            self.num_workers_training += 1
            self.num_workers_waiting -= 1
            if RANK == 0:
                logging.info(
                    f"Trajectory worker {idx} is adding a point to the training set."
                )
            # while the initial datasets are different for each ensemble member we add the new points to
            # all ensemble member datasets
            for tag in self.ensemble_ase_sets.keys():
                self.ensemble_ase_sets[tag]["train"] += [self.point]
                self.ensemble_mace_sets[tag]["train"] += self.mace_point
            if len(self.ensemble_ase_sets[tag]["train"]) > self.max_set_size:
                return True
            if RANK == 0:
                logging.info(
                    f"Size of the training and validation set: {len(self.ensemble_ase_sets[tag]['train'])}, {len(self.ensemble_ase_sets[tag]['valid'])}."
                )
        self.point_added += 1

    def training_task(
            self,
            idx: int
            ):
        """
        Creates the dataloader of the updated dataset, updates
        the average number of neighbors, shifts and scaling factor
        and trains the ensemble members. Saves the models and checkpoints.

        Args:
            idx (int): Index of the trajectory worker.
        """

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
            # because the dataset size is dynamically changing we have to update the average number of neighbors
            # and shifts and the scaling factor for the models
            # usually they converge pretty fast
            update_model_auxiliaries(
                model=model,
                mace_sets=self.ensemble_mace_sets[tag],
                atomic_energies=self.ensemble_atomic_energies[tag],
                scaling=self.scaling,
                update_atomic_energies=self.update_atomic_energies,
                z_table=self.z_table,
                atomic_energies_dict=self.ensemble_atomic_energies_dict[tag],
                dtype=self.dtype,
            )
        if RANK == 0:
            logging.info(f"Trajectory worker {idx} is training.")
        # we train only for some epochs before we move to the next worker which may be running MD
        # all workers train on the same models with the respective training settings for
        # each ensemble member
        if RANK == 0:
            for _ in range(self.intermediate_epochs):
                for tag, model in self.ensemble.items():

                    if self.ensemble_reset_opt[tag]:
                        #TODO: reduce this to resetting the optimizer; reset EMA also????
                        logging.info(f'Resetting training setup for model {tag}.')
                        self.training_setups[tag] = setup_mace_training(
                                        settings=self.mace_settings,
                                        model=self.ensemble[tag],
                                        tag=tag,
                                        )
                        self.ensemble_reset_opt[tag] = False

                    logger = tools.MetricsLogger(
                        directory=self.mace_settings["GENERAL"]["results_dir"],
                        tag=tag + "_train",
                    )
                    train_epoch(
                        model=model,
                        train_loader=self.ensemble_mace_sets[tag]["train_loader"],
                        loss_fn=self.training_setups[tag]["loss_fn"],
                        optimizer=self.training_setups[tag]["optimizer"],
                        lr_scheduler=self.training_setups[tag]['lr_scheduler'] if self.use_scheduler else None,  # no scheduler used here
                        epoch=self.trajectory_epochs[idx],
                        start_epoch=None,
                        valid_loss=None,
                        logger=logger,
                        device=self.training_setups[tag]["device"],
                        max_grad_norm=self.training_setups[tag]["max_grad_norm"],
                        output_args=self.training_setups[tag]["output_args"],
                        ema=self.training_setups[tag]["ema"],
                    )
                self.total_epoch += 1
                

                    
                if (
                    self.trajectory_epochs[idx] % self.valid_skip == 0
                    or self.trajectory_epochs[idx] == self.max_epochs_worker - 1
                ):
                    ensemble_valid_loss, _, metrics = validate_epoch_ensemble(
                        ensemble=self.ensemble,
                        training_setups=self.training_setups,
                        ensemble_set=self.ensemble_mace_sets,
                        logger=logger,
                        log_errors=self.mace_settings["MISC"]["error_table"],
                        epoch=self.trajectory_epochs[idx],
                    )
                    
                    self.current_valid_error = metrics["mae_f"]

                    for tag in ensemble_valid_loss.keys():
                        if ensemble_valid_loss[tag] < self.ensemble_best_valid[tag]:
                            self.ensemble_best_valid[tag] = ensemble_valid_loss[tag]
                        else:
                            self.ensemble_no_improvement[tag] += 1
                        
                        if self.ensemble_no_improvement[tag] > self.patience:
                            logging.info(
                                f"No improvements for {self.patience} epochs at ensemble member {tag}."
                            )
                            self.ensemble_reset_opt[tag] = True
                            self.ensemble_no_improvement[tag] = 0
                        
                    
                    save_checkpoint(
                        checkpoint_handler=self.training_setups[tag][
                            "checkpoint_handler"
                        ],
                        training_setup=self.training_setups[tag],
                        model=model,
                        epoch=self.trajectory_epochs[idx],
                        keep_last=True,
                    )
                # to here, can be made into a class
                #############
                self.trajectory_epochs[idx] += 1
        # update calculator
        MPI.COMM_WORLD.Barrier()
        self.current_valid_error = MPI.COMM_WORLD.bcast(self.current_valid_error, root=0)
        self.ensemble = MPI.COMM_WORLD.bcast(self.ensemble, root=0)
        MPI.COMM_WORLD.Barrier()
        
        for trajectory in self.trajectories.values():
            trajectory.calc.models = [self.ensemble[tag] for tag in self.ensemble.keys()]
            # TH: do we need this? could make it faster
            #for model in trajectory.calc.models:
                #for param in model.parameters():
                    #param.requires_grad = False

        MPI.COMM_WORLD.Barrier()
        self.total_epoch = MPI.COMM_WORLD.bcast(self.total_epoch, root=0)
        self.trajectory_epochs[idx] = MPI.COMM_WORLD.bcast(self.trajectory_epochs[idx], root=0)
        MPI.COMM_WORLD.Barrier()

        if self.trajectory_epochs[idx] == self.max_epochs_worker:
            self.trajectory_training[idx] = "running"
            self.num_workers_training -= 1
            self.trajectory_epochs[idx] = 0
            if RANK == 0:
                logging.info(f"Trajectory worker {idx} finished training.")
            # calculate true error and uncertainty on validation set

    def running_task(
            self,
            idx: int
            ):
        """
        Runs the molecular dynamics simulation using the MLFF and
        checks the uncertainty. If the uncertainty is above the threshold
        the point is calculated using FHI aims and sent to the waiting task.

        Args:
            idx (int): Index of the trajectory worker.

        """

        current_MD_step = self.trajectory_MD_steps[idx]
        
        # kill the worker if the maximum number of MD steps is reached
        if (
            current_MD_step > self.max_MD_steps
            and self.trajectory_training[idx] == "running"
        ):
            if RANK == 0:
                logging.info(
                    f"Trajectory worker {idx} reached maximum MD steps and is killed."
                )
            self.num_MD_limits_reached += 1
            self.trajectory_training[idx] = "killed"
            return "killed"

        else:
            # TODO:
            # ideally we would first check the uncertainty, then optionally 
            # calculate the aims forces and use them to propagate
            # currently the mace forces are used even if the uncertainty is too high
            # but ase is weird and i don't want to change it so whatever. when we have our own 
            # MD engine we can adress this.
            if RANK == 0:
                self.md_drivers[idx].step()
                logging.info(f"MD step {current_MD_step} done for trajectory worker {idx}.")

            MPI.COMM_WORLD.Barrier()
            self.trajectories[idx] = MPI.COMM_WORLD.bcast(self.trajectories[idx], root=0)
            MPI.COMM_WORLD.Barrier()

            self.trajectory_MD_steps[idx] += 1
            if current_MD_step % self.skip_step == 0:
                if RANK == 0:
                    logging.info(
                        f"Trajectory worker {idx} at MD step {current_MD_step}."
                    )
                    
                self.point = self.trajectories[idx].copy()
                prediction = None
                if RANK == 0:
                    prediction = self.trajectories[idx].calc.results["forces_comm"]
                
                MPI.COMM_WORLD.Barrier()
                prediction = MPI.COMM_WORLD.bcast(prediction, root=0)
                MPI.COMM_WORLD.Barrier()

                uncertainty = max_sd_2(prediction)
                # compute moving average of uncertainty
                self.uncertainties.append(uncertainty)
                # TODO: Remove hardcode
                # limit the history to 400
                if len(self.uncertainties) > 400:
                    self.uncertainties = self.uncertainties[-400:]
                if len(self.uncertainties) > 10:
                    mov_avg_uncert = np.mean(self.uncertainties)
                    self.threshold = mov_avg_uncert * (1.0 + self.c_x)

                if uncertainty > self.threshold:
                    if RANK == 0:
                        logging.info(
                            f"Uncertainty of point is beyond threshold {np.round(self.threshold,3)} at worker {idx}: {round(uncertainty.item(),3)}."
                        )
                    
                    # at the moment the process waits for the calculation to finish
                    # ideally it should calculate in the background and the other
                    # workers sample/train in the meantime
                    
                    MPI.COMM_WORLD.Barrier()
                    self.aims_calculator.calculate(self.point, properties=["energy","forces"])
                    self.point.info['energy'] = self.aims_calculator.results['energy']
                    self.point.arrays['forces'] = self.aims_calculator.results['forces']
                    self.mace_point = create_mace_dataset(
                        data=[self.point],
                        z_table=self.z_table,
                        seed=None,
                        r_max=self.r_max,
                    )
                    # it sends the job and does not wait for the result but
                    # continues with the next worker. only if the job is done
                    # the worker is set to training mode
                    self.trajectory_training[idx] = "waiting"
                    self.num_workers_waiting += 1
                    self.waiting_task(idx)

                    # for analysis
                    self.t_intervals[idx].append(current_MD_step)
                    if RANK == 0:
                        logging.info(
                            f"Trajectory worker {idx} is waiting for job to finish."
                        )

            if (
                current_MD_step % self.sanity_skip == 0
            ):  
                # TODO:
                # should not be static but increase with time, based on how many uninterrupted
                #  MD steps have been taken or if all workes are running and currently
                # we are not doing anything with this. Maybe check correlation between real error
                # and uncertainty?
                if RANK == 0:
                    logging.info(f"Trajectory worker {idx} doing a sanity check.")
                
                if current_MD_step % self.skip_step == 0:
                    sanity_prediction = prediction
                    sanity_uncertainty = uncertainty
                else:
                    if RANK == 0:
                        sanity_prediction = ensemble_prediction(
                            models=list(self.ensemble.values()),
                            atoms_list=[self.point],
                            device=self.device,
                            dtype=self.mace_settings["GENERAL"]["default_dtype"],
                        )
                    MPI.COMM_WORLD.Barrier()
                    sanity_prediction = MPI.COMM_WORLD.bcast(sanity_prediction, root=0)
                    MPI.COMM_WORLD.Barrier()

                    sanity_uncertainty = max_sd_2(sanity_prediction)
                    
                if self.aims_calculator.results.get("forces") is None:
                    MPI.COMM_WORLD.Barrier()
                    self.aims_calculator.calculate(self.point, properties=["energy","forces"])

                sanity_uncertainty, max_error = self.sanity_check(
                    sanity_prediction=sanity_prediction,
                    true_forces = self.aims_calculator.results['forces']
                )
                self.sanity_checks[idx].append((sanity_uncertainty, max_error))
                self.check += 1

    def run(self):
        """
        Main function to run the active learning procedure. Initializes variables and
        controls the workers tasks.
        """

        if RANK == 0:
            logging.info("Starting active learning procedure.")
        self.current_valid_error = np.inf
        self.ensemble_no_improvement = {
            tag: 0 for tag in self.ensemble.keys()
        }
        self.ensemble_best_valid = {
            tag: np.inf for tag in self.ensemble.keys()
        }
        self.threshold = np.inf
        self.point_added = 0  # counts how many points have been added to the training set to decide when to add a point to the validation set
        self.num_MD_limits_reached = 0
        self.num_workers_training = 0  # maybe useful lateron to give CPU some to work if all workers are training
        self.num_workers_waiting = 0
        self.total_epoch = 0
        self.check = 0

        MPI.COMM_WORLD.Barrier()
        
        while True:
            for trajectory_idx, _ in enumerate(self.trajectories):

                if self.trajectory_training[trajectory_idx] == "waiting":

                    set_limit = self.waiting_task(trajectory_idx)
                    if set_limit: # stops the process if the maximum dataset size is reached
                        break

                if (
                    self.trajectory_training[trajectory_idx] == "training"
                ):  # and training_job: # the idea is to let a worker train only if new points
                    # have been added. e.g. it can happen that one worker is
                    # beyond its MD limit but there is no new point that has been added

                    self.training_task(trajectory_idx)

                if self.trajectory_training[trajectory_idx] == "running":

                    self.running_task(trajectory_idx)

                if (
                    self.num_workers_training == self.num_trajectories
                ):  
                    if RANK == 0:
                        logging.info(
                            "All workers are in training mode."
                        )  
                
                if self.num_workers_waiting == self.num_trajectories:
                    if RANK == 0:
                        logging.info("All workers are waiting for jobs to finish.")
            
            if self.num_MD_limits_reached == self.num_trajectories:
                if RANK == 0:
                    logging.info(
                        "All trajectories reached maximum MD steps. Training until convergence."
                    )
                break
            if (
                len(
                    self.ensemble_ase_sets[list(self.ensemble.keys())[0]][
                        "train"
                    ]
                )
                >= self.max_set_size
            ):
                if RANK == 0:
                    logging.info(
                        "Maximum size of training set reached. Training until convergence."
                    )
                break
            
            
            if self.current_valid_error < self.desired_accuracy:
                if RANK == 0:
                    logging.info(
                        "Desired accuracy reached. Training until convergence."
                    )
                break

        # turn keys which are ints into strings
        # save the datasets and the intervals for analysis
        save_datasets(
            ensemble=self.ensemble,
            ensemble_ase_sets=self.ensemble_ase_sets,
            path=self.dataset_dir / "final",
        )

    def converge(self):
        """
        Converges the ensemble on the acquired dataset. Trains the ensemble members
        until the validation loss does not improve anymore.
        """
        if RANK == 0:
            logging.info("Converging ensemble on acquired dataset.")
            for _, (tag, model) in enumerate(self.ensemble.items()):
                train_set = create_mace_dataset(
                    data=self.ensemble_ase_sets[tag]["train"],
                    z_table=self.z_table,
                    seed=self.seeds[tag],
                    r_max=self.r_max,
                )
                valid_set = create_mace_dataset(
                    data=self.ensemble_ase_sets[tag]["valid"],
                    z_table=self.z_table,
                    seed=self.seeds[tag],
                    r_max=self.r_max,
                )
                (
                    self.ensemble_ase_sets[tag]["train_loader"],
                    self.ensemble_ase_sets[tag]["valid_loader"],
                ) = create_dataloader(
                    train_set,
                    valid_set,
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
                )
                training_setups = {}
            # resetting optimizer and scheduler
            for tag in self.ensemble.keys():
                training_setups[tag] = setup_mace_training(
                    settings=self.mace_settings,
                    model=self.ensemble[tag],
                    tag=tag,
                )
            best_valid_loss = np.inf
            epoch = 0

            no_improvement = 0
            ensemble_valid_losses = {tag: np.inf for tag in self.ensemble.keys()}
            for j in range(self.max_final_epochs):
                # ensemble_loss = 0
                for tag, model in self.ensemble.items():
                    training_setup = training_setups[tag]
                    logger = tools.MetricsLogger(
                        directory=self.mace_settings["GENERAL"]["results_dir"],
                        tag=tag + "_train",
                    )
                    train_epoch(
                        model=model,
                        train_loader=self.ensemble_ase_sets[tag]["train_loader"],
                        loss_fn=training_setup["loss_fn"],
                        optimizer=training_setup["optimizer"],
                        lr_scheduler=training_setup["lr_scheduler"],
                        valid_loss=ensemble_valid_losses[tag],
                        epoch=epoch,
                        start_epoch=0,
                        logger=logger,
                        device=training_setup["device"],
                        max_grad_norm=training_setup["max_grad_norm"],
                        output_args=training_setup["output_args"],
                        ema=training_setup["ema"],
                    )
                    # ensemble_loss += loss
                # ensemble_loss /= len(ensemble)

                if epoch % self.valid_skip == 0 or epoch == self.max_final_epochs - 1:
                    (
                        ensemble_valid_losses,
                        valid_loss,
                        _,
                    ) = validate_epoch_ensemble(
                        ensemble=self.ensemble,
                        training_setups=training_setups,
                        ensemble_set=self.ensemble_mace_sets,
                        logger=logger,
                        log_errors=self.mace_settings["MISC"]["error_table"],
                        epoch=epoch,
                    )
                    if best_valid_loss > valid_loss and (best_valid_loss - valid_loss) > 0.01:
                        best_valid_loss = valid_loss
                        best_epoch = epoch
                        no_improvement = 0
                        for tag, model in self.ensemble.items():
                            torch.save(
                                model,
                                Path(self.mace_settings["GENERAL"]["model_dir"])
                                / (tag + ".model"),
                            )
                            save_checkpoint(
                                checkpoint_handler=training_setups[tag][
                                    "checkpoint_handler_convergence"
                                ],
                                training_setup=training_setups[tag],
                                model=model,
                                epoch=epoch,
                                keep_last=True,
                            )
                    else:
                        no_improvement += 1

                epoch += 1
                if no_improvement > self.patience:
                    logging.info(
                        f"No improvements for {self.patience} epochs. Training converged. Best model (Epoch {best_epoch}) based on validation loss saved."
                    )
                    break
                if j == self.max_final_epochs - 1:
                    logging.info(
                        f"Maximum number of epochs reached. Best model (Epoch {best_epoch}) based on validation loss saved."
                    )
        MPI.COMM_WORLD.Barrier()    


class StandardMACEEnsembleProcedure:
    """
    Just a simple Class to train ensembles from scratch using existing
    dataset in order to compare the continuous learning procedure with
    learning from scratch.
    """
    def __init__(
        self, 
        mace_settings: dict,
        path_to_aims_lib: str,
        species_dir: str = None,
        path_to_control: str = "./control.in",
        dataset_dir_train: str = None,
        dataset_dir_valid: str = None,
        num_members: int = None,
        ensemble_ase_sets: dict = None,
        seeds: list = None,
        ) -> None:
        self.handle_mace_settings(mace_settings)
        logging.basicConfig(
            filename="standard_ensemble.log",
            encoding="utf-8",
            level=logging.DEBUG,
            force=True,
        )
        tools.setup_logger(
            level=self.mace_settings["MISC"]["log_level"],
            #    tag=tag,
            directory=self.mace_settings["GENERAL"]["log_dir"],
        )
        self.handle_aims_settings(
            path_to_control=path_to_control,
            species_dir=species_dir
        )
        self.ASI_path = path_to_aims_lib
        self.create_folders()
        self.get_atomic_energies()
        if seeds is None:
            self.ensemble_seeds = np.random.randint(
                        0, 1000, size=num_members
                            )
        else:
            self.ensemble_seeds = np.array(seeds)    
    
        (
        self.seeds_tags_dict,
        self.ensemble,
        self.training_setups,
        ) = setup_ensemble_dicts(
            seeds=self.ensemble_seeds,
            mace_settings=self.mace_settings,
            al_settings=None,
            atomic_energies_dict=self.atomic_energies_dict,
            save_seeds_tags_dict=False
        )
        if ensemble_ase_sets is not None:
            self.ensemble_ase_sets = ensemble_ase_sets
        else:
            train_set = read(dataset_dir_train)
            valid_set = read(dataset_dir_valid)
            self.ensemble_ase_sets = {
                tag: {"train": train_set, "valid": valid_set} for tag in self.ensemble.keys()
            }
            
    def train(self):
        
        for _, (tag, model) in enumerate(self.ensemble.items()):
            train_set = create_mace_dataset(
                data=self.ensemble_ase_sets[tag]["train"],
                z_table=self.z_table,
                seed=self.seeds_tags_dict[tag],
                r_max=self.r_max,
            )
            valid_set = create_mace_dataset(
                data=self.ensemble_ase_sets[tag]["valid"],
                z_table=self.z_table,
                seed=self.seeds_tags_dict[tag],
                r_max=self.r_max,
            )
            (
                self.ensemble_ase_sets[tag]["train_loader"],
                self.ensemble_ase_sets[tag]["valid_loader"],
            ) = create_dataloader(
                train_set,
                valid_set,
                self.set_batch_size,
                self.set_valid_batch_size,
            )

            update_model_auxiliaries(
                model=model,
                train_loader=self.ensemble_ase_sets[tag]["train_loader"],
                atomic_energies=self.atomic_energies,
                scaling=self.scaling,
            )
            
        best_valid_loss = np.inf
        epoch = 0
        no_improvement = 0
        ensemble_valid_losses = {tag: np.inf for tag in self.ensemble.keys()}
        for j in range(self.max_num_epochs):
            # ensemble_loss = 0
            for tag, model in self.ensemble.items():
                training_setup = self.training_setups[tag]
                logger = tools.MetricsLogger(
                    directory=self.mace_settings["GENERAL"]["results_dir"],
                    tag=tag + "_standard_train",
                )
                train_epoch(
                    model=model,
                    train_loader=self.ensemble_ase_sets[tag]["train_loader"],
                    loss_fn=training_setup["loss_fn"],
                    optimizer=training_setup["optimizer"],
                    lr_scheduler=training_setup["lr_scheduler"],
                    valid_loss=ensemble_valid_losses[tag],
                    epoch=epoch,
                    start_epoch=0,
                    logger=logger,
                    device=training_setup["device"],
                    max_grad_norm=training_setup["max_grad_norm"],
                    output_args=training_setup["output_args"],
                    ema=training_setup["ema"],
                )
                # ensemble_loss += loss
            # ensemble_loss /= len(ensemble)

            if epoch % self.eval_interval == 0:
                (
                    ensemble_valid_losses,
                    valid_loss,
                    _,
                ) = validate_epoch_ensemble(
                    ensemble=self.ensemble,
                    training_setups=self.training_setups,
                    ensemble_set=self.ensemble_ase_sets,
                    logger=logger,
                    log_errors=self.mace_settings["MISC"]["error_table"],
                    epoch=epoch,
                )
                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    no_improvement = 0
                    for tag, model in self.ensemble.items():
                        torch.save(
                            model,
                            self.standard_model_dir / (tag + ".model"),
                        )
                        save_checkpoint(
                            checkpoint_handler=self.training_setups[tag][
                                "checkpoint_handler_convergence"
                            ],
                            training_setup=self.training_setups[tag],
                            model=model,
                            epoch=epoch,
                            keep_last=True,
                        )
                else:
                    no_improvement += 1

            epoch += 1
            if no_improvement > self.patience:
                logging.info(
                    f"No improvements for {self.patience} epochs. Training converged. Best model (Epoch {best_epoch}) based on validation loss saved."
                )
                break
            if j == self.max_num_epochs - 1:
                logging.info(
                    f"Maximum number of epochs reached. Best model (Epoch {best_epoch}) based on validation loss saved."
                )
    
    def handle_aims_settings(
        self,
        path_to_control: str,
        species_dir: str
        ):
        """
        Parses the AIMS control file to get the settings for the AIMS calculator.

        Args:
            path_to_control (str): Path to the AIMS control file.
            species_dir (str): Path to the species directory of AIMS.
        """
        #TODO: make this more flexible
        self.aims_settings = {}
        with open(path_to_control, "r") as f: 
            for line in f:
                if "xc" in line and '#' not in line:
                    self.aims_settings['xc'] = line.split()[1]
                if "relativistic" in line and '#' not in line:
                    self.aims_settings['relativistic'] = line.split()[1] + " " + line.split()[2]
                if "charge" in line and '#' not in line:
                    self.aims_settings['charge'] = line.split()[1]
                if "many_body_dispersion" in line and '#' not in line:
                    self.aims_settings['many_body_dispersion'] = '' # TODO: i think this is not working
        self.aims_settings['species_dir'] = species_dir

    def setup_calculator(self, atoms: ase.Atoms) -> ase.Atoms:
        """
        Attaches the AIMS calculator to the atoms object. Uses the AIMS settings
        from the control.in to set up the calculator.

        Args:
            atoms (ase.Atoms): Atoms object to attach the calculator to.

        Returns:
            ase.Atoms: Atoms object with the calculator attached.
        """
        def init_via_ase(asi):
            
            from ase.calculators.aims import Aims
            #TODO: make this more flexible
            if self.aims_settings.get("many_body_dispersion") is not None:
                calc = Aims(xc=self.aims_settings["xc"],
                    relativistic=self.aims_settings["relativistic"],
                    species_dir=self.aims_settings["species_dir"],
                    compute_forces=True,
                    many_body_dispersion='',) # mbd in aims has only the
                                              # keyword, so it must be
                                              # an empty string here
            else:
                calc = Aims(xc=self.aims_settings["xc"],
                    relativistic=self.aims_settings["relativistic"],
                    species_dir=self.aims_settings["species_dir"],
                    compute_forces=True,
                    )
            
            calc.write_input(asi.atoms)
        atoms.calc = ASI_ASE_calculator(
            self.ASI_path,
            init_via_ase,
            MPI.COMM_WORLD,
            atoms
            )
        return atoms
    
    def get_atomic_energies(self):
        """
        Calculates the isolated atomic energies for the elements in the geometry using AIMS.
        TODO: make it possible to provide the numbers yourself or use the average atomic
              energies (then we'd have to update them continously like the shift and scaling factor)
              The function get_atomic_energies_from_ensemble(self) in the AL procedure has to be changed then.
        """     
        if RANK == 0:
            logging.info('Calculating isolated atomic energies.')
        self.atomic_energies_dict = {}        
        unique_atoms = np.unique(self.z)
        for element in unique_atoms:
            if RANK == 0:
                logging.info(f'Calculating energy for element {element}.')
            atom = ase.Atoms([int(element)],positions=[[0,0,0]])
            self.setup_calculator(atom)
            self.atomic_energies_dict[element] = atom.get_potential_energy() 
            atom.calc.close() # kills AIMS process so we can start a new one later
 
        self.atomic_energies = np.array(
            [
                self.atomic_energies_dict[z]
                for z in self.atomic_energies_dict.keys()
            ]
        )
        self.z_table = tools.get_atomic_number_table_from_zs(
            z for z in self.atomic_energies_dict.keys()
        )
        logging.info(f'{self.atomic_energies_dict}')

    def handle_mace_settings(self, mace_settings: dict):
        self.mace_settings = mace_settings
        self.seed = self.mace_settings["GENERAL"]["seed"]
        self.r_max = self.mace_settings["ARCHITECTURE"]["r_max"]
        self.set_batch_size = self.mace_settings["TRAINING"]["batch_size"]
        self.set_valid_batch_size = self.mace_settings["TRAINING"][
            "valid_batch_size"
        ]
        self.scaling = self.mace_settings["TRAINING"]["scaling"]
        self.model_dir = self.mace_settings["GENERAL"]["model_dir"]
        self.eval_interval = self.mace_settings['MISC']['eval_interval']
        self.max_num_epochs = self.mace_settings['TRAINING']['max_num_epochs']
        self.patience = self.mace_settings['TRAINING']['patience']
    
    def create_folders(self):
        
        os.makedirs(self.model_dir, exist_ok=True)
        self.standard_model_dir = Path(self.model_dir) / "standard"
        self.standard_model_dir.mkdir(
            parents=True, exist_ok=True
        )
        
