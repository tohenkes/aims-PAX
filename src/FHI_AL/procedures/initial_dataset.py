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
    save_ensemble,
    AIMSControlParser,
)
from FHI_AL.tools.train_epoch_mace import train_epoch, validate_epoch_ensemble
import ase
from ase.io import read
import logging
import random
from mpi4py import MPI
from asi4py.asecalc import ASI_ASE_calculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.npt import NPT
from ase import units
from contextlib import nullcontext
import threading


WORLD_COMM = MPI.COMM_WORLD
WORLD_SIZE = WORLD_COMM.Get_size()
RANK = WORLD_COMM.Get_rank()


#TODO: remove, but keeping it as a reference for now
class ReqHandler:
    def __init__(self, source, tag):
        self.source = source
        self.tag = tag
        self.received_data = None
        self.req = None
        self.thread = None
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()  # Event to signal thread termination

    def start_wait_thread(self):
        if self.thread is None or not self.thread.is_alive():
            logging.info('Starting wait thread.')
            self.req = MPI.COMM_WORLD.irecv(source=self.source, tag=self.tag)
            self.stop_flag.clear()  # Reset the stop flag
            self.thread = threading.Thread(target=self._wait_and_store)
            self.thread.start()

    def _wait_and_store(self):
        data = None
        while not self.stop_flag.is_set():
            s, data = self.req.test()
            if s:
                with self.lock:
                    self.received_data = data
                break  # Exit loop once data is received

    def get_received_data(self):
        with self.lock:
            data = self.received_data
            self.received_data = None
        return data

    def stop_thread(self):
        if self.thread and self.thread.is_alive():
            self.stop_flag.set()  # Signal the thread to stop
            self.thread.join()   # Wait for the thread to terminate


            
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
            logging.info(f"Procedure runs on {WORLD_SIZE} workers.")
            
        self.control_parser = AIMSControlParser()
        self.handle_mace_settings(mace_settings)
        self.handle_al_settings(al_settings)
        self.handle_aims_settings(path_to_control)
        self.create_folders()
            
        if self.restart:
            if RANK == 0:
                logging.info('Restarting initial dataset acquisition from checkpoint.')
            self.init_ds_restart_dict = np.load(
                "restart/initial_ds/initial_ds_restart.npy",
                allow_pickle=True
                ).item()
            self.atoms = self.init_ds_restart_dict["last_geometry"]
            self.step = self.init_ds_restart_dict["step"]
        else:
            self.atoms = read(path_to_geometry)
            self.step = 0
        
        self.z = Z_from_geometry_in()
        self.n_atoms = len(self.z)
        self.z_table = create_ztable(self.z)

        np.random.seed(self.seed)
        random.seed(self.seed) # this influences the shuffling of the sampled geometries
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
                restart=self.restart,
                checkpoints_dir=self.checkpoints_dir,
            )
            if self.restart:
                self.epoch = self.training_setups[list(self.ensemble.keys())[0]]["epoch"] + 1
        
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
            if self.restart:
                self.collect_losses = self.init_ds_restart_dict["last_initial_losses"]
            else:
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
        self.checkpoints_dir = self.mace_settings["GENERAL"]["checkpoints_dir"] + "/initial"
        self.scaling = self.mace_settings["TRAINING"]["scaling"]
        self.dtype = self.mace_settings["GENERAL"]["default_dtype"]
        self.device = self.mace_settings["MISC"]["device"]
        self.atomic_energies_dict = self.mace_settings[
            "ARCHITECTURE"].get("atomic_energies", None)
        self.compute_stress = self.mace_settings.get("compute_stress", False)       

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
        self.valid_ratio = self.al["valid_ratio"]
        self.ASI_path = self.al["aims_lib_path"]
        self.species_dir = self.al["species_dir"]
        self.analysis = self.al.get("analysis", False)
        self.margin = self.al.get("margin", 0.001)
        if not self.al["scheduler_initial"]:
            self.mace_settings["lr_scheduler"] = None
        
        self.initial_foundational_size = self.al.get("initial_foundational_size", None)
        if self.initial_foundational_size is not None:
            assert self.al["initial_foundational_size"] in ("small", "medium", "large"), "Initial foundational size not recognized."
        
        self.restart = os.path.exists("restart/initial_ds/initial_ds_restart.npy")
        self.create_restart = self.misc.get("create_restart", False)
        if self.create_restart:
            self.init_ds_restart_dict = {
                "last_geometry": None,
                "last_initial_losses": None,
                "initial_ds_done": False,           
            }
    
    def update_restart_dict(self):
        self.init_ds_restart_dict["last_geometry"] = self.last_point
        self.init_ds_restart_dict["step"] = self.step
        if self.analysis:
            self.init_ds_restart_dict["last_initial_losses"] = self.collect_losses
        
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
            os.makedirs("restart/initial_ds", exist_ok=True)
        
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
        self.properties = ['energy', 'forces']
        if self.compute_stress:
            self.properties.append('stress')

        def init_via_ase(asi):
            from ase.calculators.aims import Aims, AimsProfile
            aims_settings["profile"] = AimsProfile(command="asi-doesnt-need-command")
            calc = Aims(**aims_settings)
            calc.write_inputfiles(asi.atoms, properties=self.properties)

        calc = ASI_ASE_calculator(
            self.ASI_path,
            init_via_ase,
            MPI.COMM_WORLD,
            atoms
            )
        return calc
    
    def handle_aims_settings(
        self,
        path_to_control: str
        ):
        """
        Parses the AIMS control file to get the settings for the AIMS calculator.

        Args:
            path_to_control (str): Path to the AIMS control file.
            species_dir (str): Path to the species directory of AIMS.
        """
        
        self.aims_settings = self.control_parser(path_to_control)
        self.aims_settings['compute_forces'] = True
        self.aims_settings['species_dir'] = self.species_dir
        self.aims_settings['postprocess_anyway'] = True # this is necesssary to check for convergence
        

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
        
        if md_settings["stat_ensemble"].lower() == 'nvt':
            if md_settings['thermostat'].lower() == 'langevin':
                dyn = Langevin(
                    atoms,
                    timestep=md_settings['timestep'] * units.fs,
                    friction=md_settings['friction'] / units.fs,
                    temperature_K=md_settings['temperature'],
                    rng=np.random.RandomState(md_settings['seed'])
                )  
        elif md_settings["stat_ensemble"].lower() == 'npt':
            if md_settings['barostat'].lower() == 'berendsen':
                npt_settings = {
                    'atoms': atoms,
                    'timestep': md_settings['timestep'] * units.fs,
                    'temperature': md_settings['temperature'],
                    'pressure_au': md_settings['pressure_au'],
                }
                
                if md_settings.get('taup',False):
                    npt_settings['taup'] = md_settings['taup'] * units.fs
                if md_settings.get('taut', False):
                    npt_settings['taut'] = md_settings['taut'] * units.fs
                if md_settings.get('compressibility_au', False):
                    npt_settings['compressibility_au'] = md_settings['compressibility_au']
                if md_settings.get('fixcm', False):
                    npt_settings['fixcm'] = md_settings['fixcm']
                
                dyn = NPTBerendsen(**npt_settings)
            if md_settings['barostat'].lower() == 'npt':
                npt_settings = {
                    'atoms': atoms,
                    'timestep': md_settings['timestep'] * units.fs,
                    'temperature_K': md_settings['temperature'],
                    'externalstress': md_settings['externalstress'] * units.bar,
                    'ttime': md_settings['ttime'] * units.fs,
                    'pfactor': md_settings['pfactor'] * units.fs,
                }
                
                if md_settings.get('mask', False):
                    npt_settings['mask'] = md_settings['mask']
                
                dyn = NPT(**npt_settings)
        
        if not self.restart:
            MaxwellBoltzmannDistribution(atoms, temperature_K=md_settings['temperature'])    
            
        return dyn

    def handle_atomic_energies(
        self,
    ):
        
        self.ensemble_atomic_energies = None
        self.ensemble_atomic_energies_dict = None
        self.update_atomic_energies = False
        if RANK == 0:
            if self.atomic_energies_dict is None:
                if self.restart:
                    
                    logging.info("Loading atomic energies from checkpoint.")
                    (
                        self.ensemble_atomic_energies,
                        self.ensemble_atomic_energies_dict
                    ) = get_atomic_energies_from_pt(
                        path_to_checkpoints=self.checkpoints_dir,
                        z=self.z,
                        seeds_tags_dict=self.seeds_tags_dict,
                    )
                else:
                    
                    logging.info("No atomic specified. Initializing with 0 and fit to training data.")
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

                logging.info("Using specified atomic energies.")
                self.ensemble_atomic_energies_dict = {
                    tag: self.atomic_energies_dict for tag in self.seeds_tags_dict.keys()
                }

                self.ensemble_atomic_energies = {tag: np.array(
                    [
                        self.ensemble_atomic_energies_dict[tag][z]
                        for z in self.ensemble_atomic_energies_dict[tag].keys()
                    ]
                ) for tag in self.seeds_tags_dict.keys()}


            logging.info(f'{self.ensemble_atomic_energies_dict[list(self.seeds_tags_dict.keys())[0]]}')

    def check_initial_ds_done(self):
        if self.create_restart:
            check = self.init_ds_restart_dict.get("initial_ds_done", False)
            if check:
                if RANK==0:
                    logging.info('Initial dataset generation is already done. Closing')
            return check
        else:
            return False     

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
        
        dyn.run(self.skip_step)
        current_energy = np.array(atoms.get_potential_energy())
        current_forces = np.array(atoms.get_forces())
        current_point = atoms.copy()
        # MACE reads energies and forces from the info & arrays dictionary
        current_point.info['REF_energy'] = current_energy
        current_point.arrays['REF_forces'] = current_forces 

        if self.create_restart:
            current_point.set_velocities(atoms.get_velocities())
            current_point.set_masses(atoms.get_masses())
            self.last_point = current_point
        
        return current_point
    
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
                    convergence=True,
                    checkpoints_dir=self.checkpoints_dir,
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
                        best_epoch = epoch
                        no_improvement = 0
                        for tag, model in self.ensemble.items():
                            param_context = (
                                self.training_setups_convergence[tag]['ema'].average_parameters()
                                if self.training_setups_convergence[tag]['ema'] is not None
                                else nullcontext()
                            )
                            with param_context:
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
                        f"Maximum number of epochs reached. Best model (Epoch {best_epoch}) based on validation loss saved."
                    )
                    break

class InitialDatasetProcedure(PrepareInitialDatasetProcedure):
    """
    Class to generate the initial dataset for the active learning procedure. Handles the
    molecular dynamics simulations, the sampling of points, the training of the ensemble
    members and the saving of the datasets.

    """
    
    def sample_points(
            self
            ):
        """
        Dummy function to sample points. This function should be overwritten in the
        derived classes.
        """
        raise NotImplementedError
    
    def train(
            self
            ):
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
                        self.handle_analysis(
                            valid_loss=valid_loss,
                            ensemble_valid_losses=ensemble_valid_losses
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
                        self.update_restart_dict()
                        np.save(
                            "restart/initial_ds/initial_ds_restart.npy",
                            self.init_ds_restart_dict
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

    
    def sample_and_train(
            self
            ):
        if RANK == 0:
            logging.info(f"Sampling new points at step {self.step}.")

        self.sampled_points = []    
        # in case SCF fails to converge no point is returned
        # TODO: come up with a better solution. maybe restart trajectory
        while len(self.sampled_points) == 0:
            self.sampled_points = self.sample_points()
        
        self.step += 1 
        self.train()
    
    def setup_calcs(
            self
            ):
        """
        Dummy function to set up the calculators. This function should be overwritten in the
        derived classes.
        """
        raise NotImplementedError
    
    def close_aims(
            self
            ):
        """
        Dummy function to close the AIMS calculators. This function should be overwritten in the
        derived classes.
        """
        raise NotImplementedError 
    
    def handle_analysis(
        self,
        valid_loss,
        ensemble_valid_losses,
        save_path: str = "analysis/initial_losses.npz"
    ):
        self.collect_losses["epoch"].append(
            self.epoch
        )
        self.collect_losses["avg_losses"].append(
            valid_loss
        )
        self.collect_losses["ensemble_losses"].append(
            ensemble_valid_losses
        )
        np.savez(
            save_path,
            **self.collect_losses
        )
    
    def run(self):
        """
        Main function to run the initial dataset generation procedure.
        It samples points and trains the ensemble members until the desired accuracy
        is reached or the maximum number of epochs is reached.

        """
        
        # initializing md and FHI aims
        self.dyn = self.setup_md(self.atoms, md_settings=self.md_settings)
        
        self.setup_calcs()
        
        self.current_valid = np.inf
        # criterion for initial dataset is multiple of the desired accuracy
        # TODO: add maximum initial dataset len criterion
        while (
            self.desired_acc * self.lamb <= self.current_valid 
            and self.epoch < self.max_initial_epochs
        ):
            
            self.sample_and_train()
            # only one worker is doing the training right now,
            # so we have to broadcast the criterion so they
            # don't get stuck in the while loop
            MPI.COMM_WORLD.Barrier()
            self.current_valid = MPI.COMM_WORLD.bcast(self.current_valid, root=0)
            self.epoch = MPI.COMM_WORLD.bcast(self.epoch, root=0)
            MPI.COMM_WORLD.Barrier()
        
        if RANK == 0:
            
            save_ensemble(
                ensemble=self.ensemble,
                training_setups=self.training_setups,
                mace_settings=self.mace_settings
            )
            
            if self.create_restart:
                self.update_restart_dict()
                self.init_ds_restart_dict["initial_ds_done"] = True
                np.save(
                    "restart/initial_ds/initial_ds_restart.npy",
                    self.init_ds_restart_dict
                )
        self.close_aims()
        return 0

class InitialDatasetAIMD(InitialDatasetProcedure):
    """
    Class to generate the initial dataset for the active learning procedure. Handles the
    molecular dynamics simulations, the sampling of points, the training of the ensemble
    members and the saving of the datasets.

    """
    def sample_points(
            self
            )-> list:
        """
        Samples geometries solely using AIMD.
                
        Returns:
            list: List of ASE Atoms objects.
        """
        return [
                self.run_MD(
                    atoms=self.atoms,
                    dyn=self.dyn
                    ) for _ in range(self.ensemble_size * self.n_samples)
                ]
    
    def setup_calcs(
            self
            ):
        """
        Sets up the calculators for the initial dataset generation. In this case it sets up the
        AIMS calculators for AIMD.
        """
        self.atoms.calc = self.setup_aims_calculator(
            self.atoms
            )
        
    def close_aims(
            self
            ):
        """
        Kills the AIMS calculators.
        """
        self.atoms.calc.close()

class InitialDatasetFoundational(InitialDatasetProcedure):

    def setup_foundational(self):
        calc = mace_mp(
            model=self.initial_foundational_size,
            dispersion=False,
            default_dtype=self.dtype,
            device=self.device)
        return calc

    def recalc_aims(
            self,
            current_point: ase.Atoms
            ) -> ase.Atoms:
        self.aims_calc.calculate(current_point, properties=self.properties)
        if self.aims_calc.asi.is_scf_converged:
            current_point.info["REF_energy"] = self.aims_calc.results["energy"]
            current_point.arrays["REF_forces"] = self.aims_calc.results["forces"]
            if self.compute_stress:
                current_point.arrays["REF_stress"] = self.aims_calc.results["stress"]
            return current_point
        else:
            if RANK == 0:
                logging.info("SCF not converged.")
            return None

    def sample_points(
            self
            )-> list:
        """
        Samples geometries using foundational model and recalculates the energies and forces with DFT.
        
        Returns:
            list: List of ASE Atoms objects.
        """
        sampled_points = []
        for _ in range(self.ensemble_size * self.n_samples):
            if RANK == 0:
                current_point = self.run_MD(self.atoms, self.dyn)
                sampled_points.append(
                    current_point
                )
        MPI.COMM_WORLD.Barrier()
        sampled_points = MPI.COMM_WORLD.bcast(sampled_points, root=0)
        MPI.COMM_WORLD.Barrier()
        if RANK == 0:
            logging.info(f"Recalculating energies and forces with DFT.")
        recalculated_points = []
        for atoms in sampled_points:
            temp = self.recalc_aims(atoms)
            if temp is not None:
                recalculated_points.append(temp)
        return recalculated_points
    
    def setup_calcs(
            self
            ):
        """
        Sets up the calculators for the initial dataset generation. In this case it sets up the
        AIMS calculators for recalculating the energies and forces and the foundational model for
        MD.
        """
        self.aims_calc = self.setup_aims_calculator(
            self.atoms
            )
        if RANK == 0:
            logging.info(f"Initial dataset generation with foundational model of size: {self.initial_foundational_size}.")
            self.atoms.calc = self.setup_foundational()
        
    
    def close_aims(
            self
            ):
        """
        Kills the AIMS calculator.
        """
        self.aims_calc.close()

class InitialDatasetFoundationalParallel(InitialDatasetFoundational):
    
    def __init__(                 
        self,
        mace_settings: dict,
        al_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",):
        
        # this is necessary because of the way the MPI communicator is split
        super().__init__(
            mace_settings=mace_settings,
            al_settings=al_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry
        )
        
        # one for ML and one for DFT
        if RANK == 0:
            self.color = 0
        else:
            self.color = 1

        self.comm = MPI.COMM_WORLD.Split(
            color=self.color,
            key=RANK
            )
    
    
    def close_aims(self):
        # this is just to overwrite the function in the parent class
        # due to the communicators we are closing it inside the
        # sample_and_train function
        return None
    
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
        # only one communictor initializes aims
        if self.color == 1:
            self.properties = ['energy', 'forces']
            if self.compute_stress:
                self.properties.append('stress')

            def init_via_ase(asi):
                from ase.calculators.aims import Aims, AimsProfile
                aims_settings["profile"] = AimsProfile(command="asi-doesnt-need-command")
                calc = Aims(**aims_settings)
                calc.write_inputfiles(asi.atoms, properties=self.properties)

            calc = ASI_ASE_calculator(
                self.ASI_path,
                init_via_ase,
                self.comm,
                atoms
                )
            return calc
        else:
            return None
        
    def sample_and_train(
        self
        ) -> list:
        
        self.sampled_points = []
        temp_sampled_points = []
        
        req = None # handling data communication
        criterion_req = None # handling the communication regarding stopping
        current_point = None
        recieved_points = None
        criterion_met = False
        
        if RANK == 0:
            logging.info('Starting sampling and training using parallel mode.')
        while not criterion_met:
            if self.color == 0:
                current_point = self.run_MD(self.atoms, self.dyn)
                for dest in range(1, MPI.COMM_WORLD.Get_size()):
                    # using isend to create a queue of messages
                    sample_send = MPI.COMM_WORLD.isend(current_point, dest=dest, tag=0)
                    sample_send.Wait()

                # checking if training data recieved
                if req is None:
                    req = MPI.COMM_WORLD.irecv(source=1, tag=1)
                status, recieved_points = req.test() # non-blocking recieve
                if status:
                    # checking if the criterion is met
                    criterion_met = (
                        self.desired_acc * self.lamb >= self.current_valid or
                        self.epoch >= self.max_initial_epochs
                        )
                    if criterion_met:
                        for dest in range(1, MPI.COMM_WORLD.Get_size()):
                            criterion_send = MPI.COMM_WORLD.isend(None, dest=dest, tag=2)
                            criterion_send.Wait()
                        break
                    logging.info('Recieved points from DFT worker; training.')
                    self.sampled_points.extend(recieved_points)
                    recieved_points = None
                    self.train()

                    req = None
                           
            if self.color == 1:
                # recieving the criterion
                if criterion_req is None:
                    criterion_req = MPI.COMM_WORLD.irecv(source=0, tag=2)
                criterion_met = criterion_req.Test()
                if criterion_met:
                    break
                
                # recieving sampled point to recompute
                if req is None:
                    req = MPI.COMM_WORLD.irecv(source=0, tag=0)
                current_point = req.wait() # blocking recieve
                req = None
                # dft results are computed here and collected
                dft_result = self.recalc_aims(current_point)
                if dft_result is not None:
                    temp_sampled_points.append(dft_result)
                
                if RANK == 1:
                    # if enough are computed send them to training worker    
                    if len(temp_sampled_points) % (self.n_samples * self.ensemble_size) == 0 and len(temp_sampled_points) != 0:
                        logging.info(f'Computed {len(temp_sampled_points)} points with DFT and sending them to training worker.')
                        req_send = MPI.COMM_WORLD.isend(temp_sampled_points, dest=0, tag=1)
                        req_send.Wait()
                        temp_sampled_points = []
        if RANK == 0:
            logging.info('Closing down MPI communicators.')
            
        MPI.COMM_WORLD.Barrier()
        self.current_valid = MPI.COMM_WORLD.bcast(self.current_valid, root=0)
        self.epoch = MPI.COMM_WORLD.bcast(self.epoch, root=0)
        MPI.COMM_WORLD.Barrier()
    
        if self.color == 1:
            self.aims_calc.close()
            
        self.comm.Free()
        
    