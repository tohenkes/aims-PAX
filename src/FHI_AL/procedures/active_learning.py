import os
from pathlib import Path
import ase.build
import torch
import numpy as np
from mace import tools
from FHI_AL.tools.custom_MACECalculator import MACECalculator
from FHI_AL.tools.uncertainty import (
    HandleUncertainty,
    MolForceUncertainty,
    get_threshold
)
from FHI_AL.tools.utilities import (
    create_dataloader,
    ensemble_training_setups,
    ensemble_from_folder,
    setup_ensemble_dicts,
    update_model_auxiliaries,
    save_checkpoint,
    save_datasets,
    load_ensemble_sets_from_folder,
    ase_to_mace_ensemble_sets,
    create_mace_dataset,
    ensemble_prediction,
    ensemble_prediction_v2,
    max_sd_2,
    avg_sd,
    atom_wise_sd,
    atom_wise_f_error,
    Z_from_geometry_in,
    list_files_in_directory,
    get_atomic_energies_from_ensemble,
    create_ztable,
    get_atomic_energies_from_pt,
    select_best_member,
    AIMSControlParser,
    ModifyMD
)
from FHI_AL.tools.setup_MACE_training import (
    setup_mace_training,
    reset_optimizer
    )
from FHI_AL.tools.train_epoch_mace import train_epoch, validate_epoch_ensemble
import ase
from ase.io import read, write
import logging
from mpi4py import MPI
from asi4py.asecalc import ASI_ASE_calculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
from contextlib import nullcontext

WORLD_COMM = MPI.COMM_WORLD
WORLD_SIZE = WORLD_COMM.Get_size()
RANK = WORLD_COMM.Get_rank()



#TODO: refactor this. too much in one class. maybe restart class etc.
class PrepareALProcedure:
    """
    Class to prepare the active learning procedure. It handles all the input files,
    prepares the calculators, models, directories etc.
    """
    def __init__(
        self,
        mace_settings,
        al_settings,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in"
    ) -> None:
        
        logging.basicConfig(
            filename="AL.log",
            encoding="utf-8",
            level=logging.INFO,
            force=True,
        )
        tools.setup_logger(
            level=mace_settings["MISC"]["log_level"],
            #    tag=tag,
            directory=mace_settings["GENERAL"]["log_dir"],
        )

        if RANK == 0:
            logging.info("Initializing active learning procedure.")
            logging.info(f"Procedure runs on {WORLD_SIZE} workers.")
        
        self.control_parser = AIMSControlParser()
        self.md_settings = al_settings["MD"]
        self.handle_al_settings(al_settings)
        self.handle_mace_settings(mace_settings)
        self.handle_aims_settings(path_to_control)
        self.setup_uncertainty()
        np.random.seed(self.mace_settings["GENERAL"]["seed"])
        self.create_folders()
        
        
        #TODO: will break when using multiple settings for different trajectories and 
        # it should be adapted to the way the initial dataset procecure treats md settings
        self.setup_md_modify()
        
        #TODO: this would change with multiple species
        self.z = Z_from_geometry_in()
        self.z_table = create_ztable(self.z)
        self.n_atoms = len(self.z)
        
        if self.seeds_tags_dict is None:
            try:
                self.seeds_tags_dict = dict(
                    np.load(
                        self.dataset_dir / "seeds_tags_dict.npz",
                        allow_pickle=True
                    )
                )
            except:
                logging.error(
                    f"Could not load seeds_tags_dict.npz. Either specify it in the active "
                    f"learning settings or put it as seeds_tags_dict.npz in {self.dataset_dir}."
                )

        # TODO: remove hardcode
        self.use_scheduler = False
        
        
        if RANK == 0:
            self.ensemble = ensemble_from_folder(
                path_to_models=self.model_dir,
                device=self.device,
            )

            self.training_setups = ensemble_training_setups(
                ensemble=self.ensemble,
                mace_settings=self.mace_settings,
                restart=self.restart,
                checkpoints_dir=self.checkpoints_dir,
            )

            if not self.restart:
                logging.info("Loading initial datasets.")
                self.ensemble_ase_sets = load_ensemble_sets_from_folder(
                    ensemble=self.ensemble,
                    path_to_folder=Path(self.al["dataset_dir"] + "/initial"),
                )
            else:
                logging.info("Loading datasets from checkpoint.")
                self.ensemble_ase_sets = load_ensemble_sets_from_folder(
                    ensemble=self.ensemble,
                    path_to_folder=Path(self.al["dataset_dir"] + "/final"),
                )
            self.ensemble_mace_sets = ase_to_mace_ensemble_sets(
                ensemble_ase_sets=self.ensemble_ase_sets,
                z_table=self.z_table,
                r_max=self.r_max,
                seed=self.seed,
            )
            self.train_dataset_len = len(
                    self.ensemble_ase_sets[list(self.ensemble.keys())[0]][
                        "train"
                    ]
                )
        if RANK != 0:
            self.train_dataset_len = None
        
        MPI.COMM_WORLD.Barrier()
        self.train_dataset_len = MPI.COMM_WORLD.bcast(self.train_dataset_len, root=0)
        MPI.COMM_WORLD.Barrier()
        
        self.handle_atomic_energies()
        
        # this initializes the FHI aims process
        self.aims_calculator = self.setup_aims_calc(
            atoms=read(path_to_geometry)
        )
        
        if self.restart:
            self.handle_al_restart()

        else:
            #TODO: tidy up and put in a separate function/class?
            self.trajectories = {
                trajectory: read(path_to_geometry) for trajectory in range(self.num_trajectories)
            }
            self.trajectory_status = {
                trajectory: "running"
                for trajectory in range(self.num_trajectories)
            }
            self.trajectory_MD_steps = {
                trajectory: 0 for trajectory in range(self.num_trajectories)
            }
            self.trajectory_total_epochs = {
                trajectory: 0 for trajectory in range(self.num_trajectories)
            }
            self.trajectory_intermediate_epochs = {
                trajectory: 0 for trajectory in range(self.num_trajectories)
            }

            self.ensemble_reset_opt = {
                tag: False for tag in self.seeds_tags_dict.keys()
            }
            self.ensemble_no_improvement = {
                tag: 0 for tag in self.seeds_tags_dict.keys()
            }
            self.ensemble_best_valid = {
                tag: np.inf for tag in self.seeds_tags_dict.keys()
            }
            
            self.current_valid_error = np.inf
            self.threshold = np.inf
            self.total_points_added = 0  # counts how many points have been added to the training set to decide when to add a point to the validation set
            self.train_points_added = 0
            self.valid_points_added = 0
            self.num_MD_limits_reached = 0
            self.num_workers_training = 0  # maybe useful later on to give CPU some to work if all workers are training
            self.num_workers_waiting = 0
            self.total_epoch = 0
            self.check = 0
            self.uncertainties = []  # for moving average
            
        #TODO: put this somewhere else:        
        if self.analysis and not self.restart:
            # this saves the intervals between points that cross the uncertainty threshold
            self.t_intervals = {
                trajectory: [] for trajectory in range(self.num_trajectories)
            }
            # this saves uncertainty and true errors for each trajectory
            self.sanity_checks = {
                trajectory: {
                    "atom_wise_uncertainty": [],
                    "uncertainty_via_max": [],
                    "uncertainty_via_mean": [],
                    "max_error": [],
                    "mean_error": [],
                    "atom_wise_error": [],
                    "threshold": [],
                    } for trajectory in range(self.num_trajectories)
            }
            if self.mol_idxs is not None:
                for trajectory in range(self.num_trajectories):
                    self.sanity_checks[trajectory].update({
                        "total_uncertainty": [],
                        "mol_forces_uncertainty": [],
                        "mol_wise_error": [],
                        "max_mol_error": [],
                        "mean_mol_error": [],
                    })
                self.uncertainty_checks = []
            # this saves the validation losses for each trajectory
            self.collect_losses = {
                "epoch": [],
                "avg_losses": [],
                "ensemble_losses": [],
            }
            
            self.collect_thresholds = {
                trajectory: [] for trajectory in range(self.num_trajectories)
            }
            

        if RANK == 0:
            self.setup_mace_calc()
            for trajectory in self.trajectories.values():
                trajectory.calc = self.mace_calc

            #TODO: make this more flexible, to allow different drivers per trajectory
            # just make a dictionary of different md settings and pass them here
            self.md_drivers = {
                trajectory: self.setup_md_al(
                    atoms=self.trajectories[trajectory],
                    md_settings=self.md_settings,
                    idx=trajectory
                ) for trajectory in range(self.num_trajectories)
            }
            
    def handle_mace_settings(self, mace_settings: dict):
        self.mace_settings = mace_settings
        self.seed = self.mace_settings["GENERAL"]["seed"]
        self.r_max = self.mace_settings["ARCHITECTURE"]["r_max"]
        self.set_batch_size = self.mace_settings["TRAINING"]["batch_size"]
        self.set_valid_batch_size = self.mace_settings["TRAINING"][
            "valid_batch_size"
        ]
        self.checkpoints_dir = self.mace_settings["GENERAL"]["checkpoints_dir"] + "/al"
        self.scaling = self.mace_settings["TRAINING"]["scaling"]
        self.device = self.mace_settings["MISC"]["device"]
        self.model_dir = self.mace_settings["GENERAL"]["model_dir"]
        self.dtype = self.mace_settings["GENERAL"]["default_dtype"]
        self.device = self.mace_settings["MISC"]["device"]
        self.atomic_energies_dict = self.mace_settings["ARCHITECTURE"].get("atomic_energies", None)
    
    def handle_al_settings(self, al_settings):
        
        self.al = al_settings['ACTIVE_LEARNING']
        self.misc = al_settings.get('MISC', {})
        
        self.max_MD_steps = self.al["max_MD_steps"]
        self.max_epochs_worker = self.al["max_epochs_worker"]
        self.max_final_epochs = self.al["max_final_epochs"]
        self.desired_accuracy = self.al["desired_acc"]
        self.num_trajectories = self.al["num_trajectories"]
        self.skip_step = self.al["skip_step_mlff"]
        self.valid_skip = self.al["valid_skip"]
        self.sanity_skip = self.al["sanity_skip"]
        self.valid_ratio = self.al["valid_ratio"]
        self.max_set_size = self.al["max_set_size"]
        self.num_trajectories = self.al["num_trajectories"]
        self.c_x = self.al["c_x"]
        self.intermediate_epochs = self.al["intermediate_epochs"]
        self.dataset_dir = Path(self.al["dataset_dir"])
        self.patience = self.al["patience"]
        self.species_dir = self.al["species_dir"]
        self.ASI_path = self.al["aims_lib_path"]
        self.analysis = self.al.get("analysis", False)
        self.seeds_tags_dict = self.al.get("seeds_tags_dict", None)
        self.margin = self.al.get("margin", 0.001)
        self.converge_best = self.al.get("converge_best", True)
        self.mol_idxs = np.load(self.al['mol_idxs'],allow_pickle=True)['arr_0'].tolist() if self.al.get("mol_idxs", None) is not None else None
        self.uncertainty_type = self.al.get("uncertainty_type", "max_atomic_sd")
        
        self.restart = os.path.exists("restart/al/al_restart.npy")
        self.create_restart = self.misc.get("create_restart", False)

        #TODO: put this somewhere else
        if self.create_restart:
            self.al_restart_dict = {
                'trajectories': None,
                'trajectory_status': None,
                'trajectory_MD_steps': None,
                'trajectory_epochs': None,
                'ensemble_reset_opt': None,
                'ensemble_no_improvement': None,
                'ensemble_best_valid': None,
                'current_valid_error': None,
                'threshold': None,
                'total_points_added': None,
                'train_points_added': None,
                'valid_points_added': None,
                'num_MD_limits_reached': None,
                'num_workers_training': None,
                'num_workers_waiting': None,
                'total_epoch': None,
                'check': None,
                'uncertainties': None,
                'al_done': False,
                'best_member': None,
            }
            self.save_restart = False # is set to true in training when checkpoint is saved
            
            if self.md_settings['stat_ensemble'].lower() == 'nvt':
                self.al_restart_dict.update({
                    'current_temperatures': None,
                })
            
            if self.analysis:#
                self.al_restart_dict.update({
                    't_intervals': None,
                    'sanity_checks': None,
                    'collect_losses': None,
                    'collect_thresholds': None
                })
                
    def handle_aims_settings(
        self,
        path_to_control: str
        ):
        """
        Loads and parses the AIMS control file to get the settings for the AIMS calculator.

        Args:
            path_to_control (str): Path to the AIMS control file.
            species_dir (str): Path to the species directory of AIMS.

        """
        
        self.aims_settings = self.control_parser(path_to_control)
        self.aims_settings['compute_forces'] = True
        self.aims_settings['species_dir'] = self.species_dir
        
    def handle_atomic_energies(
            self,
        ):
            self.update_atomic_energies = False
            
            if RANK==0:
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
                        logging.info("Loading atomic energies from existing ensemble.")
                        (
                            self.ensemble_atomic_energies,
                            self.ensemble_atomic_energies_dict
                        ) = get_atomic_energies_from_ensemble(
                            ensemble=self.ensemble,
                            z=self.z,
                        )
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
    
    def handle_al_restart(self):

        attributes = [
            'trajectories', 'trajectory_intermediate_epochs', 'ensemble_reset_opt',
            'ensemble_no_improvement', 'ensemble_best_valid', 'threshold',
            'trajectory_status', 'trajectory_MD_steps', 'trajectory_total_epochs',
            'current_valid_error', 'total_points_added', 'train_points_added',
            'valid_points_added', 'num_MD_limits_reached', 'num_workers_training',
            'num_workers_waiting', 'total_epoch', 'check', 'uncertainties', 'converge_best'
        ]
        if self.md_settings['stat_ensemble'].lower() == 'nvt':
            attributes.append('current_temperatures')
            
        for attr in attributes:
            setattr(self, attr, None)
        
        if self.analysis:
            analysis_attributes = [
            't_intervals', 'sanity_checks', 'collect_losses',
            'collect_thresholds', 'uncertainty_checks'
            ]
            for attr in analysis_attributes:
                setattr(self, attr, None)

        if RANK == 0:
            logging.info("Restarting active learning procedure from checkpoint.")
            self.al_restart_dict = np.load(
                "restart/al/al_restart.npy",
                allow_pickle=True
            ).item()
            restart_keys = [
                'trajectories', 'trajectory_intermediate_epochs', 'ensemble_reset_opt',
                'ensemble_no_improvement', 'ensemble_best_valid', 'threshold',
                'trajectory_status', 'trajectory_MD_steps', 'trajectory_total_epochs',
                'current_valid_error', 'total_points_added', 'train_points_added',
                'valid_points_added', 'num_MD_limits_reached', 'num_workers_training',
                'num_workers_waiting', 'total_epoch', 'check', 'uncertainties', 'converge_best'
            ]
            if self.md_settings['stat_ensemble'].lower() == 'nvt':
                restart_keys.append('current_temperatures')

            for key in restart_keys:
                setattr(self, key, self.al_restart_dict[key])

            if self.analysis:
                restart_analysis_keys = [
                    't_intervals', 'sanity_checks', 'collect_losses',
                    'collect_thresholds', 'uncertainty_checks'
                ]
                for key in restart_analysis_keys:
                    setattr(self, key, self.al_restart_dict[key])

                if self.mol_idxs is not None:
                    self.uncertainty_checks = []

        MPI.COMM_WORLD.Barrier()
        self.trajectory_status = MPI.COMM_WORLD.bcast(self.trajectory_status, root=0)
        self.trajectory_MD_steps = MPI.COMM_WORLD.bcast(self.trajectory_MD_steps, root=0)
        self.trajectory_total_epochs = MPI.COMM_WORLD.bcast(self.trajectory_total_epochs, root=0)
        self.current_valid_error = MPI.COMM_WORLD.bcast(self.current_valid_error, root=0)
        self.total_points_added = MPI.COMM_WORLD.bcast(self.total_points_added, root=0)
        self.train_points_added = MPI.COMM_WORLD.bcast(self.train_points_added, root=0)
        self.valid_points_added = MPI.COMM_WORLD.bcast(self.valid_points_added, root=0)
        self.num_MD_limits_reached = MPI.COMM_WORLD.bcast(self.num_MD_limits_reached, root=0)
        self.num_workers_training = MPI.COMM_WORLD.bcast(self.num_workers_training, root=0)
        self.num_workers_waiting = MPI.COMM_WORLD.bcast(self.num_workers_waiting, root=0)
        self.total_epoch = MPI.COMM_WORLD.bcast(self.total_epoch, root=0)
        self.check = MPI.COMM_WORLD.bcast(self.check, root=0)
        self.uncertainties = MPI.COMM_WORLD.bcast(self.uncertainties, root=0)
        self.converge_best = MPI.COMM_WORLD.bcast(self.converge_best, root=0)
        
        if self.md_settings['stat_ensemble'].lower() == 'nvt':
            self.current_temperatures = MPI.COMM_WORLD.bcast(self.current_temperatures, root=0)
            
        if self.analysis:
            self.t_intervals = MPI.COMM_WORLD.bcast(self.t_intervals, root=0)
            self.sanity_checks = MPI.COMM_WORLD.bcast(self.sanity_checks, root=0)
            self.collect_losses = MPI.COMM_WORLD.bcast(self.collect_losses, root=0)
            self.collect_thresholds = MPI.COMM_WORLD.bcast(self.collect_thresholds, root=0)
            if self.mol_idxs is not None:
                self.uncertainty_checks = MPI.COMM_WORLD.bcast(self.uncertainty_checks, root=0)
        MPI.COMM_WORLD.Barrier()

    def update_al_restart_dict(self):
        self.al_restart_dict.update({
            'trajectories': self.trajectories,
            'trajectory_status': self.trajectory_status,
            'trajectory_MD_steps': self.trajectory_MD_steps,
            'trajectory_total_epochs': self.trajectory_total_epochs,
            'trajectory_intermediate_epochs': self.trajectory_intermediate_epochs,
            'ensemble_reset_opt': self.ensemble_reset_opt,
            'ensemble_no_improvement': self.ensemble_no_improvement,
            'ensemble_best_valid': self.ensemble_best_valid,
            'current_valid_error': self.current_valid_error,
            'threshold': self.threshold,
            'total_points_added': self.total_points_added,
            'train_points_added': self.train_points_added,
            'valid_points_added': self.valid_points_added,
            'num_MD_limits_reached': self.num_MD_limits_reached,
            'num_workers_training': self.num_workers_training,
            'num_workers_waiting': self.num_workers_waiting,
            'total_epoch': self.total_epoch,
            'check': self.check,
            'uncertainties': self.uncertainties,
            'converge_best': self.converge_best
        })
        
        if self.md_settings['stat_ensemble'].lower() == 'nvt':
            self.al_restart_dict.update({
                'current_temperatures': {
                    trajectory: (self.md_drivers[trajectory].temp / units.kB) for trajectory in self.trajectories.keys()
                }
            })
            
        
        if self.analysis:
            self.al_restart_dict.update({
                't_intervals': self.t_intervals,
                'sanity_checks': self.sanity_checks,
                'collect_losses': self.collect_losses,
                'collect_thresholds': self.collect_thresholds
            })
            if self.mol_idxs is not None:
                self.al_restart_dict.update({
                    'uncertainty_checks': self.uncertainty_checks
                })
        
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
        if self.analysis:
            os.makedirs("analysis", exist_ok=True)
        if self.create_restart:
            os.makedirs("restart/al", exist_ok=True)

    def setup_aims_calc(
            self,
            atoms: ase.Atoms,
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
            atoms # TODO: must be changed when we have multiple species and then we need multiaims
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
            md_settings: dict,
            idx: None
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
            if not self.restart:
                try:
                    self.current_temperatures[idx] = md_settings['temperature']
                except AttributeError:
                    self.current_temperatures = {idx: md_settings['temperature']}
            
            if md_settings['thermostat'].lower() == 'langevin':
                dyn = Langevin(
                    atoms,
                    timestep=md_settings['timestep'] * units.fs,
                    friction=md_settings['friction'] / units.fs,
                    temperature_K=self.current_temperatures[idx],
                    rng=np.random.RandomState(md_settings['seed'])
                )
                
        # make this optional and have the possibility for different initial temperature
        if not self.restart:
            MaxwellBoltzmannDistribution(
                atoms,
                temperature_K=self.current_temperatures[idx]
                )
    
        return dyn
    
    def setup_md_modify(
        self,
    ):
        # TODO: make possible to have differrent modfiers for different trajectories
        self.md_mod_settings = self.md_settings.get("MODIFY", None)
        if self.md_mod_settings is not None:
            self.md_mod_metric = self.md_mod_settings.get("metric", None)
            assert self.md_mod_metric is not None, "No metric specified for MD modification."
            
            if  self.md_mod_metric == 'train_size':
                self.get_md_mod_metric = lambda: self.train_dataset_len
            
            self.md_modifier = ModifyMD(
                settings=self.md_mod_settings
            )
            self.mod_md = True
        else:
            self.mod_md = False
            
    def setup_uncertainty(
            self
            ):

            if self.mol_idxs is not None:
                self.get_uncertainty = MolForceUncertainty(
                    mol_idxs=self.mol_idxs,
                    uncertainty_type=self.uncertainty_type
                )
            else:
                self.get_uncertainty = HandleUncertainty(
                    uncertainty_type=self.uncertainty_type
                )
    
    def check_al_done(self):
        if self.create_restart:
            check = self.al_restart_dict.get("al_done", False)
            if check:
                if RANK == 0:
                    logging.info('Active learning procedure is already done. Closing')
            return check
        else:
            return False

class ALProcedure(PrepareALProcedure):
    """
    Class for the active learning procedure. It handles the training of the ensemble
    members, the molecular dynamics simulations, the sampling of points and the saving
    of the datasets.
    """
    def sanity_check(
    #TODO: put this somewhere else and its ugly
    #TODO: update docstring
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
        
        check_results = {}
        atom_wise_uncertainty = self.get_uncertainty.ensemble_sd(sanity_prediction)
        uncertainty_via_max = self.get_uncertainty.max_atomic_sd(sanity_prediction)
        uncertainty_via_mean = self.get_uncertainty.mean_atomic_sd(sanity_prediction)
        
        mean_sanity_prediction = sanity_prediction.mean(0).squeeze()

        diff_sq_mean = np.mean(
            (true_forces - mean_sanity_prediction)**2,
            axis=-1)
        
        max_error = np.max(np.sqrt(diff_sq_mean), axis=-1)
        mean_error = np.mean(np.sqrt(diff_sq_mean), axis=-1)
        atom_wise_error = np.sqrt(diff_sq_mean)
        
        check_results['atom_wise_uncertainty'] = atom_wise_uncertainty
        check_results['uncertainty_via_max'] = uncertainty_via_max
        check_results['uncertainty_via_mean'] = uncertainty_via_mean
        check_results['atom_wise_error'] = atom_wise_error
        check_results['max_error'] = max_error
        check_results['mean_error'] = mean_error
        
        if self.mol_idxs is not None:
            total_certainty = self.get_uncertainty(sanity_prediction)
            mol_forces_uncertainty = self.get_uncertainty.get_intermol_uncertainty(
                sanity_prediction
            )
            mol_forces_prediction = self.get_uncertainty.compute_mol_forces(
                sanity_prediction,
                self.mol_idxs
                ).mean(0).squeeze()
            mol_forces_true =  self.get_uncertainty.compute_mol_forces(
                true_forces.reshape(1, *true_forces.shape),
                self.mol_idxs
                ).squeeze()
            mol_diff_sq_mean = np.mean(
                (mol_forces_true - mol_forces_prediction)**2,
                axis=-1)
            max_mol_error = np.max(np.sqrt(mol_diff_sq_mean), axis=-1)
            mean_mol_error = np.mean(np.sqrt(mol_diff_sq_mean), axis=-1)
            mol_wise_error = np.sqrt(mol_diff_sq_mean)
            
            check_results['total_uncertainty'] = total_certainty
            check_results['mol_forces_uncertainty'] = mol_forces_uncertainty
            check_results['mol_wise_error'] = mol_wise_error
            check_results['max_mol_error'] = max_mol_error
            check_results['mean_mol_error'] = mean_mol_error
        
        return check_results
            


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

        if self.valid_points_added < self.valid_ratio * self.total_points_added:
            self.trajectory_status[idx] = "running"
            self.num_workers_waiting -= 1
            if RANK == 0:
                logging.info(
                    f"Trajectory worker {idx} is adding a point to the validation set."
                )
            # while the initial datasets are different for each ensemble member we add the new points to
            # all ensemble member datasets
            if RANK == 0:
                for tag in self.ensemble_ase_sets.keys():
                    self.ensemble_ase_sets[tag]["valid"] += [self.point]
                    self.ensemble_mace_sets[tag]["valid"] += self.mace_point
            self.valid_points_added += 1

        else:
            self.trajectory_status[idx] = "training"
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
                self.train_dataset_len = len(self.ensemble_ase_sets[tag]["train"])
            
            MPI.COMM_WORLD.Barrier()
            self.train_dataset_len = MPI.COMM_WORLD.bcast(self.train_dataset_len, root=0)
            MPI.COMM_WORLD.Barrier()
            
            if self.train_dataset_len > self.max_set_size:
                return True
            if RANK == 0:
                logging.info(
                    f"Size of the training and validation set: {self.train_dataset_len}, {len(self.ensemble_ase_sets[tag]['valid'])}."
                )
            self.train_points_added += 1
        self.total_points_added += 1

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
        if RANK == 0:
            for _, (tag, model) in enumerate(self.ensemble.items()):
                
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
                # because the dataset size is dynamically changing 
                # we have to update the average number of neighbors,
                # shifts and the scaling factor for the models
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
                    device=self.device,
                )

            logging.info(f"Trajectory worker {idx} is training.")
            # we train only for some epochs before we move to the next worker which may be running MD
            # all workers train on the same models with the respective training settings for
            # each ensemble member
            
            while self.trajectory_intermediate_epochs[idx] < self.intermediate_epochs:
                for tag, model in self.ensemble.items():

                    if self.ensemble_reset_opt[tag]:
                        logging.info(f'Resetting optimizer for model {tag}.')
                        self.training_setups[tag] = reset_optimizer(
                            model=self.ensemble[tag],
                            training_setup=self.training_setups[tag],
                            training_settings=self.mace_settings["TRAINING"],
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
                        epoch=self.trajectory_intermediate_epochs[idx],
                        start_epoch=None,
                        valid_loss=None,
                        logger=logger,
                        device=self.training_setups[tag]["device"],
                        max_grad_norm=self.training_setups[tag]["max_grad_norm"],
                        output_args=self.training_setups[tag]["output_args"],
                        ema=self.training_setups[tag]["ema"],
                    )
                if (
                    self.trajectory_intermediate_epochs[idx] % self.valid_skip == 0
                    or self.trajectory_intermediate_epochs[idx] == self.intermediate_epochs - 1
                ):
                    ensemble_valid_loss, valid_loss, metrics = validate_epoch_ensemble(
                        ensemble=self.ensemble,
                        training_setups=self.training_setups,
                        ensemble_set=self.ensemble_mace_sets,
                        logger=logger,
                        log_errors=self.mace_settings["MISC"]["error_table"],
                        epoch=self.trajectory_total_epochs[idx],
                    )
                    self.best_member = select_best_member(ensemble_valid_loss)
                    if self.analysis:
                        self.collect_losses["epoch"].append(self.total_epoch)
                        self.collect_losses['avg_losses'].append(valid_loss)
                        self.collect_losses['ensemble_losses'].append(ensemble_valid_loss)

                    self.current_valid_error = metrics["mae_f"]

                    for tag in ensemble_valid_loss.keys():
                        if ensemble_valid_loss[tag] < self.ensemble_best_valid[tag]:
                            self.ensemble_best_valid[tag] = ensemble_valid_loss[tag]
                        else:
                            self.ensemble_no_improvement[tag] += 1
                        
                        if self.ensemble_no_improvement[tag] > self.max_epochs_worker:
                            logging.info(
                                f"No improvements for {self.max_epochs_worker} epochs "
                                f"(maximum epochs per worker) at ensemble member {tag}."
                            )
                            self.ensemble_reset_opt[tag] = True
                            self.ensemble_no_improvement[tag] = 0
                                         
                        save_checkpoint(
                            checkpoint_handler=self.training_setups[tag][
                                "checkpoint_handler"
                            ],
                            training_setup=self.training_setups[tag],
                            model=model,
                            epoch=self.trajectory_intermediate_epochs[idx],
                            keep_last=False,
                        )

                        save_datasets(
                            ensemble=self.ensemble,
                            ensemble_ase_sets=self.ensemble_ase_sets,
                            path=self.dataset_dir / "final",
                        )
                        if self.create_restart:
                            self.save_restart = True

                self.trajectory_total_epochs[idx] += 1
                self.trajectory_intermediate_epochs[idx] += 1
                self.total_epoch += 1
                
                if self.save_restart and self.create_restart:
                    self.update_al_restart_dict()
                    np.save(
                        "restart/al/al_restart.npy",
                        self.al_restart_dict
                        )
                    self.save_restart = False
            self.trajectory_intermediate_epochs[idx] = 0
                    
        # update calculator
        MPI.COMM_WORLD.Barrier()
        self.current_valid_error = MPI.COMM_WORLD.bcast(self.current_valid_error, root=0)
        MPI.COMM_WORLD.Barrier()
        if RANK == 0:
            for trajectory in self.trajectories.values():
                trajectory.calc.models = [self.ensemble[tag] for tag in self.ensemble.keys()]
                # TH: do we need this? could make it faster
                #for model in trajectory.calc.models:
                    #for param in model.parameters():
                        #param.requires_grad = False

        MPI.COMM_WORLD.Barrier()
        self.total_epoch = MPI.COMM_WORLD.bcast(self.total_epoch, root=0)
        self.trajectory_total_epochs[idx] = MPI.COMM_WORLD.bcast(self.trajectory_total_epochs[idx], root=0)
        MPI.COMM_WORLD.Barrier()

        if self.trajectory_total_epochs[idx] >= self.max_epochs_worker:
            self.trajectory_status[idx] = "running"
            self.num_workers_training -= 1
            self.trajectory_total_epochs[idx] = 0
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
            and self.trajectory_status[idx] == "running"
        ):
            if RANK == 0:
                logging.info(
                    f"Trajectory worker {idx} reached maximum MD steps and is killed."
                )
            self.num_MD_limits_reached += 1
            self.trajectory_status[idx] = "killed"
            return "killed"

        else:
            # TODO:
            # ideally we would first check the uncertainty, then optionally 
            # calculate the aims forces and use them to propagate
            # currently the mace forces are used even if the uncertainty is too high
            # but ase is weird and i don't want to change it so whatever. when we have our own 
            # MD engine we can adress this.
            
            if self.mod_md:
                if RANK == 0:
                    modified = self.md_modifier(
                        driver=self.md_drivers[idx],
                        metric=self.get_md_mod_metric(),
                        idx=idx
                    )
                    if modified and self.create_restart:
                        self.update_al_restart_dict()
                        
            for _ in range(self.skip_step):
                if RANK == 0:
                    self.md_drivers[idx].step()
                    #write(f"md{idx}.xyz", self.trajectories[idx],append=True)
                self.trajectory_MD_steps[idx] += 1
            #MPI.COMM_WORLD.Barrier()
            #self.trajectories[idx] = MPI.COMM_WORLD.bcast(self.trajectories[idx], root=0)
            #MPI.COMM_WORLD.Barrier()

            # somewhat arbitrary; i just want to save checkpoints if the MD phase
            # is super long
            if RANK == 0:
                if current_MD_step % 100 == 0:
                    self.update_al_restart_dict()


            if RANK == 0:  
                logging.info(
                    f"Trajectory worker {idx} at MD step {current_MD_step}."
                )
                
                self.point = self.trajectories[idx].copy()
                prediction = self.trajectories[idx].calc.results["forces_comm"]
                uncertainty = self.get_uncertainty(prediction)

                self.uncertainties.append(uncertainty)
                    
                if len(self.uncertainties) > 10: #TODO: remove hardcode
                    
                    self.threshold = get_threshold(
                        uncertainties=self.uncertainties,
                        c_x = self.c_x,
                        max_len = 400 #TODO: remove hardcode
                    )
                    
                    if self.analysis:
                        self.collect_thresholds[
                            idx
                            ].append(self.threshold)
        
            if RANK != 0:
                uncertainty = None
                prediction = None
                self.point = None
                self.threshold = None
                
            MPI.COMM_WORLD.Barrier()
            self.threshold = MPI.COMM_WORLD.bcast(self.threshold, root=0)
            self.point = MPI.COMM_WORLD.bcast(self.point, root=0)
            uncertainty = MPI.COMM_WORLD.bcast(uncertainty, root=0)
            prediction = MPI.COMM_WORLD.bcast(prediction, root=0)
            MPI.COMM_WORLD.Barrier()
    
            
            if (uncertainty > self.threshold).any():
                if RANK == 0:
                    logging.info(
                        f"Uncertainty of point is beyond threshold {np.round(self.threshold,3)} at worker {idx}: {np.round(uncertainty,3)}."
                    )
                # for analysis
                if self.analysis:
                    self.t_intervals[idx].append(current_MD_step)           
                    if self.mol_idxs is not None:
                        self.uncertainty_checks.append(
                            uncertainty > self.threshold
                            )
                # at the moment the process waits for the calculation to finish
                # ideally it should calculate in the background and the other
                # workers sample/train in the meantime
                
                if RANK == 0:
                    logging.info(
                        f"Trajectory worker {idx} is running DFT."
                    )
                
                MPI.COMM_WORLD.Barrier()
                self.aims_calculator.calculate(self.point, properties=["energy","forces"])
                if RANK == 0:
                    self.point.info['energy'] = self.aims_calculator.results['energy']
                    self.point.arrays['forces'] = self.aims_calculator.results['forces']
                    self.mace_point = create_mace_dataset(
                        data=[self.point],
                        z_table=self.z_table,
                        seed=None,
                        r_max=self.r_max,
                    )

                self.trajectory_status[idx] = "waiting"
                self.num_workers_waiting += 1
                
                MPI.COMM_WORLD.Barrier()
                self.waiting_task(idx)

                
                if RANK == 0:
                    logging.info(
                        f"Trajectory worker {idx} is waiting for job to finish."
                    )
            if self.analysis:
                if (
                    current_MD_step % self.sanity_skip == 0
                ):  
                    #TODO: Put in a function
                    
                    # TODO:
                    # should not be static but increase with time, based on how many uninterrupted
                    #  MD steps have been taken or if all workes are running and currently
                    # we are not doing anything with this. Maybe check correlation between real error
                    # and uncertainty?
                    if RANK == 0:
                        logging.info(f"Trajectory worker {idx} doing a sanity check.")
                    
                    if current_MD_step % self.skip_step == 0:
                        sanity_prediction = prediction
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
                        
                    #TODO: sometimes already calculated above so we should not calculate it again
                    MPI.COMM_WORLD.Barrier()
                    self.aims_calculator.calculate(self.point, properties=["energy","forces"])
                    MPI.COMM_WORLD.Barrier()
                    
                    check_results = self.sanity_check(
                        sanity_prediction=sanity_prediction,
                        true_forces = self.aims_calculator.results['forces']
                    )
                    for key in check_results.keys():
                        self.sanity_checks[idx][key].append(check_results[key])

                    
                    self.collect_thresholds[idx].append(self.threshold)
                    
                    self.check += 1

    def run(self):
        """
        Main function to run the active learning procedure. Initializes variables and
        controls the workers tasks.
        """

        if RANK == 0:
            logging.info("Starting active learning procedure.")


        MPI.COMM_WORLD.Barrier()
        
        while True: 
            for trajectory_idx in range(self.num_trajectories):

                if self.trajectory_status[trajectory_idx] == "waiting":

                    set_limit = self.waiting_task(trajectory_idx)
                    if set_limit: # stops the process if the maximum dataset size is reached
                        break

                if (
                    self.trajectory_status[trajectory_idx] == "training"
                ):  # and training_job: # the idea is to let a worker train only if new points
                    # have been added. e.g. it can happen that one worker is
                    # beyond its MD limit but there is no new point that has been added

                    self.training_task(trajectory_idx)

                if self.trajectory_status[trajectory_idx] == "running":

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
                        "All trajectories reached maximum MD steps."
                    )
                break
            
            if self.train_dataset_len >= self.max_set_size:
                if RANK == 0:
                    logging.info(
                        "Maximum size of training set reached."
                    )
                break
            
            
            if self.current_valid_error < self.desired_accuracy:
                if RANK == 0:
                    logging.info(
                        "Desired accuracy reached."
                    )
                break
       
        
        # turn keys which are ints into strings
        # save the datasets and the intervals for analysis
        if RANK == 0:
            logging.info(f"Active learning procedure finished. The best ensemble member based on validation loss is {self.best_member}.")
            save_datasets(
                ensemble=self.ensemble,
                ensemble_ase_sets=self.ensemble_ase_sets,
                path=self.dataset_dir / "final",
            )
            
            #TODO: put them all into one dict
            if self.analysis:
                np.savez(
                    "analysis/sanity_checks.npz",
                    self.sanity_checks
                )
                np.savez(
                    "analysis/t_intervals.npz",
                    self.t_intervals
                )
                np.savez(
                    "analysis/al_losses.npz",
                    **self.collect_losses
                )
                np.savez(
                    "analysis/thresholds.npz",
                    self.collect_thresholds
                )
                if self.mol_idxs is not None:
                    np.savez(
                        "analysis/uncertainty_checks.npz",
                        self.uncertainty_checks
                    )
            
            if self.create_restart:
                self.update_al_restart_dict()
                self.al_restart_dict["al_done"] = True
                np.save(
                    "restart/al/al_restart.npy",
                    self.al_restart_dict
                    )

    def converge(self):
        """
        Converges the ensemble on the acquired dataset. Trains the ensemble members
        until the validation loss does not improve anymore.
        """
        if RANK == 0:
            if self.converge_best:
                logging.info(f"Converging best model ({self.best_member}) on acquired dataset.")
                ensemble = {self.best_member: self.ensemble[self.best_member]}
            else:
                ensemble = self.ensemble
                logging.info("Converging ensemble on acquired dataset.")
                
            for _, (tag, model) in enumerate(ensemble.items()):
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
                    self.ensemble_mace_sets[tag]["train_loader"],
                    self.ensemble_mace_sets[tag]["valid_loader"],
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
                    device=self.device,
                )

            # resetting optimizer and scheduler
            self.training_setups_convergence = {}
            for tag in ensemble.keys():
                self.training_setups_convergence[tag] = setup_mace_training(
                    settings=self.mace_settings,
                    model=ensemble[tag],
                    tag=tag,
                    restart=self.restart,
                    convergence=True,
                    checkpoints_dir=self.checkpoints_dir,
                )
            best_valid_loss = np.inf
            epoch = 0
            if self.restart:
                epoch = self.training_setups_convergence[list(ensemble.keys())[0]]["epoch"]
            no_improvement = 0
            ensemble_valid_losses = {tag: np.inf for tag in ensemble.keys()}
            for j in range(self.max_final_epochs):
                # ensemble_loss = 0
                for tag, model in ensemble.items():
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

                if epoch % self.valid_skip == 0 or epoch == self.max_final_epochs - 1:
                    (
                        ensemble_valid_losses,
                        valid_loss,
                        _,
                    ) = validate_epoch_ensemble(
                        ensemble=ensemble,
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
                        for tag, model in ensemble.items():
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
                                checkpoint_handler=self.training_setups_convergence[tag][
                                    "checkpoint_handler"
                                ],
                                training_setup=self.training_setups_convergence[tag],
                                model=model,
                                epoch=epoch,
                                keep_last=False,
                            )
                    else:
                        no_improvement += 1

                epoch += 1
                if no_improvement > self.patience:
                    logging.info(
                        f"No improvements for {self.patience} epochs. Training converged. Best model(s) (Epoch {best_epoch}) based on validation loss saved."
                    )
                    break
                if j == self.max_final_epochs - 1:
                    logging.info(
                        f"Maximum number of epochs reached. Best model (Epoch {best_epoch}) based on validation loss saved."
                    )
        MPI.COMM_WORLD.Barrier()    

# TODO: This is not done yet
class StandardMACEEnsembleProcedure:
    """
    Just a simple Class to train ensembles from scratch using existing
    dataset in order to compare the continuous learning procedure with
    learning from scratch.
    """
    def __init__(
        self, 
        mace_settings: dict,
        active_learning_settings: dict,
        train_set_dir: str = "data/final/training",
        valid_set_dir: str = "data/final/validation",
        ) -> None:

        raise NotImplementedError("This is not done yet.")
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
        self.create_folders()
        self.z = Z_from_geometry_in()

        if self.atomic_energies_dict is None:
            self.atomic_energies_dict = {
                z: 0 for z in np.unique(self.z)
            }
            self.update_atomic_energies = True

        self.atomic_energies = np.array(
            [
                self.atomic_energies_dict[z]
                for z in self.atomic_energies_dict.keys()
            ]
        )
        self.z_table = tools.get_atomic_number_table_from_zs(
            z for z in self.atomic_energies_dict.keys()
        )

        np.random.seed(self.seed)
        #random.seed(self.seed)
        self.ensemble_seeds = np.random.randint(
            0, 1000, size=active_learning_settings["ACTIVE_LEARNING"]["ensemble_size"]
        )
    
        (
        self.seeds_tags_dict,
        self.ensemble,
        self.training_setups,
        ) = setup_ensemble_dicts(
            seeds=self.ensemble_seeds,
            mace_settings=self.mace_settings,
            atomic_energies_dict=self.atomic_energies_dict,
            save_seeds_tags_dict=False
        )
        for tag in self.ensemble.keys():
            train_set = read(train_set_dir + f"/train_set_{tag}.xyz",index=":")
            valid_set = read(valid_set_dir + f"/valid_set_{tag}.xyz",index=":")
            logging.info(
                f"Training set {tag} contains {len(train_set)} structures."
            )
            self.ensemble_ase_sets = {
                tag: {"train": train_set, "valid": valid_set} for tag in self.ensemble.keys()
            }
        self.ensemble_mace_sets = {
            tag: {
                    "train": create_mace_dataset(
                    data=self.ensemble_ase_sets[tag]["train"],
                    z_table=self.z_table,
                    seed=self.seeds_tags_dict[tag],
                    r_max=self.r_max
                    ),
                    "valid": create_mace_dataset(
                    data=self.ensemble_ase_sets[tag]["valid"],
                    z_table=self.z_table,
                    seed=self.seeds_tags_dict[tag],
                    r_max=self.r_max
                    )
                } for tag in self.ensemble.keys()
        }
            
    def train(self):
        
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
                atomic_energies=self.atomic_energies,
                scaling=self.scaling,
                update_atomic_energies=self.update_atomic_energies,
                z_table=self.z_table,
                atomic_energies_dict=self.atomic_energies_dict,
                dtype=self.dtype,
                device=self.device,
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
                    train_loader=self.ensemble_mace_sets[tag]["train_loader"],
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

            if epoch % self.eval_interval == 0 or epoch == self.max_num_epochs - 1:
                (
                    ensemble_valid_losses,
                    valid_loss,
                    _,
                ) = validate_epoch_ensemble(
                    ensemble=self.ensemble,
                    training_setups=self.training_setups,
                    ensemble_set=self.ensemble_mace_sets,
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
        self.eval_interval = self.mace_settings['TRAINING']['eval_interval']
        self.max_num_epochs = self.mace_settings['TRAINING']['max_num_epochs']
        self.patience = self.mace_settings['TRAINING']['patience']
        self.device = self.mace_settings["MISC"]["device"]
        self.dtype = self.mace_settings["GENERAL"]["default_dtype"]
        self.atomic_energies_dict = self.mace_settings["ARCHITECTURE"].get("atomic_energies_dict", None)

    
    def create_folders(self):
        
        os.makedirs(self.model_dir, exist_ok=True)
        self.standard_model_dir = Path(self.model_dir) / "standard"
        self.standard_model_dir.mkdir(
            parents=True, exist_ok=True
        )

    def setup_mace_calc(self):
        """
        Loads the models of the existing ensemble and creates the MACE calculator.
        """
        model_paths = list_files_in_directory(Path(self.model_dir) / "standard")
        self.models = [
            torch.load(f=model_path, map_location=self.device) for model_path in model_paths
        ]
        # the calculator needs to be updated consistently see below
        self.mace_calc = MACECalculator(
            models=self.models,
            device=self.device,
            default_dtype=self.dtype)
        
    def sanity_check(
    #TODO: put this somewhere else and its ugly
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
        uncertainty_via_max = max_sd_2(sanity_prediction)
        uncertainty_via_avg = avg_sd(sanity_prediction)
        atom_wise_uncertainty = atom_wise_sd(sanity_prediction)
        mean_sanity_prediction = sanity_prediction.mean(0).squeeze()
        difference = true_forces - mean_sanity_prediction
        diff_sq = difference**2
        diff_sq_mean = np.mean(diff_sq, axis=-1)
        
        max_error = np.max(np.sqrt(diff_sq_mean), axis=-1)
        mean_error = np.mean(np.sqrt(diff_sq_mean), axis=-1)
        atom_wise_error = atom_wise_f_error(mean_sanity_prediction, true_forces)

        return {
            "atom_wise_uncertainty": atom_wise_uncertainty,
            "uncertainty_via_max": uncertainty_via_max,
            "uncertainty_via_avg": uncertainty_via_avg,
            "max_error": max_error,
            "mean_error": mean_error,
            "atom_wise_error": atom_wise_error
        }

    def test(
        self,
        path_to_ds: str,
    ):
        """
        Tests the ensemble on a dataset.

        Args:
            path_to_ds (str): Path to the dataset.
        """
        test_set = read(path_to_ds, index=":")
        test_mace_ds = create_mace_dataset(
            data=test_set,
            z_table=self.z_table,
            seed=None,
            r_max=self.r_max,
        )
        logging.info("Created DS.")
        ensemble_energies, ensemble_forces = ensemble_prediction_v2(
            models=list(self.ensemble.values()),
            mace_ds=test_mace_ds,
            device=self.device,
            dtype=self.dtype,
            return_energies=True
        )
        reference_energies = np.array([atoms.info["energy"] for atoms in test_set])
        reference_forces = np.array([atoms.arrays["forces"] for atoms in test_set])
        check = self.sanity_check(ensemble_forces, reference_forces)
        np.savez("check.npz", **check)
