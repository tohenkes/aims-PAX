import os
from pathlib import Path
import ase.build
import torch
import numpy as np
from mace import tools
from mace.calculators import MACECalculator
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
    atoms_full_copy,
    CommHandler,
    dtype_mapping,
    AIMSControlParser,
    ModifyMD
)
from FHI_AL.tools.utilities_parsl import (
    prepare_parsl,
    recalc_aims_parsl,
    handle_parsl_logger
)
import shutil
from FHI_AL.tools.setup_MACE_training import (
    setup_mace_training,
    reset_optimizer
    )
from FHI_AL.tools.train_epoch_mace import train_epoch, validate_epoch_ensemble
import ase
from ase.io import read, write
import logging
from asi4py.asecalc import ASI_ASE_calculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.npt import NPT
from ase import units
from contextlib import nullcontext
import threading
import queue
import time
try:
    import parsl
except ImportError:
    parsl = None




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
        path_to_geometry: str = "./geometry.in",
        use_mpi: bool = True,
        comm_handler: CommHandler = None
    ) -> None:
        
        if comm_handler is not None:
            self.comm_handler = comm_handler
        else:
            self.comm_handler = CommHandler(
                use_mpi=use_mpi,
            )
            self.rank = self.comm_handler.get_rank()
            self.world_size = self.comm_handler.get_size()
        
        # purge all existing loggers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # basic logger is being set up here
        logger_level = logging.DEBUG if mace_settings["MISC"]["log_level"].lower() == "debug" else logging.INFO 
        self.log_dir = Path(mace_settings["GENERAL"]["log_dir"])
        tools.setup_logger(
            level=logger_level,
            tag=f'active_learning',
            directory=self.log_dir,
        )
                
        if self.rank == 0:
            logging.info("Initializing active learning procedure.")
            logging.info(f"Procedure runs on {self.world_size} workers.")

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
        self.freeze_threshold = False
        
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
        
        
        if self.rank == 0:
            self.ensemble = ensemble_from_folder(
                path_to_models=self.model_dir,
                device=self.device,
                dtype=dtype_mapping[self.dtype],
            )

            self.training_setups = ensemble_training_setups(
                ensemble=self.ensemble,
                mace_settings=self.mace_settings,
                restart=self.restart,
                checkpoints_dir=self.checkpoints_dir,
                al_settings=self.al,
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
        if self.rank != 0:
            self.train_dataset_len = None
        self.comm_handler.barrier()
        self.train_dataset_len = self.comm_handler.bcast(self.train_dataset_len, root=0)
        self.comm_handler.barrier()
        
        self.handle_atomic_energies()
        # this initializes the FHI aims process
        self.aims_calculator = self.setup_aims_calculator(
            atoms=read(path_to_geometry)
        )
        
        if self.restart:
            self.handle_al_restart()
        else:
            #TODO: tidy up and put in a separate function/class?
            self.trajectories = {
                trajectory: read(path_to_geometry) for trajectory in range(self.num_trajectories)
            }
            self.MD_checkpoints = {
                trajectory: self.trajectories[trajectory] for trajectory in range(self.num_trajectories)
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
            self.uncert_not_crossed = {
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
            self.uncertainties = []  # for moving average # This times self.skip_step sets the limit TODO: make this a setting
            
        #TODO: put this somewhere else:        
        if self.analysis :

            self.trajectories_analysis_prediction = {
                idx: None for idx in range(self.num_trajectories)
            }
            if not self.restart:
                # this saves the intervals between points that cross the uncertainty threshold
                self.t_intervals = {
                    trajectory: [] for trajectory in range(self.num_trajectories)
                }
                # this saves uncertainty and true errors for each trajectory
                self.analysis_checks = {
                    trajectory: {
                        "prediction": [],
                        "true_forces": [],
                        "atom_wise_uncertainty": [],
                        "uncertainty_via_max": [],
                        "uncertainty_via_mean": [],
                        "max_error": [],
                        "mean_error": [],
                        "atom_wise_error": [],
                        "threshold": [],
                        "train_set_length": []
                        } for trajectory in range(self.num_trajectories)
                }
                if self.mol_idxs is not None:
                    for trajectory in range(self.num_trajectories):
                        self.analysis_checks[trajectory].update({
                            "mol_forces_prediction": [],
                            "mol_forces_true": [],
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

        if self.rank == 0:
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
        torch.set_default_dtype(dtype_mapping[self.dtype])
        self.device = self.mace_settings["MISC"]["device"]
        self.atomic_energies_dict = self.mace_settings["ARCHITECTURE"].get("atomic_energies", None)
        self.compute_stress = self.mace_settings.get("compute_stress", False)
        self.properties = ['energy', 'forces']
        if self.compute_stress:
            self.properties.append('stress')
        
    def handle_al_settings(self, al_settings):
        
        # TODO: use setattr here
        self.al = al_settings['ACTIVE_LEARNING']
        self.cluster_settings = al_settings.get('CLUSTER', None)
        self.misc = al_settings.get('MISC', {})
        self.max_MD_steps = self.al["max_MD_steps"]
        self.max_epochs_worker = self.al["max_epochs_worker"]
        self.max_final_epochs = self.al["max_final_epochs"]
        self.desired_accuracy = self.al["desired_acc"]
        self.num_trajectories = self.al["num_trajectories"]
        self.skip_step = self.al["skip_step_mlff"]
        self.valid_skip = self.al["valid_skip"]
        self.analysis_skip = self.al["analysis_skip"]
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
        if self.rank == 0:
            logging.info(f"Using molecular indices: {self.mol_idxs}")
        self.uncertainty_type = self.al.get("uncertainty_type", "max_atomic_sd")
        self.uncert_not_crossed_limit = self.al.get("uncert_not_crossed_limit", 500)
        self.freeze_threshold_dataset = self.al.get("freeze_threshold_dataset", np.inf)
        if self.rank == 0:
            if self.freeze_threshold_dataset != np.inf:
                logging.info(f"Freezing threshold at dataset size of {self.freeze_threshold_dataset}")
        
        
        self.restart = os.path.exists("restart/al/al_restart.npy")
        self.create_restart = self.misc.get("create_restart", False)

        #TODO: put this somewhere else
        if self.create_restart:
            self.al_restart_dict = {
                'trajectories': None,
                'MD_checkppoints': None,
                'trajectory_status': None,
                'trajectory_MD_steps': None,
                'trajectory_epochs': None,
                'ensemble_reset_opt': None,
                'ensemble_no_improvement': None,
                'ensemble_best_valid': None,
                'uncert_not_crossed': None,
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
            
            if self.md_settings['stat_ensemble'].lower() in ['nvt', 'npt']:
                self.al_restart_dict.update({
                    'current_temperatures': None,
                })
            
            if self.analysis:#
                self.al_restart_dict.update({
                    't_intervals': None,
                    'analysis_checks': None,
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
        self.aims_settings['postprocess_anyway'] = True # this is necesssary to check for convergence with ASI
        
    def handle_atomic_energies(
            self,
        ):
            self.update_atomic_energies = False
            
            if self.rank==0:
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
                            dtype=self.dtype
                        )
                    else:
                        logging.info("Loading atomic energies from existing ensemble.")
                        (
                            self.ensemble_atomic_energies,
                            self.ensemble_atomic_energies_dict
                        ) = get_atomic_energies_from_ensemble(
                            ensemble=self.ensemble,
                            z=self.z,
                            dtype=self.dtype
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
                        ],
                        dtype=self.dtype
                    ) for tag in self.seeds_tags_dict.keys()}

                logging.info(f'{self.ensemble_atomic_energies_dict[list(self.seeds_tags_dict.keys())[0]]}')
    
    def handle_al_restart(self):

        attributes = [
            'trajectories','MD_checkpoints', 'trajectory_intermediate_epochs', 'ensemble_reset_opt',
            'ensemble_no_improvement', 'ensemble_best_valid', 'threshold',
            'trajectory_status', 'trajectory_MD_steps', 'trajectory_total_epochs',
            'current_valid_error', 'total_points_added', 'train_points_added',
            'valid_points_added', 'num_MD_limits_reached', 'num_workers_training',
            'num_workers_waiting', 'total_epoch', 'check', 'uncertainties', 'converge_best',
            'uncert_not_crossed'
        ]
        if self.md_settings['stat_ensemble'].lower() in ['nvt', 'npt']:
            attributes.append('current_temperatures')
            
        for attr in attributes:
            setattr(self, attr, None)
        
        if self.analysis:
            analysis_attributes = [
            't_intervals', 'analysis_checks', 'collect_losses',
            'collect_thresholds'
            ]
            if self.mol_idxs is not None:
                analysis_attributes.append('uncertainty_checks')
                
            for attr in analysis_attributes:
                setattr(self, attr, None)

        if self.rank == 0:
            logging.info("Restarting active learning procedure from checkpoint.")
            self.al_restart_dict = np.load(
                "restart/al/al_restart.npy",
                allow_pickle=True
            ).item()
            restart_keys = [
                'trajectories', 'MD_checkpoints', 'trajectory_intermediate_epochs', 'ensemble_reset_opt',
                'ensemble_no_improvement', 'ensemble_best_valid', 'threshold',
                'trajectory_status', 'trajectory_MD_steps', 'trajectory_total_epochs',
                'current_valid_error', 'total_points_added', 'train_points_added',
                'valid_points_added', 'num_MD_limits_reached', 'num_workers_training',
                'num_workers_waiting', 'total_epoch', 'check', 'uncertainties', 'converge_best',
                'uncert_not_crossed'
            ]
            if self.md_settings['stat_ensemble'].lower() in ['nvt', 'npt']:
                restart_keys.append('current_temperatures')

            for key in restart_keys:
                setattr(self, key, self.al_restart_dict[key])

            if self.analysis:
                restart_analysis_keys = [
                    't_intervals', 'analysis_checks', 'collect_losses',
                    'collect_thresholds'
                ]
                if self.mol_idxs is not None:
                    restart_analysis_keys.append('uncertainty_checks')
                    
                for key in restart_analysis_keys:
                    setattr(self, key, self.al_restart_dict[key])

                if self.mol_idxs is not None:
                    self.uncertainty_checks = []

        self.comm_handler.barrier()
        self.trajectory_status = self.comm_handler.bcast(self.trajectory_status, root=0)
        self.trajectory_MD_steps = self.comm_handler.bcast(self.trajectory_MD_steps, root=0)
        self.trajectory_total_epochs = self.comm_handler.bcast(self.trajectory_total_epochs, root=0)
        self.current_valid_error = self.comm_handler.bcast(self.current_valid_error, root=0)
        self.total_points_added = self.comm_handler.bcast(self.total_points_added, root=0)
        self.train_points_added = self.comm_handler.bcast(self.train_points_added, root=0)
        self.valid_points_added = self.comm_handler.bcast(self.valid_points_added, root=0)
        self.num_MD_limits_reached = self.comm_handler.bcast(self.num_MD_limits_reached, root=0)
        self.num_workers_training = self.comm_handler.bcast(self.num_workers_training, root=0)
        self.num_workers_waiting = self.comm_handler.bcast(self.num_workers_waiting, root=0)
        self.total_epoch = self.comm_handler.bcast(self.total_epoch, root=0)
        self.check = self.comm_handler.bcast(self.check, root=0)
        self.uncertainties = self.comm_handler.bcast(self.uncertainties, root=0)
        self.converge_best = self.comm_handler.bcast(self.converge_best, root=0)
        self.uncert_not_crossed = self.comm_handler.bcast(self.uncert_not_crossed, root=0)
        
        if self.md_settings['stat_ensemble'].lower() in ['nvt','npt']:
            self.current_temperatures = self.comm_handler.bcast(self.current_temperatures, root=0)
            
        if self.analysis:
            self.t_intervals = self.comm_handler.bcast(self.t_intervals, root=0)
            self.analysis_checks = self.comm_handler.bcast(self.analysis_checks, root=0)
            self.collect_losses = self.comm_handler.bcast(self.collect_losses, root=0)
            self.collect_thresholds = self.comm_handler.bcast(self.collect_thresholds, root=0)
            if self.mol_idxs is not None:
                self.uncertainty_checks = self.comm_handler.bcast(self.uncertainty_checks, root=0)
        self.comm_handler.barrier()

    def update_al_restart_dict(self, save_restart: str = None):
        self.al_restart_dict.update({
            'trajectories': self.trajectories,
            'MD_checkpoints': self.MD_checkpoints,
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
            'converge_best': self.converge_best,
            'uncert_not_crossed': self.uncert_not_crossed
        })
        
        if self.md_settings['stat_ensemble'].lower() in ['nvt', 'npt']:
            if self.md_settings['stat_ensemble'].lower() == 'nvt':
                get_temp = lambda x: x.temp
            elif self.md_settings['stat_ensemble'].lower() == 'npt':
                get_temp = lambda x: x.temperature

            self.al_restart_dict.update({
                'current_temperatures': {
                    trajectory: (get_temp(self.md_drivers[trajectory]) / units.kB) for trajectory in self.trajectories.keys()
                }
            })
            
        
        if self.analysis:
            self.al_restart_dict.update({
                't_intervals': self.t_intervals,
                'analysis_checks': self.analysis_checks,
                'collect_losses': self.collect_losses,
                'collect_thresholds': self.collect_thresholds
            })
            if self.mol_idxs is not None:
                self.al_restart_dict.update({
                    'uncertainty_checks': self.uncertainty_checks
                })
        if save_restart is not None:
            np.save(
                save_restart,
                self.al_restart_dict
            )
        
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

    def setup_aims_calculator(
            self,
            atoms: ase.Atoms,
            ):
        """
        Creates and returns the AIMS calculator for a given atoms object.

        Args:
            path_to_geometry (str, optional): Path to geometry file. Defaults to "./geometry.in".
        """
        
        aims_settings = self.aims_settings.copy()

        def init_via_ase(asi):
            from ase.calculators.aims import Aims, AimsProfile
            aims_settings["profile"] = AimsProfile(command="asi-doesnt-need-command")
            calc = Aims(**aims_settings)
            calc.write_inputfiles(asi.atoms, properties=self.properties)

        calc = ASI_ASE_calculator(
            self.ASI_path,
            init_via_ase,
            self.comm_handler.comm,
            atoms
            )
        return calc

    def recalc_aims(
            self,
            current_point: ase.Atoms
            ) -> ase.Atoms:
        self.aims_calculator.calculate(current_point, properties=self.properties)
        if self.aims_calculator.asi.is_scf_converged:
            current_point.info["REF_energy"] = self.aims_calculator.results["energy"]
            current_point.arrays["REF_forces"] = self.aims_calculator.results["forces"]
            if self.compute_stress:
                current_point.arrays["REF_stress"] = self.aims_calculator.results["stress"]
            return current_point
        else:
            if self.rank == 0:
                logging.info("SCF not converged.")
            return None

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
        if md_settings["stat_ensemble"].lower() in ['nvt', 'npt']:
            if not self.restart:
                try:
                    self.current_temperatures[idx] = md_settings['temperature']
                except AttributeError:
                    self.current_temperatures = {idx: md_settings['temperature']}
            
            if md_settings["stat_ensemble"].lower() == 'nvt':
                if md_settings['thermostat'].lower() == 'langevin':
                    dyn = Langevin(
                        atoms,
                        timestep=md_settings['timestep'] * units.fs,
                        friction=md_settings['friction'] / units.fs,
                        temperature_K=self.current_temperatures[idx],
                        rng=np.random.RandomState(md_settings['seed'])
                    )  
            elif md_settings["stat_ensemble"].lower() == 'npt':
                if md_settings['barostat'].lower() == 'berendsen':
                    npt_settings = {
                        'atoms': atoms,
                        'timestep': md_settings['timestep'] * units.fs,
                        'temperature': self.current_temperatures[idx],
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
                        'temperature_K': self.current_temperatures[idx],
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
                if self.rank == 0:
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

    def _finalize_ab_initio(
        self
    ):
        self.aims_calculator.asi.close()
        
    
    def _al_loop(self):
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
                    if self.rank == 0:
                        logging.info(
                            "All workers are in training mode."
                        )  
    
            
            if self.num_MD_limits_reached == self.num_trajectories:
                if self.rank == 0:
                    logging.info(
                        "All trajectories reached maximum MD steps."
                    )
                break
            
            if self.train_dataset_len >= self.max_set_size:
                if self.rank == 0:
                    logging.info(
                        "Maximum size of training set reached."
                    )
                break
            
            
            if self.current_valid_error < self.desired_accuracy:
                if self.rank == 0:
                    logging.info(
                        "Desired accuracy reached."
                    )
                break           
        self._finalize_ab_initio()

    def analysis_check(
    #TODO: put this somewhere else and its ugly
    #TODO: update docstring
            self,
            analysis_prediction: np.ndarray,
            true_forces: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the force uncertainty and the maximum 
        force error for a given prediction and true forces.

        Args:
            analysis_prediction (np.ndarray): Ensemble prediction. [n_members, n_points, n_atoms, 3]
            true_forces (np.ndarray): True forces. [n_points, n_atoms, 3]

        Returns:
            tuple[np.ndarray, np.ndarray]: force uncertainty, true force error
        """
        
        check_results = {}
        atom_wise_uncertainty = self.get_uncertainty.ensemble_sd(analysis_prediction)
        uncertainty_via_max = self.get_uncertainty.max_atomic_sd(atom_wise_uncertainty)
        uncertainty_via_mean = self.get_uncertainty.mean_atomic_sd(atom_wise_uncertainty)
        
        mean_analysis_prediction = analysis_prediction.mean(0).squeeze()

        diff_sq_mean = np.mean(
            (true_forces - mean_analysis_prediction)**2,
            axis=-1)
        
        max_error = np.max(np.sqrt(diff_sq_mean), axis=-1)
        mean_error = np.mean(np.sqrt(diff_sq_mean), axis=-1)
        atom_wise_error = np.sqrt(diff_sq_mean)
        
        check_results['prediction'] = mean_analysis_prediction
        check_results['true_forces'] = true_forces
        check_results['atom_wise_uncertainty'] = atom_wise_uncertainty
        check_results['uncertainty_via_max'] = uncertainty_via_max
        check_results['uncertainty_via_mean'] = uncertainty_via_mean
        check_results['atom_wise_error'] = atom_wise_error
        check_results['max_error'] = max_error
        check_results['mean_error'] = mean_error
        check_results['train_set_length'] = self.train_dataset_len
        
        if self.mol_idxs is not None:
            total_certainty = self.get_uncertainty(analysis_prediction)
            mol_forces_uncertainty = self.get_uncertainty.get_intermol_uncertainty(
                analysis_prediction
            )
            mol_forces_prediction = self.get_uncertainty.compute_mol_forces_ensemble(
                analysis_prediction,
                self.mol_idxs
                ).mean(0).squeeze()
            mol_forces_true =  self.get_uncertainty.compute_mol_forces_ensemble(
                true_forces.reshape(1, *true_forces.shape),
                self.mol_idxs
                ).squeeze()
            mol_diff_sq_mean = np.mean(
                (mol_forces_true - mol_forces_prediction)**2,
                axis=-1)
            max_mol_error = np.max(np.sqrt(mol_diff_sq_mean), axis=-1)
            mean_mol_error = np.mean(np.sqrt(mol_diff_sq_mean), axis=-1)
            mol_wise_error = np.sqrt(mol_diff_sq_mean)
            
            check_results['mol_forces_prediction'] = mol_forces_prediction
            check_results['mol_forces_true'] = mol_forces_true
            check_results['total_uncertainty'] = total_certainty
            check_results['mol_forces_uncertainty'] = mol_forces_uncertainty
            check_results['mol_wise_error'] = mol_wise_error
            check_results['max_mol_error'] = max_mol_error
            check_results['mean_mol_error'] = mean_mol_error
        
        return check_results
    
    def _handle_recieved_point(
            self,
            idx,
            recieved_point
        ):
        mace_point = create_mace_dataset(
                    data=[recieved_point],
                    z_table=self.z_table,
                    seed=None,
                    r_max=self.r_max,
                )
        if self.valid_points_added < self.valid_ratio * self.total_points_added:
            self.trajectory_status[idx] = "running"
            self.num_workers_waiting -= 1
            if self.rank == 0:
                logging.info(
                    f"Trajectory worker {idx} is adding a point to the validation set."
                )
            # while the initial datasets are different for each ensemble member we add the new points to
            # all ensemble member datasets
            if self.rank == 0:
                for tag in self.ensemble_ase_sets.keys():
                    self.ensemble_ase_sets[tag]["valid"] += [recieved_point]
                    self.ensemble_mace_sets[tag]["valid"] += mace_point
            self.valid_points_added += 1

        else:
            self.trajectory_status[idx] = "training"
            self.num_workers_training += 1
            self.num_workers_waiting -= 1
            if self.rank == 0:
                logging.info(
                    f"Trajectory worker {idx} is adding a point to the training set."
                )
                # while the initial datasets are different for each ensemble member we add the new points to
                # all ensemble member datasets
                for tag in self.ensemble_ase_sets.keys():
                    self.ensemble_ase_sets[tag]["train"] += [recieved_point]
                    self.ensemble_mace_sets[tag]["train"] += mace_point
                self.train_dataset_len = len(self.ensemble_ase_sets[tag]["train"])
            
            self.comm_handler.barrier()
            self.train_dataset_len = self.comm_handler.bcast(self.train_dataset_len, root=0)
            self.comm_handler.barrier()
            
            if self.train_dataset_len > self.max_set_size:
                return True
            if self.rank == 0:
                logging.info(
                    f"Size of the training and validation set: {self.train_dataset_len}, {len(self.ensemble_ase_sets[tag]['valid'])}."
                )
            self.train_points_added += 1
        self.total_points_added += 1
        
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
        self._handle_recieved_point(
            idx,
            recieved_point=self.point  #TODO: change self.point, i dont like it
        )


    
    def _check_batch_size(self, set_batch_size, tag):
        batch_size = (
            1
            if len(self.ensemble_mace_sets[tag]["train"])
            < set_batch_size
            else set_batch_size
                )
        return batch_size

    def _prepare_training(
            self,
            mace_sets: dict,
            
    ):
        for _, (tag, model) in enumerate(self.ensemble.items()):            
            
            train_batch_size = self._check_batch_size(self.set_batch_size, tag)
            valid_batch_size = self._check_batch_size(self.set_valid_batch_size, tag)
            
            (
                mace_sets[tag]["train_loader"],
                mace_sets[tag]["valid_loader"],
            ) = create_dataloader(
                mace_sets[tag]["train"],
                mace_sets[tag]["valid"],
                train_batch_size,
                valid_batch_size,
            )
            # because the dataset size is dynamically changing 
            # we have to update the average number of neighbors,
            # shifts and the scaling factor for the models
            # usually they converge pretty fast
            update_model_auxiliaries(
                model=model,
                mace_sets=mace_sets[tag],
                atomic_energies=self.ensemble_atomic_energies[tag],
                scaling=self.scaling,
                update_atomic_energies=self.update_atomic_energies,
                z_table=self.z_table,
                atomic_energies_dict=self.ensemble_atomic_energies_dict[tag],
                dtype=self.dtype,
                device=self.device,
            )
        return mace_sets

    def _perform_training(
            self,
            idx: int = 0
    ):
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
                self.update_al_restart_dict(
                    save_restart="restart/al/al_restart.npy"
                )
                self.save_restart = False

        self.trajectory_intermediate_epochs[idx] = 0

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
        if self.rank == 0:

            self.ensemble_mace_sets = self._prepare_training(
                mace_sets=self.ensemble_mace_sets
            )

            logging.info(f"Trajectory worker {idx} is training.")
            # we train only for some epochs before we move to the next worker which may be running MD
            # all workers train on the same models with the respective training settings for
            # each ensemble member
            
            self._perform_training(idx)
                    
        # update calculators with the new models
        self.comm_handler.barrier()
        self.current_valid_error = self.comm_handler.bcast(self.current_valid_error, root=0)
        self.comm_handler.barrier()
        if self.rank == 0:
            for trajectory in self.trajectories.values():
                trajectory.calc.models = [self.ensemble[tag] for tag in self.ensemble.keys()]

        self.comm_handler.barrier()
        self.total_epoch = self.comm_handler.bcast(self.total_epoch, root=0)
        self.trajectory_total_epochs[idx] = self.comm_handler.bcast(self.trajectory_total_epochs[idx], root=0)
        self.comm_handler.barrier()

        if self.trajectory_total_epochs[idx] >= self.max_epochs_worker:
            self.trajectory_status[idx] = "running"
            self.num_workers_training -= 1
            self.trajectory_total_epochs[idx] = 0
            if self.rank == 0:
                logging.info(f"Trajectory worker {idx} finished training.")
            # calculate true error and uncertainty on validation set

    def _analysis_dft_call(
            self,
            point: ase.Atoms,
            idx: int = None
    ):
        self.aims_calculator.calculate(
            point,
            properties=self.properties
        )
        return self.aims_calculator.asi.is_scf_converged
    
    def _save_analysis(
            self
    ):
        np.savez(
            "analysis/analysis_checks.npz",
            self.analysis_checks
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

    def _process_analysis(
            self,
            idx: int,
            converged: bool,
            analysis_prediction: np.ndarray
    ):
        if converged:
            check_results = self.analysis_check(
                analysis_prediction=analysis_prediction,
                true_forces = self.aims_calculator.results["forces"]
            )
            for key in check_results.keys():
                self.analysis_checks[idx][key].append(check_results[key])
            self.analysis_checks[idx]['threshold'].append(self.threshold)

            
            self.collect_thresholds[idx].append(self.threshold)
            
            self.check += 1
            
            self._save_analysis()

        else:
            if self.rank == 0:
                logging.info(
                    f"SCF not converged at worker {idx} for analysis. Discarding point."
                )

    def _perform_analysis(
            self,
            idx,
            prediction,
            current_MD_step,
            uncertainty,
            ):

        self.t_intervals[idx].append(current_MD_step)           
        if self.mol_idxs is not None:
            self.uncertainty_checks.append(
                uncertainty > self.threshold
                )
        if (
            current_MD_step % self.analysis_skip == 0
        ):  
            if self.rank == 0:
                logging.info(f"Trajectory worker {idx} is sending a point to DFT for analysis.")
            
            if current_MD_step % self.skip_step == 0:
                self.trajectories_analysis_prediction[idx] = prediction
            else:
                if self.rank == 0:
                    self.trajectories_analysis_prediction[idx] = ensemble_prediction(
                        models=list(self.ensemble.values()),
                        atoms_list=[self.point],
                        device=self.device,
                        dtype=self.mace_settings["GENERAL"]["default_dtype"],
                    )
                self.comm_handler.barrier()
                self.trajectories_analysis_prediction[idx] = self.comm_handler.bcast(
                    self.trajectories_analysis_prediction[idx], root=0
                    )
                self.comm_handler.barrier()

            #TODO: sometimes already calculated above so we should not calculate it again
            self.comm_handler.barrier()
            converged = self._analysis_dft_call(point=self.point, idx=idx)
            self.comm_handler.barrier()

            self._process_analysis(
                idx=idx,
                converged=converged,
                analysis_prediction=self.trajectories_analysis_prediction[idx]
            )

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
            if self.rank == 0:
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
                if self.rank == 0:
                    modified = self.md_modifier(
                        driver=self.md_drivers[idx],
                        metric=self.get_md_mod_metric(),
                        idx=idx
                    )
                    if modified and self.create_restart:
                        self.update_al_restart_dict()

            if self.rank == 0:
                self.md_drivers[idx].run(self.skip_step)
                self.trajectory_MD_steps[idx] += self.skip_step
                current_MD_step += self.skip_step

            # somewhat arbitrary; i just want to save checkpoints if the MD phase
            # is super long
            if self.rank == 0:
                if current_MD_step % (self.skip_step * 100) == 0:
                    self.update_al_restart_dict(
                        save_restart="restart/al/al_restart.npy"
                    )

                logging.info(
                    f"Trajectory worker {idx} at MD step {current_MD_step}."
                )
                
                self.point = self.trajectories[idx].copy()
                prediction = self.trajectories[idx].calc.results["forces_comm"]
                uncertainty = self.get_uncertainty(prediction)

                self.uncertainties.append(uncertainty)

                if len(self.uncertainties) > 10: #TODO: remove hardcode
                    if (
                        self.train_dataset_len >= self.freeze_threshold_dataset
                        ) and not self.freeze_threshold:
                        if self.rank == 0:
                            logging.info(f'Train data has reached size {self.train_dataset_len}: freezing threshold at {self.threshold:.3f}.')
                        self.freeze_threshold = True
                         
                    if not self.freeze_threshold:
                        self.threshold = get_threshold(
                            uncertainties=self.uncertainties,
                            c_x = self.c_x,
                            max_len = 400 #TODO: remove hardcode
                        )
                    
                    if self.analysis:
                        self.collect_thresholds[
                            idx
                            ].append(self.threshold)
        
            if self.rank != 0:
                uncertainty = None
                prediction = None
                self.point = None
                self.threshold = None
                current_MD_step = None
                
            self.comm_handler.barrier()
            self.threshold = self.comm_handler.bcast(self.threshold, root=0)
            self.point = self.comm_handler.bcast(self.point, root=0)
            uncertainty = self.comm_handler.bcast(uncertainty, root=0)
            prediction = self.comm_handler.bcast(prediction, root=0)
            current_MD_step = self.comm_handler.bcast(current_MD_step, root=0)
            self.comm_handler.barrier()
            
            if (uncertainty > self.threshold).any() or self.uncert_not_crossed[idx] > self.skip_step * self.uncert_not_crossed_limit:
                self.uncert_not_crossed[idx] = 0
                if self.rank == 0:
                    if (uncertainty > self.threshold).any():
                        logging.info(
                            f"Uncertainty of point is beyond threshold {np.round(self.threshold,3)} at worker {idx}: {np.round(uncertainty,3)}."
                        )
                    else:
                        logging.info(
                            f"Threshold not crossed at trajectory worker {idx} for {self.skip_step*self.uncert_not_crossed_limit} steps."
                        )

                # at the moment the process waits for the calculation to finish
                # ideally it should calculate in the background and the other
                # workers sample/train in the meantime
                
                self._handle_dft_call(
                    idx
                )
      
            else:
                self.uncert_not_crossed[idx] += 1
            
            if self.analysis:
                self._perform_analysis(
                    idx=idx,
                    prediction=prediction,
                    current_MD_step=current_MD_step,
                    uncertainty=uncertainty
                )
    
    def _handle_dft_call(
            self,
            idx
    ):
        if self.rank == 0:
            logging.info(
                f"Trajectory worker {idx} is running DFT."
            )
        
        self.comm_handler.barrier()
        self.point = self.recalc_aims(self.point)

        if not self.aims_calculator.asi.is_scf_converged:
            if self.rank == 0:
                logging.info(
                    f"SCF not converged at worker {idx}. Discarding point and restarting MD from last checkpoint."
                )
                self.trajectories[idx] = atoms_full_copy(self.MD_checkpoints[idx])
            self.trajectory_status[idx] = "running"
        else:
            # we are updating the MD checkpoint here because then we make sure
            # that the MD is restarted from a point that is inside the training set
            # so the MLFF should be able to handle this and lead to a better trajectory
            # that does not lead to convergence issues
            if self.rank == 0:
                self.MD_checkpoints[idx] = atoms_full_copy(self.point)
            self.trajectory_status[idx] = "waiting"
            self.num_workers_waiting += 1
        
            self.comm_handler.barrier()
            self.waiting_task(idx)
            if self.rank == 0:
                logging.info(
                    f"Trajectory worker {idx} is going to add point to the dataset."
                )

    def run(self):
        """
        Main function to run the active learning procedure. Initializes variables and
        controls the workers tasks.
        """

        if self.rank == 0:
            logging.info("Starting active learning procedure.")


        self.comm_handler.barrier()
        self._al_loop()
        # turn keys which are ints into strings
        # save the datasets and the intervals for analysis
        if self.rank == 0:
            logging.info(f"Active learning procedure finished. The best ensemble member based on validation loss is {self.best_member}.")
            save_datasets(
                ensemble=self.ensemble,
                ensemble_ase_sets=self.ensemble_ase_sets,
                path=self.dataset_dir / "final",
            )

            if self.analysis:
                self._save_analysis()
            
            if self.create_restart:
                self.update_al_restart_dict(
                    save_restart="restart/al/al_restart.npy"
                )
                self.al_restart_dict["al_done"] = True

    def converge(self):
        """
        Converges the ensemble on the acquired dataset. Trains the ensemble members
        until the validation loss does not improve anymore.
        """
        if self.rank == 0:
            if self.converge_best:
                logging.info(f"Converging best model ({self.best_member}) on acquired dataset.")
                self.ensemble = {self.best_member: self.ensemble[self.best_member]}
            else:
                logging.info("Converging ensemble on acquired dataset.")
                
            temp_mace_sets = {}
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
                temp_mace_sets[tag] = {"train": train_set, "valid": valid_set}

            self.ensemble_mace_sets = self._prepare_training(
                mace_sets=temp_mace_sets
            )

            # resetting optimizer and scheduler
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
                epoch = self.training_setups_convergence[list(self.ensemble.keys())[0]]["epoch"]
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

                if epoch % self.valid_skip == 0 or epoch == self.max_final_epochs - 1:
                    (
                        ensemble_valid_losses,
                        valid_loss,
                        _,
                    ) = validate_epoch_ensemble(
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


class ALProcedureParallel(ALProcedure):
    def __init__(                 
        self,
        mace_settings: dict,
        al_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
    ):
        
        self.comm_handler = CommHandler()
        self.world_comm = self.comm_handler.comm
        self.rank = self.comm_handler.get_rank()
        self.world_size = self.comm_handler.get_size()
        # one for ML and one for DFT
        if self.rank == 0:
            self.color = 0
        else:
            self.color = 1
        
        comm = self.comm_handler.comm.Split(
            color=self.color,
            key=self.rank
            )
        self.comm_handler.comm = comm
        self.comm_handler.size = comm.Get_size()
        
        # this is necessary because of the way the MPI communicator is split
        super().__init__(
            mace_settings=mace_settings,
            al_settings=al_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
            comm_handler=self.comm_handler
        )


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
                self.comm_handler.comm,
                atoms
                )
            return calc
        else:
            return None

    def _send_kill(self):
        """
        Sends a kill signal
        """

        for dest in range(1, self.world_size):
            self.kill_send = self.world_comm.isend(True, dest=dest, tag=422)
            self.kill_send.Wait()
        
        #for trajectory_idx in range(self.num_trajectories):
        #    if self.worker_reqs[trajectory_idx] is not None:
        #        self.worker_reqs[trajectory_idx].cancel()

        #if self.point_send is not None:
        #    self.point_send.cancel()

    def _al_loop(
        self
    ):
        self.worker_reqs = {idx: None for idx in range(self.num_trajectories)}
        self.req = None

        if self.analysis:
            self.req_analysis = None


        if self.analysis:
            self.analysis_worker_reqs = {idx: None for idx in range(self.num_trajectories)}

        if self.color == 1:
            self.req_kill = self.world_comm.irecv(source=0, tag=422)
        
        while True:
            if self.color == 0: 
                for trajectory_idx in range(self.num_trajectories):
                    
                    if self.trajectory_status[trajectory_idx] == "running":

                        self.running_task(trajectory_idx)

                    if (
                        self.trajectory_status[trajectory_idx] == "training"
                    ):  # and training_job: # the idea is to let a worker train only if new points
                        # have been added. e.g. it can happen that one worker is
                        # beyond its MD limit but there is no new point that has been added

                        self.training_task(trajectory_idx)

                    if self.trajectory_status[trajectory_idx] == "waiting":

                        set_limit = self.waiting_task(trajectory_idx)
                        if set_limit: # stops the process if the maximum dataset size is reached
                            self._send_kill()
                            break
                    
                    if self.analysis:
                        if self.trajectory_status[trajectory_idx] == "analysis_waiting":
                            self._analysis_waiting_task(trajectory_idx)

                    if (
                        self.num_workers_training == self.num_trajectories
                    ):  
                        if self.rank == 0:
                            logging.info(
                                "All workers are in training mode."
                            )  
                    
                    #if self.num_workers_waiting == self.num_trajectories:
                    #    if self.rank == 0:
                    #        logging.info("All workers are waiting for jobs to finish.")
                
                if self.num_MD_limits_reached == self.num_trajectories:
                    if self.rank == 0:
                        logging.info(
                            "All trajectories reached maximum MD steps."
                        )
                        self._send_kill()
                    break
                
                if self.train_dataset_len >= self.max_set_size:
                    if self.rank == 0:
                        logging.info(
                            "Maximum size of training set reached."
                        )
                        self._send_kill()
                    break
                
                
                if self.current_valid_error < self.desired_accuracy:
                    if self.rank == 0:
                        logging.info(
                            "Desired accuracy reached."
                        )
                        self._send_kill()
                    break
                
            if self.color == 1:
                # TODO: put everything into a function
                kill_signal = self.req_kill.Test()
                if kill_signal:
                    self.aims_calculator.asi.close()
                    #self.req.cancel() if self.req is not None else None
                    #self.req_send.cancel() if self.req_send is not None and not self.req_send.Test() else None
                    break
                
                if self.req is None:
                    self.req = self.world_comm.irecv(source=0, tag=1234)

                status, data = self.req.test()
                
                if status:
                    self.req = None
                    idx, point = data
                    if self.rank == 1:
                        logging.info(f"Recieved point from worker {idx} and running DFT calculation.")
                    # change return from none to false when scf not converged
                    dft_result = self.recalc_aims(point)
                    if dft_result is not None:
                        if self.rank == 1:
                            logging.info(f"DFT calculation for worker {idx} finished and sending point back.")
                            self.req_send = self.world_comm.isend(dft_result, dest=0, tag=idx)
                            self.req_send.Wait()
                    else:
                        if self.rank == 1:
                            self.req_send = self.world_comm.isend(False, dest=0, tag=idx)
                            self.req_send.Wait()

                if self.analysis:
                    if self.req_analysis is None:
                        self.req_analysis = self.world_comm.irecv(source=0, tag=80545)
                    
                    status_analysis, data_analysis = self.req_analysis.test()

                    if status_analysis:
                        self.req_analysis = None
                        idx, point_analysis = data_analysis
                        dft_result_analysis = self.recalc_aims(point_analysis)
                        if dft_result_analysis is not None:

                            if self.rank == 1:
                                logging.info(f"DFT calculation for worker {idx} analysis finished and sending point back.")
                                self.req_send_analysis = self.world_comm.isend(dft_result_analysis, dest=0, tag=idx)
                                self.req_send_analysis.Wait()
                        else:
                            if self.rank == 1:
                                self.req_send_analysis = self.world_comm.isend(False, dest=0, tag=idx)
                                self.req_send_analysis.Wait()        

        if self.color == 1:
            self.aims_calculator.asi.close()
            
    def waiting_task(
            self,
            idx: int
            ):
        """
        TODO

        Args:
            idx (int): Index of the trajectory worker.

        """

        if self.worker_reqs[idx] is None:
            self.worker_reqs[idx] = self.world_comm.irecv(source=1, tag=idx)
        
        status, recieved_point = self.worker_reqs[idx].test()
        
        if status:
            self.worker_reqs[idx] = None

            if not recieved_point: # DFT not converged
                if self.rank == 0:
                    logging.info(
                        f"SCF not converged at worker {idx}. Discarding point and restarting MD from last checkpoint."
                    )
                    self.trajectories[idx] = atoms_full_copy(self.MD_checkpoints[idx])
                self.trajectory_status[idx] = "running"
            
            else:
                logging.info(f"Worker {idx} recieved a point.")
                
                # we are updating the MD checkpoint here because then we make sure
                # that the MD is restarted from a point that is inside the training set
                # and using the DFT forces
                if self.rank == 0:
                    self.MD_checkpoints[idx] = atoms_full_copy(recieved_point)

                self._handle_recieved_point(
                    idx=idx,
                    recieved_point=recieved_point
                )
        
    def _handle_dft_call(
            self,
            idx
    ):
        self.comm_handler.barrier()
        if self.rank == 0:
            logging.info(f"Trajectory worker {idx} is sending point to DFT.")
            for dest in range(1, self.world_size):
                self.point_send = self.world_comm.isend((idx, self.point), dest=dest, tag=1234)
                self.point_send.Wait()

        self.comm_handler.barrier()
        self.trajectory_status[idx] = "waiting"
        self.num_workers_waiting += 1
        self.comm_handler.barrier()

        if self.rank == 0:
            logging.info(
                f"Trajectory worker {idx} is waiting for job to finish."
            )        

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
        # TODO: why is this here? why can't i use the parents method?
        self.ensemble_mace_sets = self._prepare_training(
            mace_sets=self.ensemble_mace_sets
        )

        logging.info(f"Trajectory worker {idx} is training.")
        # we train only for some epochs before we move to the next worker which may be running MD
        # all workers train on the same models with the respective training settings for
        # each ensemble member
        
        self._perform_training(idx)

        for trajectory in self.trajectories.values():
            trajectory.calc.models = [self.ensemble[tag] for tag in self.ensemble.keys()]

        if self.trajectory_total_epochs[idx] >= self.max_epochs_worker:
            self.trajectory_status[idx] = "running"
            self.num_workers_training -= 1
            self.trajectory_total_epochs[idx] = 0
            if self.rank == 0:
                logging.info(f"Trajectory worker {idx} finished training.")
            # calculate true error and uncertainty on validation set

    def _process_analysis(
            self,
            idx: int,
            converged: bool,
            analysis_prediction: np.ndarray
    ):
        # Dummy to overwrite the parent method
        return

    def _analysis_waiting_task(
            self,
            idx: int
            ):
        if self.analysis_worker_reqs[idx] is None:
            self.analysis_worker_reqs[idx] = self.world_comm.irecv(source=1, tag=idx)

        status, recieved_points = self.analysis_worker_reqs[idx].test()        

        if status:
            self.analysis_worker_reqs[idx] = None
            
            if not recieved_points:
                if self.rank == 0:
                    logging.info(
                        f"SCF not converged at worker {idx} for analysis. Discarding point."
                    )
                self.trajectory_status[idx] = "running"
            else:
                self.analysis_true_forces = recieved_points.arrays["REF_forces"]

                check_results = self.analysis_check(
                    analysis_prediction=self.trajectories_analysis_prediction[idx],
                    true_forces = self.analysis_true_forces
                    )

                for key in check_results.keys():
                    self.analysis_checks[idx][key].append(check_results[key])

                self.analysis_checks[idx]['threshold'].append(self.threshold)
                self.collect_thresholds[idx].append(self.threshold)
                self.check += 1
                self._save_analysis()
                self.trajectory_status[idx] = "running"

    def _analysis_dft_call(self, idx, point):
        self.comm_handler.barrier()
        if self.rank == 0:
            for dest in range(1, self.world_size):
                self.point_send = self.world_comm.isend((idx, point), dest=dest, tag=80545)
                self.point_send.Wait()
        self.comm_handler.barrier()
        self.trajectory_status[idx] = "analysis_waiting"
        self.comm_handler.barrier()
        if self.rank == 0:
            logging.info(
                f"Trajectory worker {idx} is waiting for analysis job to finish."
            )
        return None


class ALProcedurePARSL(ALProcedure):
    """
    This class is a placeholder for the PARSL implementation of the active learning procedure.
    It is not implemented yet and will raise a NotImplementedError if used.
    """

    def __init__(
        self,
        mace_settings: dict,
        al_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
    ):
        
        if parsl is None:
            raise ImportError(
                "PARSL is not installed. Please install PARSL to use this feature."
            )
        super().__init__(
            mace_settings=mace_settings,
            al_settings=al_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
            use_mpi=False
        )
        try:
            parsl.dfk()
            logging.info("PARSL is already initialized. Using existing PARSL context.")
        except parsl.errors.NoDataFlowKernelError:
            parsl_setup_dict = prepare_parsl(
                cluster_settings=self.cluster_settings
            )
            self.config = parsl_setup_dict["config"]
            self.calc_dir = parsl_setup_dict["calc_dir"]
            self.clean_dirs = parsl_setup_dict["clean_dirs"]
            self.launch_str = parsl_setup_dict["launch_str"]
            
            handle_parsl_logger(
                log_dir=self.log_dir / "al_parsl.log"
            )
            parsl.load(self.config)
                        
        logging.info("Launching ab initio manager thread for PARSL.")
        self.ab_initio_queue = queue.Queue()
        self.ab_intio_results = {}
        self.results_lock = threading.Lock()
        self.kill_thread = False
        threading.Thread(
            target=self.ab_initio_manager,
            daemon=True
        ).start()
    
        
    def _handle_dft_call(
        self,
        idx: int
    ):
        logging.info(f"Trajectory worker {idx} is sending point to DFT.")
        self.trajectory_status[idx] = "waiting"
        self.num_workers_waiting += 1
        self.ab_initio_queue.put((idx, self.trajectories[idx]))
        logging.info(f"Trajectory worker {idx} is waiting for job to finish.")
    
    def ab_initio_manager(
        self
    ):
        # collect parsl futures
        futures = {}
        # constantly check the queue for new jobs
        while True:
            
            if self.kill_thread:
                logging.info("Ab initio manager thread is stopping.")
                break
            try:
                idx, data = self.ab_initio_queue.get(timeout=1)
                futures[idx] = recalc_aims_parsl(
                    positions=data.get_positions(),
                    species=data.get_chemical_symbols(),
                    cell=data.get_cell(),
                    pbc=data.pbc,
                    aims_settings=self.aims_settings,
                    directory=self.calc_dir / f"worker_{idx}",
                    properties=self.properties,
                    ase_aims_command=self.launch_str
                )
                self.ab_initio_queue.task_done()
            
            # is raised when the queue is empty after the timeout
            except queue.Empty:
                pass
            
            # goes through the futures and checks if they are done
            done_jobs = [
                job_idx for job_idx, future in futures.items() if future.done()
            ]
            # if there are done jobs, get the results and store them in dict
            for job_idx in done_jobs:
                with self.results_lock:
                    temp_result = futures[job_idx].result()
                    if temp_result is None:
                        # if the result is None, it means the DFT calculation did not converge
                        self.ab_intio_results[job_idx] = False
                    else:
                        # the DFT calculation converged
                        self.ab_intio_results[job_idx] = temp_result
                        logging.info(f"DFT calculation for worker {job_idx} finished.")
                    # remove the job from the futures dict to avoid double counting
                    del futures[job_idx]
                    # remove folder with results
                    try:
                        shutil.rmtree(self.calc_dir / f"worker_{job_idx}")
                    except FileNotFoundError:
                        logging.warning(
                            f"Directory {self.calc_dir / f'worker_{job_idx}'} not found. Skipping removal."
                        )

    def setup_aims_calculator(self, atoms):
        pass
    
    
    def recalc_aims(self, current_point):
        pass
    
    def _finalize_ab_initio(self):
        with threading.Lock():
            self.kill_thread = True
        time.sleep(5)
        parsl.dfk().cleanup()
    
    def waiting_task(self, idx):
        
        with self.results_lock:
            job_result = self.ab_intio_results.get(idx, "not_done")
        
        if job_result == "not_done":
            # if the job is not done, we return None and wait for the next iteration
            return None
        else:
            if not job_result:
                logging.info(
                    f"SCF not converged at worker {idx}. Discarding point and restarting MD from last checkpoint."
                )
                self.trajectories[idx] = atoms_full_copy(self.MD_checkpoints[idx])
                self.trajectory_status[idx] = "running"
        
            else:
                logging.info(f"Worker {idx} recieved a point.")
                recieved_point = self.trajectories[idx].copy()
                recieved_point.info["REF_energy"] = job_result['energy']
                recieved_point.arrays["REF_forces"] = job_result['forces']
                if self.compute_stress:
                    recieved_point.arrays["REF_stress"] = job_result['stress']
                self.MD_checkpoints[idx] = atoms_full_copy(recieved_point)
                
                self._handle_recieved_point(
                    idx=idx,
                    recieved_point=recieved_point
                )
            with self.results_lock:
                # remove the job from the results dict to avoid double counting
                del self.ab_intio_results[idx]
                
                
                
                
                
            
    
    
if False:

    class ALProcedureGPUParallel(ALProcedureParallel):
        def __init__(
            self,
            mace_settings: dict,
            al_settings: dict,
            path_to_control: str = "./control.in",
            path_to_geometry: str = "./geometry.in",
        ):
            
            raise NotImplementedError(
                "This class is not implemented yet. Please use ALProcedureParallel."
            )
            super().__init__(
                mace_settings=mace_settings,
                al_settings=al_settings,
                path_to_control=path_to_control,
                path_to_geometry=path_to_geometry,
            )
            # get number of GPUs
            self.num_gpus = torch.cuda.device_count()
            self.devices = [torch.device(f"cuda:{i}") for i in range(self.num_gpus)]
            # create one mpi communcatior per gpu and assign rest to extra communicator
            self.world_size = self.world_size
            self.gpu_world_size = self.num_gpus
            self.gpu_world_comm = self.comm_handler.comm.Split(color=0, key=0)
            self.extra_comm = self.comm_handler.comm.Split(color=1, key=0)
            self.gpu_ranks = self.gpu_world_comm.Get_size()
            self.extra_ranks = self.extra_comm.Get_size()
            self.gpu_rank = self.gpu_world_comm.Get_rank()
            self.extra_rank = self.extra_comm.Get_rank()

    class ALProcedureParsl(ALProcedureParallel):
        raise NotImplementedError(
            "This class is not implemented yet. Please use ALProcedureParallel or ALProcedureGPUParallel."
        )
            
            

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
            
        def analysis_check(
        #TODO: put this somewhere else and its ugly
                self,
                analysis_prediction: np.ndarray,
                true_forces: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
            """
            Calculates the force uncertainty and the maximum 
            force error for a given prediction and true forces.

            Args:
                analysis_prediction (np.ndarray): Ensemble prediction. [n_members, n_points, n_atoms, 3]
                true_forces (np.ndarray): True forces. [n_points, n_atoms, 3]

            Returns:
                tuple[np.ndarray, np.ndarray]: force uncertainty, true force error
            """
            uncertainty_via_max = max_sd_2(analysis_prediction)
            uncertainty_via_avg = avg_sd(analysis_prediction)
            atom_wise_uncertainty = atom_wise_sd(analysis_prediction)
            mean_analysis_prediction = analysis_prediction.mean(0).squeeze()
            difference = true_forces - mean_analysis_prediction
            diff_sq = difference**2
            diff_sq_mean = np.mean(diff_sq, axis=-1)
            
            max_error = np.max(np.sqrt(diff_sq_mean), axis=-1)
            mean_error = np.mean(np.sqrt(diff_sq_mean), axis=-1)
            atom_wise_error = atom_wise_f_error(mean_analysis_prediction, true_forces)

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
            check = self.analysis_check(ensemble_forces, reference_forces)
            np.savez("check.npz", **check)
