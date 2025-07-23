import os
from pathlib import Path
import torch
import numpy as np
from mace import tools
from mace.calculators import MACECalculator
from aims_PAX.tools.uncertainty import (
    HandleUncertainty,
    MolForceUncertainty,
    get_threshold,
)
from aims_PAX.tools.utilities.data_handling import (
    create_dataloader,
    save_datasets,
    load_ensemble_sets_from_folder,
    ase_to_mace_ensemble_sets,
    create_mace_dataset,
)
from aims_PAX.tools.utilities.eval_tools import (
    ensemble_prediction,
)
from aims_PAX.tools.utilities.utilities import (
    ensemble_training_setups,
    ensemble_from_folder,
    update_model_auxiliaries,
    save_checkpoint,
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
    ModifyMD,
)
from aims_PAX.tools.utilities.parsl_tools import (
    prepare_parsl,
    recalc_aims_parsl,
    handle_parsl_logger,
)
import shutil
from aims_PAX.tools.setup_MACE_training import (
    setup_mace_training,
    reset_optimizer,
)
from aims_PAX.tools.train_epoch_mace import (
    train_epoch,
    validate_epoch_ensemble,
)
import ase
from ase.io import read
import logging
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
try:
    import asi4py
except Exception as e:
    asi4py = None


class ALConfigurationManager:
    """Handles all configuration setup for active learning."""
    
    def __init__(self, mace_settings: dict, al_settings: dict):
        self.mace_settings = mace_settings
        self.al_settings = al_settings
        self.al = al_settings["ACTIVE_LEARNING"]
        self.md_settings = al_settings["MD"]
        self.cluster_settings = al_settings.get("CLUSTER", None)
        self.misc = al_settings.get("MISC", {})
        
        self._setup_mace_configuration()
        self._setup_al_configuration()
    
    def _setup_mace_configuration(self):
        """Setup MACE-specific configuration."""
        general = self.mace_settings["GENERAL"]
        self.seed = general["seed"]
        self.checkpoints_dir = f"{general['checkpoints_dir']}/al"
        self.model_dir = general["model_dir"]
        self.dtype = general["default_dtype"]
        self.compute_stress = general.get("compute_stress", False)
        
        architecture = self.mace_settings["ARCHITECTURE"]
        self.r_max = architecture["r_max"]
        self.atomic_energies_dict = architecture.get("atomic_energies", None)
        
        training = self.mace_settings["TRAINING"]
        self.set_batch_size = training["batch_size"]
        self.set_valid_batch_size = training["valid_batch_size"]
        self.scaling = training["scaling"]
        
        self.device = self.mace_settings["MISC"]["device"]
        torch.set_default_dtype(dtype_mapping[self.dtype])
        
        self.properties = ["energy", "forces"]
        if self.compute_stress:
            self.properties.append("stress")
    
    def _setup_al_configuration(self):
        """Setup active learning configuration."""
        # Training parameters
        self.max_MD_steps = self.al["max_MD_steps"]
        self.max_epochs_worker = self.al["max_epochs_worker"]
        self.max_final_epochs = self.al["max_final_epochs"]
        self.intermediate_epochs = self.al["intermediate_epochs"]
        self.patience = self.al["patience"]
        self.desired_accuracy = self.al["desired_acc"]
        
        # Data parameters
        self.num_trajectories = self.al["num_trajectories"]
        self.skip_step = self.al["skip_step_mlff"]
        self.valid_skip = self.al["valid_skip"]
        self.analysis_skip = self.al["analysis_skip"]
        self.valid_ratio = self.al["valid_ratio"]
        self.max_set_size = self.al["max_set_size"]
        self.c_x = self.al["c_x"]
        
        # Paths
        self.dataset_dir = Path(self.al["dataset_dir"])
        self.species_dir = self.al["species_dir"]
        self.ASI_path = self.al["aims_lib_path"]
        
        # Optional settings
        self.analysis = self.al.get("analysis", False)
        self.seeds_tags_dict = self.al.get("seeds_tags_dict", None)
        
        # Uncertainty parameters
        self.margin = self.al.get("margin", 0.001)
        self.converge_best = self.al.get("converge_best", True)
        self.uncertainty_type = self.al.get("uncertainty_type", "max_atomic_sd")
        self.uncert_not_crossed_limit = self.al.get("uncert_not_crossed_limit", 500)
        self.freeze_threshold_dataset = self.al.get("freeze_threshold_dataset", np.inf)
        self.freeze_threshold = False
        
        # Molecular indices
        self._setup_molecular_indices()
        
        # Restart handling
        self.restart = os.path.exists("restart/al/al_restart.npy")
        self.create_restart = self.misc.get("create_restart", False)
    
    def _setup_molecular_indices(self):
        """Setup molecular indices configuration."""
        mol_idxs_path = self.al.get("mol_idxs", None)
        self.mol_idxs = (
            np.load(mol_idxs_path, allow_pickle=True)["arr_0"].tolist()
            if mol_idxs_path is not None
            else None
        )
        
        if self.mol_idxs is not None:
            self.intermol_crossed = 0
            self.intermol_crossed_limit = self.al.get("intermol_crossed_limit", 10)
            self.intermol_forces_weight = self.al.get("intermol_forces_weight", 100)
            self.switched_on_intermol = False
            
            # Check if using intermolecular loss
            loss_type = self.mace_settings["TRAINING"]["loss"].lower()
            self.using_intermol_loss = loss_type == "intermol"


class ALStateManager:
    """Manages the state of trajectories, ensembles, and analysis data."""
    
    def __init__(self, config: ALConfigurationManager, comm_handler):
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
        
        # Ensemble state
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
        
        # Analysis state
        if self.config.analysis:
            self._initialize_analysis_state()
    
    def initialize_fresh_state(self, path_to_geometry: str):
        """Initialize state for a fresh run."""
        geometry = read(path_to_geometry)
        num_traj = self.config.num_trajectories
        
        # Initialize trajectory data
        self.trajectories = {i: geometry.copy() for i in range(num_traj)}
        self.MD_checkpoints = {i: self.trajectories[i].copy() for i in range(num_traj)}
        self.trajectory_status = {i: "running" for i in range(num_traj)}
        self.trajectory_MD_steps = {i: 0 for i in range(num_traj)}
        self.trajectory_total_epochs = {i: 0 for i in range(num_traj)}
        self.trajectory_intermediate_epochs = {i: 0 for i in range(num_traj)}
        self.uncert_not_crossed = {i: 0 for i in range(num_traj)}
        
        # Initialize ensemble data
        if hasattr(self, 'seeds_tags_dict'):
            tags = self.config.seeds_tags_dict.keys()
            self.ensemble_reset_opt = {tag: False for tag in tags}
            self.ensemble_no_improvement = {tag: 0 for tag in tags}
            self.ensemble_best_valid = {tag: np.inf for tag in tags}
    
    def _initialize_analysis_state(self):
        """Initialize analysis-specific state."""
        num_traj = self.config.num_trajectories
        
        self.trajectories_analysis_prediction = {i: None for i in range(num_traj)}
        self.t_intervals = {i: [] for i in range(num_traj)}
        self.analysis_checks = self._create_analysis_checks_dict(num_traj)
        
        if self.config.mol_idxs is not None:
            self._add_molecular_analysis_fields()
            self.uncertainty_checks = []
        
        self.collect_losses = {"epoch": [], "avg_losses": [], "ensemble_losses": []}
        self.collect_thresholds = {i: [] for i in range(num_traj)}
    
    def _create_analysis_checks_dict(self, num_trajectories: int) -> dict:
        """Create analysis checks dictionary structure."""
        base_fields = [
            "prediction", "true_forces", "atom_wise_uncertainty",
            "uncertainty_via_max", "uncertainty_via_mean", "max_error",
            "mean_error", "atom_wise_error", "threshold", "train_set_length",
            "current_md_step",
        ]
        
        return {
            trajectory: {field: [] for field in base_fields}
            for trajectory in range(num_trajectories)
        }
    
    def _add_molecular_analysis_fields(self):
        """Add molecular-specific analysis fields."""
        mol_fields = [
            "mol_forces_prediction", "mol_forces_true", "total_uncertainty",
            "mol_forces_uncertainty", "mol_wise_error", "max_mol_error",
            "mean_mol_error",
        ]
        
        for trajectory in range(self.config.num_trajectories):
            for field in mol_fields:
                self.analysis_checks[trajectory][field] = []


class ALEnsembleManager:
    """Manages ensemble setup and dataset handling for active learning."""
    
    def __init__(self, config: ALConfigurationManager, comm_handler):
        self.config = config
        self.comm_handler = comm_handler
        self.rank = comm_handler.get_rank()
        
        # System properties
        self.z = None
        self.z_table = None
        self.n_atoms = None
        
        # Ensemble and datasets (rank 0 only)
        self.ensemble = None
        self.training_setups = None
        self.ensemble_ase_sets = None
        self.ensemble_mace_sets = None
        self.train_dataset_len = None
        
        # Setup system properties
        self._setup_system_properties()
        
        # Load seeds_tags_dict if not provided
        self._load_seeds_tags_dict()
    
    def _setup_system_properties(self):
        """Setup system-specific properties."""
        self.z = Z_from_geometry_in()
        self.z_table = create_ztable(self.z)
        self.n_atoms = len(self.z)
    
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
                    f"Could not load seeds_tags_dict.npz. Either specify it in the active "
                    f"learning settings or put it as seeds_tags_dict.npz in {self.config.dataset_dir}."
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
        self.ensemble = ensemble_from_folder(
            path_to_models=self.config.model_dir,
            device=self.config.device,
            dtype=dtype_mapping[self.config.dtype],
        )
        
        self.training_setups = ensemble_training_setups(
            ensemble=self.ensemble,
            mace_settings=self.config.mace_settings,
            restart=self.config.restart,
            checkpoints_dir=self.config.checkpoints_dir,
            al_settings=self.config.al,
        )
    
    def _setup_datasets(self):
        """Setup initial datasets (rank 0 only)."""
        dataset_subdir = "final" if self.config.restart else "initial"
        log_message = (
            "Loading datasets from checkpoint."
            if self.config.restart
            else "Loading initial datasets."
        )
        
        logging.info(log_message)
        
        self.ensemble_ase_sets = load_ensemble_sets_from_folder(
            ensemble=self.ensemble,
            path_to_folder=Path(self.config.al["dataset_dir"] + f"/{dataset_subdir}"),
        )
        
        self.ensemble_mace_sets = ase_to_mace_ensemble_sets(
            ensemble_ase_sets=self.ensemble_ase_sets,
            z_table=self.z_table,
            r_max=self.config.r_max,
            seed=self.config.seed,
        )
        
        self.train_dataset_len = len(
            self.ensemble_ase_sets[list(self.ensemble.keys())[0]]["train"]
        )
    
    def _broadcast_dataset_info(self):
        """Broadcast dataset information to all ranks."""
        self.comm_handler.barrier()
        self.train_dataset_len = self.comm_handler.bcast(self.train_dataset_len, root=0)
        self.comm_handler.barrier()


class ALCalculatorManager:
    """Manages calculators for active learning procedure."""
    
    def __init__(
        self,
        config: ALConfigurationManager, 
        ensemble_manager: ALEnsembleManager, 
        comm_handler
    ):
        self.config = config
        self.ensemble_manager = ensemble_manager
        self.comm_handler = comm_handler
        self.rank = comm_handler.get_rank()
        
        # AIMS settings
        self.control_parser = AIMSControlParser()
        self.aims_settings = None
        self.aims_calculator = None
        
        # MACE calculator
        self.models = None
        self.mace_calc = None
        
        # Atomic energies handling
        self.update_atomic_energies = False
        self.ensemble_atomic_energies = None
        self.ensemble_atomic_energies_dict = None
    
    def setup_calculators(self, path_to_control: str, path_to_geometry: str, setup_aims: bool = True):
        """Setup all required calculators."""
        self._handle_aims_settings(path_to_control)
        self.handle_atomic_energies()
        if setup_aims:
            self.aims_calculator = self._setup_aims_calculator(atoms=read(path_to_geometry))
        else:
            self.aims_calculator = None

        if self.rank == 0:
            self._setup_mace_calculator()
    
    def _handle_aims_settings(self, path_to_control: str):
        """Load and parse AIMS control file."""
        self.aims_settings = self.control_parser(path_to_control)
        self.aims_settings["compute_forces"] = True
        self.aims_settings["species_dir"] = self.config.species_dir
        self.aims_settings["postprocess_anyway"] = True
    
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
        )
    
    def _load_from_ensemble(self):
        """Load atomic energies from existing ensemble."""
        (
            self.ensemble_atomic_energies,
            self.ensemble_atomic_energies_dict,
        ) = get_atomic_energies_from_ensemble(
            ensemble=self.ensemble_manager.ensemble,
            z=self.ensemble_manager.z,
            dtype=self.config.dtype
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
    
    def _setup_mace_calculator(self):
        """Setup MACE calculator with ensemble models."""
        model_paths = list_files_in_directory(self.config.model_dir)
        self.models = [
            torch.load(f=model_path, map_location=self.config.device, weights_only=False)
            for model_path in model_paths
        ]
        
        self.mace_calc = MACECalculator(
            models=self.models,
            device=self.config.device,
            default_dtype=self.config.dtype
        )
    
    def _setup_aims_calculator(self, atoms: ase.Atoms):
        """Setup AIMS calculator."""
        aims_settings = self.aims_settings.copy()
        
        def init_via_ase(asi):
            from ase.calculators.aims import Aims, AimsProfile
            
            aims_settings["profile"] = AimsProfile(command="asi-doesnt-need-command")
            calc = Aims(**aims_settings)
            calc.write_inputfiles(asi.atoms, properties=self.config.properties)
        
        if asi4py is None:
            raise ImportError(
                "asi4py is not properly installed. "
                "Please install it to use the AIMS calculator."
            )
        
        calc = asi4py.asecalc.ASI_ASE_calculator(
            self.config.ASI_path, init_via_ase, self.comm_handler.comm, atoms
        )
        return calc
    
    def recalc_aims(self, current_point: ase.Atoms) -> ase.Atoms:
        """Recalculate with AIMS and return updated atoms object."""
        self.aims_calculator.calculate(current_point, properties=self.config.properties)
        
        if self.aims_calculator.asi.is_scf_converged:
            current_point.info["REF_energy"] = self.aims_calculator.results["energy"]
            current_point.arrays["REF_forces"] = self.aims_calculator.results["forces"]
            
            if self.config.compute_stress:
                current_point.info["REF_stress"] = self.aims_calculator.results["stress"]
            
            return current_point
        else:
            if self.rank == 0:
                logging.info("SCF not converged.")
            return None


class ALMDManager:
    """Manages molecular dynamics setup and execution for active learning."""
    
    def __init__(self, config: ALConfigurationManager, state_manager: ALStateManager, comm_handler):
        self.config = config
        self.state_manager = state_manager
        self.comm_handler = comm_handler
        self.rank = comm_handler.get_rank()
        
        # MD drivers will be stored here
        self.md_drivers = {}
        self.current_temperatures = {}
        
        # Setup MD modification if configured
        self._setup_md_modify()
        
        # Setup uncertainty handling
        self._setup_uncertainty()
    
    def setup_md_drivers(self, trajectories: dict, mace_calculator):
        """Setup MD drivers for all trajectories."""
        if self.rank != 0:
            return
        
        # Assign calculator to all trajectories
        for trajectory in trajectories.values():
            trajectory.calc = mace_calculator
        
        # Setup MD drivers for each trajectory
        self.md_drivers = {
            trajectory_idx: self._setup_md_dynamics(
                atoms=trajectories[trajectory_idx],
                md_settings=self.config.md_settings,
                idx=trajectory_idx,
            )
            for trajectory_idx in range(self.config.num_trajectories)
        }
    
    def _setup_md_dynamics(self, atoms: ase.Atoms, md_settings: dict, idx: int):
        """Setup ASE molecular dynamics object for given atoms."""
        self._initialize_temperature(md_settings, idx)
        
        ensemble = md_settings["stat_ensemble"].lower()
        dyn = self._create_dynamics_engine(atoms, md_settings, ensemble, idx)
        
        self._initialize_velocities(atoms, md_settings)
        
        return dyn
    
    def _initialize_temperature(self, md_settings: dict, idx: int):
        """Initialize temperature tracking for trajectory."""
        if md_settings["stat_ensemble"].lower() not in ["nvt", "npt"]:
            return
        
        if self.config.restart:
            return
        
        self.current_temperatures[idx] = md_settings["temperature"]
    
    def _create_dynamics_engine(self, atoms: ase.Atoms, md_settings: dict, ensemble: str, idx: int):
        """Create appropriate dynamics engine based on ensemble type."""
        if ensemble == "nvt":
            return self._create_nvt_dynamics(atoms, md_settings, idx)
        elif ensemble == "npt":
            return self._create_npt_dynamics(atoms, md_settings, idx)
        else:
            raise ValueError(f"Unsupported ensemble type: {ensemble}")
    
    def _create_nvt_dynamics(self, atoms: ase.Atoms, md_settings: dict, idx: int):
        """Create NVT dynamics engine."""
        thermostat = md_settings["thermostat"].lower()
        
        if thermostat == "langevin":
            return Langevin(
                atoms,
                timestep=md_settings["timestep"] * units.fs,
                friction=md_settings["friction"] / units.fs,
                temperature_K=self.current_temperatures[idx],
                rng=np.random.RandomState(md_settings["seed"]),
            )
        else:
            raise ValueError(f"Unsupported thermostat: {thermostat}")
    
    def _create_npt_dynamics(self, atoms: ase.Atoms, md_settings: dict, idx: int):
        """Create NPT dynamics engine."""
        barostat = md_settings["barostat"].lower()
        
        if barostat == "berendsen":
            return self._create_berendsen_npt(atoms, md_settings, idx)
        elif barostat == "npt":
            return self._create_standard_npt(atoms, md_settings, idx)
        else:
            raise ValueError(f"Unsupported barostat: {barostat}")
    
    def _create_berendsen_npt(self, atoms: ase.Atoms, md_settings: dict, idx: int) -> NPTBerendsen:
        """Create Berendsen NPT dynamics engine."""
        npt_settings = {
            "atoms": atoms,
            "timestep": md_settings["timestep"] * units.fs,
            "temperature": self.current_temperatures[idx],
            "pressure_au": md_settings["pressure_au"],
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
    
    def _create_standard_npt(self, atoms: ase.Atoms, md_settings: dict, idx: int) -> NPT:
        """Create standard NPT dynamics engine."""
        npt_settings = {
            "atoms": atoms,
            "timestep": md_settings["timestep"] * units.fs,
            "temperature_K": self.current_temperatures[idx],
            "externalstress": md_settings["externalstress"] * units.bar,
            "ttime": md_settings["ttime"] * units.fs,
            "pfactor": md_settings["pfactor"] * units.fs,
        }
        
        # Add optional mask parameter
        if md_settings.get("mask"):
            npt_settings["mask"] = md_settings["mask"]
        
        return NPT(**npt_settings)
    
    def _initialize_velocities(self, atoms: ase.Atoms, md_settings: dict):
        """Initialize Maxwell-Boltzmann velocity distribution if not restarting."""
        if not self.config.restart:
            MaxwellBoltzmannDistribution(
                atoms, temperature_K=md_settings["temperature"]
            )
    
    def _setup_md_modify(self):
        """Setup MD modification if configured."""
        self.md_mod_settings = self.config.md_settings.get("MODIFY", None)
        if self.md_mod_settings is not None:
            self.md_mod_metric = self.md_mod_settings.get("metric", None)
            assert self.md_mod_metric is not None, "No metric specified for MD modification."
            
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
                uncertainty_type=self.config.uncertainty_type
            )
        else:
            self.get_uncertainty = HandleUncertainty(
                uncertainty_type=self.config.uncertainty_type
            )
    
    def get_temperature_function(self, ensemble_type: str):
        """Get appropriate temperature extraction function for ensemble type."""
        temperature_functions = {
            "nvt": lambda driver: driver.temp,
            "npt": lambda driver: driver.temperature,
        }
        return temperature_functions.get(ensemble_type, lambda driver: driver.temp)


class ALRestartManager:
    """Handles active learning restart functionality."""
    
    def __init__(
        self, 
        config: ALConfigurationManager,
        state_manager: ALStateManager, 
        comm_handler,
        md_manager: ALMDManager
    ):
        self.config = config
        self.state_manager = state_manager
        self.md_manager = md_manager
        self.comm_handler = comm_handler
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
        }
        
        self.save_restart = False
        self._add_conditional_restart_keys()
    
    def _add_conditional_restart_keys(self):
        """Add conditional restart keys based on configuration."""
        if self.config.md_settings["stat_ensemble"].lower() in ["nvt", "npt"]:
            self.al_restart_dict.update({"current_temperatures": None})
        
        if self.config.analysis:
            self.al_restart_dict.update({
                "t_intervals": None,
                "analysis_checks": None,
                "collect_losses": None,
                "collect_thresholds": None,
            })
    
    def handle_restart(self):
        """Handle active learning restart."""
        self._initialize_restart_attributes()
        
        if self.rank == 0:
            self._load_restart_checkpoint()
        
        self._broadcast_restart_state()
    
    def _initialize_restart_attributes(self):
        """Initialize all restart attributes to None."""
        base_attributes = [
            "trajectories", "MD_checkpoints", "trajectory_intermediate_epochs",
            "ensemble_reset_opt", "ensemble_no_improvement", "ensemble_best_valid",
            "threshold", "trajectory_status", "trajectory_MD_steps",
            "trajectory_total_epochs", "current_valid_error", "total_points_added",
            "train_points_added", "valid_points_added", "num_MD_limits_reached",
            "num_workers_training", "num_workers_waiting", "total_epoch",
            "check", "uncertainties", "uncert_not_crossed",
        ]
        
        # Add conditional attributes
        if self._is_nvt_or_npt_ensemble():
            base_attributes.append("current_temperatures")
        
        if self.config.analysis:
            analysis_attributes = [
                "t_intervals", "analysis_checks", "collect_losses", "collect_thresholds"
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
    
    def _broadcast_restart_state(self):
        """Broadcast restart state from rank 0 to all processes."""
        self.comm_handler.barrier()
        
        # Define attributes to broadcast
        broadcast_attributes = [
            "trajectory_status", "trajectory_MD_steps", "trajectory_total_epochs",
            "current_valid_error", "total_points_added", "train_points_added",
            "valid_points_added", "num_MD_limits_reached", "num_workers_training",
            "num_workers_waiting", "total_epoch", "check", "uncertainties",
            "uncert_not_crossed",
        ]
        
        # Add conditional attributes
        if self._is_nvt_or_npt_ensemble():
            broadcast_attributes.append("current_temperatures")
        
        if self.config.analysis:
            analysis_broadcast = [
                "t_intervals", "analysis_checks", "collect_losses", "collect_thresholds"
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
        return self.config.md_settings["stat_ensemble"].lower() in ["nvt", "npt"]
    
    def update_restart_dict(
        self, 
        trajectories_keys,
        md_drivers,
        save_restart: str = None
        ):
        """Update restart dictionary with current state."""
        self._update_base_restart_attributes()
        self._update_temperature_attributes(
            md_drivers,
            trajectories_keys
        )
        self._update_analysis_attributes()
        
        if save_restart is not None:
            np.save(save_restart, self.al_restart_dict)
    
    def _update_base_restart_attributes(self):
        """Update base restart attributes."""
        base_attributes = [
            "trajectories", "MD_checkpoints", "trajectory_status",
            "trajectory_MD_steps", "trajectory_total_epochs", "trajectory_intermediate_epochs",
            "ensemble_reset_opt", "ensemble_no_improvement", "ensemble_best_valid",
            "current_valid_error", "threshold", "total_points_added",
            "train_points_added", "valid_points_added", "num_MD_limits_reached",
            "num_workers_training", "num_workers_waiting", "total_epoch",
            "check", "uncertainties", "uncert_not_crossed",
        ]
        
        update_dict = {attr: getattr(self.state_manager, attr) for attr in base_attributes}
        self.al_restart_dict.update(update_dict)
     
    def _get_temperature_function(self, ensemble_type: str):
        """Get the appropriate temperature extraction function for the ensemble type."""
        temperature_functions = {
            "nvt": lambda driver: driver.temp,
            "npt": lambda driver: driver.temperature,
        }

        return temperature_functions.get(
            ensemble_type, lambda driver: driver.temp
        )
    
    def _update_temperature_attributes(self, md_drivers, trajectories_keys):
        """Update temperature-related restart attributes for NVT/NPT ensembles."""
        if not self._is_nvt_or_npt_ensemble():
            return

        ensemble_type = self.config.md_settings["stat_ensemble"].lower()
        temp_getter = self._get_temperature_function(ensemble_type)

        current_temperatures = {
            trajectory: temp_getter(md_drivers[trajectory]) / units.kB
            for trajectory in trajectories_keys
        }

        self.al_restart_dict.update(
            {"current_temperatures": current_temperatures}
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
            analysis_update["uncertainty_checks"] = self.state_manager.uncertainty_checks
        
        self.al_restart_dict.update(analysis_update)


class PrepareALProcedure:
    """Refactored active learning preparation class with clear separation of concerns."""
    
    def __init__(
        self,
        mace_settings: dict,
        al_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
        use_mpi: bool = True,
        comm_handler: CommHandler = None,
    ):
        """Initialize the active learning procedure."""
        
        # Setup communication first
        self._setup_communication(comm_handler, use_mpi)
        
        # Initialize configuration
        self.config = ALConfigurationManager(mace_settings, al_settings)
        
        # Initialize all managers
        self.ensemble_manager = ALEnsembleManager(self.config, self.comm_handler)
        self.state_manager = ALStateManager(self.config, self.comm_handler)
        self.calc_manager = ALCalculatorManager(
            self.config, 
            self.ensemble_manager, 
            self.comm_handler
            )
        self.md_manager = ALMDManager(self.config, self.state_manager, self.comm_handler)
        self.restart_manager = ALRestartManager(
            self.config,
            self.state_manager,
            self.comm_handler,
            self.md_manager
            )
        
        # Setup logging and basic configuration
        self._setup_logging()
        self._create_folders()
        
        # Set random seed
        np.random.seed(self.config.seed)
        
        # Setup calculators and datasets
        self.ensemble_manager.setup_ensemble_and_datasets()
        self.calc_manager.setup_calculators(
            path_to_control, 
            path_to_geometry,
            setup_aims=use_mpi
            )
        
        # Pass train_dataset_len to MD manager for modification
        self.md_manager.train_dataset_len = self.ensemble_manager.train_dataset_len

        if self.config.restart:
            self.restart_manager.handle_restart()
        else:
            self.state_manager.initialize_fresh_state(path_to_geometry)
            # Make sure state manager has access to seeds_tags_dict
            self.state_manager.seeds_tags_dict = self.config.seeds_tags_dict
            self.state_manager.initialize_fresh_state(path_to_geometry)

        if self.rank == 0:
            self.md_manager.setup_md_drivers(
                self.state_manager.trajectories,
                self.calc_manager.mace_calc
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
            if self.config.mace_settings["MISC"]["log_level"].lower() == "debug"
            else logging.INFO
        )
        
        self.log_dir = Path(self.config.mace_settings["GENERAL"]["log_dir"])
        tools.setup_logger(
            level=logger_level,
            tag="active_learning",
            directory=self.log_dir,
        )
        
        if self.rank == 0:
            logging.info("Initializing active learning procedure.")
            logging.info(f"Procedure runs on {self.world_size} workers.")
    
    def _create_folders(self):
        """Create necessary directories."""
        (self.config.dataset_dir / "final" / "training").mkdir(parents=True, exist_ok=True)
        (self.config.dataset_dir / "final" / "validation").mkdir(parents=True, exist_ok=True)
        
        if self.config.analysis:
            os.makedirs("analysis", exist_ok=True)
        
        if self.config.create_restart:
            os.makedirs("restart/al", exist_ok=True)
    
    def check_al_done(self) -> bool:
        """Check if active learning is done."""
        if self.config.create_restart:
            check = self.restart_manager.al_restart_dict.get("al_done", False)
            if check and self.rank == 0:
                logging.info("Active learning procedure is already done. Closing")
            return check
        return False
    
    @property
    def trajectories(self):
        return self.state_manager.trajectories

    @property
    def md_drivers(self):
        return self.md_manager.md_drivers
    
    @property
    def mace_calc(self):
        return self.calc_manager.mace_calc
    
    @property
    def aims_calculator(self):
        return self.calc_manager.aims_calculator
    
    @property
    def ensemble(self):
        return self.ensemble_manager.ensemble
    
    @property
    def get_uncertainty(self):
        return self.md_manager.get_uncertainty
    
    def _recalc_aims(self, current_point):
        """Delegate AIMS recalculation to calculator manager."""
        return self.calc_manager.recalc_aims(current_point)


class ALProcedure(PrepareALProcedure):
    """
    Class for the active learning procedure. It handles the training of the ensemble
    members, the molecular dynamics simulations, the sampling of points and the saving
    of the datasets.
    """

    def _finalize_ab_initio(self):
        self.aims_calculator.asi.close()

    def _al_loop(self):
        while True:
            for trajectory_idx in range(self.config.num_trajectories):

                if self.state_manager.trajectory_status[trajectory_idx] == "waiting":

                    set_limit = self.waiting_task(trajectory_idx)
                    if (
                        set_limit
                    ):  # stops the process if the maximum dataset size is reached
                        break

                if (
                    self.state_manager.trajectory_status[trajectory_idx] == "training"
                ):  # and training_job: # the idea is to let a worker train only if new points
                    # have been added. e.g. it can happen that one worker is
                    # beyond its MD limit but there is no new point that has been added

                    self.training_task(trajectory_idx)

                if self.state_manager.trajectory_status[trajectory_idx] == "running":

                    self.running_task(trajectory_idx)


            if self.state_manager.num_MD_limits_reached == self.config.num_trajectories:
                if self.rank == 0:
                    logging.info("All trajectories reached maximum MD steps.")
                break

            if self.ensemble_manager.train_dataset_len >= self.config.max_set_size:
                if self.rank == 0:
                    logging.info("Maximum size of training set reached.")
                break

            if self.state_manager.current_valid_error  < self.config.desired_accuracy:
                if self.rank == 0:
                    logging.info("Desired accuracy reached.")
                break
        self._finalize_ab_initio()

    def analysis_check(
        # TODO: put this somewhere else and its ugly
        # TODO: update docstring
        self,
        analysis_prediction: np.ndarray,
        true_forces: np.ndarray,
        current_md_step: int,
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
        atom_wise_uncertainty = self.get_uncertainty.ensemble_sd(
            analysis_prediction
        )
        uncertainty_via_max = self.get_uncertainty.max_atomic_sd(
            atom_wise_uncertainty
        )
        uncertainty_via_mean = self.get_uncertainty.mean_atomic_sd(
            atom_wise_uncertainty
        )

        mean_analysis_prediction = analysis_prediction.mean(0).squeeze()

        diff_sq_mean = np.mean(
            (true_forces - mean_analysis_prediction) ** 2, axis=-1
        )

        max_error = np.max(np.sqrt(diff_sq_mean), axis=-1)
        mean_error = np.mean(np.sqrt(diff_sq_mean), axis=-1)
        atom_wise_error = np.sqrt(diff_sq_mean)

        check_results["prediction"] = mean_analysis_prediction
        check_results["true_forces"] = true_forces
        check_results["atom_wise_uncertainty"] = atom_wise_uncertainty
        check_results["uncertainty_via_max"] = uncertainty_via_max
        check_results["uncertainty_via_mean"] = uncertainty_via_mean
        check_results["atom_wise_error"] = atom_wise_error
        check_results["max_error"] = max_error
        check_results["mean_error"] = mean_error
        check_results["train_set_length"] = self.ensemble_manager.train_dataset_len
        check_results["current_md_step"] = current_md_step

        if self.config.mol_idxs is not None:
            total_certainty = self.get_uncertainty(analysis_prediction)
            mol_forces_uncertainty = (
                self.get_uncertainty.get_intermol_uncertainty(
                    analysis_prediction
                )
            )
            mol_forces_prediction = (
                self.get_uncertainty.compute_mol_forces_ensemble(
                    analysis_prediction, self.config.mol_idxs
                )
                .mean(0)
                .squeeze()
            )
            mol_forces_true = self.get_uncertainty.compute_mol_forces_ensemble(
                true_forces.reshape(1, *true_forces.shape), self.config.mol_idxs
            ).squeeze()
            mol_diff_sq_mean = np.mean(
                (mol_forces_true - mol_forces_prediction) ** 2, axis=-1
            )
            max_mol_error = np.max(np.sqrt(mol_diff_sq_mean), axis=-1)
            mean_mol_error = np.mean(np.sqrt(mol_diff_sq_mean), axis=-1)
            mol_wise_error = np.sqrt(mol_diff_sq_mean)

            check_results["mol_forces_prediction"] = mol_forces_prediction
            check_results["mol_forces_true"] = mol_forces_true
            check_results["total_uncertainty"] = total_certainty
            check_results["mol_forces_uncertainty"] = mol_forces_uncertainty
            check_results["mol_wise_error"] = mol_wise_error
            check_results["max_mol_error"] = max_mol_error
            check_results["mean_mol_error"] = mean_mol_error

        return check_results

    def _handle_received_point(self, idx, received_point):
        mace_point = create_mace_dataset(
            data=[received_point],
            z_table=self.ensemble_manager.z_table,
            seed=None,
            r_max=self.config.r_max,
        )
        if (
            self.state_manager.valid_points_added
            < self.config.valid_ratio * self.state_manager.total_points_added
        ):
            self.state_manager.trajectory_status[idx] = "running"
            self.state_manager.num_workers_training -= 1
            if self.rank == 0:
                logging.info(
                    f"Trajectory worker {idx} is adding a point to the validation set."
                )
            # while the initial datasets are different for each ensemble member we add the new points to
            # all ensemble member datasets
            if self.rank == 0:
                for tag in self.ensemble_manager.ensemble_ase_sets.keys():
                    self.ensemble_manager.ensemble_ase_sets[tag]["valid"] += [received_point]
                    self.ensemble_manager.ensemble_mace_sets[tag]["valid"] += mace_point
            self.state_manager.valid_points_added += 1

        else:
            self.state_manager.trajectory_status[idx] = "training"
            self.state_manager.num_workers_training += 1
            self.state_manager.num_workers_training -= 1
            if self.rank == 0:
                logging.info(
                    f"Trajectory worker {idx} is adding a point to the training set."
                )
                # while the initial datasets are different for each ensemble member we add the new points to
                # all ensemble member datasets
                for tag in self.ensemble_manager.ensemble_ase_sets.keys():
                    self.ensemble_manager.ensemble_ase_sets[tag]["train"] += [received_point]
                    self.ensemble_manager.ensemble_mace_sets[tag]["train"] += mace_point
                self.ensemble_manager.train_dataset_len = len(
                    self.ensemble_manager.ensemble_ase_sets[tag]["train"]
                )

            self.comm_handler.barrier()
            self.ensemble_manager.train_dataset_len = self.comm_handler.bcast(
                self.ensemble_manager.train_dataset_len, root=0
            )
            self.comm_handler.barrier()

            if self.ensemble_manager.train_dataset_len > self.config.max_set_size:
                return True
            if self.rank == 0:
                logging.info(
                    f"Size of the training and validation set: {self.ensemble_manager.train_dataset_len}, {len(self.ensemble_manager.ensemble_ase_sets[tag]['valid'])}."
                )
            self.state_manager.train_points_added += 1
        self.state_manager.total_points_added += 1

    def waiting_task(self, idx: int):
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
        self._handle_received_point(
            idx,
            received_point=self.point,  # TODO: change self.point, i dont like it
        )

    def _check_batch_size(self, set_batch_size, tag):
        batch_size = (
            1
            if len(self.ensemble_manager.ensemble_mace_sets[tag]["train"]) < set_batch_size
            else set_batch_size
        )
        return batch_size

    def _prepare_training(
        self,
        mace_sets: dict,
    ):
        for _, (tag, model) in enumerate(self.ensemble.items()):

            train_batch_size = self._check_batch_size(self.config.set_batch_size, tag)
            valid_batch_size = self._check_batch_size(
                self.config.set_valid_batch_size, tag
            )

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
                atomic_energies_list=self.calc_manager.ensemble_atomic_energies[tag],
                scaling=self.config.scaling,
                update_atomic_energies=self.calc_manager.update_atomic_energies,
                z_table=self.ensemble_manager.z_table,
                atomic_energies_dict=self.calc_manager.ensemble_atomic_energies_dict[tag],
                dtype=self.config.dtype,
                device=self.config.device,
            )
        return mace_sets

    def _perform_training(self, idx: int = 0):
        while (
            self.state_manager.trajectory_intermediate_epochs[idx] < self.config.intermediate_epochs
        ):
            for tag, model in self.ensemble.items():

                if self.state_manager.ensemble_reset_opt[tag]:
                    logging.info(f"Resetting optimizer for model {tag}.")
                    self.ensemble_manager.training_setups[tag] = reset_optimizer(
                        model=self.ensemble[tag],
                        training_setup=self.ensemble_manager.training_setups[tag],
                        training_settings=self.config.mace_settings["TRAINING"],
                    )
                    self.state_manager.ensemble_reset_opt[tag] = False

                logger = tools.MetricsLogger(
                    directory=self.config.mace_settings["GENERAL"]["results_dir"],
                    tag=tag + "_train",
                )
                train_epoch(
                    model=model,
                    train_loader=self.ensemble_manager.ensemble_mace_sets[tag]["train_loader"],
                    loss_fn=self.ensemble_manager.training_setups[tag]["loss_fn"],
                    optimizer=self.ensemble_manager.training_setups[tag]["optimizer"],
                    lr_scheduler=(
                        self.ensemble_manager.training_setups[tag]["lr_scheduler"]
                        if self.use_scheduler
                        else None
                    ),  # no scheduler used here
                    epoch=self.state_manager.trajectory_intermediate_epochs[idx],
                    start_epoch=None,
                    valid_loss=None,
                    logger=logger,
                    device=self.ensemble_manager.training_setups[tag]["device"],
                    max_grad_norm=self.ensemble_manager.training_setups[tag]["max_grad_norm"],
                    output_args=self.ensemble_manager.training_setups[tag]["output_args"],
                    ema=self.ensemble_manager.training_setups[tag]["ema"],
                )
            if (
                self.state_manager.trajectory_intermediate_epochs[idx] % self.config.valid_skip == 0
                or self.state_manager.trajectory_intermediate_epochs[idx]
                == self.config.intermediate_epochs - 1
            ):
                ensemble_valid_loss, valid_loss, metrics = (
                    validate_epoch_ensemble(
                        ensemble=self.ensemble,
                        training_setups=self.ensemble_manager.training_setups,
                        ensemble_set=self.ensemble_manager.ensemble_mace_sets,
                        logger=logger,
                        log_errors=self.config.mace_settings["MISC"]["error_table"],
                        epoch=self.state_manager.trajectory_total_epochs[idx],
                    )
                )
                self.best_member = select_best_member(ensemble_valid_loss)
                if self.config.analysis:
                    self.state_manager.collect_losses["epoch"].append(self.state_manager.total_epoch)
                    self.state_manager.collect_losses["avg_losses"].append(valid_loss)
                    self.state_manager.collect_losses["ensemble_losses"].append(
                        ensemble_valid_loss
                    )

                self.state_manager.current_valid_error  = metrics["mae_f"]

                for tag in ensemble_valid_loss.keys():
                    if (
                        ensemble_valid_loss[tag]
                        < self.state_manager.ensemble_best_valid[tag]
                    ):
                        self.state_manager.ensemble_best_valid[tag] = ensemble_valid_loss[
                            tag
                        ]
                    else:
                        self.state_manager.ensemble_no_improvement[tag] += 1

                    if (
                        self.state_manager.ensemble_no_improvement[tag]
                        > self.config.max_epochs_worker
                    ):
                        logging.info(
                            f"No improvements for {self.config.max_epochs_worker} epochs "
                            f"(maximum epochs per worker) at ensemble member {tag}."
                        )
                        self.state_manager.ensemble_reset_opt[tag] = True
                        self.state_manager.ensemble_no_improvement[tag] = 0

                    save_checkpoint(
                        checkpoint_handler=self.ensemble_manager.training_setups[tag][
                            "checkpoint_handler"
                        ],
                        training_setup=self.ensemble_manager.training_setups[tag],
                        model=model,
                        epoch=self.state_manager.trajectory_intermediate_epochs[idx],
                        keep_last=False,
                    )

                    save_datasets(
                        ensemble=self.ensemble,
                        ensemble_ase_sets=self.ensemble_manager.ensemble_ase_sets,
                        path=self.config.dataset_dir / "final",
                    )
                    if self.config.create_restart:
                        self.save_restart = True

            self.state_manager.trajectory_total_epochs[idx] += 1
            self.state_manager.trajectory_intermediate_epochs[idx] += 1
            self.state_manager.total_epoch += 1

            if self.save_restart and self.config.create_restart:
                self.restart_manager.update_restart_dict(
                    trajectories_keys=self.trajectories.keys(),
                    md_drivers=self.md_drivers,
                    save_restart="restart/al/al_restart.npy"
                )
                self.save_restart = False
        self.models = [self.ensemble[tag] for tag in self.ensemble.keys()]
        self.state_manager.trajectory_intermediate_epochs[idx] = 0

    def training_task(self, idx: int):
        """
        Creates the dataloader of the updated dataset, updates
        the average number of neighbors, shifts and scaling factor
        and trains the ensemble members. Saves the models and checkpoints.

        Args:
            idx (int): Index of the trajectory worker.
        """
        if self.rank == 0:

            self.ensemble_manager.ensemble_mace_sets = self._prepare_training(
                mace_sets=self.ensemble_manager.ensemble_mace_sets
            )

            logging.info(f"Trajectory worker {idx} is training.")
            # we train only for some epochs before we move to the next worker which may be running MD
            # all workers train on the same models with the respective training settings for
            # each ensemble member

            self._perform_training(idx)

        # update calculators with the new models
        self.comm_handler.barrier()
        self.state_manager.current_valid_error  = self.comm_handler.bcast(
            self.state_manager.current_valid_error , root=0
        )
        self.comm_handler.barrier()
        if self.rank == 0:
            for trajectory in self.trajectories.values():
                trajectory.calc.models = [
                    self.ensemble[tag] for tag in self.ensemble.keys()
                ]

        self.comm_handler.barrier()
        self.state_manager.total_epoch = self.comm_handler.bcast(self.state_manager.total_epoch, root=0)
        self.state_manager.trajectory_total_epochs[idx] = self.comm_handler.bcast(
            self.state_manager.trajectory_total_epochs[idx], root=0
        )
        self.comm_handler.barrier()

        if self.state_manager.trajectory_total_epochs[idx] >= self.config.max_epochs_worker:
            self.state_manager.trajectory_status[idx] = "running"
            self.state_manager.num_workers_training -= 1
            self.state_manager.trajectory_total_epochs[idx] = 0
            if self.rank == 0:
                logging.info(f"Trajectory worker {idx} finished training.")
            # calculate true error and uncertainty on validation set

    def _analysis_dft_call(self, point: ase.Atoms, idx: int = None):
        self.aims_calculator.calculate(point, properties=self.config.properties)
        return self.aims_calculator.asi.is_scf_converged

    def _save_analysis(self):
        np.savez("analysis/analysis_checks.npz", self.state_manager.analysis_checks)
        np.savez("analysis/t_intervals.npz", self.state_manager.t_intervals)
        np.savez("analysis/al_losses.npz", **self.state_manager.collect_losses)
        np.savez("analysis/thresholds.npz", self.state_manager.collect_thresholds)
        if self.config.mol_idxs is not None:
            np.savez(
                "analysis/uncertainty_checks.npz", self.state_manager.uncertainty_checks
            )

    def _process_analysis(
        self, idx: int, converged: bool, analysis_prediction: np.ndarray
    ):
        if converged:
            check_results = self.analysis_check(
                current_md_step=self.state_manager.trajectory_MD_steps[idx],
                analysis_prediction=analysis_prediction,
                true_forces=self.aims_calculator.results["forces"],
            )
            for key in check_results.keys():
                self.state_manager.analysis_checks[idx][key].append(check_results[key])
            self.state_manager.analysis_checks[idx]["threshold"].append(self.state_manager.threshold )

            self.state_manager.collect_thresholds[idx].append(self.state_manager.threshold )

            self.state_manager.check += 1

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

        self.state_manager.t_intervals[idx].append(current_MD_step)
        if self.config.mol_idxs is not None:
            self.state_manager.uncertainty_checks.append(uncertainty > self.state_manager.threshold )
        if current_MD_step % self.config.analysis_skip == 0:
            if self.rank == 0:
                logging.info(
                    f"Trajectory worker {idx} is sending a point to DFT for analysis."
                )

            if current_MD_step % self.config.skip_step == 0:
                self.state_manager.trajectories_analysis_prediction[idx] = prediction
            else:
                if self.rank == 0:
                    self.state_manager.trajectories_analysis_prediction[idx] = (
                        ensemble_prediction(
                            models=list(self.ensemble.values()),
                            atoms_list=[self.point],
                            device=self.device,
                            dtype=self.config.mace_settings["GENERAL"][
                                "default_dtype"
                            ],
                        )
                    )
                self.comm_handler.barrier()
                self.state_manager.trajectories_analysis_prediction[idx] = (
                    self.comm_handler.bcast(
                        self.state_manager.trajectories_analysis_prediction[idx], root=0
                    )
                )
                self.comm_handler.barrier()

            # TODO: sometimes already calculated above so we should not calculate it again
            self.comm_handler.barrier()
            send_point = atoms_full_copy(self.point)
            send_point.arrays["forces_comm"] = (
                self.state_manager.trajectories_analysis_prediction[idx]
            )
            send_point.info["current_MD_step"] = current_MD_step
            converged = self._analysis_dft_call(point=send_point, idx=idx)
            self.comm_handler.barrier()

            self._process_analysis(
                idx=idx,
                converged=converged,
                analysis_prediction=self.state_manager.trajectories_analysis_prediction[idx],
            )

    def running_task(self, idx: int):
        """
        Runs the molecular dynamics simulation using the MLFF and
        checks the uncertainty. If the uncertainty is above the threshold
        the point is calculated using FHI aims and sent to the waiting task.

        Args:
            idx (int): Index of the trajectory worker.

        """

        current_MD_step = self.state_manager.trajectory_MD_steps[idx]

        # kill the worker if the maximum number of MD steps is reached
        if (
            current_MD_step > self.config.max_MD_steps
            and self.state_manager.trajectory_status[idx] == "running"
        ):
            if self.rank == 0:
                logging.info(
                    f"Trajectory worker {idx} reached maximum MD steps and is killed."
                )
            self.state_manager.num_MD_limits_reached += 1
            self.state_manager.trajectory_status[idx] = "killed"
            return "killed"

        else:
            # TODO:
            # ideally we would first check the uncertainty, then optionally
            # calculate the aims forces and use them to propagate
            # currently the mace forces are used even if the uncertainty is too high
            # but ase is weird and i don't want to change it so whatever. when we have our own
            # MD engine we can adress this.

            if self.md_manager.mod_md:
                if self.rank == 0:
                    modified = self.md_manager.md_modifier(
                        driver=self.md_drivers[idx],
                        metric=self.md_manager.get_md_mod_metric(),
                        idx=idx,
                    )
                    if modified and self.config.create_restart:
                        self.restart_manager.update_restart_dict(    
                            trajectories_keys=self.trajectories.keys(),
                            md_drivers=self.md_drivers,
                            save_restart="restart/al/al_restart.npy"
                        )

            if self.rank == 0:
                self.md_drivers[idx].run(self.config.skip_step)
            self.state_manager.trajectory_MD_steps[idx] += self.config.skip_step
            current_MD_step += self.config.skip_step

            # somewhat arbitrary; i just want to save checkpoints if the MD phase
            # is super long
            if self.rank == 0:
                if current_MD_step % (self.config.skip_step * 100) == 0:
                    self.restart_manager.update_restart_dict(
                        trajectories_keys=self.trajectories.keys(),
                        md_drivers=self.md_drivers,
                        save_restart="restart/al/al_restart.npy"
                    )

                logging.info(
                    f"Trajectory worker {idx} at MD step {current_MD_step}."
                )

                self.point = self.trajectories[idx].copy()
                prediction = self.trajectories[idx].calc.results["forces_comm"]
                uncertainty = self.get_uncertainty(prediction)

                self.state_manager.uncertainties.append(uncertainty)

                if len(self.state_manager.uncertainties) > 10:  # TODO: remove hardcode
                    if (
                        self.ensemble_manager.train_dataset_len >= self.config.freeze_threshold_dataset
                    ) and not self.config.freeze_threshold:
                        if self.rank == 0:
                            logging.info(
                                f"Train data has reached size {self.ensemble_manager.train_dataset_len}: freezing threshold at {self.state_manager.threshold :.3f}."
                            )
                        self.config.freeze_threshold = True

                    if not self.config.freeze_threshold:
                        self.state_manager.threshold  = get_threshold(
                            uncertainties=self.state_manager.uncertainties,
                            c_x=self.config.c_x,
                            max_len=400,  # TODO: remove hardcode
                        )

                    if self.config.analysis:
                        self.state_manager.collect_thresholds[idx].append(self.state_manager.threshold )

            if self.rank != 0:
                uncertainty = None
                prediction = None
                self.point = None
                self.state_manager.threshold  = None
                current_MD_step = None

            self.comm_handler.barrier()
            self.state_manager.threshold  = self.comm_handler.bcast(self.state_manager.threshold , root=0)
            self.point = self.comm_handler.bcast(self.point, root=0)
            uncertainty = self.comm_handler.bcast(uncertainty, root=0)
            prediction = self.comm_handler.bcast(prediction, root=0)
            current_MD_step = self.comm_handler.bcast(current_MD_step, root=0)
            self.comm_handler.barrier()

            if (uncertainty > self.state_manager.threshold ).any() or self.state_manager.uncert_not_crossed[
                idx
            ] > self.config.skip_step * self.config.uncert_not_crossed_limit:
                self.state_manager.uncert_not_crossed[idx] = 0
                if self.rank == 0:
                    if (uncertainty > self.state_manager.threshold ).any():
                        logging.info(
                            f"Uncertainty of point is beyond threshold {np.round(self.state_manager.threshold ,3)} at worker {idx}: {np.round(uncertainty,3)}."
                        )
                if self.config.mol_idxs is not None:
                    crossings = uncertainty > self.state_manager.threshold 
                    cross_global = crossings[0]
                    cross_inter = crossings[1]

                    if cross_inter and not cross_global:
                        self.config.intermol_crossed += 1

                    if cross_global:
                        self.config.intermol_crossed = 0

                    if self.config.intermol_crossed != 0:
                        if self.rank == 0:
                            logging.info(
                                f"Intermolecular uncertainty crossed {self.config.intermol_crossed} consecutive times."
                            )

                    if (
                        self.config.intermol_crossed >= self.config.intermol_crossed_limit
                        and not self.config.switched_on_intermol
                        and self.config.using_intermol_loss
                    ):
                        if self.rank == 0:
                            logging.info(
                                f"Intermolecular uncertainty crossed "
                                f"{self.config.intermol_crossed_limit} consecutive "
                                "times. Turning intermol_loss weight to "
                                f"{self.config.intermol_forces_weight}."
                            )
                            for tag in self.ensemble.keys():
                                self.ensemble_manager.training_setups[tag][
                                    "loss_fn"
                                ].intermol_forces_weight = (
                                    self.config.intermol_forces_weight
                                )
                            self.config.switched_on_intermol = True

                self._handle_dft_call(idx)

            else:
                self.state_manager.uncert_not_crossed[idx] += 1

            if self.config.analysis:
                self._perform_analysis(
                    idx=idx,
                    prediction=prediction,
                    current_MD_step=current_MD_step,
                    uncertainty=uncertainty,
                )

    def _handle_dft_call(self, idx):
        if self.rank == 0:
            logging.info(f"Trajectory worker {idx} is running DFT.")

        self.comm_handler.barrier()
        self.point = self._recalc_aims(self.point)

        if not self.aims_calculator.asi.is_scf_converged:
            if self.rank == 0:
                logging.info(
                    f"SCF not converged at worker {idx}. Discarding point and restarting MD from last checkpoint."
                )
                self.trajectories[idx] = atoms_full_copy(
                    self.state_manager.MD_checkpoints[idx]
                )
            self.state_manager.trajectory_status[idx] = "running"
        else:
            # we are updating the MD checkpoint here because then we make sure
            # that the MD is restarted from a point that is inside the training set
            # so the MLFF should be able to handle this and lead to a better trajectory
            # that does not lead to convergence issues
            if self.rank == 0:
                received_point = self.trajectories[idx].copy()
                received_point.info["REF_energy"] = self.point.info[
                    "REF_energy"
                ]
                received_point.arrays["REF_forces"] = self.point.arrays[
                    "REF_forces"
                ]
                if self.config.compute_stress:
                    received_point.info["REF_stress"] = self.point.info[
                        "REF_stress"
                    ]

                self.state_manager.MD_checkpoints[idx] = atoms_full_copy(received_point)
                self.state_manager.MD_checkpoints[idx].calc = self.trajectories[idx].calc
            self.state_manager.trajectory_status[idx] = "waiting"
            self.state_manager.num_workers_training += 1

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
            logging.info(
                f"Active learning procedure finished. The best ensemble member based on validation loss is {self.best_member}."
            )
            save_datasets(
                ensemble=self.ensemble,
                ensemble_ase_sets=self.ensemble_manager.ensemble_ase_sets,
                path=self.config.dataset_dir / "final",
            )

            if self.config.analysis:
                self._save_analysis()

            if self.config.create_restart:
                self.restart_manager.update_restart_dict(
                    trajectories_keys=self.trajectories.keys(),
                    md_drivers=self.md_drivers,
                    save_restart="restart/al/al_restart.npy"
                )
                self.restart_manager.al_restart_dict["al_done"] = True

    def converge(self):
        """
        Converges the ensemble on the acquired dataset. Trains the ensemble members
        until the validation loss does not improve anymore.
        """
        if self.rank == 0:
            if self.config.converge_best:
                logging.info(
                    f"Converging best model ({self.best_member}) on acquired dataset."
                )
                self.ensemble = {
                    self.best_member: self.ensemble[self.best_member]
                }
            else:
                logging.info("Converging ensemble on acquired dataset.")

            temp_mace_sets = {}
            for _, (tag, model) in enumerate(self.ensemble.items()):
                train_set = create_mace_dataset(
                    data=self.ensemble_manager.ensemble_ase_sets[tag]["train"],
                    z_table=self.ensemble_manager.z_table,
                    seed=self.config.seeds_tags_dict[tag],
                    r_max=self.config.r_max,
                )
                valid_set = create_mace_dataset(
                    data=self.ensemble_manager.ensemble_ase_sets[tag]["valid"],
                    z_table=self.ensemble_manager.z_table,
                    seed=self.config.seeds_tags_dict[tag],
                    r_max=self.config.r_max,
                )
                temp_mace_sets[tag] = {"train": train_set, "valid": valid_set}

            self.ensemble_manager.ensemble_mace_sets = self._prepare_training(
                mace_sets=temp_mace_sets
            )

            # resetting optimizer and scheduler
            self.training_setups_convergence = {}
            for tag in self.ensemble.keys():
                self.training_setups_convergence[tag] = setup_mace_training(
                    settings=self.config.mace_settings,
                    model=self.ensemble[tag],
                    tag=tag,
                    restart=self.config.restart,
                    convergence=True,
                    checkpoints_dir=self.config.checkpoints_dir,
                    al_settings=self.config.al,
                )
            best_valid_loss = np.inf
            epoch = 0
            if self.config.restart:
                epoch = self.training_setups_convergence[
                    list(self.ensemble.keys())[0]
                ]["epoch"]
            no_improvement = 0
            ensemble_valid_losses = {
                tag: np.inf for tag in self.ensemble.keys()
            }
            for j in range(self.config.max_final_epochs):
                # ensemble_loss = 0
                for tag, model in self.ensemble.items():
                    logger = tools.MetricsLogger(
                        directory=self.config.mace_settings["GENERAL"]["results_dir"],
                        tag=tag + "_train",
                    )
                    train_epoch(
                        model=model,
                        train_loader=self.ensemble_manager.ensemble_mace_sets[tag][
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
                    # ensemble_loss += loss
                # ensemble_loss /= len(ensemble)

                if (
                    epoch % self.config.valid_skip == 0
                    or epoch == self.config.max_final_epochs - 1
                ):
                    (
                        ensemble_valid_losses,
                        valid_loss,
                        _,
                    ) = validate_epoch_ensemble(
                        ensemble=self.ensemble,
                        training_setups=self.training_setups_convergence,
                        ensemble_set=self.ensemble_manager.ensemble_mace_sets,
                        logger=logger,
                        log_errors=self.config.mace_settings["MISC"]["error_table"],
                        epoch=epoch,
                    )

                    if (
                        best_valid_loss > valid_loss
                        and (best_valid_loss - valid_loss) > self.config.margin
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
                                        self.config.mace_settings["GENERAL"][
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
                if no_improvement > self.config.patience:
                    logging.info(
                        f"No improvements for {self.config.patience} epochs. Training converged. Best model(s) (Epoch {best_epoch}) based on validation loss saved."
                    )
                    break
                if j == self.config.max_final_epochs - 1:
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

        self.comm = self.world_comm.Split(color=self.color, key=self.rank)
        self.comm_handler = CommHandler()
        self.comm_handler.rank = self.rank
        self.comm_handler.size = self.comm.Get_size()
        self.comm_handler.comm = self.comm

        super().__init__(
            mace_settings=mace_settings,
            al_settings=al_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
            comm_handler=self.comm_handler,
        )
        self.first_wait_after_restart = {
            idx: True for idx in range(self.config.num_trajectories)
        }

    def _setup_aims_calculator(
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
        aims_settings = self.calc_manager.aims_settings.copy()
        # only one communictor initializes aims
        if self.color == 1:
            self.properties = ["energy", "forces"]
            if self.config.compute_stress:
                self.properties.append("stress")

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
        else:
            return None

    def _send_kill(self):
        """
        Sends a kill signal
        """

        for dest in range(1, self.world_size):
            self.kill_send = self.world_comm.isend(True, dest=dest, tag=422)
            self.kill_send.Wait()

        # for trajectory_idx in range(self.config.num_trajectories):
        #    if self.worker_reqs[trajectory_idx] is not None:
        #        self.worker_reqs[trajectory_idx].cancel()

        # if self.point_send is not None:
        #    self.point_send.cancel()

    def _al_loop(self):
        self.worker_reqs = {
            "energy": {idx: None for idx in range(self.config.num_trajectories)},
            "forces": {idx: None for idx in range(self.config.num_trajectories)},
            "stress": {idx: None for idx in range(self.config.num_trajectories)},
        }

        self.worker_reqs_bufs = {
            "energy": {idx: None for idx in range(self.config.num_trajectories)},
            "forces": {idx: None for idx in range(self.config.num_trajectories)},
            "stress": {idx: None for idx in range(self.config.num_trajectories)},
        }

        if self.config.analysis:
            self.worker_reqs_analysis = {
                "energy": {idx: None for idx in range(self.config.num_trajectories)},
                "forces": {idx: None for idx in range(self.config.num_trajectories)},
                "stress": {idx: None for idx in range(self.config.num_trajectories)},
            }
            self.worker_reqs_analysis_bufs = {
                "energy": {idx: None for idx in range(self.config.num_trajectories)},
                "forces": {idx: None for idx in range(self.config.num_trajectories)},
                "stress": {idx: None for idx in range(self.config.num_trajectories)},
            }

        self.req_sys_info = None
        self.req_geo_info = None
        self.current_num_atoms = None

        if self.config.analysis:
            self.req_sys_info_analysis = None
            self.req_geo_info_analysis = None
            self.current_num_atoms_analysis = None
            self.received_analysis = None
            self.analysis_worker_reqs = {
                idx: None for idx in range(self.config.num_trajectories)
            }

        if self.color == 1:
            self.req_kill = self.world_comm.irecv(source=0, tag=422)

        while True:
            if self.color == 0:
                for trajectory_idx in range(self.config.num_trajectories):

                    if self.state_manager.trajectory_status[trajectory_idx] == "running":

                        self.running_task(trajectory_idx)

                    if self.state_manager.trajectory_status[trajectory_idx] == "training":

                        self.training_task(trajectory_idx)

                    if self.state_manager.trajectory_status[trajectory_idx] == "waiting":

                        set_limit = self.waiting_task(trajectory_idx)
                        if set_limit:
                            self._send_kill()
                            break

                    if self.config.analysis:
                        if (
                            self.state_manager.trajectory_status[trajectory_idx]
                            == "analysis_waiting"
                        ):
                            self._analysis_waiting_task(trajectory_idx)

                if self.state_manager.num_MD_limits_reached == self.config.num_trajectories:
                    if self.rank == 0:
                        logging.info(
                            "All trajectories reached maximum MD steps."
                        )
                        self._send_kill()
                    break

                if self.ensemble_manager.train_dataset_len >= self.config.max_set_size:
                    if self.rank == 0:
                        logging.info("Maximum size of training set reached.")
                        self._send_kill()
                    break

                if self.state_manager.current_valid_error  < self.config.desired_accuracy:
                    if self.rank == 0:
                        logging.info("Desired accuracy reached.")
                        self._send_kill()
                    break

            if self.color == 1:
                kill_signal = self.req_kill.Test()
                if kill_signal:
                    return None

                if self.rank != 1:
                    self.geo_info_buf = None
                    self.sys_info_buf = None
                    received = False

                if self.rank == 1:
                    received = self._listening_task()

                self.comm_handler.barrier()
                received = self.comm_handler.bcast(
                    received, root=0
                )  # global rank 1 is 0 of split comm
                self.current_num_atoms = self.comm_handler.bcast(
                    self.current_num_atoms, root=0
                )
                self.comm_handler.barrier()

                if received:
                    if self.rank != 1:
                        self.sys_info_buf = np.empty(
                            shape=(14,), dtype=np.float64
                        )
                        self.geo_info_buf = np.empty(
                            shape=(2, self.current_num_atoms, 3),
                            dtype=np.float64,
                        )
                    self.comm_handler.barrier()
                    self.comm_handler.comm.Bcast(buf=self.sys_info_buf, root=0)
                    self.comm_handler.comm.Bcast(buf=self.geo_info_buf, root=0)
                    self.comm_handler.barrier()

                    self.comm_handler.barrier()
                    dft_result = self._calculate_received()

                    if self.rank == 1:
                        self._send_result_back(
                            idx=int(self.sys_info_buf[0]),
                            dft_result=dft_result,
                            num_atoms=self.current_num_atoms,
                        )

                if self.config.analysis:

                    if self.rank != 1:
                        self.geo_info_buf_analysis = None
                        self.sys_info_buf_analysis = None
                        received_analysis = False

                    if self.rank == 1:
                        received_analysis = self._analysis_listening_task()

                    self.comm_handler.barrier()
                    received_analysis = self.comm_handler.bcast(
                        received_analysis, root=0
                    )  # global rank 1 is 0 of split comm
                    self.current_num_atoms_analysis = self.comm_handler.bcast(
                        self.current_num_atoms_analysis, root=0
                    )
                    self.comm_handler.barrier()

                    if received_analysis:
                        if self.rank != 1:
                            self.sys_info_buf_analysis = np.empty(
                                shape=(14,), dtype=np.float64
                            )
                            self.geo_info_buf_analysis = np.empty(
                                shape=(2, self.current_num_atoms_analysis, 3),
                                dtype=np.float64,
                            )
                        self.comm_handler.barrier()
                        self.comm_handler.comm.Bcast(
                            buf=self.sys_info_buf_analysis, root=0
                        )
                        self.comm_handler.comm.Bcast(
                            buf=self.geo_info_buf_analysis, root=0
                        )
                        self.comm_handler.barrier()

                        dft_result_analysis = (
                            self._analysis_calculate_received()
                        )

                        if self.rank == 1:
                            self._send_result_back(
                                idx=int(self.sys_info_buf_analysis[0]),
                                dft_result=dft_result_analysis,
                                num_atoms=self.current_num_atoms_analysis,
                            )

        if self.color == 1:
            self.aims_calculator.asi.close()

    def waiting_task(self, idx: int):
        """
        TODO

        Args:
            idx (int): Index of the trajectory worker.

        """

        if self.config.restart and self.first_wait_after_restart[idx]:
            # if the worker is waiting and we just restarted the
            # procedure, we have to relaunch the dft job and then
            # leave the function
            self._handle_dft_call(idx)
            self.first_wait_after_restart[idx] = False
            return None

        if self.worker_reqs["energy"][idx] is None:
            self.worker_reqs_bufs["energy"][idx] = np.empty(
                shape=(2,), dtype=np.float64
            )
            self.worker_reqs["energy"][idx] = self.world_comm.Irecv(
                buf=self.worker_reqs_bufs["energy"][idx], source=1, tag=idx
            )
        status, _ = self.worker_reqs["energy"][idx].test()

        if status:
            scf_failed = np.isnan(self.worker_reqs_bufs["energy"][idx][0])
            # check if the energy is NaN
            if not scf_failed:
                if self.worker_reqs["forces"][idx] is None:
                    self.worker_reqs_bufs["forces"][idx] = np.empty(
                        shape=(
                            int(self.worker_reqs_bufs["energy"][idx][1]),
                            3,
                        ),
                        dtype=np.float64,
                    )
                    self.worker_reqs["forces"][idx] = self.world_comm.Irecv(
                        buf=self.worker_reqs_bufs["forces"][idx],
                        source=1,
                        tag=idx + 10000,
                    )
                    if self.config.compute_stress:
                        self.worker_reqs_bufs["stress"][idx] = np.empty(
                            shape=(6,), dtype=np.float64
                        )
                        self.worker_reqs["stress"][idx] = (
                            self.world_comm.Irecv(
                                buf=self.worker_reqs_bufs["stress"][idx],
                                source=1,
                                tag=idx + 20000,
                            )
                        )
                status_forces = self.worker_reqs["forces"][idx].Wait()
                if self.config.compute_stress:
                    status_stress = self.worker_reqs["stress"][idx].Wait()
                if status_forces or (self.config.compute_stress and status_stress):
                    self.worker_reqs["energy"][idx] = None
                    self.worker_reqs["forces"][idx] = None
                    if self.config.compute_stress:
                        self.worker_reqs["stress"][idx] = None

                    if self.rank == 0:
                        logging.info(
                            f"Worker {idx} received a point from DFT."
                        )
                    # we are updating the MD checkpoint here because then we make sure
                    # that the MD is restarted from a point that is inside the training set
                    # so the MLFF should be able to handle this and lead to a better trajectory
                    if self.rank == 0:
                        self.state_manager.MD_checkpoints[idx] = atoms_full_copy(
                            self.trajectories[idx]
                        )

                    received_point = self.trajectories[idx].copy()
                    received_point.info["REF_energy"] = self.worker_reqs_bufs[
                        "energy"
                    ][idx][0]
                    received_point.arrays["REF_forces"] = (
                        self.worker_reqs_bufs["forces"][idx]
                    )
                    if self.config.compute_stress:
                        received_point.info["REF_stress"] = (
                            self.worker_reqs_bufs["stress"][idx]
                        )

                    self._handle_received_point(
                        idx=idx, received_point=received_point
                    )
            else:
                if self.rank == 0:
                    logging.info(
                        f"SCF not converged at worker {idx}. Discarding point and restarting MD from last checkpoint."
                    )
                    self.worker_reqs["energy"][idx] = None
                    self.worker_reqs["forces"][idx] = None
                    if self.config.compute_stress:
                        self.worker_reqs["stress"][idx] = None
                    self.trajectories[idx] = atoms_full_copy(
                        self.state_manager.MD_checkpoints[idx]
                    )
                self.state_manager.trajectory_status[idx] = "running"

    def _listening_task(
        self,
    ) -> bool:
        """
        Listens for incoming data from the DFT worker.

        Returns:
            bool: True if data was received, False otherwise.
        """
        received = False
        if self.req_sys_info is None:
            self.sys_info_buf = np.empty(shape=(14,), dtype=np.float64)
            self.req_sys_info = self.world_comm.Irecv(
                buf=self.sys_info_buf, source=0, tag=1234
            )

        status, _ = self.req_sys_info.test()
        if status:
            idx = int(self.sys_info_buf[0])
            self.current_num_atoms = int(self.sys_info_buf[1])
            self.req_sys_info = None
            if self.req_geo_info is None:
                self.geo_info_buf = np.empty(
                    shape=(2, self.current_num_atoms, 3), dtype=np.float64
                )

                self.req_geo_info = self.world_comm.Irecv(
                    buf=self.geo_info_buf, source=0, tag=1235
                )
            status_pos_spec = self.req_geo_info.Wait()
            if status_pos_spec:
                self.req_geo_info = None
                received = True

        return received

    def _calculate_received(self):
        current_idx = int(self.sys_info_buf[0])
        current_num_atoms = int(self.sys_info_buf[1])
        current_pbc = self.sys_info_buf[2:5].astype(np.bool_).reshape((3,))
        current_species = self.geo_info_buf[0].astype(np.int32)
        # transform current_species to a list of species
        current_species = current_species[:, 0].tolist()
        current_positions = self.geo_info_buf[1]
        current_cell = self.sys_info_buf[5:14].reshape((3, 3))

        point = ase.Atoms(
            positions=current_positions,
            numbers=current_species,
            pbc=current_pbc,
            cell=current_cell,
        )
        self.comm_handler.barrier()
        dft_result = self._recalc_aims(point)
        return dft_result

    def _send_result_back(self, idx, dft_result, num_atoms):
        if dft_result is not None:
            logging.info(
                f"DFT calculation for worker {idx} finished and sending point back."
            )
            dft_energies = dft_result.info["REF_energy"]
            dft_energies_num_atoms = np.array(
                [dft_energies, num_atoms], dtype=np.float64
            )
            dft_forces = dft_result.arrays["REF_forces"]
            if self.config.compute_stress:
                dft_stress = dft_result.info["REF_stress"]
        else:
            dft_energies_num_atoms = np.array(
                [np.nan, np.nan], dtype=np.float64
            )
            dft_forces = np.empty(shape=(num_atoms, 3), dtype=np.float64).fill(
                np.nan
            )
            if self.config.compute_stress:
                dft_stress = np.empty(shape=(6,), dtype=np.float64).fill(
                    np.nan
                )
            logging.info(
                f"DFT calculation for worker {idx} failed. Sending NaN values back."
            )

        self.world_comm.Isend(buf=dft_energies_num_atoms, dest=0, tag=idx)
        self.world_comm.Isend(
            buf=np.asarray(dft_forces, dtype=np.float64),
            dest=0,
            tag=idx + 10000,
        )
        if self.config.compute_stress:
            self.world_comm.Isend(
                buf=np.asarray(dft_stress, dtype=np.float64),
                dest=0,
                tag=idx + 20000,
            )

    def _handle_dft_call(self, idx):
        self.comm_handler.barrier()
        if self.rank == 0:
            logging.info(f"Trajectory worker {idx} is sending point to DFT.")
            self.send_points_non_blocking(
                idx=idx, point_data=self.trajectories[idx], tag=1234
            )
        self.comm_handler.barrier()
        self.state_manager.trajectory_status[idx] = "waiting"
        self.state_manager.num_workers_training += 1
        self.comm_handler.barrier()

        if self.rank == 0:
            logging.info(
                f"Trajectory worker {idx} is waiting for job to finish."
            )

    def training_task(self, idx: int):
        """
        Creates the dataloader of the updated dataset, updates
        the average number of neighbors, shifts and scaling factor
        and trains the ensemble members. Saves the models and checkpoints.

        Args:
            idx (int): Index of the trajectory worker.
        """
        # TODO: why is this here? why can't i use the parents method?
        self.ensemble_manager.ensemble_mace_sets = self._prepare_training(
            mace_sets=self.ensemble_manager.ensemble_mace_sets
        )

        logging.info(f"Trajectory worker {idx} is training.")
        # we train only for some epochs before we move to the next worker which may be running MD
        # all workers train on the same models with the respective training settings for
        # each ensemble member

        self._perform_training(idx)

        for trajectory in self.trajectories.values():
            trajectory.calc.models = [
                self.ensemble[tag] for tag in self.ensemble.keys()
            ]

        if self.state_manager.trajectory_total_epochs[idx] >= self.config.max_epochs_worker:
            self.state_manager.trajectory_status[idx] = "running"
            self.state_manager.num_workers_training -= 1
            self.state_manager.trajectory_total_epochs[idx] = 0
            if self.rank == 0:
                logging.info(f"Trajectory worker {idx} finished training.")
            # calculate true error and uncertainty on validation set

    def _process_analysis(
        self, idx: int, converged: bool, analysis_prediction: np.ndarray
    ):
        # Dummy to overwrite the parent method
        return

    def _analysis_waiting_task(self, idx: int):

        if self.config.restart and self.first_wait_after_restart[idx]:
            # if the worker is waiting and we just restarted the
            # procedure, we have to relaunch the dft job and then
            # leave the function
            logging.info(
                f"Trajectory worker {idx} is restarting DFT analysis."
            )
            self._analysis_dft_call(idx, self.trajectories[idx])
            self.first_wait_after_restart[idx] = False
            return None

        if self.worker_reqs_analysis["energy"][idx] is None:
            self.worker_reqs_analysis_bufs["energy"][idx] = np.empty(
                shape=(2,), dtype=np.float64
            )
            self.worker_reqs_analysis["energy"][idx] = self.world_comm.Irecv(
                buf=self.worker_reqs_analysis_bufs["energy"][idx],
                source=1,
                tag=idx,
            )
        status, _ = self.worker_reqs_analysis["energy"][idx].test()

        if status:
            scf_failed = np.isnan(
                self.worker_reqs_analysis_bufs["energy"][idx][0]
            )
            # check if the energy is NaN
            if not scf_failed:
                if self.worker_reqs_analysis["forces"][idx] is None:
                    self.worker_reqs_analysis_bufs["forces"][idx] = np.empty(
                        shape=(
                            int(
                                self.worker_reqs_analysis_bufs["energy"][idx][
                                    1
                                ]
                            ),
                            3,
                        ),
                        dtype=np.float64,
                    )
                    self.worker_reqs_analysis["forces"][idx] = (
                        self.world_comm.Irecv(
                            buf=self.worker_reqs_analysis_bufs["forces"][idx],
                            source=1,
                            tag=idx + 10000,
                        )
                    )
                    if self.config.compute_stress:
                        self.worker_reqs_analysis_bufs["stress"][idx] = (
                            np.empty(shape=(6,), dtype=np.float64)
                        )
                        self.worker_reqs_analysis["stress"][idx] = (
                            self.world_comm.Irecv(
                                buf=self.worker_reqs_analysis_bufs["stress"][
                                    idx
                                ],
                                source=1,
                                tag=idx + 20000,
                            )
                        )
                status_forces = self.worker_reqs_analysis["forces"][idx].Wait()
                if self.config.compute_stress:
                    status_stress = self.worker_reqs_analysis["stress"][
                        idx
                    ].Wait()
                if status_forces or (self.config.compute_stress and status_stress):
                    self.worker_reqs_analysis["energy"][idx] = None
                    self.worker_reqs_analysis["forces"][idx] = None
                    if self.config.compute_stress:
                        self.worker_reqs_analysis["stress"][idx] = None

                    if self.rank == 0:
                        logging.info(
                            f"Worker {idx} received a point from DFT for analysis."
                        )

                    analysis_forces = self.worker_reqs_analysis_bufs["forces"][
                        idx
                    ]
                    analysis_predicted_forces = (
                        self.state_manager.trajectories_analysis_prediction[idx]
                    )
                    check_results = self.analysis_check(
                        analysis_prediction=analysis_predicted_forces,
                        true_forces=analysis_forces,
                        current_md_step=self.state_manager.trajectory_MD_steps[idx],
                    )

                    for key in check_results.keys():
                        self.state_manager.analysis_checks[idx][key].append(
                            check_results[key]
                        )

                    self.state_manager.analysis_checks[idx]["threshold"].append(
                        self.state_manager.threshold 
                    )
                    self.state_manager.collect_thresholds[idx].append(self.state_manager.threshold )
                    self.state_manager.check += 1
                    self._save_analysis()
                    self.state_manager.trajectory_status[idx] = "running"

    def _analysis_dft_call(self, idx, point):

        self.comm_handler.barrier()
        if self.rank == 0:
            self.send_points_non_blocking(
                idx=idx, point_data=self.trajectories[idx], tag=80545
            )
        self.comm_handler.barrier()
        self.state_manager.trajectory_status[idx] = "analysis_waiting"
        self.comm_handler.barrier()
        if self.rank == 0:
            logging.info(
                f"Trajectory worker {idx} is waiting for analysis job to finish."
            )
        return None

    def _analysis_listening_task(self):
        received = False
        if self.req_sys_info_analysis is None:
            self.sys_info_buf_analysis = np.empty(
                shape=(14,), dtype=np.float64
            )
            self.req_sys_info_analysis = self.world_comm.Irecv(
                buf=self.sys_info_buf_analysis, source=0, tag=80545
            )

        status, _ = self.req_sys_info_analysis.test()
        if status:
            idx = int(self.sys_info_buf_analysis[0])
            self.current_num_atoms_analysis = int(
                self.sys_info_buf_analysis[1]
            )
            self.req_sys_info_analysis = None
            if self.req_geo_info_analysis is None:
                self.geo_info_buf_analysis = np.empty(
                    shape=(2, self.current_num_atoms_analysis, 3),
                    dtype=np.float64,
                )

                self.req_geo_info_analysis = self.world_comm.Irecv(
                    buf=self.geo_info_buf_analysis, source=0, tag=80546
                )
            status_pos_spec = self.req_geo_info_analysis.Wait()
            if status_pos_spec:
                self.req_geo_info_analysis = None
                received = True

        return received

    def _analysis_calculate_received(self):
        current_pbc = (
            self.sys_info_buf_analysis[2:5].astype(np.bool_).reshape((3,))
        )
        current_species = self.geo_info_buf_analysis[0].astype(np.int32)
        # transform current_species to a list of species
        current_species = current_species[:, 0].tolist()
        current_positions = self.geo_info_buf_analysis[1]
        current_cell = self.sys_info_buf_analysis[5:14].reshape((3, 3))

        point = ase.Atoms(
            positions=current_positions,
            numbers=current_species,
            pbc=current_pbc,
            cell=current_cell,
        )
        self.comm_handler.barrier()
        dft_result = self._recalc_aims(point)
        return dft_result

    def send_points_non_blocking(self, idx, point_data, tag):
        # send idx, pbc, cell, positions, species in a non-blocking way

        positions = np.asarray(point_data.get_positions(), dtype=np.float64)
        species = point_data.get_atomic_numbers()
        pbc = point_data.pbc
        cell = point_data.get_cell()
        num_atoms = len(positions)

        idx_num_atoms_pbc_cell = np.array(
            [idx, num_atoms, *pbc.flatten(), *cell.flatten()], dtype=np.float64
        )
        species_array = np.array(
            [[element, element, element] for element in species],
            dtype=np.float64,
        )

        positions_species = np.empty(
            shape=(2, positions.shape[0], 3),
        )
        positions_species[0] = species_array
        positions_species[1] = positions

        self.world_comm.Isend(buf=idx_num_atoms_pbc_cell, dest=1, tag=tag)
        self.world_comm.Isend(buf=positions_species, dest=1, tag=tag + 1)


class ALProcedurePARSL(ALProcedure):
    """
    This class is for the PARSL implementation of the active learning procedure.
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
            use_mpi=False,
        )
        parsl_setup_dict = prepare_parsl(
            cluster_settings=self.config.cluster_settings
        )
        self.parsl_config = parsl_setup_dict["config"]
        self.calc_dir = parsl_setup_dict["calc_dir"]
        self.clean_dirs = parsl_setup_dict["clean_dirs"]
        self.launch_str = parsl_setup_dict["launch_str"]
        try:
            parsl.dfk()
            logging.info(
                "PARSL is already initialized. Using existing PARSL context."
            )
        except parsl.errors.NoDataFlowKernelError:
            handle_parsl_logger(log_dir=self.log_dir / "parsl_al.log")
            parsl.load(self.parsl_config)

        logging.info("Launching ab initio manager thread for PARSL.")
        self.ab_initio_queue = queue.Queue()
        self.ab_intio_results = {}
        self.ab_initio_counter = {
            idx: 0 for idx in range(self.config.num_trajectories)
        }
        self.results_lock = threading.Lock()
        self.kill_thread = False
        threading.Thread(target=self.ab_initio_manager, daemon=True).start()
        self.first_wait_after_restart = {
            idx: True for idx in range(self.config.num_trajectories)
        }

        if self.config.analysis:
            logging.info("Launching analysis manager thread for PARSL.")
            self.analysis_queue = queue.Queue()
            self.analysis_kill_thread = False
            self.analysis_done = False
            self.analysis_results = {}
            self.analysis_counter = {
                idx: 0 for idx in range(self.config.num_trajectories)
            }
            threading.Thread(
                target=self._analysis_manager, daemon=True
            ).start()

    def _handle_dft_call(self, idx: int):
        logging.info(f"Trajectory worker {idx} is sending point to DFT.")
        self.state_manager.trajectory_status[idx] = "waiting"
        self.state_manager.num_workers_training += 1
        self.ab_initio_queue.put((idx, self.trajectories[idx]))
        logging.info(f"Trajectory worker {idx} is waiting for job to finish.")

    def ab_initio_manager(self):
        # collect parsl futures
        futures = {}
        for idx in range(self.config.num_trajectories):
            futures[idx] = {}
        # constantly check the queue for new jobs
        while True:

            if self.kill_thread:
                logging.info("Ab initio manager thread is stopping.")
                break
            try:
                idx, data = self.ab_initio_queue.get(timeout=1)
                with self.results_lock:
                    curr_job_no = self.ab_initio_counter[idx]
                futures[idx][curr_job_no] = recalc_aims_parsl(
                    positions=data.get_positions(),
                    species=data.get_chemical_symbols(),
                    cell=data.get_cell(),
                    pbc=data.pbc,
                    aims_settings=self.calc_manager.aims_settings,
                    directory=self.calc_dir / f"worker_{idx}_no_{curr_job_no}",
                    properties=self.config.properties,
                    ase_aims_command=self.launch_str,
                )
                with self.results_lock:
                    self.ab_initio_counter[idx] += 1
                self.ab_initio_queue.task_done()

            # is raised when the queue is empty after the timeout
            except queue.Empty:
                pass

            # goes through the futures and checks if they are done
            done_jobs = []
            for job_idx in futures.keys():
                for job_no, future in futures[job_idx].items():
                    if future.done():
                        done_jobs.append((job_idx, job_no))

            # if there are done jobs, get the results and store them in dict
            for job_idx, job_no in done_jobs:
                with self.results_lock:
                    temp_result = futures[job_idx][job_no].result()
                    if temp_result is None:
                        # if the result is None, it means the DFT calculation did not converge
                        self.ab_intio_results[job_idx] = False
                    else:
                        # the DFT calculation converged
                        self.ab_intio_results[job_idx] = temp_result
                        logging.info(
                            f"DFT calculation number {job_no} for worker {job_idx} finished."
                        )
                    # remove the job from the futures dict to avoid double counting
                    del futures[job_idx][job_no]
                    # remove folder with results
                    if self.clean_dirs:
                        try:
                            shutil.rmtree(
                                self.calc_dir / f"worker_{job_idx}_no_{job_no}"
                            )
                        except FileNotFoundError:
                            logging.warning(
                                f"Directory {self.calc_dir / f'worker_{job_idx}_{job_no}'} not found. Skipping removal."
                            )

    def _setup_aims_calculator(self, atoms):
        pass

    def _recalc_aims(self, current_point):
        pass

    def _finalize_ab_initio(self):
        with threading.Lock():
            self.kill_thread = True
            if self.config.analysis:
                self.analysis_kill_thread = True
                while not self.analysis_done:
                    time.sleep(0.1)
        time.sleep(5)
        parsl.dfk().cleanup()

    def waiting_task(self, idx):

        if self.config.restart and self.first_wait_after_restart[idx]:
            # if the worker is waiting and we just restarted the
            # procedure, we have to relaunch the dft job and then
            # leave the function
            logging.info(f"Worker {idx} is restarting DFT job after restart.")
            self._handle_dft_call(idx)
            self.first_wait_after_restart[idx] = False
            return None

        # with self.results_lock:
        job_result = self.ab_intio_results.get(idx, "not_done")

        if job_result == "not_done":
            # if the job is not done, we return None and wait for the next iteration
            return None
        else:
            if not job_result:
                logging.info(
                    f"SCF not converged at worker {idx}. Discarding point and restarting MD from last checkpoint."
                )
                self.trajectories[idx] = atoms_full_copy(
                    self.state_manager.MD_checkpoints[idx]
                )
                self.state_manager.trajectory_status[idx] = "running"

            else:
                logging.info(f"Worker {idx} received a point.")
                received_point = self.trajectories[idx].copy()
                received_point.info["REF_energy"] = job_result["energy"]
                received_point.arrays["REF_forces"] = job_result["forces"]
                if self.config.compute_stress:
                    received_point.info["REF_stress"] = job_result["stress"]

                self.state_manager.MD_checkpoints[idx] = atoms_full_copy(received_point)
                self.state_manager.MD_checkpoints[idx].calc = self.trajectories[idx].calc

                self._handle_received_point(
                    idx=idx, received_point=received_point
                )
            with self.results_lock:
                # remove the job from the results dict to avoid double counting
                del self.ab_intio_results[idx]

    def _analysis_manager(self):
        futures = {}
        predicted_forces = {}
        current_md_steps = {}

        for idx in range(self.config.num_trajectories):
            predicted_forces[idx] = {}
            futures[idx] = {}
            current_md_steps[idx] = {}

        kill_requested = False

        while True:
            if self.analysis_kill_thread and not kill_requested:
                logging.info(
                    "Analysis manager kill switch triggered. Waiting for pending analysis jobs..."
                )
                kill_requested = True

            # Try to get new data unless kill has been requested
            if not kill_requested:
                try:
                    idx, data = self.analysis_queue.get(timeout=1)
                    self.analysis_counter[idx] += 1
                    current_idx = self.analysis_counter[idx]
                    predicted_forces[idx][current_idx] = data.arrays[
                        "forces_comm"
                    ]
                    current_md_steps[idx][current_idx] = data.info[
                        "current_MD_step"
                    ]
                    futures[idx][current_idx] = recalc_aims_parsl(
                        positions=data.get_positions(),
                        species=data.get_chemical_symbols(),
                        cell=data.get_cell(),
                        pbc=data.pbc,
                        aims_settings=self.calc_manager.aims_settings,
                        directory=self.calc_dir
                        / f"worker_analysis{idx}_no_{current_idx}",
                        properties=self.config.properties,
                        ase_aims_command=self.launch_str,
                    )
                    self.analysis_queue.task_done()
                except queue.Empty:
                    pass

            # Check all futures for completion
            done_jobs = []
            for job_idx in list(futures.keys()):
                for job_no, future in list(futures[job_idx].items()):
                    if future.done():
                        done_jobs.append((job_idx, job_no))

            # Process done jobs
            for job_idx, job_no in done_jobs:
                with self.results_lock:
                    temp_result = futures[job_idx][job_no].result()
                if temp_result is None:
                    logging.info(
                        f"SCF during analysis for worker {job_idx} no {job_no} failed. Discarding point."
                    )
                else:
                    analysis_forces = predicted_forces[job_idx][job_no]
                    true_forces = temp_result["forces"]
                    check_results = self.analysis_check(
                        analysis_prediction=analysis_forces,
                        true_forces=true_forces,
                        current_md_step=current_md_steps[job_idx][job_no],
                    )
                    with self.results_lock:
                        for key in check_results:
                            self.state_manager.analysis_checks[job_idx][key].append(
                                check_results[key]
                            )
                        self.state_manager.analysis_checks[job_idx]["threshold"].append(
                            self.state_manager.threshold 
                        )
                        self.state_manager.collect_thresholds[job_idx].append(self.state_manager.threshold )
                        self.state_manager.check += 1
                        self._save_analysis()
                # Remove only the completed job
                del futures[job_idx][job_no]
                del predicted_forces[job_idx][job_no]
                del current_md_steps[job_idx][job_no]
                if self.clean_dirs:
                    try:
                        shutil.rmtree(
                            self.calc_dir
                            / f"worker_analysis{job_idx}_no_{job_no}"
                        )
                    except FileNotFoundError:
                        logging.warning(
                            f"Directory {self.calc_dir / f'worker_analysis{job_idx}_no_{job_no}'} not found. Skipping removal."
                        )

            # Check for final shutdown
            if kill_requested and not any(futures.values()):
                logging.info(
                    "All pending analysis jobs completed. Shutting down thread."
                )
                self.analysis_done = True
                break

            time.sleep(0.1)

    def _analysis_dft_call(self, idx: int, point: ase.Atoms):
        # we can go back to running here as the analysis is not blocking
        self.state_manager.trajectory_status[idx] = "running"
        self.analysis_queue.put((idx, point))

    def _process_analysis(
        self, idx: int, converged: bool, analysis_prediction: np.ndarray
    ):
        # Dummy to overwrite the parent method
        # processing is done in the analysis manager thread
        return
