import os
from pathlib import Path
import torch
import numpy as np
from mace import tools
from mace.calculators import MACECalculator
from aims_PAX.tools.uncertainty import (
    HandleUncertainty,
    MolForceUncertainty,
)
from aims_PAX.tools.utilities.data_handling import (
    load_ensemble_sets_from_folder,
    ase_to_mace_ensemble_sets,
)
from aims_PAX.tools.utilities.utilities import (
    ensemble_training_setups,
    ensemble_from_folder,
    Z_from_geometry_in,
    list_files_in_directory,
    get_atomic_energies_from_ensemble,
    create_ztable,
    get_atomic_energies_from_pt,
    dtype_mapping,
    ModifyMD,
)
from aims_PAX.tools.utilities.mpi_utils import CommHandler
import ase
from ase.io import read
import logging
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.npt import NPT
from ase import units

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
        self.uncertainty_type = self.al.get(
            "uncertainty_type", "max_atomic_sd"
        )
        self.uncert_not_crossed_limit = self.al.get(
            "uncert_not_crossed_limit", 500
        )
        self.freeze_threshold_dataset = self.al.get(
            "freeze_threshold_dataset", np.inf
        )
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
            self.intermol_crossed_limit = self.al.get(
                "intermol_crossed_limit", 10
            )
            self.intermol_forces_weight = self.al.get(
                "intermol_forces_weight", 100
            )
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
            path_to_folder=Path(
                self.config.al["dataset_dir"] + f"/{dataset_subdir}"
            ),
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
        self.train_dataset_len = self.comm_handler.bcast(
            self.train_dataset_len, root=0
        )
        self.comm_handler.barrier()


class ALCalculatorManager:
    """Manages calculators for active learning procedure."""

    def __init__(
        self,
        config: ALConfigurationManager,
        ensemble_manager: ALEnsembleManager,
        comm_handler,
    ):
        self.config = config
        self.ensemble_manager = ensemble_manager
        self.comm_handler = comm_handler
        self.rank = comm_handler.get_rank()

        # MACE calculator
        self.models = None
        self.mace_calc = None

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
            self._setup_mace_calculator()

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

    def _setup_mace_calculator(self):
        """Setup MACE calculator with ensemble models."""
        model_paths = list_files_in_directory(self.config.model_dir)
        self.models = [
            torch.load(
                f=model_path,
                map_location=self.config.device,
                weights_only=False,
            )
            for model_path in model_paths
        ]

        self.mace_calc = MACECalculator(
            models=self.models,
            device=self.config.device,
            default_dtype=self.config.dtype,
        )


class ALMDManager:
    """Manages molecular dynamics setup and execution for active learning."""

    def __init__(
        self,
        config: ALConfigurationManager,
        state_manager: ALStateManager,
        comm_handler,
    ):
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

    def _setup_md_dynamics(
        self, atoms: ase.Atoms, md_settings: dict, idx: int
    ):
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

    def _create_dynamics_engine(
        self, atoms: ase.Atoms, md_settings: dict, ensemble: str, idx: int
    ):
        """Create appropriate dynamics engine based on ensemble type."""
        if ensemble == "nvt":
            return self._create_nvt_dynamics(atoms, md_settings, idx)
        elif ensemble == "npt":
            return self._create_npt_dynamics(atoms, md_settings, idx)
        else:
            raise ValueError(f"Unsupported ensemble type: {ensemble}")

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
                temperature_K=self.current_temperatures[idx],
                rng=np.random.RandomState(md_settings["seed"]),
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
        elif barostat == "npt":
            return self._create_standard_npt(atoms, md_settings, idx)
        else:
            raise ValueError(f"Unsupported barostat: {barostat}")

    def _create_berendsen_npt(
        self, atoms: ase.Atoms, md_settings: dict, idx: int
    ) -> NPTBerendsen:
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

    def _create_standard_npt(
        self, atoms: ase.Atoms, md_settings: dict, idx: int
    ) -> NPT:
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
        self.md_mod_settings = self.config.md_settings.get("MODIFY", None)
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


class ALRestartManager:
    """Handles active learning restart functionality."""

    def __init__(
        self,
        config: ALConfigurationManager,
        state_manager: ALStateManager,
        comm_handler,
        md_manager: ALMDManager,
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
        ]

        # Add conditional attributes
        if self._is_nvt_or_npt_ensemble():
            base_attributes.append("current_temperatures")

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

        # Add conditional attributes
        if self._is_nvt_or_npt_ensemble():
            broadcast_attributes.append("current_temperatures")

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
        return self.config.md_settings["stat_ensemble"].lower() in [
            "nvt",
            "npt",
        ]

    def update_restart_dict(
        self, trajectories_keys, md_drivers, save_restart: str = None
    ):
        """Update restart dictionary with current state."""
        self._update_base_restart_attributes()
        self._update_temperature_attributes(md_drivers, trajectories_keys)
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

    def _update_temperature_attributes(self, md_drivers, trajectories_keys):
        """
        Update temperature-related restart
        attributes for NVT/NPT ensembles.
        """
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
        self.ensemble_manager = ALEnsembleManager(
            self.config, self.comm_handler
        )
        self.state_manager = ALStateManager(self.config, self.comm_handler)
        self.calc_manager = ALCalculatorManager(
            self.config, self.ensemble_manager, self.comm_handler
        )
        self.md_manager = ALMDManager(
            self.config, self.state_manager, self.comm_handler
        )
        self.restart_manager = ALRestartManager(
            self.config, self.state_manager, self.comm_handler, self.md_manager
        )

        # Setup logging and basic configuration
        self._setup_logging()
        self._create_folders()

        # Set random seed
        np.random.seed(self.config.seed)

        # Setup calculators and datasets
        self.ensemble_manager.setup_ensemble_and_datasets()
        self.calc_manager.setup_ml_calculators()
        # Pass train_dataset_len to MD manager for modification
        self.md_manager.train_dataset_len = (
            self.ensemble_manager.train_dataset_len
        )

        if self.config.restart:
            self.restart_manager.handle_restart()
        else:
            self.state_manager.initialize_fresh_state(path_to_geometry)
            # Make sure state manager has access to seeds_tags_dict
            self.state_manager.seeds_tags_dict = self.config.seeds_tags_dict
            self.state_manager.initialize_fresh_state(path_to_geometry)

        if self.rank == 0:
            self.md_manager.setup_md_drivers(
                self.state_manager.trajectories, self.calc_manager.mace_calc
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
            if self.config.mace_settings["MISC"]["log_level"].lower()
            == "debug"
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
    def md_drivers(self):
        return self.md_manager.md_drivers

    @property
    def mace_calc(self):
        return self.calc_manager.mace_calc

    @property
    def ensemble(self):
        return self.ensemble_manager.ensemble

    @property
    def get_uncertainty(self):
        return self.md_manager.get_uncertainty
