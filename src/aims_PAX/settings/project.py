"""
Pydantic settings for aimsPAX project.
"""
import io
import sys
import yaml
from pathlib import Path
from typing import Union, Dict, Any, Annotated

from pydantic import BaseModel, RootModel, Field, DirectoryPath,  FilePath, model_validator, field_validator, \
    ValidationError
from typing_extensions import Literal


class ProjectBaseModel(BaseModel):
    """Base class for settings in the project"""

    @classmethod
    def from_file(cls, f: str | Path | io.FileIO) -> "ProjectBaseModel":
        """Load settings from a file."""
        if isinstance(f, (str, Path)):
            path = Path(f)
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {path}")

            with open(path, "r") as file_stream:
                data = yaml.safe_load(file_stream)
        else:
            data = yaml.safe_load(f)
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            # error in validation, can be logged in any way
            print(f"\n[!] Configuration error in {cls.__name__}:")
            print(e)
            raise


class FMSettings(ProjectBaseModel):
    """Base class for foundational model settings."""


class MaceFMSettings(FMSettings):
    """Pydantic settings for Mace foundational model."""

    mace_model: str = Field(
        default="small",
        description="Type of `MACE` foundational model. See "
                    "[here](https://github.com/ACEsuit/mace/blob/main/mace/calculators/foundations_models.py) "
                    "for their names.",
    )


class So3lrFMSettings(FMSettings):
    """Pydantic settings for SO3LR foundational model."""

    r_max_lr: float | None = Field(
        default=None, description="Cutoff of long-range modules of `SO3LR`."
    )
    dispersion: bool = Field(default=False)
    dispersion_xc: str = Field(default="pbe")
    dispersion_cutoff: float = Field(default=12.0, gt=0.)
    damping: str = Field(default="bj")
    dispersion_lr_damping: float | None = Field(
        default=None,
        description="Damping parameter for dispersion interaction in `SO3LR`. Needed if `r_max_lr` is not `None`!",
    )


MODEL_MAP = {
    "mace-mp": MaceFMSettings,
    "so3lr": So3lrFMSettings,
}


class IDGSettings(ProjectBaseModel):
    """Pydantic settings for Initial Dataset Generation"""

    species_dir: DirectoryPath = Field(
        ...,
        description="Path to the directory containing the FHI AIMS species defaults.",
    )
    n_points_per_sampling_step_idg: int = Field(
        ...,
        description="Number of points that is sampled at each step for each model in the ensemble and each geometry.",
        gt=0,
    )
    analysis: bool = Field(
        default=False,
        description="Whether to save metrics such as losses during initial dataset generation",
    )
    desired_acc: float = Field(
        default=0.0,
        description="Force MAE (eV/Å) that the ensemble should reach on the validation set. Needs to be combined with "
        "`desired_acc_scale_idg`.",
    )
    desired_acc_scale_idg: float = Field(
        default=10.0,
        description="Scales `desired_acc` during initial dataset generation. Resulting product is accuracy "
        "that the model has to reach on the validation set before stopping the procedure at this stage.",
    )
    distinct_model_sets: bool = Field(
        default=True,
        description="Whether to sample enough points so that every model in the ensemble gets distinct data sets. "
        "If set to `False`, the same dataset is used for all ensemble members.",
    )
    ensemble_size: int = Field(
        default=4,
        description="Number of models in the ensemble for uncertainty estimation.",
        gt=0,
    )
    foundational_model: Literal["mace-mp", "so3lr"] = Field(
        default="mace-mp",
        description="Which foundational model to use for structure generation. Possible options: `mace-mp` or `so3lr`.",
    )
    foundational_model_settings: Union[Dict[str, Any], FMSettings] = Field(
        default_factory=dict,
        description="Foundational model settings"
    )
    intermediate_epochs_idg: int = Field(
        default=5,
        description="Number of intermediate epochs between dataset growth steps in initial training.",
        gt=0,
    )
    max_initial_epochs: int = Field(
        default=sys.maxsize,
        description="Maximum number of epochs for the initial training stage.",
        gt=0,
    )
    max_initial_set_size: int = Field(
        default=sys.maxsize,
        description="Maximum size of the initial training dataset.",
        gt=0,
    )
    progress_dft_update: int = Field(
        default=10,
        description="Intervals at which progress of DFT calculations is logged.",
        gt=0,
    )
    scheduler_initial: bool = Field(
        default=True,
        description="Whether to use a learning rate scheduler during initial training.",
    )
    skip_step_initial: int = Field(
        default=25,
        description="Intervals at which a structure is taken from the MD simulation either for the dataset "
                    "in case of AIMD or for DFT in case of using an MLFF.",
        gt=0,
    )
    valid_ratio: float = Field(
        default=0.1, description="Fraction of data reserved for validation.", gt=0
    )
    valid_skip: int = Field(
        default=1,
        description="Number of training steps between validation runs in initial training.",
    )
    # Fields populated from your provided table
    converge_initial: bool = Field(
        default=False,
        description="Whether to converge the model(s) on the initial training set after a stopping criterion was met.",
    )
    convergence_patience: int = Field(
        default=50,
        description="Number of epochs without improvement before halting convergence.",
    )
    margin: float = Field(
        default=0.002,
        description="Margin to decide if a model has improved over the previous training epoch.",
    )
    max_convergence_epochs: int = Field(
        default=500,
        description="Maximum total epochs allowed before halting convergence.",
    )
    initial_sampling: Literal["aimd", "foundational"] = Field(default="foundational")
    aims_lib_path: str | None = Field(default=None, description="Path to the compiled FHI-aims library for direct force and energy evaluation.")

    @model_validator(mode="after")
    def check_at_least_one_required(self) -> "IDGSettings":
        """Ensures stopping criteria are set."""
        # If all these are at their 'unlimited' defaults, the simulation runs forever
        is_acc_set = self.desired_acc > 0
        is_epochs_set = self.max_initial_epochs < sys.maxsize
        is_size_set = self.max_initial_set_size < sys.maxsize

        if not any([is_acc_set, is_epochs_set, is_size_set]):
            raise ValueError(
                "INITIAL_DATASET_GENERATION: You must provide a limit for at least one stopping criterion: "
                "desired_acc, max_initial_epochs, or max_initial_set_size."
            )
        return self

    @model_validator(mode="after")
    def validate_nested_settings(self) -> "IDGSettings":
        """Ensures foundational_model_settings matches the foundational_model type."""
        model_class = MODEL_MAP.get(self.foundational_model)
        self.foundational_model_settings = model_class(**self.foundational_model_settings)
        return self

class ALSettings(ProjectBaseModel):
    species_dir: str = Field(
        ...,
        description="Path to the directory containing the FHI-aims species defaults."
    )
    num_trajectories: int = Field(
        ...,
        description="How many trajectories are sampled from during the active learning phase.",
        gt=0
    )
    analysis: bool = Field(
        default=False,
        description="Whether to run DFT calculations at specified intervals and save predictions, uncertainties etc."
    )
    analysis_skip: int = Field(
        default=50,
        description="Interval (in MD steps) at which analysis DFT calculations are performed."
    )
    c_x: float = Field(
        default=0.0,
        description="Weighting factor for the uncertainty threshold (see Eq. 2 in the paper). < 0 tightens, > 0 relaxes the threshold."
    )
    desired_acc: float = Field(
        default=0.0,
        description="Force MAE (eV/Å) that the ensemble should reach on the validation set."
    )
    ensemble_size: int = Field(
        default=4,
        description="Number of models in the ensemble for uncertainty estimation."
    )
    epochs_per_worker: int = Field(
        default=2,
        description="Number of training epochs per worker after DFT is done."
    )
    freeze_threshold_dataset: float = Field(
        default=float("inf"),
        description="Training set size at which the uncertainty threshold is frozen; inf disables freezing."
    )
    intermediate_epochs_al: int = Field(
        default=1,
        description="Number of intermediate training epochs after DFT is done."
    )
    margin: float = Field(
        default=0.002,
        description="Margin to decide if a model has improved over the previous training epoch."
    )
    max_MD_steps: int = Field(
        default=sys.maxsize,
        description="Maximum number of steps taken using the MLFF during active learning per trajectory."
    )
    max_train_set_size: int = Field(
        default=sys.maxsize,
        description="Maximum size of training set before procedure is stopped."
    )
    seeds_tags_dict: dict | None = Field(
        default=None,
        description="Optional mapping of seed indices to trajectory tags for reproducible runs."
    )
    skip_step_mlff: int = Field(
        default=25,
        description="Step interval for evaluating the uncertainty criterion during MD in active learning."
    )
    uncertainty_type: str = Field(
        default="max_atomic_sd",
        description="Method for estimating prediction uncertainty. Default is max force standard deviation."
    )
    uncert_not_crossed_limit: int = Field(
        default=50000,
        description="Max consecutive steps without crossing uncertainty threshold after which a point is treated as if it crossed."
    )
    valid_ratio: float = Field(
        default=0.1,
        description="Fraction of data reserved for validation during active learning."
    )
    valid_skip: int = Field(
        default=1,
        description="Rate at which validation of model during training is performed."
    )
    replay_strategy: str = Field(
        default="full_dataset",
        description="Method for replaying data during training (e.g., 'full_dataset' or 'random_batch')."
    )
    # --- Convergence ---
    converge_al: bool = Field(
        default=True,
        description="Whether to converge the model(s) on the final training set."
    )
    converge_best: bool = Field(
        default=True,
        description="Whether to only converge the best performing model of the ensemble."
    )
    convergence_patience: int = Field(
        default=50,
        description="Number of epochs without improvement before halting convergence."
    )
    max_convergence_epochs: int = Field(
        default=500,
        description="Maximum total epochs allowed before halting convergence."
    )

    # --- Not actively used, but are working ---
    aims_lib_path: str | None = Field(
        default=None,
        description="Path to the compiled FHI-aims library for direct force and energy evaluation via API."
    )
    parallel: bool = Field(
        default=False,
        description="Whether to run multiple active learning trajectories in parallel using multiprocessing."
    )
    intermol_crossed_limit: int = Field(
        default=10,
        description="Max uncertainty threshold crossings allowed for intermolecular interactions before stopping."
    )
    intermol_forces_weight: float = Field(
        default=100.0,
        description="Weight factor applied to intermolecular force contributions in the loss function."
    )
    extend_existing_final_ds: bool = Field(
        default=False
    )


    @model_validator(mode="after")
    def check_at_least_one_required(self) -> "ALSettings":
        """Ensures stopping criteria are set."""
        # If all these are at their 'unlimited' defaults, the simulation runs forever
        is_acc_set = self.desired_acc > 0
        is_steps_set = self.max_MD_steps < sys.maxsize
        is_size_set = self.max_train_set_size < sys.maxsize

        if not any([is_acc_set, is_steps_set, is_size_set]):
            raise ValueError(
                "ACTIVE_LEARNING: You must provide a limit for at least one stopping criterion: "
                "desired_acc, max_MD_steps, or max_train_set_size."
            )
        return self


class TrajectoryMDBase(ProjectBaseModel):
    timestep: float = Field(
        default=0.5,
        description="Time step for molecular dynamics (in femtoseconds)."
    )
    temperature: float = Field(
        default=300.0,
        description="Target temperature."
    )


class LangevinNVT(TrajectoryMDBase):
    stat_ensemble: Literal["nvt"]
    thermostat: Literal["langevin"]
    friction: float = Field(
        default=0.001,
        description="Friction coefficient for Langevin dynamics (in fs<sup>-1</sup>)."
    )
    MD_seed: int = Field(
        default=42,
        description="Random number generator seed for Langevin dynamics."
    )


class BerendsenNPT(TrajectoryMDBase):
    stat_ensemble: Literal["npt"]
    barostat: Literal["berendsen"]
    pressure: float = Field(
        default=101325.0,
        description="Pressure used for `NPT` in Pa"
    )


class MTKNPT(TrajectoryMDBase):
    stat_ensemble: Literal["npt"]
    barostat: Literal["mtk"] = "mtk"
    pressure: float = Field(
        default=101325.0,
        description="Pressure used for `NPT` in Pa"
    )
    tdamp: float = Field(
        default=0.5 * 100,
        description="Temperature damping for MTK dynamics (`100*timestep`)."
    )
    pdamp: float = Field(
        default=0.5 * 1000,
        description="Pressure damping for MTK dynamics (`1000*timestep`)."
    )
    tchain: int = Field(
        default=3,
        description="Number of thermostats in the thermostat chain for MTK dynamics."
    )
    pchain: int = Field(
        default=3,
        description="Number of thermostats in the barostat chain for MTK dynamics."
    )
    tloop: int = Field(
        default=1,
        description="Number of loops for thermostat integration in MTK dynamics."
    )
    ploop: int = Field(
        default=1,
        description="Number of loops for barostat integration in MTK dynamics."
    )


NPTEnsemble = Annotated[
    Union[BerendsenNPT, MTKNPT],
    Field(
        discriminator="barostat",
        description="Barostat used when `NPT` is chosen, either `mtk` or `berendsen`. MTK stands for Full"
                    " [Martyna-Tobias-Klein barostat](https://doi.org/10.1063/1.467468)."
    )
]


NVTEnsemble = Annotated[
    Union[LangevinNVT],
    Field(
        discriminator="thermostat",
        description="Thermostat used when `NVT` is chosen.")
]


TrajectoryMDSettings = Annotated[
    Union[NVTEnsemble, NPTEnsemble],
    Field(
        discriminator="stat_ensemble",
        description="Statistical ensemble for molecular dynamics (e.g., `NVT`, `NPT`)."
    )
]


class MDSettings(RootModel):
    root: Union[Dict[int, TrajectoryMDSettings], TrajectoryMDSettings]

    def get_for_index(self, index: int) -> TrajectoryMDSettings:
        """Helper to retrieve settings regardless of which format was used."""
        if isinstance(self.root, dict):
            if index not in self.root:
                raise KeyError(f"No MD settings found for trajectory index {index}.")
            return self.root[index]
        return self.root

    @field_validator("root", mode="before")
    @classmethod
    def normalize_case(cls, v: Any) -> Any:
        def fix_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            # keys to normalize to lowercase
            for key in ["stat_ensemble", "thermostat", "barostat"]:
                if key in d and isinstance(d[key], str):
                    d[key] = d[key].lower()
            return d

        if isinstance(v, dict):
            # indexed format
            if any(isinstance(k, int) for k in v.keys()):
                return {k: fix_dict(val) if isinstance(val, dict) else val for k, val in v.items()}
            return fix_dict(v)
        return v


class ParslSettings(ProjectBaseModel):
    nodes_per_block: int = Field(
        ...,
        description="Number of nodes per block."
    )
    init_blocks: int = Field(
        ...,
        description="Initial number of blocks to launch."
    )
    min_blocks: int = Field(
        ...,
        description="Minimum number of blocks allowed."
    )
    max_blocks: int = Field(
        ...,
        description="Maximum number of blocks allowed."
    )
    label: str = Field(
        ...,
        description="Unique label for this Parsl configuration. Must be unique for each aims-PAX instance."
    )
    run_dir: str | None = Field(
        default=None,
        description="Directory to store runtime files."
    )
    function_dir: str | None = Field(
        default=None,
        description="Directory for Parsl function storage."
    )


class ClusterSettings(ProjectBaseModel):
    type: Literal["slurm"] = Field(
        default="slurm",
        description="Cluster backend type. Currently only 'slurm' is available."
    )
    parsl_options: ParslSettings = Field(
        ...,
        description="Parsl configuration options including block scaling and labeling."
    )
    slurm_str: str = Field(
        ...,
        description="SLURM job script header specifying job resources and options. Can be multiline."
    )
    worker_str: str = Field(
        ...,
        description=(
            "Shell commands to configure the environment for each worker process. Can be multiline. "
            "Ensure 'export WORK_QUEUE_DISABLE_SHARED_PORT=1' is included if necessary."
        )
    )
    launch_str: str = Field(
        ...,
        description="Command to run FHI aims, e.g., 'srun path/to/aims >> aims.out'."
    )
    calc_dir: str = Field(
        ...,
        description="Path to the directory used for calculation outputs."
    )
    clean_dirs: bool = Field(
        default=True,
        description="Whether to remove calculation directories after DFT computations."
    )


PathOrIndexedPaths = Union[FilePath, Dict[int, FilePath]]

class MiscSettings(ProjectBaseModel):
    create_restart: bool = Field(
        default=True,
        description="Whether to create restart files during the run."
    )
    dataset_dir: Path = Field(
        default=Path("./data"),
        description="Directory where dataset files will be stored."
    )
    log_dir: Path = Field(
        default=Path("./logs"),
        description="Directory where log files are saved."
    )
    path_to_control: PathOrIndexedPaths = Field(
        default=Path("./control.in"),
        description=(
            "Path to the FHI-aims control input file. Can be a single path "
            "or a dictionary mapping system indices to specific control files."
        )
    )
    path_to_geometry: Union[PathOrIndexedPaths | DirectoryPath] = Field(
        default=Path("./geometry.in"),
        description=(
            "Path to the geometry input file or folder. Can be a single path, "
            "a folder, or a dictionary mapping indices to specific geometry files."
        )
    )
    energy_key: str = "REF_energy"
    forces_key: str = "REF_forces"
    stress_key: str = "REF_stress"
    dipole_key: str = "REF_dipole"
    polarizability_key: str = "REF_polarizability"
    head_key: str = "head"
    charges_key: str = "REF_charges"
    total_charge_key: str = "total_charge"
    total_spin_key: str = "total_spin"
    mol_idxs: list[int] | None  = Field(
        default=None,
        description="Specific molecule indices to include/exclude."
    )

    @model_validator(mode="after")
    def validate_indices_match(self) -> "MiscSettings":
        """
        Cross-checks that if indexed paths are used, the indices
        overlap correctly between geometry and control files.
        """
        if isinstance(self.path_to_geometry, dict) and isinstance(self.path_to_control, dict):
            geo_indices = set(self.path_to_geometry.keys())
            ctrl_indices = set(self.path_to_control.keys())

            if not geo_indices.issubset(ctrl_indices):
                missing = geo_indices - ctrl_indices
                raise ValueError(f"Geometry indices {missing} are missing corresponding control files.")

        return self


class AimsPAXSettings(ProjectBaseModel):
    """Pydantic settings for aimsPAX project."""
    INITIAL_DATASET_GENERATION: IDGSettings | None = Field(
        default=None,
        description="Initial dataset generation Settings"
    )
    ACTIVE_LEARNING: ALSettings | None = Field(
        default=None,
        description="Active Learning settings"
    )
    MD: MDSettings = Field(..., description="Molecular Dynamics settings.")
    CLUSTER: ClusterSettings = Field(..., description="PARSL/HPC infrastructure settings.")
    MISC: MiscSettings = Field(default_factory=MiscSettings)

    @model_validator(mode="after")
    def check_operation_mode(self) -> "AimsPAXSettings":
        """
        Enforces that at least one functional mode is active.
        """
        al_active = self.ACTIVE_LEARNING is not None
        gen_active = self.INITIAL_DATASET_GENERATION is not None

        if not (al_active or gen_active):
            raise ValueError(
                "Incomplete Configuration: You must provide settings for "
                "'ACTIVE_LEARNING' or 'INITIAL_DATASET_GENERATION', or both."
            )
        return self

