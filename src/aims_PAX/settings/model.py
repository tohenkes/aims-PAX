"""
Model settings for aims-PAX project
"""
from pathlib import Path
from typing import Literal, Annotated, Union

from pydantic import Field, field_validator

from .project import ProjectBaseModel


class GeneralSettings(ProjectBaseModel):
    name_exp: str = Field(
        ...,
        description="The name given to the experiment, models, and datasets."
    )
    checkpoints_dir: Path = Field(
        default=Path("./checkpoints"),
        description="Directory path for storing model checkpoints.",
        validate_default=True
    )
    compute_stress: bool = Field(
        default=False,
        description="Whether to compute stress tensors."
    )
    default_dtype: Literal["float32", "float64"] = Field(
        default="float32",
        description="Default data type for model parameters."
    )
    loss_dir: Path = Field(
        default=Path("./losses"),
        description="Directory path for storing training losses.",
        validate_default=True
    )
    model_dir: Path = Field(
        default=Path("./model"),
        description="Directory path for storing final trained models.",
        validate_default=True
    )
    seed: int = Field(
        default=42,
        description="Random seed for ensemble member generation."
    )

    @field_validator("checkpoints_dir", "loss_dir", "model_dir", mode="after")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Automatically create the training directories if they don't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class MACEArchitectureSettings(ProjectBaseModel):
    model: Literal["MACE"]
    atomic_energies: dict[int, float] | None = Field(
        default=None,
        description="Atomic energy references {atomic_number: energy}. "
                    "If None, determined via linear least squares on training set."
    )
    compute_avg_num_neighbors: bool = Field(
        default=True,
        description="Whether to compute average number of neighbors."
    )
    correlation: int = Field(
        default=3,
        description="Correlation order for many-body interactions."
    )
    gate: str = Field(
        default="silu",
        description="Activation function (e.g., 'silu', 'tanh')."
    )
    interaction: str = Field(
        default="RealAgnosticResidualInteractionBlock",
        description="Type of interaction block."
    )
    interaction_first: str = Field(
        default="RealAgnosticResidualInteractionBlock",
        description="Type of first interaction block."
    )
    max_ell: int = Field(
        default=3,
        description="Maximum degree of direction embeddings."
    )
    max_L: int = Field(
        default=1,
        description="Maximum degree for equivariant features."
    )
    MLP_irreps: str = Field(
        default="16x0e",
        description="Irreps of the MLP in the last readout (e3nn format)."
    )
    num_channels: int = Field(
        default=128,
        description="Number of channels (features)."
    )
    num_cutoff_basis: int = Field(
        default=5,
        description="Number of cutoff basis functions."
    )
    num_interactions: int = Field(
        2,
        description="Number of interaction layers."
    )
    num_radial_basis: int = Field(
        default=8,
        description="Number of radial basis functions."
    )
    r_max: float = Field(
        default=5.0,
        description="Cutoff radius in Angstroms."
    )
    radial_MLP: list[int] = Field(
        default=[64, 64, 64],
        description="Architecture of the radial MLP (hidden layer sizes)."
    )
    radial_type: str = Field(
        default="bessel",
        description="Type of radial basis functions."
    )
    scaling: str = Field(
        default="rms_forces_scaling",
        description="Scaling method used."
    )
    # extra fields from check_inputs
    use_multihead_model: bool = Field(
        default=False,
        description="Whether to use a multihead model architecture."
    )
    num_multihead_heads: int | None = Field(
        default=None,
        description="Number of heads if using a multihead model."
    )


class SO3LRArchitectureSettings(ProjectBaseModel):
    model: Literal["SO3LR"]
    r_max: float = 4.5
    num_features: int = 128
    num_radial_basis_fn: int = 32
    degrees: list[int] = [1, 2, 3, 4]
    num_heads: int = 4
    num_layers: int = 3
    final_mlp_layers: int = 2
    energy_regression_dim: int = 128
    message_normalization: str = "avg_num_neighbors"
    initialize_ev_to_zeros: bool = True
    radial_basis_fn: str = "bernstein"
    trainable_rbf: bool = False
    energy_learn_atomic_type_shifts: bool = True
    energy_learn_atomic_type_scales: bool = True
    layer_normalization_1: bool = True
    layer_normalization_2: bool = True
    residual_mlp_1: bool = True
    residual_mlp_2: bool = False
    use_charge_embed: bool = True
    use_spin_embed: bool = True
    interaction_bias: bool = True
    qk_non_linearity: str = "identity"
    cutoff_fn: str = "phys"
    cutoff_p: int = 5
    activation_fn: str = "silu"
    energy_activation_fn: str = "identity"
    layers_behave_like_identity_fn_at_init: bool = False
    output_is_zero_at_init: bool = False
    input_convention: str = "positions"
    zbl_repulsion_bool: bool = True
    electrostatic_energy_bool: bool = True
    electrostatic_energy_scale: float = 4.0
    dispersion_energy_bool: bool = True
    dispersion_energy_scale: float = 1.2
    dispersion_energy_cutoff_lr_damping: float = 2.0
    r_max_lr: float | None = None
    neighborlist_format_lr: str = "sparse"
    atomic_energies: dict[int, float] | None = None
    use_multihead_model: bool = False
    num_multihead_heads: int | None = None


ArchitectureSettings = Annotated[
    Union[MACEArchitectureSettings, SO3LRArchitectureSettings],
    Field(
        discriminator="model",
        description="Type of model architecture to use."
    )
]


class TrainingSettings(ProjectBaseModel):
    optimizer: Literal["adam", "adamw"] = Field(
        default="adam",
        description="Optimizer type (adam/adamw)."
    )
    lr: float = Field(
        default=0.01,
        description="Initial learning rate for optimizer."
    )
    weight_decay: float = Field(
        default=5.e-07,
        description="L2 regularization weight decay factor."
    )
    amsgrad: bool = Field(
        default=True,
        description="Whether to use AMSGrad variant of Adam optimizer."
    )
    clip_grad: float = Field(
        default=10.0,
        description="Gradient clipping threshold."
    )
    batch_size: int = Field(
        default=5,
        description="Batch size for training data."
    )
    valid_batch_size: int = Field(
        default=5,
        description="Batch size for validation data."
    )
    seed: int = Field(
        default=42,
        description="Random seed for ensemble generation."
    )
    loss: str = Field(
        default="weighted",
        description="Loss function type."
    )
    energy_weight: float = Field(
        default=1.0,
        description="Weight for energy loss component."
    )
    forces_weight: float = Field(
        default=1000.0,
        description="Weight for forces loss component."
    )
    stress_weight: float = Field(
        default=1.0,
        description="Weight for stress loss component."
    )
    virials_weight: float = Field(
        default=1.0,
        description="Weight for virials loss component."
    )
    config_type_weights: dict[str, float] = Field(
        default_factory=lambda: {"Default": 1.0},
        description="Weights for different configuration types (e.g., TS vs Ground State)."
    )
    scheduler: Literal["ReduceLROnPlateau", "ExponentialLR"] = Field(
        default="ReduceLROnPlateau",
        description="Learning rate scheduler type."
    )
    lr_factor: float = Field(
        default=0.8,
        description="Factor by which learning rate is reduced (for ReduceLROnPlateau)."
    )
    lr_scheduler_gamma: float = Field(
        default=0.9993,
        description="Learning rate decay factor (for ExponentialLR)."
    )
    scheduler_patience: int = Field(
        default=5,
        description="Number of epochs to wait before reducing LR."
    )
    ema: bool = Field(
        default=True,
        description="Whether to use Exponential Moving Average."
    )
    ema_decay: float = Field(
        default=0.99,
        description="Decay factor for exponential moving average."
    )
    swa: bool = Field(
        default=False,
        description="Whether to use Stochastic Weight Averaging."
    )
    # extra fields
    pretrained_model: str | None = None
    pretrained_weights: str | None = None
    update_avg_num_neighbors: bool = True
    perform_finetuning: bool = False
    finetuning_choice: Literal["naive", "lora"] = "naive"
    freeze_embedding: bool = False
    freeze_zbl: bool = False
    freeze_hirshfeld: bool = False
    freeze_partial_charges: bool = False
    freeze_shifts: bool = False
    freeze_scales: bool = False
    convert_to_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_freeze_A: bool = False
    convert_to_multihead: bool = False


class MiscSettings(ProjectBaseModel):
    device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description="Device used for training and inference (cpu or cuda)."
    )
    error_table: str = Field(
        default="PerAtomMAE",
        description="Type of error metrics to compute and display in logs."
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity level."
    )
    keep_checkpoints: bool = Field(
        default=False,
        description="Whether to keep all checkpoint files or only the best/latest."
    )
    restart_latest: bool = Field(
        default=False,
        description="Whether to restart training from the latest available checkpoint."
    )
    compute_stress: bool = Field(
        default=False,
        description="Whether to compute stress tensors."
    )
    compute_dipole: bool = Field(
        default=False,
        description="Whether to compute dipole moments."
    )
    enable_cueq: bool = Field(
        default=False,
        description="Enable Charge Equilibration (CuEq) module."
    )
    enable_cueq_train: bool = Field(
        default=False,
        description="Enable training for the CuEq parameters."
    )


class ModelSettings(ProjectBaseModel):
    GENERAL: GeneralSettings = Field(
        ...,
        description="General model settings"
    )
    ARCHITECTURE: ArchitectureSettings = Field(
        ...,
        description="Architecture settings"
    )
    TRAINING: TrainingSettings = Field(default_factory=TrainingSettings)
    MISC: MiscSettings = Field(default_factory=MiscSettings)



