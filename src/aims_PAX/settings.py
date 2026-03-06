"""
Pydantic settings for aimsPAX project.
"""

import sys
from typing import Union, Dict, Any

from pydantic import BaseModel, Field, DirectoryPath, model_validator
from typing_extensions import Literal


class ProjectBaseModel(BaseModel):
    """Base class for settings in the project"""


class AimsPAXSettings(ProjectBaseModel):
    """Pydantic settings for aimsPAX project."""


class MaceSettings(ProjectBaseModel):
    """Pydantic settings for Mace foundational model."""

    mace_model: str = Field(
        default="small",
        description="Type of `MACE` foundational model. See "
                    "[here](https://github.com/ACEsuit/mace/blob/main/mace/calculators/foundations_models.py) "
                    "for their names.",
    )


class So3lrSettings(ProjectBaseModel):
    """Pydantic settings for SO3LR foundational model."""

    r_max_lr: float | None = Field(
        default=None, description="Cutoff of long-range modules of `SO3LR`."
    )
    dispersion: bool = Field(default=False)
    dispersion_xc: str = Field(default="pbe")
    dispersion_cutoff: float = Field(default=12.0)
    damping: str = Field(default="bj")
    dispersion_lr_damping: float | None = Field(
        default=None,
        description="Damping parameter for dispersion interaction in `SO3LR`. Needed if `r_max_lr` is not `None`!",
    )


MODEL_MAP = {
    "mace-mp": MaceSettings,
    "so3lr": So3lrSettings,
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
    foundational_model: Literal["mace-mp", "so3lr"] = (
        Field(
            default="mace-mp",
            description="Which foundational model to use for structure generation. Possible options: `mace-mp` or `so3lr`.",
        ),
    )
    foundational_model_settings: Union[MaceSettings, So3lrSettings, Dict[str, Any]]
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
    initial_sampling: str = Field(default="foundational")
    aims_lib_path: str | None = Field(default=None, description="Path to the compiled FHI-aims library for direct force and energy evaluation.")

    @model_validator(mode="after")
    def validate_nested_settings(self) -> "IDGSettings":
        """Ensures foundational_model_settings matches the foundational_model type."""
        model_class = MODEL_MAP.get(self.foundational_model)
        self.foundational_model_settings = model_class(**self.foundational_model_settings)
        return self

    @model_validator(mode="after")
    def check_at_least_one_required(self) -> "IDGSettings":
        """Ensures stopping criteria are set."""
        # If all these are at their 'unlimited' defaults, the simulation runs forever
        is_acc_set = self.desired_acc > 0
        is_epochs_set = self.max_initial_epochs < sys.maxsize
        is_size_set = self.max_initial_set_size < sys.maxsize

        if not any([is_acc_set, is_epochs_set, is_size_set]):
            raise ValueError(
                "You must provide a limit for at least one stopping criterion: "
                "desired_acc, max_initial_epochs, or max_initial_set_size."
            )
        return self
