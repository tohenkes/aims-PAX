from typing import Optional
import numpy as np
import torch.nn.functional
from e3nn import o3
from mace import modules, tools


#############################################################################
############ This part is mostly taken from the MACE source code ############
############ with slight modifications to fit the needs of AL    ############
#############################################################################


# this function is basically from the MACE run_train script but only
# contains the parts important for setting up the model.


def setup_mace(
    settings: dict,
    z_table: tools.AtomicNumberTable,
    atomic_energies_dict: dict,
    avg_num_neighbors: Optional[np.ndarray] = 1.0,
    atomic_inter_scale: Optional[float] = 0.0,
    atomic_inter_shift: Optional[float] = 0.0,
):
    """
    Setup the MACE model according to the settings and return it.

    Args:
        settings (dict): MACE model settings
        z_table (tools.AtomicNumberTable): Table of atomic numbers
        atomic_energies_dict (dict): Dictionary of atomic energies
        avg_num_neighbors (Optional[np.ndarray], optional): Average number of neighbors. Defaults to 1..
        atomic_inter_scale (Optional[float], optional): Scaling factor for energies and forces. Defaults to 0..
        atomic_inter_shift (Optional[float], optional): Shift for energies and forces. Defaults to 0..

    Returns:
        MACE model
    """

    general_settings = settings["GENERAL"]
    architecture_settings = settings["ARCHITECTURE"]
    misc_settings = settings["MISC"]

    tools.set_default_dtype(general_settings["default_dtype"])

    atomic_energies: np.ndarray = np.array(
        [atomic_energies_dict[z] for z in z_table.zs]
    )

    if (
        architecture_settings["num_channels"] is not None
        and architecture_settings["max_L"] is not None
    ):
        assert (
            architecture_settings["num_channels"] > 0
        ), "num_channels must be positive integer"
        assert (
            architecture_settings["max_L"] >= 0
            and architecture_settings["max_L"] < 4
        ), "max_L must be between 0 and 3, if you want to use larger specify"
        "it via the hidden_irrpes keyword"

        hidden_irreps = f"{architecture_settings['num_channels']:d}x0e"
        if architecture_settings["max_L"] > 0:
            hidden_irreps += f" + {architecture_settings['num_channels']:d}x1o"
        if architecture_settings["max_L"] > 1:
            hidden_irreps += f" + {architecture_settings['num_channels']:d}x2e"
        if architecture_settings["max_L"] > 2:
            hidden_irreps += f" + {architecture_settings['num_channels']:d}x3o"

    model_config = dict(
        r_max=architecture_settings["r_max"],
        num_bessel=architecture_settings["num_radial_basis"],
        num_polynomial_cutoff=architecture_settings["num_cutoff_basis"],
        max_ell=architecture_settings["max_ell"],
        interaction_cls=modules.interaction_classes[
            architecture_settings["interaction"]
        ],
        num_interactions=architecture_settings["num_interactions"],
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=z_table.zs,
        radial_type=architecture_settings["radial_type"],
        radial_MLP=architecture_settings["radial_MLP"],
    )

    model: torch.nn.Module

    model = modules.ScaleShiftMACE(
        **model_config,
        correlation=architecture_settings["correlation"],
        gate=modules.gate_dict[architecture_settings["gate"]],
        interaction_cls_first=modules.interaction_classes[
            architecture_settings["interaction_first"]
        ],
        MLP_irreps=o3.Irreps(architecture_settings["MLP_irreps"]),
        atomic_inter_scale=atomic_inter_scale,
        atomic_inter_shift=atomic_inter_shift,
    )
    model.to(misc_settings["device"])

    return model
