from typing import Optional
import numpy as np
from so3krates_torch.modules.models import So3krates, SO3LR
from mace import tools
from mace.tools import AtomicNumberTable
import torch


def create_base_so3krates_settings(
    settings: dict,
    z_table: tools.AtomicNumberTable,
    atomic_energies_dict: dict,
    avg_num_neighbors: Optional[np.ndarray] = 1.0
):
    general_settings = settings["GENERAL"]
    architecture_settings = settings["ARCHITECTURE"]
    misc_settings = settings["MISC"]

    tools.set_default_dtype(general_settings["default_dtype"])
    tools.set_seeds(general_settings["seed"])

    atomic_energies: np.ndarray = np.array(
        [atomic_energies_dict[z] for z in z_table.zs]
    )

    model_config = dict(
        r_max=architecture_settings["r_max"],
        features_dim=architecture_settings["features_dim"],
        num_radial_basis=architecture_settings["num_radial_basis"],
        degrees=architecture_settings["degrees"],
        num_att_heads=architecture_settings["num_att_heads"],
        num_interactions=architecture_settings["num_interactions"],
        num_elements=len(z_table),
        avg_num_neighbors=avg_num_neighbors,
        final_mlp_layers=architecture_settings["final_mlp_layers"],
        energy_regression_dim=architecture_settings["energy_regression_dim"],
        message_normalization=architecture_settings["message_normalization"],
        initialize_ev_to_zeros=architecture_settings["initialize_ev_to_zeros"],
        radial_basis_fn=architecture_settings["radial_basis_fn"],
        trainable_rbf=architecture_settings["trainable_rbf"],
        atomic_type_shifts=atomic_energies,
        learn_atomic_type_shifts=misc_settings["learn_atomic_type_shifts"],
        learn_atomic_type_scales=misc_settings["learn_atomic_type_scales"],
        layer_normalization_1=architecture_settings["layer_normalization_1"],
        layer_normalization_2=architecture_settings["layer_normalization_2"],
        residual_mlp_1=architecture_settings["residual_mlp_1"],
        residual_mlp_2=architecture_settings["residual_mlp_2"],
        use_charge_embed=architecture_settings["use_charge_embed"],
        use_spin_embed=architecture_settings["use_spin_embed"],
        interaction_bias=architecture_settings["interaction_bias"],
        qk_non_linearity=architecture_settings["qk_non_linearity"],
        cutoff_fn=architecture_settings["cutoff_fn"],
        cutoff_p=architecture_settings["cutoff_p"],
        activation_fn=architecture_settings["activation_fn"],
        energy_activation_fn=architecture_settings["energy_activation_fn"],
        device=torch.device(misc_settings["device"]),
        dtype=general_settings["default_dtype"],
        layers_behave_like_identity_fn_at_init=architecture_settings["layers_behave_like_identity_fn_at_init"],
        output_is_zero_at_init=architecture_settings["output_is_zero_at_init"],
        input_convention=architecture_settings["input_convention"],
    )

    return model_config


def setup_so3krates(
    settings: dict,
    atomic_energies_dict: dict,
    z_table: Optional[tools.AtomicNumberTable] = None,
    avg_num_neighbors: Optional[np.ndarray] = 1.0
) -> So3krates:
    """
    Setup the So3krates model according to the settings and return it.

    Args:
        settings (dict): Model settings
        z_table (tools.AtomicNumberTable): Table of atomic numbers
        atomic_energies_dict (dict): Dictionary of atomic energies
        avg_num_neighbors (Optional[np.ndarray], optional): Average number of neighbors. Defaults to 1.0.

    Returns:
        So3krates: So3krates model
    """
    if z_table is None:
        z_table = AtomicNumberTable([int(z) for z in range(1, 119)])

    model_config = create_base_so3krates_settings(
        settings,
        z_table,
        atomic_energies_dict,
        avg_num_neighbors
    )

    model = So3krates(**model_config).to(settings["MISC"]["device"])

    return model


def setup_so3lr(
    settings: dict,
    z_table: tools.AtomicNumberTable,
    atomic_energies_dict: dict,
    avg_num_neighbors: Optional[np.ndarray] = 1.0
) -> SO3LR:
    """
    Setup the SO3LR model according to the settings and return it.

    Args:
        settings (dict): Model settings
        z_table (tools.AtomicNumberTable): Table of atomic numbers
        atomic_energies_dict (dict): Dictionary of atomic energies
        avg_num_neighbors (Optional[np.ndarray], optional): Average number of neighbors. Defaults to 1.0.

    Returns:
        SO3LR: SO3LR model
    """

    model_config = create_base_so3krates_settings(
        settings,
        z_table,
        atomic_energies_dict,
        avg_num_neighbors
    )

    general_settings = settings["GENERAL"]
    architecture_settings = settings["ARCHITECTURE"]

    tools.set_default_dtype(general_settings["default_dtype"])
    tools.set_seeds(general_settings["seed"])

    so3lr_settings = dict(
        zbl_repulsion_bool=architecture_settings["zbl_repulsion_bool"],
        electrostatic_energy_bool=architecture_settings["electrostatic_energy_bool"],
        electrostatic_energy_scale=architecture_settings["electrostatic_energy_scale"],
        dispersion_energy_bool=architecture_settings["dispersion_energy_bool"],
        dispersion_energy_scale=architecture_settings["dispersion_energy_scale"],
        dispersion_damping_bool=architecture_settings["dispersion_damping_bool"],
        r_max_lr=architecture_settings["r_max_lr"],
        neighborlist_format=architecture_settings["neighborlist_format"],
    )

    total_settings = {**model_config, **so3lr_settings}
    model = SO3LR(**total_settings).to(settings["MISC"]["device"])
    return model
