from typing import Optional
import numpy as np
from so3krates_torch.modules.models import So3krates, SO3LR, MultiHeadSO3LR
from mace import tools
from mace.tools import AtomicNumberTable
import torch

from aims_PAX.settings import ModelSettings


def create_base_so3_settings(
    settings: ModelSettings,
    z_table: tools.AtomicNumberTable,
    atomic_energies_dict: dict,
    avg_num_neighbors: Optional[np.ndarray] = 1.0
):
    """Create settings for so3krates or so3lr model"""
    tools.set_default_dtype(settings.GENERAL.default_dtype)
    tools.set_seeds(settings.GENERAL.seed)

    model_config = settings.ARCHITECTURE.model_dump()
    model_config.pop("model")
    model_config.update({
        "num_elements": len(z_table),
        "avg_num_neighbors": avg_num_neighbors,
        "atomic_type_shifts": atomic_energies_dict,
        "device": torch.device(settings.MISC.device),
        "dtype": settings.GENERAL.default_dtype,
    })
    return model_config


def setup_so3krates(
    settings: ModelSettings,
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

    model_config = create_base_so3_settings(
        settings,
        z_table,
        atomic_energies_dict,
        avg_num_neighbors
    )

    model = So3krates(**model_config).to(settings.MISC.device)

    return model

def setup_so3lr(
    settings: ModelSettings,
    z_table: tools.AtomicNumberTable,
    atomic_energies_dict: dict,
    avg_num_neighbors: Optional[np.ndarray] = 1.0
) -> SO3LR:
    """
    Setup the SO3LR model according to the settings and return it.

    Args:
        settings (ModelSettings): Model settings
        z_table (tools.AtomicNumberTable): Table of atomic numbers
        atomic_energies_dict (dict): Dictionary of atomic energies
        avg_num_neighbors (Optional[np.ndarray], optional): Average number of neighbors. Defaults to 1.0.

    Returns:
        SO3LR: SO3LR model
    """

    model_config = create_base_so3_settings(
        settings,
        z_table,
        atomic_energies_dict,
        avg_num_neighbors
    )
    
    model = SO3LR(**model_config).to(settings.MISC.device)
    return model

def setup_multihead_so3lr(
    settings: ModelSettings,
    z_table: tools.AtomicNumberTable,
    atomic_energies_dict: dict,
    avg_num_neighbors: Optional[np.ndarray] = 1.0
) -> MultiHeadSO3LR:
    """
    Setup the MultiHeadSO3LR model according to the settings and return it.

    Args:
        settings (ModelSettings): Model settings
        z_table (tools.AtomicNumberTable): Table of atomic numbers
        atomic_energies_dict (dict): Dictionary of atomic energies
        avg_num_neighbors (Optional[np.ndarray], optional): Average number of neighbors. Defaults to 1.0.

    Returns:
        MultiHeadSO3LR: MultiHeadSO3LR model
    """
    
    model_config = create_base_so3_settings(
        settings,
        z_table,
        atomic_energies_dict,
        avg_num_neighbors
    )
    model_config[
        "num_output_heads"
    ] = settings.ARCHITECTURE.num_multihead_heads
    
    model = MultiHeadSO3LR(**model_config).to(settings.MISC.device)
    model.select_heads = True
    return model