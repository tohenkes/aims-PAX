import logging
import random
import re
import sys
import os

import torch
import numpy as np
from typing import Optional, Any, Dict, Union
from pathlib import Path
from mace import tools, modules
from mace.cli.convert_e3nn_cueq import run as convert_e3nn_cueq
from mace.cli.convert_cueq_e3nn import run as convert_cueq_e3nn
from mace.tools import AtomicNumberTable
from so3krates_torch.modules.models import So3krates, SO3LR
from so3krates_torch.tools.multihead_utils import reduce_mh_model_to_sh
from so3krates_torch.tools.finetune import setup_finetuning

from aims_PAX.settings import ModelSettings
from aims_PAX.settings.model import BaseArchitectureSettings
from aims_PAX.settings.project import MDSettings
from aims_PAX.tools.model_tools.setup_MACE import setup_mace
from aims_PAX.tools.model_tools.setup_so3 import setup_so3krates, setup_so3lr, setup_multihead_so3lr
from aims_PAX.tools.model_tools.training_tools import setup_model_training
import ase.data
from ase.io import read
from ase import units
from contextlib import nullcontext
import time
from threading import Thread
import pandas as pd
import yaml
from copy import deepcopy
from mace.data.utils import (
    KeySpecification
)
from ase.io import write


# Many functions are taken from or inspired by the MACE code:
# https://github.com/ACEsuit/mace


Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]
Stress = np.ndarray  # [6, ]
Virials = np.ndarray  # [3,3]
Charges = np.ndarray  # [..., 1]
Cell = np.ndarray  # [3,3]
Pbc = tuple  # (3,)

DEFAULT_CONFIG_TYPE = "Default"
DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}

#TODO: Refactor all model setup to a separate file

def mh_to_sh_model(
    mh_model_state_dict,
    settings: dict,
    z_table: tools.AtomicNumberTable,
    atomic_energies_dict: dict,
    avg_num_neighbors: float,
    head_idx: int,
    model_choice: str = "so3lr",
    device: str = "cpu",
    dtype: str = "float32"
):
    settings["num_elements"] = len(z_table)
    settings["avg_num_neighbors"] = avg_num_neighbors
    settings["atomic_type_shifts"] = atomic_energies_dict
    
    sh_model = reduce_mh_model_to_sh(
        mh_model_state_dict,
        settings,
        head_idx,
        model_choice,
        device,
        dtype
    )
    return sh_model


def get_free_vols(lines):
    active = False
    freevol = []
    for line in lines:
       if not active:
          if "Performing Hirshfeld analysis of fragment charges and moments" in line:
                active = True
       else:
          if "Free atom volume" in line:
             fav = float(re.findall("\-?\d+(?:\.\d+)", line)[0])
             freevol.append(fav)
    return freevol


def get_hirshfeld_charges(lines):
    active = False
    charges = []
    for line in lines:
        if not active:
            if "Performing Hirshfeld analysis of fragment charges and moments" in line:
                active = True
        else:
            if "Hirshfeld charge" in line:
                charges.append(
                    float(re.findall(r"\-?\d+(?:\.\d+)", line)[0])
                )
    return charges


def apply_model_settings(
    target,
    model_settings: ModelSettings,
) -> None:

    target.model_settings = model_settings
    model_choice = model_settings.GENERAL.model_choice.lower()
    attrs = {
        "model_settings": model_settings,
        "model_choice": model_choice,
        "seed": model_settings.GENERAL.seed,
        "checkpoints_dir": model_settings.GENERAL.checkpoints_dir,
        "model_dir": model_settings.GENERAL.model_dir,
        "dtype": model_settings.GENERAL.default_dtype,
        "r_max": model_settings.ARCHITECTURE.r_max,
        "r_max_lr": model_settings.ARCHITECTURE.r_max_lr if model_choice == "so3lr" else None,
        "dispersion_energy_cutoff_lr_damping": (model_settings.ARCHITECTURE.dispersion_energy_cutoff_lr_damping
                                                if model_choice == "so3lr" else None),
        "atomic_energies_dict": None if model_choice != "mace" else model_settings.ARCHITECTURE.atomic_energies,
        "scaling": None if model_choice != "mace" else model_settings.ARCHITECTURE.scaling,
        "set_batch_size": model_settings.TRAINING.batch_size,
        "set_valid_batch_size": model_settings.TRAINING.valid_batch_size,
        "update_avg_num_neighbors": model_settings.TRAINING.update_avg_num_neighbors,
        "device": model_settings.MISC.device,
        "compute_stress": model_settings.MISC.compute_stress,
        "compute_dipole": model_settings.MISC.compute_dipole,
        "properties": ["energy", "forces"],
        "enable_cueq": model_settings.MISC.enable_cueq,
        "enable_cueq_train": model_settings.MISC.enable_cueq_train,
        "use_multihead_model": False if model_choice != "so3lr" else model_settings.ARCHITECTURE.use_multihead_model,
        "num_multihead_heads": 0 if model_choice != "so3lr" else model_settings.ARCHITECTURE.num_multihead_heads,
    }

    if model_settings.MISC.compute_stress:
        attrs["properties"].append("stress")
    if model_settings.MISC.compute_dipole:
        attrs["properties"].append("dipole")

    # Setting actual attributes
    for k, v in attrs.items():
        target.__setattr__(k, v)

    torch.set_default_dtype(getattr(torch, target.dtype))

    # Multihead attributes
    target.use_pretrained_model = (
            isinstance(model_settings.TRAINING.pretrained_model, str) or
            isinstance(model_settings.TRAINING.pretrained_weights, str)
        )
    if target.use_pretrained_model and target.update_avg_num_neighbors:
        logging.warning(
            "Using a pretrained model/weights with "
            "update_avg_num_neighbors=True is not recommended."
            "This can lead to high errors in the beginning of training!"
        )
    if target.use_multihead_model and target.num_multihead_heads is not None:
        target.all_heads = [
            f"head_{i}" for i in range(target.num_multihead_heads) 
        ]
    else:
        target.all_heads = None


def is_multi_trajectory_md(
    md_settings: MDSettings
) -> bool:
    """Check if MD settings use multi-trajectory format."""
    if not md_settings:
        return False

    return isinstance(md_settings.root, dict)


def normalize_md_settings(
    md_settings: MDSettings,
    num_trajectories: int
) -> tuple[dict, bool]:
    """Normalize MD settings to multi-trajectory format."""
    return ({i: md_settings.get_for_index(i).model_dump() for i in range(num_trajectories)},
            is_multi_trajectory_md(md_settings))


def create_keyspec(
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: str = "REF_stress",
    dipole_key: str = "REF_dipole",
    polarizability_key: str = "REF_polarizability",
    head_key: str = "head",
    charges_key: str = "REF_charges",
    hirshfeld_ratios_key: str = "REF_hirshfeld_ratios",
    total_charge_key: str = "total_charge",
    total_spin_key: str = "total_spin"
) -> KeySpecification:

    arrays_keys = {
        "forces": forces_key,
        "charges": charges_key,
        "hirshfeld_ratios": hirshfeld_ratios_key,
    }
    
    info_keys = {
        "energy": energy_key,
        "stress": stress_key,
        "dipole": dipole_key,
        "polarizability": polarizability_key,
        "head": head_key,
        "total_charge": total_charge_key,
        "total_spin": total_spin_key
    }
    
    return KeySpecification(
        info_keys=info_keys,
        arrays_keys=arrays_keys
    )


def compute_average_E0s(
    energies_train, zs_train, z_table: AtomicNumberTable
) -> Dict[int, float]:
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s
    """
    len_train = len(energies_train)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.zeros(len_train)
    for i in range(len_train):
        B[i] = energies_train[i]
        for j, z in enumerate(z_table.zs):
            A[i, j] = np.count_nonzero(np.array(zs_train[i]) == z)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = E0s[i]
    except np.linalg.LinAlgError:
        logging.warning(
            "Failed to compute E0s using least squares regression, using the same for all atoms"
        )
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = 0.0
    return atomic_energies_dict


def ensemble_from_folder(
        path_to_models: str,
        device: str,
        dtype,
        convert_to_cueq: bool = False
            ) -> list:
    """
    Load an ensemble of models from a folder.
    (Can't handle other file formats than .pt at the moment.)

    Args:
        path_to_models (str): Path to the folder containing the models.
        device (str): Device to load the models on.
        dtype: Data type to load the models with.
        convert_to_cueq (bool, optional): Whether to convert the models to CuEQ format. Defaults to False.
    Returns:
        list: List of models.
    """

    assert os.path.exists(path_to_models)
    assert os.listdir(Path(path_to_models))

    ensemble = {}
    for filename in os.listdir(path_to_models):
        # check if filename is for model
        if filename.endswith(".model"):
            if os.path.isfile(os.path.join(path_to_models, filename)):
                complete_path = os.path.join(path_to_models, filename)
                model = torch.load(
                    complete_path, map_location=device, weights_only=False
                ).to(dtype)
                
                if convert_to_cueq:
                    model = convert_e3nn_cueq(
                        input_model=model,
                        device=device,
                        return_model=True
                    )

                filename_without_suffix = os.path.splitext(filename)[0]
                ensemble[filename_without_suffix] = model
    return ensemble


def pre_trajectories_from_folder(
    path: str,
    num_trajectories: int,
) -> list:
    """
    Load pre-existing trajectories from a folder.
    ASE readable formats are supported.

    Args:
        path (str): Path to the folder containing the trajectories.
        num_trajectories (int): Number of trajectories to load.

    Returns:
        list: List of ASE atoms objects.
    """
    trajectories = []
    for i, filename in enumerate(os.listdir(path)):
        if os.path.isfile(os.path.join(path, filename)):
            complete_path = os.path.join(path, filename)
            trajectory = read(complete_path, index=":")
            trajectories.append(trajectory)
        if i == num_trajectories:
            break
    return trajectories


def compute_max_error(delta: np.ndarray) -> float:
    return np.max(np.abs(delta)).item()


def select_best_member(
    ensemble_valid_loss: dict,
):
    """
    Selects the best member of an ensemble based on the validation loss.

    Args:
        ensemble_valid_loss (dict): Dictionary of validation losses for
                                    each ensemble member.

    Returns:
        str: Tag of the best ensemble member.
    """
    return min(ensemble_valid_loss, key=ensemble_valid_loss.get)


def get_ensemble_training_setups(
    ensemble: dict,
    model_settings: ModelSettings,
    checkpoints_dir: str = None,
    restart: bool = False,
    mol_idxs: Optional[np.ndarray] = None,
) -> dict:
    """
    Creates a dictionary of training setups for each model in the ensemble.

    Args:
        ensemble (dict): Dictionary of models in the ensemble.
        model_settings (ModelSettings): Model settings dictionary containing
                              the experiment name and all model and training
                              settings.
        checkpoints_dir (str, optional): Path to the folder where
                        the training checkpoints are saved. Defaults to None.
        restart (bool, optional): Wether this function is called during
                                        an aims PAX restart. Defaults to False.
        al_settings (dict, optional): The active learning settings,
                    only important if intermol_uncertainty is used as
                    the molecular indices are stored there. Defaults to None.

    Returns:
        dict: Dictionary of training setups for each model in the ensemble.
    """
    
    model_choice = model_settings.GENERAL.model_choice.lower()
    training_setups = {}
    for tag, model in ensemble.items():
        training_setups[tag] = setup_model_training(
            settings=model_settings,
            model=model,
            model_choice=model_choice,
            tag=tag,
            restart=restart,
            checkpoints_dir=checkpoints_dir,
            mol_idxs=mol_idxs,
        )
    return training_setups


def create_seeds_tags_dict(
    seeds: np.ndarray,
    model_settings: ModelSettings,
    dataset_dir,
    save_seeds_tags_dict: str = "seeds_tags_dict.npz",
) -> dict:
    """
    Creates a dict where the keys (tags) are the names of the model
    models in the ensemble and the values are the respective seeds.

    Args:
        seeds (np.array): Array of seeds for the ensemble.
        model_settings (ModelSettings): Model settings dictionary containing
                              the experiment name.
        dataset_dir: Path to the dataset directory where the dict is saved.
        save_seeds_tags_dict (str, optional): Name of the resulting dict.
                                Defaults to "seeds_tags_dict.npz".

    Returns:
        dict: _description_
    """
    seeds_tags_dict = {}
    for seed in seeds:
        tag = model_settings.GENERAL.name_exp + "-" + str(seed)
        seeds_tags_dict[tag] = seed
    if save_seeds_tags_dict:
        np.savez(
            dataset_dir / save_seeds_tags_dict,
            **seeds_tags_dict,
        )
    return seeds_tags_dict


def create_ztable(
    zs: np.ndarray,
):
    z_table = tools.get_atomic_number_table_from_zs(z for z in zs)
    return z_table


def setup_ensemble_dicts(
    seeds_tags_dict: dict[str, int],
    z_table: tools.AtomicNumberTable,
    model_settings: ModelSettings,
    ensemble_atomic_energies_dict: dict[str, dict[int, int]],
    num_elements: int = 118,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Creates dictionaries for the ensemble members i.e. a dictionary of models.
    Also, creates a dictionary for the training setups for each model and
    a dictionary for the seeds and their corresponding tags. All of these
    are returned as a tuple.

    Args:
        seeds_tags_dict (dict[str, int]): Tags with corresponding seeds.
        z_table (tools.AtomicNumberTable): Atomic numbers of elements in the model
        model_settings (ModelSettings): Model settings
        ensemble_atomic_energies_dict (dict[str, dict[int, int]]): Dictionary of atomic energies
        for each tag


    Returns:
        A dict connecting tag to ensemble.
    """

    ensemble = {}
    for tag, seed in seeds_tags_dict.items():
        model_settings.GENERAL.seed = seed
        tag = model_settings.GENERAL.name_exp + "-" + str(seed)
        seeds_tags_dict[tag] = seed
        
        model_choice = model_settings.GENERAL.model_choice.lower()
        if model_choice == "mace":
            ensemble[tag] = setup_mace(
                settings=model_settings,
                z_table=z_table,
                atomic_energies_dict=ensemble_atomic_energies_dict[tag],
            )
        elif model_choice == "so3krates":
            ensemble[tag] = setup_so3krates(
                settings=model_settings,
                z_table=z_table,
                atomic_energies_dict=ensemble_atomic_energies_dict[tag],
            )
        elif model_choice == "so3lr":
            # TODO: Move the check to where it belongs
            if model_settings.ARCHITECTURE.use_multihead_model:
                assert model_settings.ARCHITECTURE.num_multihead_heads is not None, (
                    "num_multihead_heads must be specified when using "
                    "a multihead SO3LR model."
                )
                assert len(seeds_tags_dict) == 1, (
                    "When using a multihead model ('use_multihead_model: True'), "
                    "'ensemble_size' must be 1."
                )

                pretrained_model = setup_pretrained(
                    model_settings=model_settings,
                    z_table=z_table,
                    atomic_energies_dict=ensemble_atomic_energies_dict[tag],
                    num_elements=num_elements
                )
                if pretrained_model is not None:
                    ensemble[tag] = pretrained_model
                    
                else:
                    ensemble[tag] = setup_multihead_so3lr(
                        settings=model_settings,
                        z_table=z_table,
                        atomic_energies_dict=ensemble_atomic_energies_dict[tag],
                    )
            else:
                ensemble[tag] = setup_so3lr(
                    settings=model_settings,
                    z_table=z_table,
                    atomic_energies_dict=ensemble_atomic_energies_dict[tag],
                )
    return ensemble


def setup_pretrained(
    model_settings: ModelSettings,
    z_table: tools.AtomicNumberTable,
    atomic_energies_dict: dict,
    num_elements: int = 118
):
    pretrained_model = model_settings.TRAINING.pretrained_model
    pretrained_weights = model_settings.TRAINING.pretrained_weights
    num_output_heads = model_settings.ARCHITECTURE.num_multihead_heads
    device = model_settings.MISC.device
    dtype = model_settings.GENERAL.default_dtype
    
    if not pretrained_model and not pretrained_weights:
        return None
    
    if pretrained_model or pretrained_weights:
        assert model_settings.ARCHITECTURE.use_multihead_model, (
            "Pretrained model or weights can only be used with multihead SO3LR."
        )
        logging.info("Using pretrained model or weights for multihead SO3LR.")
        #TODO: Find a better way to do this. Extract settings from model (weights)?
        logging.warning(
            "Make sure that the architecture settings corresponding "
            "to the pretrained model have been set correctly in the config file."
            " Otherwise, non-matching hyperparameters will be saved!"
        )
        
    model = None
    if pretrained_model:
        model = torch.load(
            pretrained_model,
            map_location=device,
            weights_only=False
        )
        model.device = device
        model.atomic_energy_output_block.device = device
        model.to(dtype_mapping[dtype])
        model.select_heads = True
        
        assert num_output_heads == model.num_output_heads, (
            "Number of output heads in the pretrained model does not match "
            "the number of output heads specified in the settings."
        )
    elif pretrained_weights:
        weights = torch.load(
            pretrained_weights,
            map_location=device,
            weights_only=True
        )
        
        assert num_output_heads == weights[
            "atomic_energy_output_block.num_output_heads"
        ], (
            "Number of output heads in the pretrained weights does not match "
            "the number of output heads specified in the settings."
        )
        
        model = setup_multihead_so3lr(
            settings=model_settings,
            z_table=z_table,
            atomic_energies_dict=atomic_energies_dict,
        )
        
        model.load_state_dict(weights)
    
    if model_settings.TRAINING.perform_finetuning:
        model = apply_finetuning_settings(
            model=model,
            model_settings=model_settings,
            num_elements=num_elements
        )

    if model_settings.GENERAL.model_choice == "so3lr":
        model.r_max_lr = model_settings.ARCHITECTURE.r_max_lr
        logging.info(
            f"Set r_max_lr to {model.r_max_lr} for pretrained SO3LR model."
        )
    return model


def apply_finetuning_settings(
    model: torch.nn.Module,
    model_settings: ModelSettings,
    num_elements: int = 118
) -> torch.nn.Module:
    
    logging.info("Applying fine-tuning settings to pretrained model.")

    finetune_choice = model_settings.TRAINING.finetuning_choice
    device_name = model_settings.MISC.device
    freeze_embedding = model_settings.TRAINING.freeze_embedding
    freeze_zbl = model_settings.TRAINING.freeze_zbl
    freeze_hirshfeld = model_settings.TRAINING.freeze_hirshfeld
    freeze_partial_charges = model_settings.TRAINING.freeze_partial_charges
    freeze_shifts = model_settings.TRAINING.freeze_shifts
    freeze_scales = model_settings.TRAINING.freeze_scales
    convert_to_lora = model_settings.TRAINING.convert_to_lora
    lora_rank = model_settings.TRAINING.lora_rank
    lora_alpha = model_settings.TRAINING.lora_alpha
    lora_freeze_A = model_settings.TRAINING.lora_freeze_A
    
    dora_scaling_to_one = True # TODO: remove dora
    
    convert_to_multihead = model_settings.TRAINING.convert_to_multihead
    seed = model_settings.GENERAL.seed
    
    return setup_finetuning(
        model=model,
        finetune_choice=finetune_choice,
        device_name=device_name,
        num_elements=num_elements,
        freeze_embedding=freeze_embedding,
        freeze_zbl=freeze_zbl,
        freeze_hirshfeld=freeze_hirshfeld,
        freeze_partial_charges=freeze_partial_charges,
        freeze_shifts=freeze_shifts,
        freeze_scales=freeze_scales,
        convert_to_lora=convert_to_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_freeze_A=lora_freeze_A,
        dora_scaling_to_one=dora_scaling_to_one,
        convert_to_multihead=convert_to_multihead,
        architecture_settings=model_settings.ARCHITECTURE.model_dump(),
        seed=seed,
        log=True
    )
    

def get_atomic_energies_from_ensemble(
    ensemble: dict,
    z,
    model_choice: str,
    dtype: str,
):
    """
    Loads the atomic energies from existing ensemble members.

    Args:
        ensemble (dict): Dictionary of model models.
        z (np.array): Array of atomic numbers for which the atomic energies
                        are needed.
        dtype (str): Data type for the atomic energies.

    """
    ensemble_atomic_energies_dict = {}
    ensemble_atomic_energies = {}
    for tag, model in ensemble.items():
        if model_choice == "mace":
            ensemble_atomic_energies[tag] = np.array(
                model.atomic_energies_fn.atomic_energies.cpu(), dtype=dtype
            )
        else:
            energy_shifts_so3 = model.atomic_energy_output_block.energy_shifts
            if energy_shifts_so3.requires_grad:
                energy_shifts_so3 = energy_shifts_so3.detach()
            ensemble_atomic_energies[tag] = np.array(
                energy_shifts_so3.cpu(),
                dtype=dtype
            )
        ensemble_atomic_energies_dict[tag] = {}
        for i, atomic_energy in enumerate(ensemble_atomic_energies[tag]):
            # TH: i don't know if the atomic energies are really sorted
            # by atomic number inside the models. TODO: check that
            ensemble_atomic_energies_dict[tag][
                np.sort(np.unique(z))[i]
            ] = atomic_energy

    return (ensemble_atomic_energies, ensemble_atomic_energies_dict)


def get_atomic_energies_from_pt(
    path_to_checkpoints: Path,
    z: np.ndarray,
    seeds_tags_dict: dict,
    dtype: str,
    model_choice: str
) -> tuple:
    """
    Loads the atomic energies (energy shifts) from an existing model
    training checkpoint. Returns a list and dictionary containing the
    atomic energies for each ensemble member.

    Args:
        path_to_checkpoints (str): Path to the directory containing the
                                    checkpoints.
        z (np.array): Array of atomic numbers for which the atomic energies
                                        are needed.
        seeds_tags_dict (dict): Dictionary mapping model tags to seeds.
        dtype (str): Data type for the atomic energies.

    Returns:
        tuple: list and dictionary containing the atomic energies for
                    each ensemble member.
    """

    ensemble_atomic_energies_dict = {}
    ensemble_atomic_energies = {}
    last_check_pt = list_latest_file(path_to_checkpoints)
    last_epoch = int(last_check_pt.split(".")[0].split("-")[-1])
    for tag in seeds_tags_dict.keys():
        check_pt = torch.load(
            path_to_checkpoints / (tag + f"_epoch-{last_epoch}.pt")
        )
        
        if model_choice == "mace":
            atomic_energies_array = check_pt["model"][
                "atomic_energies_fn.atomic_energies"
            ]
        elif model_choice in ["so3krates", "so3lr"]:
            atomic_energies_array = check_pt["model"][
                "atomic_energy_output_block.energy_shifts"
            ]
            
        ensemble_atomic_energies[tag] = np.array(
            atomic_energies_array.cpu(), dtype=dtype
        )
        ensemble_atomic_energies_dict[tag] = {}
        for i, atomic_energy in enumerate(ensemble_atomic_energies[tag]):
            # TH: i don't know if the atomic energies are really sorted by
            # atomic number inside the models. TODO: check that
            ensemble_atomic_energies_dict[tag][
                np.sort(np.unique(z))[i]
            ] = atomic_energy

    return (ensemble_atomic_energies, ensemble_atomic_energies_dict)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


def compute_avg_num_neighbors(data_loader: torch.utils.data.DataLoader) -> float:
    num_neighbors = []
    for batch in data_loader:
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item()


def update_model_auxiliaries(
    model: Union[modules.MACE, So3krates, SO3LR],
    model_choice: str,
    model_sets: dict,
    scaling: str,
    atomic_energies_list: list,
    update_avg_num_neighbors: bool = True,
    update_atomic_energies: bool = False,
    atomic_energies_dict: dict = None,
    z_table: tools.AtomicNumberTable = None,
    dtype: str = "float64",
    device: str = "cpu",
):
    """
    Update the average number of neighbors, scale and shift of
    the model with the given data

    Args:
        model (Union[modules.MACE, So3krates, SO3LR]): The model to be updated.
        model_sets (dict): Dictionary containing the training and validation
                            sets.
        scaling (str): Scaling method to be used.
        atomic_energies_list (list): List of atomic energies.
        atomic_energies_dict (dict): Dictionary of atomic energies.
        update_atomic_energies (bool): Whether to update the atomic energies.
        z_table (tools.AtomicNumberTable): Table of elements.
        dtype (str): Dtype of the model.
        device (str): Device to be used for the model.

    Returns:
         None
    """
    train_loader = model_sets["train_loader"]
    assert z_table is not None
    assert atomic_energies_dict is not None

    average_neighbors = compute_avg_num_neighbors(train_loader)
    
    energies_train = torch.stack(
        [point.energy.reshape(1) for point in model_sets["train"]]
    )
    zs_train = [
        [
            z_table.index_to_z(torch.nonzero(one_hot).item())
            for one_hot in point.node_attrs
        ]
        for point in model_sets["train"]
    ]
    new_atomic_energies_dict = compute_average_E0s(
        energies_train, zs_train, z_table
    )
    atomic_energies_dict.update(new_atomic_energies_dict)
    
    if model_choice == "mace":
        if update_atomic_energies:
            update_energy_shifts_mace(
                model,
                atomic_energies_dict,
                device,
                dtype,
            )
        mean, std = modules.scaling_classes[scaling](
            train_loader, atomic_energies_list
        )
        mean, std = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(
            device
        )
        if update_avg_num_neighbors:
            update_avg_num_neighbors_mace(
                model,
                average_neighbors,
            )
        
    elif model_choice in ["so3krates", "so3lr"]:
        if update_avg_num_neighbors:
            update_avg_num_neighbors_so3(
                model,
                average_neighbors, 
            )
        if update_atomic_energies:
            update_energy_shifts_so3(
                model,
                atomic_energies_dict,
            )
    return average_neighbors, atomic_energies_dict


def update_energy_shifts_mace(
    model: modules.MACE,
    atomic_energies_dict: dict,
    device: str,
    dtype: str = "float64",
):
    atomic_energies_list = [
        atomic_energies_dict[z] for z in atomic_energies_dict.keys()
    ]
    atomic_energies_tensor = torch.tensor(
        atomic_energies_list, dtype=dtype_mapping[dtype]
    )

    model.atomic_energies_fn.atomic_energies = atomic_energies_tensor.to(
        device
    )


def update_avg_num_neighbors_mace(
    model: modules.MACE,
    avg_num_neighbors: float,
):
    for interaction in model.interactions:
        interaction.avg_num_neighbors = avg_num_neighbors


def update_energy_shifts_so3(
    model: SO3LR,
    atomic_energies_dict: dict,
):
    model.atomic_energy_output_block.set_defined_energy_shifts(
        atomic_energies_dict
    )


def update_avg_num_neighbors_so3(
    model: SO3LR,
    avg_num_neighbors: float,
):
    model.avg_num_neighbors = avg_num_neighbors
    for layer in model.euclidean_transformers:
        layer.euclidean_attention_block.att_norm_inv = avg_num_neighbors
        layer.euclidean_attention_block.att_norm_ev = avg_num_neighbors


def save_checkpoint(
    checkpoint_handler: tools.CheckpointHandler,
    training_setup: dict,
    model,
    epoch: int,
    keep_last: bool = False,
):
    """
    Save a checkpoint of the training setup and model.

    Args:
        checkpoint_handler (tools.CheckpointHandler): model handler for saving
                                                        checkpoints.
        training_setup (dict): Training settings.
        model: model model to be saved.
        epoch (int): Current epoch.
        keep_last (bool, optional): Keep the last checkpoint.
                                        Defaults to False.

    Returns:
        None
    """
    if training_setup["ema"] is not None:
        with training_setup["ema"].average_parameters():
            checkpoint_handler.save(
                state=tools.CheckpointState(
                    model,
                    training_setup["optimizer"],
                    training_setup["lr_scheduler"],
                ),
                epochs=epoch,
                keep_last=True if keep_last else False,
            )
    else:
        checkpoint_handler.save(
            state=tools.CheckpointState(
                model,
                training_setup["optimizer"],
                training_setup["lr_scheduler"],
            ),
            epochs=epoch,
            keep_last=False,
        )
    return None


def save_models(
    ensemble: dict,
    training_setups: dict, 
    model_dir: Path,
    current_epoch: int,
    model_settings:  BaseArchitectureSettings,
    model_choice: str,
    save_state_dict: bool = True,
    convert_cueq_to_e3nn: bool = False
):
    for tag, model in ensemble.items():
        training_setup = training_setups[tag]

        param_context = (
            training_setup["ema"].average_parameters()
            if training_setup["ema"] is not None
            else nullcontext()
        )
        
        if convert_cueq_to_e3nn:
            model = convert_cueq_e3nn(
                input_model=model,
                device=training_setup["device"],
                return_model=True
            )

        with param_context:
            torch.save(
                model,
                Path(model_dir) / (tag + ".model"),
            )
            if save_state_dict:
                torch.save(
                    model.state_dict(),
                    Path(model_dir) / (tag + ".pth"),
                )

        save_checkpoint(
            checkpoint_handler=training_setup["checkpoint_handler"],
            training_setup=training_setup,
            model=model,
            epoch=current_epoch,
            keep_last=False,
        )
        
        # save hyperparams
        settings = model_settings.model_dump()
        for k in ("model", "atomic_energies", "use_multihead_model", "num_multihead_heads"):
            settings.pop(k, None)
        if model_choice == "mace":
            settings["avg_num_neighbors"] = model.interactions[0].avg_num_neighbors
        else:
            settings["avg_num_neighbors"] = model.avg_num_neighbors
        
        settings = {"ARCHITECTURE": settings}
        
        with open(
            Path(model_dir) / (tag + "_hyperparams.yaml"), "w"
        ) as file:
            yaml.dump(settings, file)


def Z_from_geometry(
    atoms: Union[ase.Atoms, dict[int, ase.Atoms]],
) -> np.ndarray:
    """
    Extracts the atomic numbers from an ASE Atoms object
    or a list of Atoms objects.

    Args:
        atoms (Union[ase.Atoms, List]): ASE Atoms object
                                or list of Atoms objects.

    Returns:
        np.ndarray: Array of atomic numbers.
    """
    if isinstance(atoms, ase.Atoms):
        return np.array(
            [ase.data.atomic_numbers[atom.symbol] for atom in atoms]
        )
    elif isinstance(atoms, dict):
        all_z = []
        for atom in atoms.values():
            current_z = atom.get_atomic_numbers()
            all_z.extend(current_z)
        return np.array(all_z)
    else:
        raise TypeError(
            "Input must be an ASE Atoms object or a list of Atoms."
        )


def list_files_in_directory(directory_path: str) -> list:
    """
    List all files in a directory.

    Args:
        directory_path (str): Path to the directory.

    Returns:
        list: List of file paths.
    """
    file_paths = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))

    return file_paths


def atoms_full_copy(atoms: ase.Atoms) -> ase.Atoms:
    """
    Creates a full copy of the ASE Atoms object, including all properties.

    Args:
        atoms (ase.Atoms): Atoms object to be copied.

    Returns:
        ase.Atoms: Copied ASE Atoms object with all properties.
    """
    
    atoms_copy = ase.Atoms(
        symbols=deepcopy(atoms.get_chemical_symbols()),
        positions=np.copy(atoms.get_positions()),
        cell=np.copy(atoms.get_cell()),
        pbc=np.copy(atoms.get_pbc()),
        momenta=np.copy(atoms.get_momenta()),
        info=deepcopy(atoms.info),
        masses=np.copy(atoms.get_masses()),
    )
    return atoms_copy


def list_latest_file(directory: Path) -> str:
    """
    List the latest file in a directory based on the last modified time.

    Args:
        directory (str): Path to the directory containing files.

    Raises:
        FileNotFoundError: If no files are found in the directory.

    Returns:
        str: Name of the latest file in the directory.
    """
    files = os.listdir(directory)

    latest_file = None
    latest_mtime = 0

    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            file_mtime = os.path.getmtime(file_path)
            if file_mtime > latest_mtime:
                latest_mtime = file_mtime
                latest_file = file

    if latest_file:
        return latest_file
    else:
        raise FileNotFoundError(f"No files found in {directory}!")


def save_ensemble(
    ensemble: dict, training_setups: dict, model_settings: dict
) -> None:
    """
    Save the ensemble models to disk. If EMA is used, the averaged parameters
    are saved, otherwise the model parameters are saved directly.
    The path to the models is defined in the model_settings under
    "GENERAL" -> "model_dir". The models are saved with the tag as the filename
    and the extension ".model".

    Args:
        ensemble (dict): Dictionary of models.
        training_setups (dict): Dictionary of training setups for each model.
        model_settings (dict): Dictionary of model settings, specificied by the
                                     user in the config file.
    """
    for tag, model in ensemble.items():
        param_context = (
            training_setups[tag]["ema"].average_parameters()
            if training_setups[tag]["ema"] is not None
            else nullcontext()
        )
        with param_context:
            torch.save(
                model,
                Path(model_settings.GENERAL.model_dir) / (tag + ".model"),
            )


def log_ensemble(
        seeds_tags_dict: dict[str, Any],
        trajectory: ase.Atoms,
        filename: str="md.extxyz"
):
    atoms = trajectory.copy()
    # get all the info together
    atoms.arrays["velocities"] = atoms.get_velocities()
    # get forces from the groups
    forces_comm = trajectory.calc.results["forces_comm"]
    for i, seed in enumerate(seeds_tags_dict):
        atoms.arrays[f"forces_{seeds_tags_dict[seed]}"] = forces_comm[i]

    # write one frame
    write(
        filename,
        atoms,
        format="extxyz",
        append=True,
    )


class ModifyMD:
    def __init__(self, settings: dict):

        self.settings = settings
        if self.settings["type"] == "temp":
            self.temp_step = settings["temp_step"]
            self.mod_interval = settings["mod_interval"]

    def change_temp(self, driver):
        driver.temp += units.kB * self.temp_step

    def __call__(self, driver, metric, idx=None) -> Any:
        raise NotImplementedError("This method is not implemented. ")
        if self.settings["type"] == "temp":
            if metric in self.mod_interval:
                self.change_temp(driver)
                if idx is not None:
                    logging.info(f"Modyfing trajectory {idx}.")
                logging.info(
                    f"Changed temperature by {self.temp_step} to "
                    "{round(driver.temp / units.kB,1)} K."
                )
                return True
            else:
                return False


dtype_mapping = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "int32": torch.int32,
    "int64": torch.int64,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
}


class GPUMonitor(Thread):
    def __init__(self, delay, output_file):
        super(GPUMonitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.output_file = output_file
        self.data = []  # Temporary storage for GPU utilization data
        self.start()

    def run(self):
        import GPUtil  # optional monitoring dependency
        while not self.stopped:
            # Get GPU utilization data
            gpus = GPUtil.getGPUs()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            for gpu in gpus:
                self.data.append(
                    {
                        "Time": timestamp,
                        "GPU ID": gpu.id,
                        "Utilization (%)": gpu.load * 100,
                        "Memory Utilization (%)": gpu.memoryUtil * 100,
                    }
                )

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        # Save collected data to a CSV file using pandas
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_file, index=False)


def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
    rank: Optional[int] = 0,
):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    # Clear existing handlers and filters to prevent duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.filters.clear()

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Messages from third-party libraries that are noisy and not useful
    _NOISY_MESSAGES = {"Using CPU"}

    # Add filter for rank and to suppress noisy third-party messages
    logger.addFilter(
        lambda record: rank == 0
        and not any(
            msg in record.getMessage() for msg in _NOISY_MESSAGES
        )
    )

    # Create console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if directory is not None and tag is not None:
        os.makedirs(name=directory, exist_ok=True)

        # Create file handler for non-debug logs
        main_log_path = os.path.join(directory, f"{tag}.log")
        fh_main = logging.FileHandler(main_log_path)
        fh_main.setLevel(level)
        fh_main.setFormatter(formatter)
        logger.addHandler(fh_main)

        # Create file handler for debug logs
        debug_log_path = os.path.join(directory, f"{tag}_debug.log")
        fh_debug = logging.FileHandler(debug_log_path)
        fh_debug.setLevel(logging.DEBUG)
        fh_debug.setFormatter(formatter)
        fh_debug.addFilter(lambda record: record.levelno >= logging.DEBUG)
        logger.addHandler(fh_debug)
    return logger


class _LiteralStr(str):
    pass


def _represent_literal_str(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


# Ensure SafeDumper knows how to emit our literal strings
yaml.SafeDumper.add_representer(_LiteralStr, _represent_literal_str)


def dump_yaml_for_log(
    data: dict,
    force_literal_keys=("slurm_str", "worker_str", "launch_str"),
    width: int = 10000,
) -> str:
    """
    Return a nicely formatted YAML string for logging.
    Multiline strings (or keys in force_literal_keys) are rendered with the '|' block style.
    """

    def convert(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                if isinstance(v, str) and (
                    "\n" in v or k in force_literal_keys
                ):
                    new_dict[k] = _LiteralStr(v)
                elif isinstance(v, (dict, list)):
                    new_dict[k] = convert(v)
                else:
                    new_dict[k] = v
            return new_dict
        elif isinstance(obj, list):
            new_list = []
            for item in obj:
                if isinstance(item, str) and "\n" in item:
                    new_list.append(_LiteralStr(item))
                elif isinstance(item, (dict, list)):
                    new_list.append(convert(item))
                else:
                    new_list.append(item)
            return new_list
        else:
            return obj

    # Create a deep copy and then convert it
    data_copy = deepcopy(data)
    converted_data = convert(data_copy)

    # Convert any non-serializable objects to strings
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__") or not isinstance(
            obj, (str, int, float, bool, type(None), _LiteralStr)
        ):
            # Convert complex objects to string representation
            return str(obj)
        else:
            return obj

    safe_data = make_serializable(converted_data)

    return yaml.safe_dump(
        safe_data,
        sort_keys=False,
        default_flow_style=False,
        indent=2,
        allow_unicode=True,
        width=width,
    )


def log_yaml_block(
    title: str,
    data: dict,
    logger: logging.Logger = logging.getLogger(__name__),
    level: int = logging.INFO,
    force_literal_keys=("slurm_str", "worker_str", "launch_str"),
    width: int = 10000,
    indent: str = "    ",  # 4 spaces; use "\t" for a tab if preferred
    indent_title: bool = False,
    no_indent_titles: tuple[str, ...] | None = None,
):
    """
    Log a YAML block so every line is prefixed by the logger (timestamp/level),
    and indented for readability.
    If the title matches one of `no_indent_titles` (or indent_title=False), the title is not indented.
    """
    no_indent_titles = no_indent_titles or ()
    title_prefix = (
        "" if (not indent_title or title in no_indent_titles) else indent
    )

    logger.log(level, "%s%s:", title_prefix, title)
    yaml_text = dump_yaml_for_log(
        data, force_literal_keys=force_literal_keys, width=width
    ).rstrip()
    for line in yaml_text.splitlines():
        logger.log(level, "%s%s", indent, line)


def get_seeds(seed: int, ensemble_size: int) -> np.ndarray:
    """
    Sets up the random seeds for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    return np.random.randint(
        0, 1000, size=ensemble_size
    )