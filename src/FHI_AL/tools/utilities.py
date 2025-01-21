import random
import logging
import re
import os
import torch
import numpy as np
from typing import Optional, Sequence, Tuple, List, Dict
from pathlib import Path
from ase.io import write
from mace import data as mace_data
from mace import tools, modules
from mace.tools import AtomicNumberTable, torch_geometric, torch_tools, utils
from mace.tools.train import evaluate
from mace.data.utils import compute_average_E0s, config_from_atoms_list, load_from_xyz
from FHI_AL.tools.setup_MACE import setup_mace
from FHI_AL.tools.setup_MACE_training import setup_mace_training
from dataclasses import dataclass
from mace.tools.utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
)
import ase.data
import ase.io
from ase.io import read
from ase import units
import dataclasses
from torchmetrics import Metric
from typing import Any, Dict, List, Optional, Tuple, Union
from mace.tools import AtomicNumberTable
from contextlib import nullcontext
import time
import GPUtil
from threading import Thread
import time
import pandas as pd

#TODO: this file combines a lot of stuff and should be split up and
        # refactored into classes


############ AIMS CONSTANTS ############
BOHR    = 0.529177210
BOHR_INV = 1.0 / BOHR
HARTREE = 27.21138450
HARTREE_INV = 1.0 / HARTREE


#############################################################################
############ This part is mostly taken from the MACE source code ############
############ with slight modifications to fit the needs of AL    ############
#############################################################################

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


@dataclasses.dataclass
class SubsetCollection:
    train: mace_data.Configurations
    valid: mace_data.Configurations
    tests: List[Tuple[str, mace_data.Configurations]]


def get_dataset_from_atoms(
    train_list: list,
    valid_list: list,
    config_type_weights: Dict,
    valid_fraction: float = None,
    test_list: list = None,
    seed: int = 1234,
    keep_isolated_atoms: bool = False,
    head_name: str = "Default",
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: str = "REF_stress",
    virials_key: str = "virials",
    dipole_key: str = "dipoles",
    charges_key: str = "charges",
    head_key: str = "head",
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    """Load training and test dataset from xyz file"""

    atomic_energies_dict, all_train_configs = load_from_atoms(
        atoms_list=train_list,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        head_key=head_key,
        extract_atomic_energies=True,
        keep_isolated_atoms=keep_isolated_atoms,
        head_name=head_name,
    )

    if valid_list is not None:
        _, valid_configs = load_from_atoms(
            atoms_list=valid_list,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            head_key=head_key,
            extract_atomic_energies=False,
            head_name=head_name,
        )

        train_configs = all_train_configs
    elif valid_fraction is not None:
        logging.info(
            "Using random %s%% of training set for validation",
            100 * valid_fraction,
        )
        train_configs, valid_configs = mace_data.random_train_valid_split(
            all_train_configs, valid_fraction, seed
        )
    else:
        valid_configs = []
        train_configs = all_train_configs
        
    test_configs = []
    if test_list is not None:
        _, all_test_configs = load_from_atoms(
            atoms_list=test_list,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            head_key=head_key,
            extract_atomic_energies=False,
            head_name=head_name,
        )
        # create list of tuples (config_type, list(Atoms))
        test_configs = mace_data.test_config_types(all_test_configs)

    return (
        SubsetCollection(
            train=train_configs, valid=valid_configs, tests=test_configs
        ),
        atomic_energies_dict,
    )


def get_single_dataset_from_atoms(
    atoms_list: list,
    config_type_weights: Dict,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipoles",
    charges_key: str = "charges",
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    """Load training and test dataset from xyz file"""
    
    atomic_energies_dict, all_configs = load_from_atoms(
        atoms_list=atoms_list,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        extract_atomic_energies=True,
    )

    return (all_configs, atomic_energies_dict)


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom
    stress: Optional[Stress] = None  # eV/Angstrom^3
    virials: Optional[Virials] = None  # eV
    dipole: Optional[Vector] = None  # Debye
    charges: Optional[Charges] = None  # atomic unit
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None
    atomic_weights: Optional[np.ndarray] = None

    weight: float = 1.0  # weight of config in loss
    energy_weight: float = 1.0  # weight of config energy in loss
    forces_weight: float = 1.0  # weight of config forces in loss
    stress_weight: float = 1.0  # weight of config stress in loss
    virials_weight: float = 1.0  # weight of config virial in loss
    config_type: Optional[str] = DEFAULT_CONFIG_TYPE  # config_type of config
    head: Optional[str] = "Default"  # head used to compute the config

Configurations = List[Configuration]


def random_train_valid_split(
    items: Sequence, valid_fraction: float, seed: int
) -> Tuple[List, List]:
    assert 0.0 < valid_fraction < 1.0

    size = len(items)
    train_size = size - int(valid_fraction * size)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    return (
        [items[i] for i in indices[:train_size]],
        [items[i] for i in indices[train_size:]],
    )

def test_config_types(
    test_configs: Configurations,
) -> List[Tuple[Optional[str], List[Configuration]]]:
    """Split test set based on config_type-s"""
    test_by_ct = []
    all_cts = []
    for conf in test_configs:
        if conf.config_type not in all_cts:
            all_cts.append(conf.config_type)
            test_by_ct.append((conf.config_type, [conf]))
        else:
            ind = all_cts.index(conf.config_type)
            test_by_ct[ind][1].append(conf)
    return test_by_ct

def load_from_atoms(
    atoms_list: list,
    config_type_weights: Dict,
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: str = "REF_stress",
    virials_key: str = "REF_virials",
    dipole_key: str = "REF_dipole",
    charges_key: str = "REF_charges",
    head_key: str = "head",
    head_name: str = "Default",
    extract_atomic_energies: bool = False,
    keep_isolated_atoms: bool = False,
) -> Tuple[Dict[int, float], Configurations]:
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if energy_key == "energy":
        logging.warning(
            "Since ASE version 3.23.0b1, using energy_key 'energy' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'energy' to 'REF_energy'. You need to use --energy_key='REF_energy' to specify the chosen key name."
        )
        energy_key = "REF_energy"
        for atoms in atoms_list:
            try:
                atoms.info["REF_energy"] = atoms.get_potential_energy()
            except Exception as e:  # pylint: disable=W0703
                logging.error(f"Failed to extract energy: {e}")
                atoms.info["REF_energy"] = None
    if forces_key == "forces":
        logging.warning(
            "Since ASE version 3.23.0b1, using forces_key 'forces' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'forces' to 'REF_forces'. You need to use --forces_key='REF_forces' to specify the chosen key name."
        )
        forces_key = "REF_forces"
        for atoms in atoms_list:
            try:
                atoms.arrays["REF_forces"] = atoms.get_forces()
            except Exception as e:  # pylint: disable=W0703
                logging.error(f"Failed to extract forces: {e}")
                atoms.arrays["REF_forces"] = None
    if stress_key == "stress":
        logging.warning(
            "Since ASE version 3.23.0b1, using stress_key 'stress' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'stress' to 'REF_stress'. You need to use --stress_key='REF_stress' to specify the chosen key name."
        )
        stress_key = "REF_stress"
        for atoms in atoms_list:
            try:
                atoms.info["REF_stress"] = atoms.get_stress()
            except Exception as e:  # pylint: disable=W0703
                atoms.info["REF_stress"] = None
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    atomic_energies_dict = {}
    if extract_atomic_energies:
        atoms_without_iso_atoms = []

        for idx, atoms in enumerate(atoms_list):
            isolated_atom_config = (
                len(atoms) == 1 and atoms.info.get("config_type") == "IsolatedAtom"
            )
            if isolated_atom_config:
                if energy_key in atoms.info.keys():
                    atomic_energies_dict[atoms.get_atomic_numbers()[0]] = atoms.info[
                        energy_key
                    ]
                else:
                    logging.warning(
                        f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                        "but does not contain an energy. Zero energy will be used."
                    )
                    atomic_energies_dict[atoms.get_atomic_numbers()[0]] = np.zeros(1)
            else:
                atoms_without_iso_atoms.append(atoms)

        if len(atomic_energies_dict) > 0:
            logging.info("Using isolated atom energies from training file")
        if not keep_isolated_atoms:
            atoms_list = atoms_without_iso_atoms

    for atoms in atoms_list:
        atoms.info[head_key] = head_name

    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        head_key=head_key,
    )
    return atomic_energies_dict, configs



def compute_average_E0s(
    collections_train: Configurations, z_table: AtomicNumberTable
) -> Dict[int, float]:
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s
    """
    len_train = len(collections_train)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.zeros(len_train)
    for i in range(len_train):
        B[i] = collections_train[i].energy
        for j, z in enumerate(z_table.zs):
            A[i, j] = np.count_nonzero(
                collections_train[i].atomic_numbers == z
            )
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


def max_sd_2(
    ensemble_prediction: np.array, 
    return_argmax: bool = False,
) -> np.array:
    """
    Compute the maximum standard deviation of the ensemble prediction.

    Args:
        ensemble_prediction (np.array): Ensemble prediction of forces: [n_ensemble_members, n_mols, n_atoms, xyz].

    Returns:
        np.array: Maximum standard deviation of atomic forces per molecule: [n_mols].
    """
    # average prediction over ensemble of models
    pred_av = np.average(ensemble_prediction, axis=0, keepdims=True)
    diff_sq = (ensemble_prediction - pred_av) ** 2.
    diff_sq_mean = np.mean(diff_sq, axis=(0,-1))
    max_sd = np.max(np.sqrt(diff_sq_mean),axis=-1)
    if return_argmax:
        return max_sd, np.argmax(np.sqrt(diff_sq_mean),axis=-1)
    else:
        return max_sd
    
def avg_sd(
        ensemble_prediction: np.array,
    ) -> np.array:
    """
    Compute the average standard deviation of the ensemble prediction.

    Args:
        ensemble_prediction (np.array): Ensemble prediction of forces: [n_ensemble_members, n_mols, n_atoms, xyz].

    Returns:
        np.array: Average standard deviation of atomic forces per molecule: [n_mols].
    """
    # average prediction over ensemble of models
    pred_av = np.average(ensemble_prediction, axis=0, keepdims=True)
    diff_sq = (ensemble_prediction - pred_av) ** 2.
    diff_sq_mean = np.mean(diff_sq, axis=(0,-1))
    avg_sd = np.mean(np.sqrt(diff_sq_mean),axis=-1)
    return avg_sd

def atom_wise_sd(
    ensemble_prediction: np.array,
) -> np.array:
    """
    Compute the atom-wise standard deviation of the ensemble prediction.

    Args:
        prediction (np.array): Prediction of forces: [n_ensemble_members, n_mols, n_atoms, xyz].

    Returns:
        np.array: Atom-wise standard deviation of atomic forces per molecule: [n_mols, n_atoms].
    """
    # average prediction over ensemble of models
    pred_av = np.average(ensemble_prediction, axis=0, keepdims=True)
    diff_sq = (ensemble_prediction - pred_av) ** 2.
    diff_sq_mean = np.mean(diff_sq, axis=(0,-1))
    atom_wise_sd = np.sqrt(diff_sq_mean)
    return atom_wise_sd

def atom_wise_f_error(
    prediction: np.array,
    target: np.array,
) -> np.array:
    """
    Compute the atom-wise force error of the ensemble prediction.

    Args:
        prediction (np.array): Prediction of forces: [n_mols, n_atoms, xyz].
        target (np.array): Target forces: [n_mols, n_atoms, xyz].

    Returns:
        np.array: Atom-wise force error of atomic forces per molecule: [n_mols, n_atoms].
    """
    atom_wise_f_error = np.sqrt(np.mean((prediction - target)**2, axis=(-1)))
    return atom_wise_f_error

def split_data(
    data: list,
    valid_fraction: float,
)-> tuple:
    """
    Split data into training and validation set.

    Args:
        data (list): List of ASE atoms objects.
        valid_fraction (float): Fraction of data to be used for validation.

    Returns:
        tuple: Tuple of training and validation data (both lists).
    """
    data = data.copy()
    random.shuffle(data)
    n_samples = len(data)
    n_valid = int(n_samples * valid_fraction)
    n_train = n_samples - n_valid
    train_data = data[:n_train]
    valid_data = data[n_train:]
    return train_data, valid_data

def create_dataloader(
    train_set,
    valid_set,
    train_batch_size: int,
    valid_batch_size: int, 
)-> tuple:
    """
    Create dataloader for training and validation data.

    Args:
        train_data (list): List of ASE atoms objects for training.
        valid_data (list): List of ASE atoms objects for validation.
        z_table (AtomicNumberTable): Table of elements.
        seed (int): Seed for training set shuffling.
        r_max (float): Cut-off radius.
        batch_size (int): Batch size for training and validation.

    Returns:
        tuple: _description_
    """
    
    
    train_loader = torch_geometric.dataloader.DataLoader(
    dataset=train_set,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        )

    valid_loader = torch_geometric.dataloader.DataLoader(
        dataset=valid_set,
        batch_size=valid_batch_size,
        shuffle=False,
        drop_last=False,
        )
    return train_loader, valid_loader

def create_mace_dataset(
    data: list,
    seed: int,
    z_table: AtomicNumberTable,
    r_max: float,
)-> list:
    """
    Create a MACE style dataset from a list of ASE atoms objects.

    Args:
        data (list): List of ASE atoms objects.
        seed (int): Seed for shuffling and splitting the dataset.
        z_table (AtomicNumberTable): Table of elements.
        r_max (float): Cut-off radius.

    Returns:
        list: MACE style dataset.
    """
    
    collections, _ = get_dataset_from_atoms(
    train_list=data,
    valid_list=None,
    seed=seed,
    config_type_weights=None,
    )
    

    data_set=[
        mace_data.AtomicData.from_config(
            config, z_table=z_table, cutoff=r_max
        )
        for config in collections.train
    ]
    return data_set

def ensemble_from_folder(
    path_to_models: str,
    device: str,
    compile_mode: str = "default",
    ) -> list:
    """
    Load an ensemble of models from a folder. 
    (Can't handle other file formats than .pt at the moment.)

    Args:
        path_to_models (str): Path to the folder containing the models.
        device (str): Device to load the models on.
        compile_mode (str, optional): Not implemented yet. Does nothing. Defaults to "default".

    Returns:
        list: List of models.
    """
    
    assert os.path.exists(path_to_models)
    assert os.listdir(Path(path_to_models))

    ensemble = {}
    for filename in os.listdir(path_to_models):
        if os.path.isfile(os.path.join(path_to_models, filename)):
            complete_path = os.path.join(path_to_models, filename)
            model = torch.load(complete_path, map_location=device)
            # One can't train compiled models so not usefull for us.
            #model = torch.compile(
            #        prepare(extract_load)(f=complete_path, map_location=device),
            #        mode=compile_mode,
            #        fullgraph=True,
            #    )
            filename_without_suffix = os.path.splitext(filename)[0]
            ensemble[filename_without_suffix] = model
    return ensemble


def pre_trajectories_from_folder(
        path: str,
        num_trajectories: int,
        ) -> list:
    """
    Load pre-existing trajectories from a folder. ASE readable formats are supported.

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


def evaluate_model(
    atoms_list: list,
    model: str,
    batch_size: int,
    device: str,
    compute_stress: bool = False,
    dtype: str = "float64",
) -> torch.tensor:
    """
    Evaluate a MACE model on a list of ASE atoms objects.
    This only handles atoms list with a single species. 

    Args:
        atoms_list (list): List of ASE atoms objects.
        model (str): MACE model to evaluate.
        batch_size (int): Batch size for evaluation.
        device (str): Device to evaluate the model on.
        compute_stress (bool, optional): Compute stress or not. Defaults to False.
        dtype (str, optional): Data type of model. Defaults to "float64".

    Returns:
        torch.tensor: _description_
    """
    torch_tools.set_default_dtype(dtype)

    # Load data and prepare input
    configs = [mace_data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            mace_data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max)
            )
            for config in configs
        ],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    stresses_list = []
    forces_collection = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), compute_stress=compute_stress)
        energies_list.append(torch_tools.to_numpy(output["energy"]))
        if compute_stress:
            stresses_list.append(torch_tools.to_numpy(output["stress"]))

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        forces_collection.append(forces[:-1])  # drop last as its emtpy

    energies = np.concatenate(energies_list, axis=0)
    # TODO: This only works for predicting a single molecule not different ones in one set
    forces_array = np.stack(forces_collection).reshape(len(energies), -1, 3)
    assert len(atoms_list) == len(energies) == len(forces_array)
    if compute_stress:
        stresses = np.concatenate(stresses_list, axis=0)
        assert len(atoms_list) == stresses.shape[0]

        return energies, forces_array, stresses
    else:
        return energies, forces_array


def ensemble_prediction(
    models: list,
    atoms_list: list,
    device: str,
    dtype: str = "float64",
    batch_size: int = 1,
    return_energies: bool = False,
) -> np.array:
    """
    Predict forces for a list of ASE atoms objects using an ensemble of models. 
    !!! Does not reduce the energies or forces to a single value. !!!
    TODO: currently only works for atoms list with a single species.

    Args:
        models (list): List of models.
        atoms_list (list): List of ASE atoms objects.
        device (str): Device to evaluate the models on.
        dtype (str, optional): Dtype of models. Defaults to "float64".
        batch_size (int, optional): Batch size of evaluation. Defaults to 1.
        return_energies (bool, optional): Whether to return energies or not. Defaults to False.

    Returns:
        np.array: Forces [n_models, n_mols, n_atoms, xyz]
        Optionally:
        np.array: Energies [n_models, n_mols], Forces [n_models, n_mols, n_atoms, xyz]

    """
    all_forces = []
    all_energies = []
    i = 0
    for model in models:
        E, F = evaluate_model(
            atoms_list=atoms_list,
            model=model,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        all_forces.append(F)
        all_energies.append(E)
        i += 1

    all_forces = np.stack(all_forces).reshape(
        (len(models), len(atoms_list), -1, 3)
    )

    all_energies = np.stack(all_energies).reshape(
        (len(models), len(atoms_list))
    )

    if return_energies:
        return all_energies, all_forces
    return all_forces


def evaluate_model_v2(
    mace_ds: list,
    model: str,
    batch_size: int,
    device: str,
    compute_stress: bool = False,
    dtype: str = "float64",
) -> torch.tensor:
    """
    Evaluate a MACE model on a list of ASE atoms objects.
    This only handles atoms list with a single species. 

    Args:
        atoms_list (list): List of ASE atoms objects.
        model (str): MACE model to evaluate.
        batch_size (int): Batch size for evaluation.
        device (str): Device to evaluate the model on.
        compute_stress (bool, optional): Compute stress or not. Defaults to False.
        dtype (str, optional): Data type of model. Defaults to "float64".

    Returns:
        torch.tensor: _description_
    """
    torch_tools.set_default_dtype(dtype)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=mace_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    stresses_list = []
    forces_collection = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), compute_stress=compute_stress)
        energies_list.append(torch_tools.to_numpy(output["energy"]))
        if compute_stress:
            stresses_list.append(torch_tools.to_numpy(output["stress"]))

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        forces_collection.append(forces[:-1])  # drop last as its emtpy

    energies = np.concatenate(energies_list, axis=0)
    # TODO: This only works for predicting a single molecule not different ones in one set
    forces_array = np.stack(forces_collection).reshape(len(energies), -1, 3)
    assert len(mace_ds) == len(energies) == len(forces_array)
    if compute_stress:
        stresses = np.concatenate(stresses_list, axis=0)
        assert len(mace_ds) == stresses.shape[0]

        return energies, forces_array, stresses
    else:
        return energies, forces_array 
    
def ensemble_prediction_v2(
    models: list,
    mace_ds: list,
    device: str,
    dtype: str = "float64",
    batch_size: int = 1,
    return_energies: bool = False,
) -> np.array:
    """
    Predict forces for a list of ASE atoms objects using an ensemble of models. 
    !!! Does not reduce the energies or forces to a single value. !!!
    TODO: currently only works for atoms list with a single species.

    Args:
        models (list): List of models.
        atoms_list (list): List of ASE atoms objects.
        device (str): Device to evaluate the models on.
        dtype (str, optional): Dtype of models. Defaults to "float64".
        batch_size (int, optional): Batch size of evaluation. Defaults to 1.
        return_energies (bool, optional): Whether to return energies or not. Defaults to False.

    Returns:
        np.array: Forces [n_models, n_mols, n_atoms, xyz]
        Optionally:
        np.array: Energies [n_models, n_mols], Forces [n_models, n_mols, n_atoms, xyz]

    """
    all_forces = []
    all_energies = []
    i = 0
    for model in models:
        E, F = evaluate_model_v2(
            mace_ds=mace_ds,
            model=model,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        all_forces.append(F)
        all_energies.append(E)
        i += 1

    all_forces = np.stack(all_forces).reshape(
        (len(models), len(mace_ds), -1, 3)
    )

    all_energies = np.stack(all_energies).reshape(
        (len(models), len(mace_ds))
    )

    if return_energies:
        return all_energies, all_forces
    return all_forces

def compute_max_error(delta: np.ndarray) -> float:
    return np.max(np.abs(delta)).item()
    

class MACEEval(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("num_data", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "stress_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_stress", default=[], dist_reduce_fx="cat")
        self.add_state("delta_stress_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state(
            "virials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_virials", default=[], dist_reduce_fx="cat")
        self.add_state("delta_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Mus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus_per_atom", default=[], dist_reduce_fx="cat")

    def update(self, batch, output):  # pylint: disable=arguments-differ
        self.num_data += batch.num_graphs

        if output.get("energy") is not None and batch.energy is not None:
            self.E_computed += 1.0
            self.delta_es.append(batch.energy - output["energy"])
            self.delta_es_per_atom.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
        if output.get("forces") is not None and batch.forces is not None:
            self.Fs_computed += 1.0
            self.fs.append(batch.forces)
            self.delta_fs.append(batch.forces - output["forces"])
        if output.get("stress") is not None and batch.stress is not None:
            self.stress_computed += 1.0
            self.delta_stress.append(batch.stress - output["stress"])
            self.delta_stress_per_atom.append(
                (batch.stress - output["stress"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("virials") is not None and batch.virials is not None:
            self.virials_computed += 1.0
            self.delta_virials.append(batch.virials - output["virials"])
            self.delta_virials_per_atom.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("dipole") is not None and batch.dipole is not None:
            self.Mus_computed += 1.0
            self.mus.append(batch.dipole)
            self.delta_mus.append(batch.dipole - output["dipole"])
            self.delta_mus_per_atom.append(
                (batch.dipole - output["dipole"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
            )

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return torch_tools.to_numpy(delta)

    def compute(self):
        aux = {}
        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            aux["mae_e"] = compute_mae(delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e"] = compute_rmse(delta_es)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            aux["max_e"] = compute_max_error(delta_es)
            aux["max_e_per_atom"] = compute_max_error(delta_es_per_atom)
            aux["q95_e"] = compute_q95(delta_es)
        if self.Fs_computed:
            fs = self.convert(self.fs)
            delta_fs = self.convert(self.delta_fs)
            aux["mae_f"] = compute_mae(delta_fs)
            aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
            aux["rmse_f"] = compute_rmse(delta_fs)
            aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
            aux["max_f"] = compute_max_error(delta_fs)
            aux["q95_f"] = compute_q95(delta_fs)
        if self.stress_computed:
            delta_stress = self.convert(self.delta_stress)
            delta_stress_per_atom = self.convert(self.delta_stress_per_atom)
            aux["mae_stress"] = compute_mae(delta_stress)
            aux["rmse_stress"] = compute_rmse(delta_stress)
            aux["rmse_stress_per_atom"] = compute_rmse(delta_stress_per_atom)
            aux["q95_stress"] = compute_q95(delta_stress)
        if self.virials_computed:
            delta_virials = self.convert(self.delta_virials)
            delta_virials_per_atom = self.convert(self.delta_virials_per_atom)
            aux["mae_virials"] = compute_mae(delta_virials)
            aux["rmse_virials"] = compute_rmse(delta_virials)
            aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
            aux["q95_virials"] = compute_q95(delta_virials)
        if self.Mus_computed:
            mus = self.convert(self.mus)
            delta_mus = self.convert(self.delta_mus)
            delta_mus_per_atom = self.convert(self.delta_mus_per_atom)
            aux["mae_mu"] = compute_mae(delta_mus)
            aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
            aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
            aux["rmse_mu"] = compute_rmse(delta_mus)
            aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
            aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
            aux["q95_mu"] = compute_q95(delta_mus)

        return aux

def test_model(
    model,
    data_loader,
    output_args: dict,
    device: str,
    return_predictions: bool = False,

):
    for param in model.parameters():
        param.requires_grad = False

    metrics = MACEEval().to(device)
    start_time = time.time()
    if return_predictions:
        predictions = {}
        if output_args.get("energy", False):
            predictions["energy"] = []
        if output_args.get("forces", False):
            predictions["forces"] = []
        if output_args.get("stress", False):
            predictions["stress"] = []
        if output_args.get("virials", False):
            predictions["virials"] = []
        if output_args.get("dipole", False):
            predictions["dipole"] = []

    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        aux = metrics(batch, output)

        if return_predictions:
            if output_args.get("energy", False):
                predictions["energy"].append(output["energy"])
            if output_args.get("forces", False):
                predictions["forces"].append(output["forces"])
            if output_args.get("stress", False):
                predictions["stress"].append(output["stress"])
            if output_args.get("virials", False):
                predictions["virials"].append(output["virials"])
            if output_args.get("dipole", False):
                predictions["dipole"].append(output["dipole"])
    
    aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    for param in model.parameters():
        param.requires_grad = True
    if return_predictions:
        for key in predictions.keys():
            predictions[key] = torch.cat(predictions[key], dim=0).detach().cpu()
        aux["predictions"] = predictions
    return aux

def test_ensemble(
    ensemble: dict,
    batch_size: int,
    output_args: dict,
    device: str,
    path_to_data: str = None,
    atoms_list: list = None,
    logger: MetricsLogger = None,
    log_errors: str = "PerAtomMAE",
    return_predictions: bool = False,
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: str = "REF_stress",
    virials_key: str = "virials",
    dipole_key: str = "dipoles",
    charges_key: str = "charges",
    head_key: str = "head",
) -> Tuple[dict, dict]:
    
    
    if atoms_list is not None:
        configs = [mace_data.config_from_atoms(
            atoms,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            dipole_key=dipole_key,
            virials_key=virials_key,
            charges_key=charges_key,
            head_key=head_key
            ) for atoms in atoms_list
                ]
    elif path_to_data is not None:
        _, configs = load_from_xyz(
            file_path=path_to_data,
            config_type_weights=None,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            head_key=head_key
        )

    else:
        raise ValueError("Either atoms_list or path_to_data must be provided")
    
    z_table = utils.AtomicNumberTable([int(z) for z in ensemble[list(ensemble.keys())[0]].atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            mace_data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(ensemble[list(ensemble.keys())[0]].r_max)
            )
            for config in configs
        ],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    ensemble_metrics = {}
    for tag, model in ensemble.items():
        metrics = test_model(
            model=model,
            data_loader=data_loader,
            output_args=output_args,
            device=device,
            return_predictions=return_predictions,
        )
        ensemble_metrics[tag] = metrics

    avg_ensemble_metrics = {}
    for key in ensemble_metrics[list(ensemble_metrics.keys())[0]].keys():
        if key not in ["mode", "epoch", "predictions"]:
            avg_ensemble_metrics[key] = np.mean([m[key] for m in ensemble_metrics.values()])
        if return_predictions:
            avg_ensemble_metrics["predictions"] = {
                key: np.mean([m["predictions"][key] for m in ensemble_metrics.values()], axis=0)
                for key in ensemble_metrics[list(ensemble_metrics.keys())[0]]["predictions"].keys()
            }
    if logger is not None:
        logger.log(avg_ensemble_metrics)
        if log_errors == "PerAtomRMSE":
            error_e = avg_ensemble_metrics["rmse_e_per_atom"] * 1e3
            error_f = avg_ensemble_metrics["rmse_f"] * 1e3
            logging.info(
                f"RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
            )
        elif (
            log_errors == "PerAtomRMSEstressvirials"
            and avg_ensemble_metrics["rmse_stress_per_atom"] is not None
        ):
            error_e = avg_ensemble_metrics["rmse_e_per_atom"] * 1e3
            error_f = avg_ensemble_metrics["rmse_f"] * 1e3
            error_stress = avg_ensemble_metrics["rmse_stress_per_atom"] * 1e3
            logging.info(
                f"RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_stress_per_atom={error_stress:.1f} meV / A^3"
            )
        elif (
            log_errors == "PerAtomRMSEstressvirials"
            and avg_ensemble_metrics["rmse_virials_per_atom"] is not None
        ):
            error_e = avg_ensemble_metrics["rmse_e_per_atom"] * 1e3
            error_f = avg_ensemble_metrics["rmse_f"] * 1e3
            error_virials = avg_ensemble_metrics["rmse_virials_per_atom"] * 1e3
            logging.info(
                f"RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_virials_per_atom={error_virials:.1f} meV"
            )
        elif log_errors == "TotalRMSE":
            error_e = avg_ensemble_metrics["rmse_e"] * 1e3
            error_f = avg_ensemble_metrics["rmse_f"] * 1e3
            logging.info(
                f"RMSE_E={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
            )
        elif log_errors == "PerAtomMAE":
            error_e = avg_ensemble_metrics["mae_e_per_atom"] * 1e3
            error_f = avg_ensemble_metrics["mae_f"] * 1e3
            logging.info(
                f"MAE_E_per_atom={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A"
            )
        elif log_errors == "TotalMAE":
            error_e = avg_ensemble_metrics["mae_e"] * 1e3
            error_f = avg_ensemble_metrics["mae_f"] * 1e3
            logging.info(
                f"MAE_E={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A"
            )
        elif log_errors == "DipoleRMSE":
            error_mu = avg_ensemble_metrics["rmse_mu_per_atom"] * 1e3
            logging.info(
                f"RMSE_MU_per_atom={error_mu:.2f} mDebye"
            )
        elif log_errors == "EnergyDipoleRMSE":
            error_e = avg_ensemble_metrics["rmse_e_per_atom"] * 1e3
            error_f = avg_ensemble_metrics["rmse_f"] * 1e3
            error_mu = avg_ensemble_metrics["rmse_mu_per_atom"] * 1e3
            logging.info(
                f"RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_Mu_per_atom={error_mu:.2f} mDebye"
            )

    return (avg_ensemble_metrics, ensemble_metrics)

def select_best_member(
        ensemble_valid_loss: dict,
):
    """
    Selects the best member of an ensemble based on the validation loss.

    Args:
        ensemble_valid_loss (dict): Dictionary of validation losses for each ensemble member.

    Returns:
        str: Tag of the best ensemble member.
    """
    return min(ensemble_valid_loss, key=ensemble_valid_loss.get)
    

def E_uncert(
        prediction: np.array,
) -> float:
    """
    Computes the standard deviation of the ensemble prediction on energies.

    Args:
        prediction (np.array): Ensemble prediction of energies: [n_ensemble_members, n_mols].

    Returns:
        float: Standard deviation of the ensemble prediction.
    """

    M = prediction.shape[0] # number of ensemble members
    prediction_avg = np.mean(prediction, axis=0, keepdims=True)
    uncert = 1/(M-1) * np.sum((prediction_avg - prediction)**2, axis=0)
    return uncert

def get_uncert_alpha(
    reference,
    ensemble_avg,
    uncertainty
    )-> float:
    """
    Estimate distribution shift for a given dataset.
    
    https://arxiv.org/abs/1809.07653

    Args:
        reference (_type_): Reference property.
        ensemble_avg (_type_): Average ensemple prediction.
        uncertainty (_type_): Uncertainty measure for the ensemble.

    Returns:
        float: Distribution shift coefficient.
    """
    
    return np.mean((reference-ensemble_avg)**2 / uncertainty **2)  

def ensemble_training_setups(
        ensemble: dict,
        mace_settings: dict,
        checkpoints_dir: str = None,
        restart: bool = False,
        ) -> dict:
    training_setups = {}
    for tag, model in ensemble.items():
        training_setups[tag] = setup_mace_training(
            settings=mace_settings,
            model=model,
            tag=tag,
            restart=restart,
            checkpoints_dir=checkpoints_dir,
        )
    return training_setups

def create_seeds_tags_dict(
    seeds: np.array,
    mace_settings: dict,
    al_settings: dict = None,
    save_seeds_tags_dict: str = "seeds_tags_dict.npz",
):
    seeds_tags_dict = {}
    for seed in seeds:
        tag = mace_settings['GENERAL']["name_exp"] + "-" + str(seed)
        seeds_tags_dict[tag] = seed
    if save_seeds_tags_dict:
        np.savez(al_settings["dataset_dir"]+ '/' + save_seeds_tags_dict, **seeds_tags_dict)
    return seeds_tags_dict

def create_ztable(
        zs: list,
):
    z_table = tools.get_atomic_number_table_from_zs(
        z for z in zs
    )
    return z_table

def setup_ensemble_dicts(
    seeds_tags_dict: dict,
    z_table: tools.AtomicNumberTable,
    mace_settings: dict,
    ensemble_atomic_energies_dict: dict,
    ) -> tuple:
    """
    Creates dictionaries for the ensemble members i.e. a dictionary of models.
    Also, creates a dictionary for the training setups for each model and
    a dictionary for the seeds and their corresponding tags. All of these
    are returned as a tuple.

    Args:
        seeds (np.array): Array of seeds for the ensemble.
        save_seeds_tags_dict (bool, optional): Seeds with corresponding tags. 
                                            Defaults to "seeds_tags_dict.npz".

    Returns:
        tuple: Tuple of seeds_tags_dict, ensemble, training_setups.
    """
    
    ensemble = {}
    for tag, seed in seeds_tags_dict.items():
        mace_settings['GENERAL']["seed"] = seed
        tag = mace_settings['GENERAL']["name_exp"] + "-" + str(seed)
        seeds_tags_dict[tag] = seed
        ensemble[tag] =  setup_mace(
                settings=mace_settings,
                z_table=z_table,
                atomic_energies_dict=ensemble_atomic_energies_dict[tag],
                )
    return ensemble

def get_atomic_energies_from_ensemble(
    ensemble: dict,
    z
):
    """
    Loads the atomic energies from existing ensemble members.
    """
    ensemble_atomic_energies_dict = {}
    ensemble_atomic_energies = {}
    for tag, model in ensemble.items():
        ensemble_atomic_energies[tag] = np.array(model.atomic_energies_fn.atomic_energies.cpu())
        ensemble_atomic_energies_dict[tag] = {}
        for i, atomic_energy in enumerate(ensemble_atomic_energies[tag]):
            # TH: i don't know if the atomic energies are really sorted by atomic number
            # inside the models. TODO: check that
            ensemble_atomic_energies_dict[tag][np.sort(np.unique(z))[i]] = atomic_energy
        
    return (ensemble_atomic_energies, ensemble_atomic_energies_dict)

def get_atomic_energies_from_pt(
    path_to_checkpoints: str,
    z,
    seeds_tags_dict: dict,
    convergence: bool = False,
):  
    ensemble_atomic_energies_dict = {}
    ensemble_atomic_energies = {}
    last_check_pt = list_latest_file(path_to_checkpoints)
    last_epoch = int(last_check_pt.split(".")[0].split('-')[-1])
    for tag in seeds_tags_dict.keys():
        check_pt = torch.load((path_to_checkpoints + '/' + tag 
                                + f'_epoch-{last_epoch}.pt'))
        
        
        atomic_energies_array = check_pt['model']['atomic_energies_fn.atomic_energies']
        ensemble_atomic_energies[tag] = np.array(atomic_energies_array.cpu())
        ensemble_atomic_energies_dict[tag] = {}
        for i, atomic_energy in enumerate(ensemble_atomic_energies[tag]):
            # TH: i don't know if the atomic energies are really sorted by atomic number
            # inside the models. TODO: check that
            ensemble_atomic_energies_dict[tag][np.sort(np.unique(z))[i]] = atomic_energy

    return (ensemble_atomic_energies, ensemble_atomic_energies_dict)
        
def update_mace_set(
    new_train_data,
    new_valid_data,
    mace_set: dict,
    z_table: tools.AtomicNumberTable,
    seed: int,
    r_max: float,
) -> dict:
    """
    Update the MACE dataset with new data. Currently needs valid and training data.

    Args:
        new_train_data (_type_): List of MACE style training data.
        new_valid_data (_type_): List of MACE style validation data.
        mace_set (dict): Dictionary containg train and valid set to be updated.
        z_table (tools.AtomicNumberTable): Table of elements.
        seed (int): Seed for shuffling and splitting the dataset.
        r_max (float): Cut-off radius.

    Returns:
        dict: Updated MACE dataset.
    """
    new_train_set = create_mace_dataset(
        data=new_train_data,
        z_table=z_table,
        seed=seed,
        r_max=r_max
    )
    
    new_valid_set = create_mace_dataset(
        data=new_valid_data,
        z_table=z_table,
        seed=seed,
        r_max=r_max
    )
    mace_set['train'] += new_train_set
    mace_set['valid'] += new_valid_set
    
    
    return mace_set

def update_datasets(
    new_points: list,
    mace_set: dict,
    ase_set: dict,
    valid_split: float,
    z_table: tools.AtomicNumberTable,
    seed: int,
    r_max: float,
) -> tuple:
    """
    Update the ASE and MACE datasets with new data.

    Args:
        new_points (list): List of ASE atoms objects.
        mace_set (dict): Dictionary of MACE style training and validation data.
        ase_set (dict): Dictionary of ASE style training and validation data.
        valid_split (float): Fraction of data to be used for validation.
        z_table (tools.AtomicNumberTable): Table of elements.
        seed (int): Seed for shuffling and splitting the dataset.
        r_max (float): Cut-off radius.

    Returns:
        tuple: _description_
    """
    new_train_data, new_valid_data = split_data(new_points, valid_split)
    ase_set['train'] += new_train_data
    ase_set['valid'] += new_valid_data
    
    mace_set = update_mace_set(
        new_train_data,
        new_valid_data,
        mace_set,
        z_table,
        seed,
        r_max
        )
    
    return ase_set,mace_set

def ase_to_mace_ensemble_sets(
    ensemble_ase_sets: dict,
    z_table,
    seed: dict,
    r_max: float,
) -> dict:    
    """
    Convert ASE style ensemble datasets to MACE style ensemble datasets.

    Args:
        ensemble_ase_sets (dict): Dictionary of ASE style ensemble datasets.
        z_table (_type_): Table of elements.
        seed (dict): Seed for shuffling and splitting the dataset.
        r_max (float): Cut-off radius.

    Returns:
        dict: Dictionary of MACE style ensemble datasets.
    """
    ensemble_mace_sets = {tag: {'train': [], 'valid': []} for tag in ensemble_ase_sets.keys()}
    for tag in ensemble_ase_sets.keys():
        ensemble_mace_sets[tag]['train'] = create_mace_dataset(
            data=ensemble_ase_sets[tag]['train'],
            z_table=z_table,
            seed=seed,
            r_max=r_max
            )
        ensemble_mace_sets[tag]['valid'] = create_mace_dataset(
            data=ensemble_ase_sets[tag]['valid'],
            z_table=z_table,
            seed=seed,
            r_max=r_max
            )
    return ensemble_mace_sets

def compute_average_E0s(
    energies_train,  
    zs_train,
    z_table: AtomicNumberTable
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

#TODO: split this up and update doc string
def update_model_auxiliaries(
    model: modules.MACE,
    mace_sets: dict,
    atomic_energies: dict,
    scaling: str,
    update_atomic_energies: bool = False,
    atomic_energies_dict: dict = None,
    z_table: tools.AtomicNumberTable = None,
    dtype: str = "float64",
    device: str = "cpu",

):
    """
    Update the average number of neighbors, scale and shift of
    the MACE model with the given data

    Args:
        model (modules.MACE): The MACE model to be updated.
        train_loader (torch_geometric.data.DataLoader): DataLoader for the training data.
        atomic_energies (dict): Dictionary of atomic energies.
        scaling (str): Scaling method to be used.

    Returns:
         None
    """
    train_loader = mace_sets["train_loader"]

    if update_atomic_energies:
        assert z_table is not None
        assert atomic_energies_dict is not None
    
        energies_train = torch.stack([point.energy.reshape(1)  for point in mace_sets["train"]])
        zs_train = [
            [
                z_table.index_to_z(torch.nonzero(one_hot).item()) for one_hot in point.node_attrs
                ] for point in mace_sets["train"]
            ]
        new_atomic_energies_dict = compute_average_E0s(energies_train, zs_train, z_table)
        atomic_energies_dict.update(new_atomic_energies_dict)
        atomic_energies = [
            atomic_energies_dict[z]
            for z in atomic_energies_dict.keys()
        ]   
        atomic_energies = torch.tensor(atomic_energies, dtype=dtype_mapping[dtype])
        
        model.atomic_energies_fn.atomic_energies = atomic_energies.to(device)

    average_neighbors = modules.compute_avg_num_neighbors(train_loader)
    for interaction_idx in range(len(model.interactions)):
        model.interactions[interaction_idx].avg_num_neighbors = average_neighbors
    mean, std = modules.scaling_classes[scaling](train_loader, atomic_energies)
    mean, std = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    model.scale_shift = modules.blocks.ScaleShiftBlock(
        scale=std, shift=mean
    )
    
def save_checkpoint(
    checkpoint_handler: tools.CheckpointHandler,
    training_setup: dict,
    model: modules.MACE,
    epoch: int,
    keep_last: bool = False,
):
    """
    Save a checkpoint of the training setup and model.

    Args:
        checkpoint_handler (tools.CheckpointHandler): MACE handler for saving checkpoints.
        training_setup (dict): Training settings.
        model (modules.MACE): MACE model to be saved.
        epoch (int): Current epoch.
        keep_last (bool, optional): Keep the last checkpoint. Defaults to False.

    Returns:
        None
    """
    if training_setup["ema"] is not None:
        with training_setup["ema"].average_parameters():
            checkpoint_handler.save(
                state=tools.CheckpointState(
                    model,
                    training_setup['optimizer'],  
                    training_setup['lr_scheduler']),
                epochs=epoch,
                keep_last=True if keep_last else False,
            )
    else:
        checkpoint_handler.save(
            state=tools.CheckpointState(
                    model,
                    training_setup['optimizer'],  
                    training_setup['lr_scheduler']),
            epochs=epoch,
            keep_last=False,
        )
    return None

def save_datasets(
    ensemble: dict,
    ensemble_ase_sets: dict,
    path: str,
    initial: bool = False
    ):
    """
    TODO: Save a complete dataset of the combined initial datasets and the rest.
    Save the ensemble datasets as xyz files in the given path.

    Args:
        ensemble (dict): Dictionary of models.
        ensemble_ase_sets (dict): Respective ASE style datasets as a dictionary.
        path (str): _description_
        initial (bool, optional): _description_. Defaults to False.
    """
    for tag in ensemble.keys():
        if initial:
            write(path/"training"/f"initial_train_set_{tag}.xyz", ensemble_ase_sets[tag]['train'])
            write(path/"validation"/f"initial_valid_set_{tag}.xyz", ensemble_ase_sets[tag]['valid'])
        else:
            write(path/"training"/f"train_set_{tag}.xyz", ensemble_ase_sets[tag]['train'])
            write(path/"validation"/f"valid_set_{tag}.xyz", ensemble_ase_sets[tag]['valid'])

def load_ensemble_sets_from_folder(
    ensemble: dict,
    path_to_folder: Path,
) -> dict:
    """
    Load ensemble datasets from a folder.

    Args:
        ensemble (dict): Dictionary of models.
        path_to_folder (Path): Path to the folder containing the datasets.

    Returns:
        dict: Dictionary of ASE style ensemble datasets.
    """

    assert os.path.exists(path_to_folder)
    assert os.path.exists(Path(path_to_folder / "training"))
    assert os.path.exists(Path(path_to_folder / "validation"))
    assert os.listdir(Path(path_to_folder / "training"))
    assert os.listdir(Path(path_to_folder / "validation"))

    training_sets = os.listdir(Path(path_to_folder / "training"))
    validation_sets = os.listdir(Path(path_to_folder / "validation"))

    ensemble_ase_sets = {tag : {'train': [], 'valid': []} for tag in ensemble.keys()}
    for tag in ensemble.keys():
        for training_set, validation_set in zip(training_sets, validation_sets):
            if tag in training_set:
                ensemble_ase_sets[tag]['train'] = (
                    read(Path(path_to_folder / "training" / training_set), index=":")
                )
            if tag in validation_set:
                ensemble_ase_sets[tag]['valid'] = (
                    read(Path(path_to_folder/"validation"/validation_set), index=":")
                )
    return ensemble_ase_sets

def Z_from_geometry_in(
        path_to_geometry: str = "geometry.in"
        ) -> list:
    """
    Extract atomic numbers from a aims geometry file.

    Args:
        path_to_geometry (str, optional): Path to the geometry file. Defaults to "geometry.in".

    Returns:
        list: List of atomic numbers (no unique).
    """
    with open(path_to_geometry, "r") as file:
        lines = file.readlines()
    Z = []
    for line in lines:
        if "atom" in line and "#" not in line:
            species = line.split()[4]
            atomic_number = ase.data.atomic_numbers[species]
            Z.append(atomic_number)
    return Z

def list_files_in_directory(
        directory_path: str
        ) -> list:
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

def atoms_full_copy(
    atoms: ase.Atoms
    ):
    atoms_copy = ase.Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc(),
        tags=atoms.get_tags(),
        momenta=atoms.get_momenta(),
        masses=atoms.get_masses(),
        magmoms=atoms.get_initial_magnetic_moments(),
        charges=atoms.get_initial_charges(),
        constraint=atoms.constraints,
        calculator=atoms.get_calculator(),
        info=atoms.info,
        )
    return atoms_copy

def list_latest_file(
    directory: str
    )-> str:
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
    ensemble: dict,
    training_setups: dict,
    mace_settings: dict
):
    for tag, model in ensemble.items():
        param_context = (
            training_setups[tag]['ema'].average_parameters()
            if training_setups[tag]['ema'] is not None
            else nullcontext()
        )
        with param_context:
            torch.save(
                model,
                Path(
                    mace_settings["GENERAL"]["model_dir"]
                )
                / (tag + ".model"),
            )    


class ModifyMD:
    def __init__(
        self,
        settings: dict
        ):
        
        self.settings = settings
        if self.settings['type'] == "temp":
            self.temp_step = settings["temp_step"]
            self.mod_interval = settings["mod_interval"]
            
    def change_temp(
        self,
        driver
        ):
        driver.temp += units.kB * self.temp_step
    
    def __call__(
        self,
        driver,
        metric,
        idx = None
        ) -> Any:
        if self.settings['type'] == "temp":
            if metric in self.mod_interval:
                self.change_temp(driver)
                if idx is not None:
                    logging.info(f'Modyfing trajectory {idx}.')
                logging.info(f"Changed temperature by {self.temp_step} to {round(driver.temp / units.kB,1)} K.")
                return True
            else:
                return False

class AIMSControlParser:
    def __init__(self) -> None:
        
        self.string_patterns = {
            'xc': re.compile(r'^\s*(xc)\s+(\S+)', re.IGNORECASE),
            'spin': re.compile(r'^\s*(spin)\s+(\S+)', re.IGNORECASE),
            'communication_type': re.compile(r'^\s*(communication_type)\s+(\S+)', re.IGNORECASE),
            'density_update_method': re.compile(r'^\s*(density_update_method)\s+(\S+)', re.IGNORECASE),
            'KS_method': re.compile(r'^\s*(KS_method)\s+(\S+)', re.IGNORECASE),
            'mixer': re.compile(r'^\s*(mixer)\s+(\S+)', re.IGNORECASE),
            'output_level': re.compile(r'^\s*(output_level)\s+(\S+)', re.IGNORECASE),
            'packed_matrix_format': re.compile(r'^\s*(packed_matrix_format)\s+(\S+)', re.IGNORECASE),
            'relax_unit_cell': re.compile(r'^\s*(relax_unit_cell)\s+(\S+)', re.IGNORECASE),
            'restart': re.compile(r'^\s*(restart)\s+(\S+)', re.IGNORECASE),
            'restart_read_only': re.compile(r'^\s*(restart_read_only)\s+(\S+)', re.IGNORECASE),
            'restart_write_only': re.compile(r'^\s*(restart_write_only)\s+(\S+)', re.IGNORECASE),
            'total_energy_method': re.compile(r'^\s*(total_energy_method)\s+(\S+)', re.IGNORECASE),
            'qpe_calc': re.compile(r'^\s*(qpe_calc)\s+(\S+)', re.IGNORECASE),
            'species_dir': re.compile(r'^\s*(species_dir)\s+(\S+)', re.IGNORECASE),
            'run_command': re.compile(r'^\s*(run_command)\s+(\S+)', re.IGNORECASE),
            'plus_u': re.compile(r'^\s*(plus_u)\s+(\S+)', re.IGNORECASE),
        }

        self.bool_patterns = {
            'collect_eigenvectors': re.compile(r'^\s*(collect_eigenvectors)\s+(\S+)', re.IGNORECASE),
            'compute_forces': re.compile(r'^\s*(compute_forces)\s+(\S+)', re.IGNORECASE),
            'compute_kinetic': re.compile(r'^\s*(compute_kinetic)\s+(\S+)', re.IGNORECASE),
            'compute_numerical_stress': re.compile(r'^\s*(compute_numerical_stress)\s+(\S+)', re.IGNORECASE),
            'compute_analytical_stress': re.compile(r'^\s*(compute_analytical_stress)\s+(\S+)', re.IGNORECASE),
            'compute_heat_flux': re.compile(r'^\s*(compute_heat_flux)\s+(\S+)', re.IGNORECASE),
            'distributed_spline_storage': re.compile(r'^\s*(distributed_spline_storage)\s+(\S+)', re.IGNORECASE),
            'evaluate_work_function': re.compile(r'^\s*(evaluate_work_function)\s+(\S+)', re.IGNORECASE),
            'final_forces_cleaned': re.compile(r'^\s*(final_forces_cleaned)\s+(\S+)', re.IGNORECASE),
            'hessian_to_restart_geometry': re.compile(r'^\s*(hessian_to_restart_geometry)\s+(\S+)', re.IGNORECASE),
            'load_balancing': re.compile(r'^\s*(load_balancing)\s+(\S+)', re.IGNORECASE),
            'MD_clean_rotations': re.compile(r'^\s*(MD_clean_rotations)\s+(\S+)', re.IGNORECASE),
            'MD_restart': re.compile(r'^\s*(MD_restart)\s+(\S+)', re.IGNORECASE),
            'override_illconditioning': re.compile(r'^\s*(override_illconditioning)\s+(\S+)', re.IGNORECASE),
            'override_relativity': re.compile(r'^\s*(override_relativity)\s+(\S+)', re.IGNORECASE),
            'restart_relaxations': re.compile(r'^\s*(restart_relaxations)\s+(\S+)', re.IGNORECASE),
            'squeeze_memory': re.compile(r'^\s*(squeeze_memory)\s+(\S+)', re.IGNORECASE),
            'symmetry_reduced_k_grid': re.compile(r'^\s*(symmetry_reduced_k_grid)\s+(\S+)', re.IGNORECASE),
            'use_density_matrix': re.compile(r'^\s*(use_density_matrix)\s+(\S+)', re.IGNORECASE),
            'use_dipole_correction': re.compile(r'^\s*(use_dipole_correction)\s+(\S+)', re.IGNORECASE),
            'use_local_index': re.compile(r'^\s*(use_local_index)\s+(\S+)', re.IGNORECASE),
            'use_logsbt': re.compile(r'^\s*(use_logsbt)\s+(\S+)', re.IGNORECASE),
            'vdw_correction_hirshfeld': re.compile(r'^\s*(vdw_correction_hirshfeld)\s+(\S+)', re.IGNORECASE),
            'postprocess_anyway': re.compile(r'^\s*(postprocess_anyway)\s+(\S+)', re.IGNORECASE), 
        }

        self.float_patterns = {
            'charge': re.compile(r'^\s*(charge)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
            'charge_mix_param': re.compile(r'^\s*(charge_mix_param)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
            'default_initial_moment': re.compile(r'^\s*(default_initial_moment)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
            'fixed_spin_moment': re.compile(r'^\s*(fixed_spin_moment)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
            'hartree_convergence_parameter': re.compile(r'^\s*(hartree_convergence_parameter)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
            'harmonic_length_scale': re.compile(r'^\s*(harmonic_length_scale)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
            'ini_linear_mix_param': re.compile(r'^\s*(ini_linear_mix_param)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
            'ini_spin_mix_parma': re.compile(r'^\s*(ini_spin_mix_parma)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
            'initial_moment': re.compile(r'^\s*(initial_moment)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
            'MD_MB_init': re.compile(r'^\s*(MD_MB_init)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
            'MD_time_step': re.compile(r'^\s*(MD_time_step)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
            'prec_mix_param': re.compile(r'^\s*(prec_mix_param)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
            'set_vacuum_level': re.compile(r'^\s*(set_vacuum_level)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
            'spin_mix_param': re.compile(r'^\s*(spin_mix_param)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
        }

        self.exp_patterns = {
            'basis_threshold': re.compile(r'^\s*(basis_threshold)\s+([-+]?\d*\.?\d*([eEdD][-+]?\d+)?)', re.IGNORECASE),
            'occupation_thr': re.compile(r'^\s*(occupation_thr)\s+([-+]?\d*\.?\d*([eEdD][-+]?\d+)?)', re.IGNORECASE),
            'sc_accuracy_eev': re.compile(r'^\s*(sc_accuracy_eev)\s+([-+]?\d*\.?\d*([eEdD][-+]?\d+)?)', re.IGNORECASE),
            'sc_accuracy_etot': re.compile(r'^\s*(sc_accuracy_etot)\s+([-+]?\d*\.?\d*([eEdD][-+]?\d+)?)', re.IGNORECASE),
            'sc_accuracy_forces': re.compile(r'^\s*(sc_accuracy_forces)\s+([-+]?\d*\.?\d*([eEdD][-+]?\d+)?)', re.IGNORECASE),
            'sc_accuracy_rho': re.compile(r'^\s*(sc_accuracy_rho)\s+([-+]?\d*\.?\d*([eEdD][-+]?\d+)?)', re.IGNORECASE),
            'sc_accuracy_stress': re.compile(r'^\s*(sc_accuracy_stress)\s+([-+]?\d*\.?\d*([eEdD][-+]?\d+)?)', re.IGNORECASE),
        }

        self.int_patterns = {
            'empty_states': re.compile(r'^\s*(empty_states)\s+(\d+)', re.IGNORECASE),
            'ini_linear_mixing': re.compile(r'^\s*(ini_linear_mixing)\s+(\d+)', re.IGNORECASE),
            'max_relaxation_steps': re.compile(r'^\s*(max_relaxation_steps)\s+(\d+)', re.IGNORECASE),
            'max_zeroin': re.compile(r'^\s*(max_zeroin)\s+(\d+)', re.IGNORECASE),
            'multiplicity': re.compile(r'^\s*(multiplicity)\s+(\d+)', re.IGNORECASE),
            'n_max_pulay': re.compile(r'^\s*(n_max_pulay)\s+(\d+)', re.IGNORECASE),
            'sc_iter_limit': re.compile(r'^\s*(sc_iter_limit)\s+(\d+)', re.IGNORECASE),
            'walltime': re.compile(r'^\s*(walltime)\s+(\d+)', re.IGNORECASE)
        }
        # TH: some of them seem unnecessary for our purposes and are complicated
        #     to put into regex which is why i commented them out
        self.list_patterns = {
            #'init_hess',
            'k_grid': re.compile(r'^\s*(k_grid)\s+(\d+)\s+(\d+)\s+(\d+)', re.IGNORECASE),
            'k_offset': re.compile(r'^\s*(k_offset)\s+(\d*\.?\d+)\s+(\d*\.?\d+)\s+(\d*\.?\d+)', re.IGNORECASE),
            #'MD_run',
            #'MD_schedule',
            #'MD_segment',
            #'mixer_threshold',
            'occupation_type': re.compile(r'^\s*(occupation_type)\s+(\S+)\s+(\d*\.?\d+)(?:\s+(\d+))?', re.IGNORECASE),
            #'output',
            #'cube',
            #'preconditioner',
            'relativistic':re.compile(r'^\s*(relativistic)\s+(\S+)\s+(\S+)(?:\s+(\d+))?', re.IGNORECASE),
            #'relax_geometry',
        }

        self.special_patterns = {
            #'many_body_dispersion': re.compile(r'^\s*(many_body_dispersion)\s', re.IGNORECASE)
            'many_body_dispersion': re.compile(r"""
                ^\s*                                               
                (many_body_dispersion)\b                              
                (?:                                                
                    \s+beta=(?P<beta>-?\d+(\.\d+)?)                
                    |\s+k_grid=(?P<k_grid>\d+:\d+:\d+)             
                    |\s+freq_grid=(?P<freq_grid>\d+)               
                    |\s+self_consistent=(?P<self_consistent>\.true\.|\.false\.) 
                    |\s+vdw_params_kind=(?P<vdw_params_kind>[^\s]+)
                )*                                                 
            """, re.IGNORECASE | re.VERBOSE),                         
            'many_body_dispersion_nl': re.compile(r"""
                ^\s*                                               
                (many_body_dispersion_nl)\b                               
                (?:                                                
                    \s+beta=(?P<beta>-?\d+(\.\d+)?)                
                    |\s+k_grid=(?P<k_grid>\d+:\d+:\d+)             
                    |\s+freq_grid=(?P<freq_grid>\d+)               
                    |\s+self_consistent=(?P<self_consistent>\.true\.|\.false\.) 
                    |\s+vdw_params_kind=(?P<vdw_params_kind>[^\s]+)
                )*                                                 
            """, re.IGNORECASE | re.VERBOSE)
        }

    def f90_bool_to_py_bool(
        self,
        f90_bool:str
        )-> bool:
        
        if f90_bool.lower() == '.true.':
            return True
        elif f90_bool.lower() == '.false.':
            return False

    def __call__(
        self,
        path_to_control: str,
        ) -> dict:
        aims_settings = {}
        with open(path_to_control, 'r') as input_file:
            for line in input_file:
                
                if '#' in line:
                    line = line.split('#')[0]
                
                for key, pattern in self.string_patterns.items():
                    match = pattern.match(line)
                    if match:
                        aims_settings[match.group(1)] = match.group(2)
                        
                for key, pattern in self.bool_patterns.items():
                    match = pattern.match(line)
                    if match:
                        aims_settings[match.group(1)] = self.f90_bool_to_py_bool(match.group(2))
                        
                for key, pattern in self.float_patterns.items():
                    match = pattern.match(line)
                    if match:
                        aims_settings[match.group(1)] = float(match.group(2))
                        
                for key, pattern in self.exp_patterns.items():
                    match = pattern.match(line)
                    if match:
                        matched_value = match.group(2).replace('d', 'e').replace('D', 'E')
                        aims_settings[match.group(1)] = float(matched_value)
                        
                for key, pattern in self.int_patterns.items():
                    match = pattern.match(line)
                    if match:
                        aims_settings[match.group(1)] = int(match.group(2))
                        
                for key, pattern in self.list_patterns.items():
                    match = pattern.match(line)
                    if match:
                        if key == 'k_grid':
                            aims_settings[match.group(1)] = [int(match.group(2)), int(match.group(3)), int(match.group(4))]
                        if key == 'k_offset':
                            aims_settings[match.group(1)] = [float(match.group(2)), float(match.group(3)), float(match.group(4))]
                        if key == 'occupation_type':
                            if match.group(4) is not None:
                                aims_settings[match.group(1)] = [match.group(2), float(match.group(3)), int(match.group(4))]
                            else:
                                aims_settings[match.group(1)] = [match.group(2), float(match.group(3))]
                        if key == 'relativistic':
                            if match.group(4) is not None:
                                aims_settings[match.group(1)] = [match.group(2), match.group(3), int(match.group(4))]
                            else:
                                aims_settings[match.group(1)] = [match.group(2), match.group(3)]
                                
                for key, pattern in self.special_patterns.items():
                    match = pattern.match(line)
                    if match:
                        if key == 'many_body_dispersion': 
                            if any(match.groupdict().values()):
                                # If parameters are found, store them in a dictionary
                                aims_settings[key] = ''
                                for param, value in match.groupdict().items():
                                    if value is not None:
                                        aims_settings[key] += f'{param}={value} '
                            else:
                                # If no parameters are found, store an empty string
                                aims_settings[key] = ''
                                
                        if key == 'many_body_dispersion_nl': 
                            if any(match.groupdict().values()):
                                # If parameters are found, store them in a dictionary
                                aims_settings[key] = ''
                                for param, value in match.groupdict().items():
                                    if value is not None:
                                        aims_settings[key] += f'{param}={value} '
                            else:
                                # If no parameters are found, store an empty string
                                aims_settings[key] = ''
        return aims_settings

dtype_mapping = {
    'float32': torch.float32,
    'float64': torch.float64,
    'float16': torch.float16,
    'int32': torch.int32,
    'int64': torch.int64,
    'int16': torch.int16,
    'int8': torch.int8,
    'uint8': torch.uint8,
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
        while not self.stopped:
            # Get GPU utilization data
            gpus = GPUtil.getGPUs()
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            
            for gpu in gpus:
                self.data.append({
                    'Time': timestamp,
                    'GPU ID': gpu.id,
                    'Utilization (%)': gpu.load * 100,
                    'Memory Utilization (%)': gpu.memoryUtil * 100
                })
                
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        # Save collected data to a CSV file using pandas
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_file, index=False)

