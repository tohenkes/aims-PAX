import random
import logging
import os
import torch
import numpy as np
from typing import Optional, Sequence, Tuple, List, Dict
from pathlib import Path
from ase.io import write
from mace import data as mace_data
from mace import tools, modules
from mace.tools import AtomicNumberTable, torch_geometric, torch_tools, utils
from ase.io import read
from FHI_AL.setup_MACE import setup_mace
from FHI_AL.setup_MACE_training import setup_mace_training
from dataclasses import dataclass
import ase.data
import ase.io
import dataclasses
from mace.tools import AtomicNumberTable
from mace.tools.compile import prepare
from mace.tools.scripts_utils import extract_load
from mpi4py import MPI
from asi4py.asecalc import ASI_ASE_calculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units


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
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipoles",
    charges_key: str = "charges",
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
        extract_atomic_energies=True,
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
            extract_atomic_energies=False,
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
            extract_atomic_energies=False,
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


def config_from_atoms_list(
    atoms_list: List[ase.Atoms],
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole",
    charges_key="charges",
    config_type_weights: Dict[str, float] = None,
) -> Configurations:
    """Convert list of ase.Atoms into Configurations"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS
    all_configs = []
    for atoms in atoms_list:
        all_configs.append(
            config_from_atoms(
                atoms,
                energy_key=energy_key,
                forces_key=forces_key,
                stress_key=stress_key,
                virials_key=virials_key,
                dipole_key=dipole_key,
                charges_key=charges_key,
                config_type_weights=config_type_weights,
            )
        )
    return all_configs


def config_from_atoms(
    atoms: ase.Atoms,
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole",
    charges_key="charges",
    config_type_weights: Dict[str, float] = None,
) -> Configuration:
    """Convert ase.Atoms to Configuration"""

    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    energy = atoms.info.get(energy_key, None)  # eV
    forces = atoms.arrays.get(forces_key, None)  # eV / Ang
    stress = atoms.info.get(stress_key, None)  # eV / Ang
    virials = atoms.info.get(virials_key, None)
    dipole = atoms.info.get(dipole_key, None)  # Debye
    # Charges default to 0 instead of None if not found
    charges = atoms.arrays.get(
        charges_key, np.zeros(len(atoms))
    )  # atomic unit
    atomic_numbers = np.array(
        [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
    )
    pbc = tuple(atoms.get_pbc())
    cell = np.array(atoms.get_cell())
    config_type = atoms.info.get("config_type", "Default")
    weight = atoms.info.get("config_weight", 1.0) * config_type_weights.get(
        config_type, 1.0
    )
    energy_weight = atoms.info.get("config_energy_weight", 1.0)
    forces_weight = atoms.info.get("config_forces_weight", 1.0)
    stress_weight = atoms.info.get("config_stress_weight", 1.0)
    virials_weight = atoms.info.get("config_virials_weight", 1.0)
    atomic_weights = atoms.info.get("atomic_weights", None)

    # fill in missing quantities but set their weight to 0.0
    if energy is None:
        energy = 0.0
        energy_weight = 0.0
    if forces is None:
        forces = np.zeros(np.shape(atoms.positions))
        forces_weight = 0.0
    if stress is None:
        stress = np.zeros(6)
        stress_weight = 0.0
    if virials is None:
        virials = np.zeros((3, 3))
        virials_weight = 0.0

    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        energy=energy,
        forces=forces,
        stress=stress,
        virials=virials,
        dipole=dipole,
        charges=charges,
        weight=weight,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
        atomic_weights=atomic_weights,
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


def load_from_xyz(
    file_path: str,
    config_type_weights: Dict,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "charges",
    extract_atomic_energies: bool = False,
) -> Tuple[Dict[int, float], Configurations]:
    atoms_list = ase.io.read(file_path, index=":")

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    atomic_energies_dict = {}
    if extract_atomic_energies:
        atoms_without_iso_atoms = []

        for idx, atoms in enumerate(atoms_list):
            if len(atoms) == 1 and atoms.info["config_type"] == "IsolatedAtom":
                if energy_key in atoms.info.keys():
                    atomic_energies_dict[
                        atoms.get_atomic_numbers()[0]
                    ] = atoms.info[energy_key]
                else:
                    logging.warning(
                        f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                        "but does not contain an energy."
                    )
            else:
                atoms_without_iso_atoms.append(atoms)

        if len(atomic_energies_dict) > 0:
            logging.info("Using isolated atom energies from training file")

        atoms_list = atoms_without_iso_atoms

    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
    )
    return atomic_energies_dict, configs


def load_from_atoms(
    atoms_list: list,
    config_type_weights: Dict,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "charges",
    extract_atomic_energies: bool = False,
) -> Tuple[Dict[int, float], Configurations]:
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    
    atomic_energies_dict = {}
    if extract_atomic_energies:
        atoms_without_iso_atoms = []

        for idx, atoms in enumerate(atoms_list):
            if len(atoms) == 1 and atoms.info["config_type"] == "IsolatedAtom":
                if energy_key in atoms.info.keys():
                    atomic_energies_dict[
                        atoms.get_atomic_numbers()[0]
                    ] = atoms.info[energy_key]
                else:
                    logging.warning(
                        f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                        "but does not contain an energy."
                    )
            else:
                atoms_without_iso_atoms.append(atoms)

        if len(atomic_energies_dict) > 0:
            logging.info("Using isolated atom energies from training file")

        atoms_list = atoms_without_iso_atoms
    
    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
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
    prediction: np.array, 
    return_argmax: bool = False,
) -> np.array:
    """
    Compute the maximum standard deviation of the ensemble prediction.

    Args:
        prediction (np.array): Ensemble prediction of forces: [n_ensemble_members, n_mols, n_atoms, xyz].

    Returns:
        np.array: Maximum standard deviation of atomic forces per molecule: [n_mols].
    """
    # average prediction over ensemble of models
    pred_av = np.average(prediction, axis=0, keepdims=True)
    diff_sq = (prediction - pred_av) ** 2.
    diff_sq_mean = np.mean(diff_sq, axis=(0,-1))
    max_sd = np.max(np.sqrt(diff_sq_mean),axis=-1)
    if return_argmax:
        return max_sd, np.argmax(np.sqrt(diff_sq_mean),axis=-1)
    else:
        return max_sd
    

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


#def split_to_member_sets(
#    
#):
#    member_sets = {}
#    return member_sets

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
    

def ensemble_training_setups(ensemble, mace_settings):
    training_setups = {}
    for tag, model in ensemble.items():
        training_setups[tag] = setup_mace_training(
            settings=mace_settings,
            model=model,
            tag=tag,
        )
    return training_setups

def setup_ensemble_dicts(
    seeds: np.array,
    mace_settings: dict,
    al_settings: dict,
    atomic_energies_dict: dict,
    save_seeds_tags_dict: str = "seeds_tags_dict.npz",
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
    
    z_table = tools.get_atomic_number_table_from_zs(
        z for z in atomic_energies_dict.keys()
    )
    seeds_tags_dict = {}
    ensemble = {}

    for seed in seeds:
        mace_settings['GENERAL']["seed"] = seed
        tag = mace_settings['GENERAL']["name_exp"] + "-" + str(seed)
        seeds_tags_dict[tag] = seed
        ensemble[tag] =  setup_mace(
                settings=mace_settings,
                z_table=z_table,
                atomic_energies_dict=atomic_energies_dict,
                )

    training_setups = ensemble_training_setups(ensemble, mace_settings)

    if save_seeds_tags_dict:
        np.savez(al_settings["dataset_dir"]+ '/' + save_seeds_tags_dict, **seeds_tags_dict)
        
    return (seeds_tags_dict, ensemble, training_setups)


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


def update_avg_neighs_shifts_scale(
    model: modules.MACE,
    train_loader,
    atomic_energies: dict,
    scaling: str,
) -> None:
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
    average_neighbors = modules.compute_avg_num_neighbors(train_loader)
    for interaction_idx in range(len(model.interactions)):
        model.interactions[interaction_idx].avg_num_neighbors = average_neighbors
    mean, std = modules.scaling_classes[scaling](train_loader, atomic_energies)
    model.scale_shift = modules.blocks.ScaleShiftBlock(
        scale=std, shift=mean
    )
    return None

                    
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

    training_sets = os.listdir(Path(path_to_folder+"/training"))
    validation_sets = os.listdir(Path(path_to_folder+"/validation"))

    ensemble_ase_sets = {tag : {'train': [], 'valid': []} for tag in ensemble.keys()}
    for tag in ensemble.keys():
        for training_set, validation_set in zip(training_sets, validation_sets):
            if tag in training_set:
                ensemble_ase_sets[tag]['train'] = (
                    read(Path(path_to_folder+"/training/"+training_set), index=":")
                )
            if tag in validation_set:
                ensemble_ase_sets[tag]['valid'] = (
                    read(Path(path_to_folder+"/validation/"+validation_set), index=":")
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