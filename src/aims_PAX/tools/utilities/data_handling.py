import random
import logging
import os
import numpy as np
from typing import (
    Optional,
    Sequence,
    Tuple,
    Dict,
    List,
)
from pathlib import Path
from ase.io import write
from mace import data as mace_data
from mace import tools
from mace.tools import AtomicNumberTable, torch_geometric
from mace.data.utils import (
    config_from_atoms_list,
    #KeySpecification,
)
from dataclasses import dataclass
from ase.io import read
import dataclasses


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
            "Since ASE version 3.23.0b1, using energy_key 'energy' is"
            "no longer safe when communicating between MACE and ASE. We"
            "recommend using a different key, rewriting 'energy' to "
            "'REF_energy'. You need to use --energy_key='REF_energy'"
            "to specify the chosen key name."
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
            "Since ASE version 3.23.0b1, using forces_key 'forces' is"
            "no longer safe when communicating between MACE and ASE. "
            "We recommend using a different key, rewriting 'forces' to"
            "'REF_forces'. You need to use --forces_key='REF_forces' to"
            "specify the chosen key name."
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
            "Since ASE version 3.23.0b1, using stress_key 'stress' is "
            "no longer safe when communicating between MACE and ASE."
            "We recommend using a different key, rewriting 'stress' to"
            "'REF_stress'. You need to use --stress_key='REF_stress' to"
            "specify the chosen key name."
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
                len(atoms) == 1
                and atoms.info.get("config_type") == "IsolatedAtom"
            )
            if isolated_atom_config:
                if energy_key in atoms.info.keys():
                    atomic_energies_dict[
                        atoms.get_atomic_numbers()[0]
                    ] = (
                        atoms.info[energy_key]
                    )
                else:
                    logging.warning(
                        f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                        "but does not contain an energy. Zero energy "
                        "will be used."
                    )
                    atomic_energies_dict[
                        atoms.get_atomic_numbers()[0]
                    ] = (
                        np.zeros(1)
                    )
            else:
                atoms_without_iso_atoms.append(atoms)

        if len(atomic_energies_dict) > 0:
            logging.info("Using isolated atom energies from training file")
        if not keep_isolated_atoms:
            atoms_list = atoms_without_iso_atoms

    for atoms in atoms_list:
        atoms.info[head_key] = head_name

    #key_spec = KeySpecification(
    #    info_keys={
    #        "energy_key": energy_key,
    #        "stress_key": stress_key,
    #        "virials_key": virials_key,
    #        "dipole_key": dipole_key,
    #        "head_key": head_key,
    #    },
    #    arrays_keys={
    #        "forces_key": forces_key,
    #        "charges_key": charges_key,
    #    },
    #)
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


def split_data(
    data: list,
    valid_fraction: float,
) -> tuple:
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
) -> tuple:
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
) -> list:
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

    data_set = [
        mace_data.AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=r_max
        )
        for config in collections.train
    ]
    return data_set


def update_mace_set(
    new_train_data,
    new_valid_data,
    mace_set: dict,
    z_table: tools.AtomicNumberTable,
    seed: int,
    r_max: float,
) -> dict:
    """
    Update the MACE dataset with new data.
    Currently needs valid and training data.

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
        data=new_train_data, z_table=z_table, seed=seed, r_max=r_max
    )

    new_valid_set = create_mace_dataset(
        data=new_valid_data, z_table=z_table, seed=seed, r_max=r_max
    )
    mace_set["train"] += new_train_set
    mace_set["valid"] += new_valid_set

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
    ase_set["train"] += new_train_data
    ase_set["valid"] += new_valid_data

    mace_set = update_mace_set(
        new_train_data, new_valid_data, mace_set, z_table, seed, r_max
    )

    return ase_set, mace_set


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
    ensemble_mace_sets = {
        tag: {"train": [], "valid": []} for tag in ensemble_ase_sets.keys()
    }
    for tag in ensemble_ase_sets.keys():
        ensemble_mace_sets[tag]["train"] = create_mace_dataset(
            data=ensemble_ase_sets[tag]["train"],
            z_table=z_table,
            seed=seed,
            r_max=r_max,
        )
        ensemble_mace_sets[tag]["valid"] = create_mace_dataset(
            data=ensemble_ase_sets[tag]["valid"],
            z_table=z_table,
            seed=seed,
            r_max=r_max,
        )
    return ensemble_mace_sets


def save_datasets(
    ensemble: dict,
    ensemble_ase_sets: dict,
    path: str, 
    initial: bool = False,
    save_combined_initial: bool = True
):
    """
    Save the ensemble datasets as xyz files in the given path.
    The datasets are saved in the "training" and "validation" subdirectories.
    
    If `initial` is True, the datasets are saved with the prefix "initial_".
   
    If `save_combined_initial` is True, a combined initial training
    and validation set is saved as "combined_initial_train_set.xyz" and
    "combined_initial_valid_set.xyz".

    Args:
        ensemble (dict): Dictionary of models.
        ensemble_ase_sets (dict): Respective ASE style datasets as a 
                                    dictionary.
        path (str): _description_
        initial (bool, optional): _description_. Defaults to False.
        save_combined_initial (bool, optional): If True, saves a combined
            initial training and validation set. Defaults to True.
    """
    
    if save_combined_initial:
        combined_init_train_set = []
        combined_init_valid_set = []
    for tag in ensemble.keys():
        if initial:
            write(
                path / "training" / f"initial_train_set_{tag}.xyz",
                ensemble_ase_sets[tag]["train"],
            )
            write(
                path / "validation" / f"initial_valid_set_{tag}.xyz",
                ensemble_ase_sets[tag]["valid"],
            )
            if save_combined_initial:
                combined_init_train_set += ensemble_ase_sets[tag]["train"]
                combined_init_valid_set += ensemble_ase_sets[tag]["valid"]
        else:
            write(
                path / "training" / f"train_set_{tag}.xyz",
                ensemble_ase_sets[tag]["train"],
            )
            write(
                path / "validation" / f"valid_set_{tag}.xyz",
                ensemble_ase_sets[tag]["valid"],
            )
    if save_combined_initial and initial:
        write(
            path / "training" / "combined_initial_train_set.xyz",
            combined_init_train_set,
        )
        write(
            path / "validation" / "combined_initial_valid_set.xyz",
            combined_init_valid_set,
        )


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

    ensemble_ase_sets = {
        tag: {"train": [], "valid": []} for tag in ensemble.keys()
    }
    for tag in ensemble.keys():
        for training_set, validation_set in zip(
            training_sets, validation_sets
        ):
            if tag in training_set:
                ensemble_ase_sets[tag]["train"] = read(
                    Path(path_to_folder / "training" / training_set), index=":"
                )
            if tag in validation_set:
                ensemble_ase_sets[tag]["valid"] = read(
                    Path(path_to_folder / "validation" / validation_set),
                    index=":",
                )
    return ensemble_ase_sets
