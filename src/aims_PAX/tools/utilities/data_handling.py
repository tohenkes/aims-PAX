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
from so3krates_torch.data.atomic_data import AtomicData as so3_data
from so3krates_torch.tools import torch_geometric as so3_torch_geometric
from mace import data as mace_data
from mace import tools
from mace.tools import AtomicNumberTable, torch_geometric, DefaultKeys
from mace.data.utils import (
    config_from_atoms_list,
)
from dataclasses import dataclass, field
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


@dataclass
class KeySpecification:
    info_keys: Dict[str, str] = field(default_factory=dict)
    arrays_keys: Dict[str, str] = field(default_factory=dict)

    def update(
        self,
        info_keys: Optional[Dict[str, str]] = None,
        arrays_keys: Optional[Dict[str, str]] = None,
    ):
        if info_keys is not None:
            self.info_keys.update(info_keys)
        if arrays_keys is not None:
            self.arrays_keys.update(arrays_keys)
        return self

    @classmethod
    def from_defaults(cls):
        instance = cls()
        return update_keyspec_from_kwargs(instance, DefaultKeys.keydict())


def update_keyspec_from_kwargs(
    keyspec: KeySpecification, keydict: Dict[str, str]
) -> KeySpecification:
    # convert command line style property_key arguments into a keyspec
    infos = [
        "energy_key",
        "stress_key",
        "virials_key",
        "dipole_key",
        "head_key",
        "elec_temp_key",
        "total_charge_key",
        "polarizability_key",
        "total_spin_key",
    ]
    arrays = ["forces_key", "charges_key"]
    info_keys = {}
    arrays_keys = {}
    for key in infos:
        if key in keydict:
            info_keys[key[:-4]] = keydict[key]
    for key in arrays:
        if key in keydict:
            arrays_keys[key[:-4]] = keydict[key]
    keyspec.update(info_keys=info_keys, arrays_keys=arrays_keys)
    return keyspec



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
    key_specification: KeySpecification = None,
    head_name: str = "Default",
    return_ase_list: bool = False,
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    """Load training and test dataset from xyz file"""

    atomic_energies_dict, all_train_configs = load_from_atoms(
        atoms_list=train_list,
        config_type_weights=config_type_weights,
        key_specification=key_specification,
        extract_atomic_energies=True,
        keep_isolated_atoms=keep_isolated_atoms,
        head_name=head_name,
    )

    if valid_list is not None:
        _, valid_configs = load_from_atoms(
            atoms_list=valid_list,
            config_type_weights=config_type_weights,
            key_specification=key_specification,
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
            key_specification=key_specification,
            extract_atomic_energies=False,
            head_name=head_name,
        )
        # create list of tuples (config_type, list(Atoms))
        test_configs = mace_data.test_config_types(all_test_configs)

    if return_ase_list:
        return (
            SubsetCollection(
                train=train_configs, valid=valid_configs, tests=test_configs
            ),
            atomic_energies_dict,
            {
                "train": train_list,
                "valid": valid_list,
                "tests": test_list,
            }
        )
    else:
        return (
            SubsetCollection(
                train=train_configs, valid=valid_configs, tests=test_configs
            ),
            atomic_energies_dict,
    )


def get_single_dataset_from_atoms(
    atoms_list: list,
    config_type_weights: Dict,
    key_specification: KeySpecification
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    """Load training and test dataset from xyz file"""

    atomic_energies_dict, all_configs = load_from_atoms(
        atoms_list=atoms_list,
        config_type_weights=config_type_weights,
        key_specification=key_specification,
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
    key_specification: KeySpecification,
    head_name: str = "Default",
    config_type_weights: Optional[Dict] = None,
    extract_atomic_energies: bool = False,
    keep_isolated_atoms: bool = False,
) -> Tuple[Dict[int, float], Configurations]:
    
    energy_key = key_specification.info_keys["energy"]
    forces_key = key_specification.arrays_keys["forces"]
    stress_key = key_specification.info_keys["stress"]
    head_key = key_specification.info_keys["head"]
    original_energy_key = energy_key
    original_forces_key = forces_key
    original_stress_key = stress_key
    if energy_key == "energy":
        logging.warning(
            "Since ASE version 3.23.0b1, using energy_key 'energy' is no "
            "longer safe when communicating between MACE and ASE. We recommend "
            "using a different key, rewriting 'energy' to 'REF_energy'. You "
            "need to use --energy_key='REF_energy' to specify the chosen key "
            "name."
        )
        key_specification.info_keys["energy"] = "REF_energy"
        for atoms in atoms_list:
            try:
                # print("OK")
                atoms.info["REF_energy"] = atoms.get_potential_energy()
                # print("atoms.info['REF_energy']:", atoms.info["REF_energy"])
            except Exception as e:  # pylint: disable=W0703
                logging.error(f"Failed to extract energy: {e}")
                atoms.info["REF_energy"] = None
    if forces_key == "forces":
        logging.warning(
            "Since ASE version 3.23.0b1, using forces_key 'forces' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'forces' to 'REF_forces'. You need to use --forces_key='REF_forces' to specify the chosen key name."
        )
        key_specification.arrays_keys["forces"] = "REF_forces"
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
        key_specification.info_keys["stress"] = "REF_stress"
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
            atoms.info[head_key] = head_name
            isolated_atom_config = (
                len(atoms) == 1 and atoms.info.get("config_type") == "IsolatedAtom"
            )
            if isolated_atom_config:
                atomic_number = int(atoms.get_atomic_numbers()[0])
                if energy_key in atoms.info.keys():
                    atomic_energies_dict[atomic_number] = float(atoms.info[energy_key])
                else:
                    logging.warning(
                        f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                        "but does not contain an energy. Zero energy will be used."
                    )
                    atomic_energies_dict[atomic_number] = 0.0
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
        key_specification=key_specification,
        head_name=head_name,
    )
    key_specification.info_keys["energy"] = original_energy_key
    key_specification.arrays_keys["forces"] = original_forces_key
    key_specification.info_keys["stress"] = original_stress_key
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
    valid_set: dict,
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

    train_loader = so3_torch_geometric.dataloader.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
    )

    valid_loaders = {}
    for head, valid_subset in valid_set.items():
        valid_loader = so3_torch_geometric.dataloader.DataLoader(
            dataset=valid_subset,
            batch_size=valid_batch_size,
            shuffle=False,
            drop_last=False,
        )
        valid_loaders[head] = valid_loader
        
    return train_loader, valid_loaders


def create_model_dataset(
    data: list,
    seed: int,
    z_table: AtomicNumberTable,
    r_max: float,
    key_specification: KeySpecification,
    r_max_lr: Optional[float] = None,
    all_heads: Optional[List[str]] = None,
    head_name: Optional[str] = "Default",
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
        key_specification=key_specification,
        head_name=head_name,
    )

    data_set = [
        so3_data.from_config(
            config,
            z_table=z_table, 
            cutoff=r_max,
            cutoff_lr=r_max_lr,
            heads=all_heads
            )
        for config in collections.train
    ]
    return data_set
 

def update_model_set(
    new_train_data,
    new_valid_data,
    model_set: dict,
    z_table: tools.AtomicNumberTable,
    seed: int,
    r_max: float,
    key_specification: KeySpecification,
    r_max_lr: Optional[float] = None,
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
    new_train_set = create_model_dataset(
        data=new_train_data,
        z_table=z_table,
        seed=seed, 
        r_max=r_max,
        r_max_lr=r_max_lr,
        key_specification=key_specification
    )

    new_valid_set = create_model_dataset(
        data=new_valid_data,
        z_table=z_table,
        seed=seed,
        r_max=r_max,
        r_max_lr=r_max_lr,
        key_specification=key_specification
    )
    model_set["train"] += new_train_set
    model_set["valid"]["Default"] += new_valid_set

    return model_set


def split_data_heads_evenly(
    data: list,
    num_heads: int,
) -> Dict[int, list]:
    """
    Split data evenly between heads. If there are not enough
    points, so that each head doesn't get at least one point,
    it re-iterates across the dataset until each head has at
    least one point.

    Args:
        data (list): List of data points to be split.
        num_heads (int): Number of heads to split the data into.

    Returns:
        Dict[int, list]: Dictionary mapping head index to list of data points.
    """
    head_data = {head: [] for head in range(num_heads)}
    n_points = len(data)
    if n_points >= num_heads:
        for idx, point in enumerate(data):
            head = idx % num_heads
            head_data[head].append(point.copy())
    else:
        idx = 0
        while any(len(points) == 0 for points in head_data.values()):
            head = idx % num_heads
            head_data[head].append(data[idx % n_points].copy())
            idx += 1
        
    return head_data


def update_datasets(
    new_points: list,
    model_set: dict,
    ase_set: dict,
    valid_split: float,
    z_table: tools.AtomicNumberTable,
    seed: int,
    r_max: float,
    key_specification: KeySpecification,
    r_max_lr: Optional[float] = None,
    all_heads: Optional[List[str]] = None,
) -> tuple:
    """
    Update the ASE and MACE/SO3LR datasets with new data.

    Args:
        new_points (list): List of ASE atoms objects.
        model_set (dict): Dictionary of MACE/SO3LR style training and validation data.
        ase_set (dict): Dictionary of ASE style training and validation data.
        valid_split (float): Fraction of data to be used for validation.
        z_table (tools.AtomicNumberTable): Table of elements.
        seed (int): Seed for shuffling and splitting the dataset.
        r_max (float): Cut-off radius.

    Returns:
        tuple: _description_
    """
    
    new_train_data_ase, new_valid_data_ase = split_data(new_points, valid_split)
    
    if all_heads is not None:
        new_train_data_head_ase = split_data_heads_evenly(new_train_data_ase, len(all_heads))
        new_valid_data_head_ase = split_data_heads_evenly(new_valid_data_ase, len(all_heads))
    
        train_collections = []
        valid_collections = {}
        new_train_data_ase_temp = []
        new_valid_data_ase_temp = []
        for head_idx, head_name in enumerate(all_heads):
            collections, _, ase_sets = get_dataset_from_atoms(
                train_list=new_train_data_head_ase[head_idx],
                valid_list=new_valid_data_head_ase[head_idx],
                seed=seed,
                config_type_weights=None,
                key_specification=key_specification,
                head_name=head_name,
                return_ase_list=True,
            )
            
            train_collections += collections.train
            valid_collections[head_name] = collections.valid
            
            new_train_data_ase_temp += ase_sets["train"]
            new_valid_data_ase_temp += ase_sets["valid"]
            
        new_train_data_ase = new_train_data_ase_temp
        new_valid_data_ase = new_valid_data_ase_temp
        
        collections.train = train_collections
            
        new_train_data_ase_head = [
            so3_data.from_config(
                config,
                z_table=z_table, 
                cutoff=r_max,
                cutoff_lr=r_max_lr,
                heads=all_heads
                )
            for config in collections.train
        ]
        model_set["train"] += new_train_data_ase_head

        for head in all_heads:
            new_valid_data_head = [
                so3_data.from_config(
                    config,
                    z_table=z_table,
                    cutoff=r_max,
                    cutoff_lr=r_max_lr,
                    heads=all_heads
                    ) 
                for config in valid_collections[head]
            ]
            model_set["valid"][head] += new_valid_data_head

    else:
        model_set = update_model_set(
            new_train_data=new_train_data_ase, 
            new_valid_data=new_valid_data_ase,
            model_set=model_set,
            z_table=z_table, 
            seed=seed, 
            r_max=r_max,
            r_max_lr=r_max_lr,
            key_specification=key_specification
        )

    ase_set["train"] += new_train_data_ase
    ase_set["valid"] += new_valid_data_ase
    return ase_set, model_set


def sort_ase_dataset_to_heads(
    ase_set: dict,
):
    head_data = {}
    for atoms in ase_set:
        head_name = atoms.info.get("head", "Default")
        if head_name not in head_data:
            head_data[head_name] = []
        head_data[head_name].append(atoms)
    return head_data
      

def ase_to_model_ensemble_sets(
    ensemble_ase_sets: dict,
    z_table,
    r_max: float,
    key_specification: KeySpecification,
    r_max_lr: Optional[float] = None,
    all_heads: Optional[List[str]] = None,
    seed: int = 1234,
) -> dict:
    """
    Convert ASE style ensemble datasets to model style ensemble datasets.

    Args:
        ensemble_ase_sets (dict): Dictionary of ASE style ensemble datasets.
        z_table (_type_): Table of elements.
        seed (dict): Seed for shuffling and splitting the dataset.
        r_max (float): Cut-off radius.

    Returns:
        dict: Dictionary of model style ensemble datasets.
    """
    ensemble_model_sets = {
        tag: {"train": [], "valid": {}} for tag in ensemble_ase_sets.keys()
    }
    
    if all_heads is not None:
        for tag in ensemble_ase_sets.keys():
            head_data_train = sort_ase_dataset_to_heads(
                ensemble_ase_sets[tag]["train"]
            )
            head_data_valid = sort_ase_dataset_to_heads(
                ensemble_ase_sets[tag]["valid"]
            )
            train_collections = []
            valid_collections = {}
            for head_name in head_data_train.keys():
                collections, _, ase_sets = get_dataset_from_atoms(
                    train_list=head_data_train[head_name],
                    valid_list=head_data_valid.get(head_name, []),
                    seed=seed,
                    config_type_weights=None,
                    key_specification=key_specification,
                    head_name=head_name,
                    return_ase_list=True,
                )
                train_collections += collections.train
                valid_collections[head_name] = collections.valid
            
            ensemble_model_sets[tag]["train"] = [
                so3_data.from_config(
                    config,
                    z_table=z_table, 
                    cutoff=r_max,
                    cutoff_lr=r_max_lr,
                    heads=all_heads
                    )
                for config in train_collections
            ]
            for head in all_heads:
                ensemble_model_sets[tag]["valid"][head] = [
                    so3_data.from_config(
                        config,
                        z_table=z_table,
                        cutoff=r_max,
                        cutoff_lr=r_max_lr,
                        heads=all_heads
                        ) 
                    for config in valid_collections[head]
                ]
        
    else:
        for tag in ensemble_ase_sets.keys():
            ensemble_model_sets[tag]["train"] = create_model_dataset(
                data=ensemble_ase_sets[tag]["train"],
                z_table=z_table,
                seed=seed,
                r_max=r_max,
                r_max_lr=r_max_lr,
                key_specification=key_specification
            )
            ensemble_model_sets[tag]["valid"] = {
                "Default":
                    create_model_dataset(
                        data=ensemble_ase_sets[tag]["valid"],
                        z_table=z_table,
                        seed=seed,
                        r_max=r_max,
                        r_max_lr=r_max_lr,
                        key_specification=key_specification
                )
            }
    
    return ensemble_model_sets


def save_datasets(
    ensemble: dict,
    ensemble_ase_sets: dict,
    path: Path,
    initial: bool = False,
    save_combined_initial: bool = True,
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

    path.mkdir(parents=True, exist_ok=True)
    training_path = path / "training"
    valid_path = path / "validation"
    training_path.mkdir(parents=True, exist_ok=True)
    valid_path.mkdir(parents=True, exist_ok=True)

    for tag in ensemble.keys():
        if initial:
            write(
                training_path / f"initial_train_set_{tag}.xyz",
                ensemble_ase_sets[tag]["train"],
            )
            write(
                valid_path / f"initial_valid_set_{tag}.xyz",
                ensemble_ase_sets[tag]["valid"],
            )
            if save_combined_initial:
                combined_init_train_set += ensemble_ase_sets[tag]["train"]
                combined_init_valid_set += ensemble_ase_sets[tag]["valid"]
        else:
            write(
                training_path / f"train_set_{tag}.xyz",
                ensemble_ase_sets[tag]["train"],
            )
            write(
                valid_path / f"valid_set_{tag}.xyz",
                ensemble_ase_sets[tag]["valid"],
            )
    if save_combined_initial and initial:
        write(
            training_path / "combined_initial_train_set.xyz",
            combined_init_train_set,
        )
        write(
            valid_path / "combined_initial_valid_set.xyz",
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
