import logging
import re
import sys
import os
import torch
import numpy as np
from typing import Optional, Any, Dict, Union, List
from pathlib import Path
from mace import tools, modules
from mace.tools import AtomicNumberTable
from aims_PAX.tools.setup_MACE import setup_mace
from aims_PAX.tools.setup_MACE_training import setup_mace_training
import ase.data
from ase.io import read
from ase import units
from contextlib import nullcontext
import time
from threading import Thread
import pandas as pd

# !!!
# Many functions are taken from the MACE code:
# https://github.com/ACEsuit/mace
# !!!

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


def read_geometry(
    path: str,
) -> Union[ase.Atoms, List]:
    """
    Checks if the path is a single ase readable file or a directory.
    If it is a directory, it reads all files in the directory and
    puts the ase.atoms in a list. If the directory contains files
    that are not ase readable, it raises an error.
    If the path is a single ase readable file, it reads the file
    and returns the ase.atoms object.

    Args:
        path (str): Path to file or directory.

    Returns:
        Union[ase.Atoms, List]: ase.atoms object or list of ase.atoms objects.
    """

    if os.path.isdir(path):
        atoms_list = []
        for filename in os.listdir(path):
            complete_path = os.path.join(path, filename)
            if os.path.isfile(complete_path):
                try:
                    atoms = read(complete_path)
                    atoms_list.append(atoms)
                except Exception as e:
                    logging.error(
                        f"File {filename} is not a valid ASE readable file: {e}"
                    )
        if not atoms_list:
            raise ValueError("No valid ASE readable files found.")
        return atoms_list
    else:
        atoms = [read(path)]
        return atoms


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


def max_sd_2(
    ensemble_prediction: np.array,
    return_argmax: bool = False,
) -> np.array:
    """
    Compute the maximum standard deviation of the ensemble prediction.

    Args:
        ensemble_prediction (np.array): Ensemble prediction of forces:
        [n_ensemble_members, n_mols, n_atoms, xyz].

    Returns:
        np.array: Maximum standard deviation of atomic forces per
                molecule: [n_mols].
    """
    # average prediction over ensemble of models
    pred_av = np.average(ensemble_prediction, axis=0, keepdims=True)
    diff_sq = (ensemble_prediction - pred_av) ** 2.0
    diff_sq_mean = np.mean(diff_sq, axis=(0, -1))
    max_sd = np.max(np.sqrt(diff_sq_mean), axis=-1)
    if return_argmax:
        return max_sd, np.argmax(np.sqrt(diff_sq_mean), axis=-1)
    else:
        return max_sd


def avg_sd(
    ensemble_prediction: np.array,
) -> np.array:
    """
    Compute the average standard deviation of the ensemble prediction.

    Args:
        ensemble_prediction (np.array): Ensemble prediction of forces:
                        [n_ensemble_members, n_mols, n_atoms, xyz].

    Returns:
        np.array: Average standard deviation of atomic forces per
                    molecule: [n_mols].
    """
    # average prediction over ensemble of models
    pred_av = np.average(ensemble_prediction, axis=0, keepdims=True)
    diff_sq = (ensemble_prediction - pred_av) ** 2.0
    diff_sq_mean = np.mean(diff_sq, axis=(0, -1))
    avg_sd = np.mean(np.sqrt(diff_sq_mean), axis=-1)
    return avg_sd


def atom_wise_sd(
    ensemble_prediction: np.array,
) -> np.array:
    """
    Compute the atom-wise standard deviation of the ensemble prediction.

    Args:
        prediction (np.array): Prediction of forces:
                            [n_ensemble_members, n_mols, n_atoms, xyz].

    Returns:
        np.array: Atom-wise standard deviation of atomic forces per
                    molecule: [n_mols, n_atoms].
    """
    # average prediction over ensemble of models
    pred_av = np.average(ensemble_prediction, axis=0, keepdims=True)
    diff_sq = (ensemble_prediction - pred_av) ** 2.0
    diff_sq_mean = np.mean(diff_sq, axis=(0, -1))
    atom_wise_sd = np.sqrt(diff_sq_mean)
    return atom_wise_sd


def atom_wise_f_error(
    prediction: np.array,
    target: np.array,
) -> np.array:
    """
    Compute the atom-wise force error of the ensemble prediction.

    Args:
        prediction (np.array): Prediction of forces:
                                [n_mols, n_atoms, xyz].
        target (np.array): Target forces: [n_mols, n_atoms, xyz].

    Returns:
        np.array: Atom-wise force error of atomic forces per molecule:
                        [n_mols, n_atoms].
    """
    atom_wise_f_error = np.sqrt(np.mean((prediction - target) ** 2, axis=(-1)))
    return atom_wise_f_error


def ensemble_from_folder(path_to_models: str, device: str, dtype) -> list:
    """
    Load an ensemble of models from a folder.
    (Can't handle other file formats than .pt at the moment.)

    Args:
        path_to_models (str): Path to the folder containing the models.
        device (str): Device to load the models on.

    Returns:
        list: List of models.
    """

    assert os.path.exists(path_to_models)
    assert os.listdir(Path(path_to_models))

    ensemble = {}
    for filename in os.listdir(path_to_models):
        if os.path.isfile(os.path.join(path_to_models, filename)):
            complete_path = os.path.join(path_to_models, filename)
            model = torch.load(
                complete_path, map_location=device, weights_only=False
            ).to(dtype)
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


def E_uncert(
    prediction: np.array,
) -> float:
    """
    Computes the standard deviation of the ensemble prediction on energies.

    Args:
        prediction (np.array): Ensemble prediction of energies:
                                        [n_ensemble_members, n_mols].

    Returns:
        float: Standard deviation of the ensemble prediction.
    """

    M = prediction.shape[0]  # number of ensemble members
    prediction_avg = np.mean(prediction, axis=0, keepdims=True)
    uncert = 1 / (M - 1) * np.sum((prediction_avg - prediction) ** 2, axis=0)
    return uncert


def get_uncert_alpha(reference, ensemble_avg, uncertainty) -> float:
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

    return np.mean((reference - ensemble_avg) ** 2 / uncertainty**2)


def ensemble_training_setups(
    ensemble: dict,
    mace_settings: dict,
    checkpoints_dir: str = None,
    restart: bool = False,
    al_settings: dict = None,
) -> dict:
    """
    Creates a dictionary of training setups for each model in the ensemble.

    Args:
        ensemble (dict): Dictionary of MACE models in the ensemble.
        mace_settings (dict): Model settings dictionary containing
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
    training_setups = {}
    for tag, model in ensemble.items():
        training_setups[tag] = setup_mace_training(
            settings=mace_settings,
            model=model,
            tag=tag,
            restart=restart,
            checkpoints_dir=checkpoints_dir,
            al_settings=al_settings,
        )
    return training_setups


def create_seeds_tags_dict(
    seeds: np.array,
    mace_settings: dict,
    al_settings: dict,
    save_seeds_tags_dict: str = "seeds_tags_dict.npz",
) -> dict:
    """
    Creates a dict where the keys (tags) are the names of the MACE
    models in the ensemble and the values are the respective seeds.

    Args:
        seeds (np.array): Array of seeds for the ensemble.
        mace_settings (dict): Model settings dictionary containing
                              the experiment name.
        al_settings (dict, optional): Active learning settings..
        save_seeds_tags_dict (str, optional): Name of the resulting dict.
                                Defaults to "seeds_tags_dict.npz".

    Returns:
        dict: _description_
    """
    seeds_tags_dict = {}
    for seed in seeds:
        tag = mace_settings["GENERAL"]["name_exp"] + "-" + str(seed)
        seeds_tags_dict[tag] = seed
    if save_seeds_tags_dict:
        np.savez(
            al_settings["dataset_dir"] + "/" + save_seeds_tags_dict,
            **seeds_tags_dict,
        )
    return seeds_tags_dict


def create_ztable(
    zs: list,
):
    z_table = tools.get_atomic_number_table_from_zs(z for z in zs)
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
        mace_settings["GENERAL"]["seed"] = seed
        tag = mace_settings["GENERAL"]["name_exp"] + "-" + str(seed)
        seeds_tags_dict[tag] = seed
        ensemble[tag] = setup_mace(
            settings=mace_settings,
            z_table=z_table,
            atomic_energies_dict=ensemble_atomic_energies_dict[tag],
        )
    return ensemble


def get_atomic_energies_from_ensemble(
    ensemble: dict,
    z,
    dtype: str,
):
    """
    Loads the atomic energies from existing ensemble members.

    Args:
        ensemble (dict): Dictionary of MACE models.
        z (np.array): Array of atomic numbers for which the atomic energies
                        are needed.
        dtype (str): Data type for the atomic energies.

    """
    ensemble_atomic_energies_dict = {}
    ensemble_atomic_energies = {}
    for tag, model in ensemble.items():
        ensemble_atomic_energies[tag] = np.array(
            model.atomic_energies_fn.atomic_energies.cpu(), dtype=dtype
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
    path_to_checkpoints: str,
    z: np.array,
    seeds_tags_dict: dict,
    dtype: str,
) -> tuple:
    """
    Loads the atomic energies (energy shifts) from an existing MACE
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
            (path_to_checkpoints + "/" + tag + f"_epoch-{last_epoch}.pt")
        )

        atomic_energies_array = check_pt["model"][
            "atomic_energies_fn.atomic_energies"
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


def update_model_auxiliaries(
    model: modules.MACE,
    mace_sets: dict,
    scaling: str,
    atomic_energies_list: list,
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
        mace_sets (dict): Dictionary containing the training and validation
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
    train_loader = mace_sets["train_loader"]

    if update_atomic_energies:
        assert z_table is not None
        assert atomic_energies_dict is not None

        energies_train = torch.stack(
            [point.energy.reshape(1) for point in mace_sets["train"]]
        )
        zs_train = [
            [
                z_table.index_to_z(torch.nonzero(one_hot).item())
                for one_hot in point.node_attrs
            ]
            for point in mace_sets["train"]
        ]
        new_atomic_energies_dict = compute_average_E0s(
            energies_train, zs_train, z_table
        )
        atomic_energies_dict.update(new_atomic_energies_dict)
        atomic_energies_list = [
            atomic_energies_dict[z] for z in atomic_energies_dict.keys()
        ]
        atomic_energies_tensor = torch.tensor(
            atomic_energies_list, dtype=dtype_mapping[dtype]
        )

        model.atomic_energies_fn.atomic_energies = atomic_energies_tensor.to(
            device
        )

    average_neighbors = modules.compute_avg_num_neighbors(train_loader)
    for interaction_idx in range(len(model.interactions)):
        model.interactions[interaction_idx].avg_num_neighbors = (
            average_neighbors
        )
    mean, std = modules.scaling_classes[scaling](
        train_loader, atomic_energies_list
    )
    mean, std = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(
        device
    )
    model.scale_shift = modules.blocks.ScaleShiftBlock(scale=std, shift=mean)


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
        checkpoint_handler (tools.CheckpointHandler): MACE handler for saving
                                                        checkpoints.
        training_setup (dict): Training settings.
        model (modules.MACE): MACE model to be saved.
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
    ensemble: dict, training_setups: dict, model_dir: str, current_epoch: int
):
    for tag, model in ensemble.items():
        training_setup = training_setups[tag]

        param_context = (
            training_setup["ema"].average_parameters()
            if training_setup["ema"] is not None
            else nullcontext()
        )

        with param_context:
            torch.save(
                model,
                Path(model_dir) / (tag + ".model"),
            )

        save_checkpoint(
            checkpoint_handler=training_setup["checkpoint_handler"],
            training_setup=training_setup,
            model=model,
            epoch=current_epoch,
            keep_last=False,
        )


def Z_from_geometry(
    atoms: Union[ase.Atoms, List],
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
    elif isinstance(atoms, list):
        all_z = []
        for atom in atoms:
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


def list_latest_file(directory: str) -> str:
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
    ensemble: dict, training_setups: dict, mace_settings: dict
) -> None:
    """
    Save the ensemble models to disk. If EMA is used, the averaged parameters
    are saved, otherwise the model parameters are saved directly.
    The path to the models is defined in the mace_settings under
    "GENERAL" -> "model_dir". The models are saved with the tag as the filename
    and the extension ".model".

    Args:
        ensemble (dict): Dictionary of models.
        training_setups (dict): Dictionary of training setups for each model.
        mace_settings (dict): Dictionary of MACE settings, specificied by the
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
                Path(mace_settings["GENERAL"]["model_dir"]) / (tag + ".model"),
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


class AIMSControlParser:
    def __init__(self) -> None:

        self.string_patterns = {
            "xc": re.compile(r"^\s*(xc)\s+(\S+)", re.IGNORECASE),
            "spin": re.compile(r"^\s*(spin)\s+(\S+)", re.IGNORECASE),
            "communication_type": re.compile(
                r"^\s*(communication_type)\s+(\S+)", re.IGNORECASE
            ),
            "density_update_method": re.compile(
                r"^\s*(density_update_method)\s+(\S+)", re.IGNORECASE
            ),
            "KS_method": re.compile(r"^\s*(KS_method)\s+(\S+)", re.IGNORECASE),
            "mixer": re.compile(r"^\s*(mixer)\s+(\S+)", re.IGNORECASE),
            "output_level": re.compile(
                r"^\s*(output_level)\s+(\S+)", re.IGNORECASE
            ),
            "packed_matrix_format": re.compile(
                r"^\s*(packed_matrix_format)\s+(\S+)", re.IGNORECASE
            ),
            "relax_unit_cell": re.compile(
                r"^\s*(relax_unit_cell)\s+(\S+)", re.IGNORECASE
            ),
            "restart": re.compile(r"^\s*(restart)\s+(\S+)", re.IGNORECASE),
            "restart_read_only": re.compile(
                r"^\s*(restart_read_only)\s+(\S+)", re.IGNORECASE
            ),
            "restart_write_only": re.compile(
                r"^\s*(restart_write_only)\s+(\S+)", re.IGNORECASE
            ),
            "total_energy_method": re.compile(
                r"^\s*(total_energy_method)\s+(\S+)", re.IGNORECASE
            ),
            "qpe_calc": re.compile(r"^\s*(qpe_calc)\s+(\S+)", re.IGNORECASE),
            "species_dir": re.compile(
                r"^\s*(species_dir)\s+(\S+)", re.IGNORECASE
            ),
            "run_command": re.compile(
                r"^\s*(run_command)\s+(\S+)", re.IGNORECASE
            ),
            "plus_u": re.compile(r"^\s*(plus_u)\s+(\S+)", re.IGNORECASE),
        }

        self.bool_patterns = {
            "collect_eigenvectors": re.compile(
                r"^\s*(collect_eigenvectors)\s+(\S+)", re.IGNORECASE
            ),
            "compute_forces": re.compile(
                r"^\s*(compute_forces)\s+(\S+)", re.IGNORECASE
            ),
            "compute_kinetic": re.compile(
                r"^\s*(compute_kinetic)\s+(\S+)", re.IGNORECASE
            ),
            "compute_numerical_stress": re.compile(
                r"^\s*(compute_numerical_stress)\s+(\S+)", re.IGNORECASE
            ),
            "compute_analytical_stress": re.compile(
                r"^\s*(compute_analytical_stress)\s+(\S+)", re.IGNORECASE
            ),
            "compute_heat_flux": re.compile(
                r"^\s*(compute_heat_flux)\s+(\S+)", re.IGNORECASE
            ),
            "distributed_spline_storage": re.compile(
                r"^\s*(distributed_spline_storage)\s+(\S+)", re.IGNORECASE
            ),
            "evaluate_work_function": re.compile(
                r"^\s*(evaluate_work_function)\s+(\S+)", re.IGNORECASE
            ),
            "final_forces_cleaned": re.compile(
                r"^\s*(final_forces_cleaned)\s+(\S+)", re.IGNORECASE
            ),
            "hessian_to_restart_geometry": re.compile(
                r"^\s*(hessian_to_restart_geometry)\s+(\S+)", re.IGNORECASE
            ),
            "load_balancing": re.compile(
                r"^\s*(load_balancing)\s+(\S+)", re.IGNORECASE
            ),
            "MD_clean_rotations": re.compile(
                r"^\s*(MD_clean_rotations)\s+(\S+)", re.IGNORECASE
            ),
            "MD_restart": re.compile(
                r"^\s*(MD_restart)\s+(\S+)", re.IGNORECASE
            ),
            "override_illconditioning": re.compile(
                r"^\s*(override_illconditioning)\s+(\S+)", re.IGNORECASE
            ),
            "override_relativity": re.compile(
                r"^\s*(override_relativity)\s+(\S+)", re.IGNORECASE
            ),
            "restart_relaxations": re.compile(
                r"^\s*(restart_relaxations)\s+(\S+)", re.IGNORECASE
            ),
            "squeeze_memory": re.compile(
                r"^\s*(squeeze_memory)\s+(\S+)", re.IGNORECASE
            ),
            "symmetry_reduced_k_grid": re.compile(
                r"^\s*(symmetry_reduced_k_grid)\s+(\S+)", re.IGNORECASE
            ),
            "use_density_matrix": re.compile(
                r"^\s*(use_density_matrix)\s+(\S+)", re.IGNORECASE
            ),
            "use_dipole_correction": re.compile(
                r"^\s*(use_dipole_correction)\s+(\S+)", re.IGNORECASE
            ),
            "use_local_index": re.compile(
                r"^\s*(use_local_index)\s+(\S+)", re.IGNORECASE
            ),
            "use_logsbt": re.compile(
                r"^\s*(use_logsbt)\s+(\S+)", re.IGNORECASE
            ),
            "vdw_correction_hirshfeld": re.compile(
                r"^\s*(vdw_correction_hirshfeld)\s+(\S+)", re.IGNORECASE
            ),
            "postprocess_anyway": re.compile(
                r"^\s*(postprocess_anyway)\s+(\S+)", re.IGNORECASE
            ),
            "override_initial_charge_check": re.compile(
                r"^\s*(override_initial_charge_check)\s+(\S+)", re.IGNORECASE
            ),
        }

        self.float_patterns = {
            "charge": re.compile(
                r"^\s*(charge)\s+([-+]?\d*\.?\d+)", re.IGNORECASE
            ),
            "charge_mix_param": re.compile(
                r"^\s*(charge_mix_param)\s+([-+]?\d*\.?\d+)", re.IGNORECASE
            ),
            "default_initial_moment": re.compile(
                r"^\s*(default_initial_moment)\s+([-+]?\d*\.?\d+)",
                re.IGNORECASE,
            ),
            "fixed_spin_moment": re.compile(
                r"^\s*(fixed_spin_moment)\s+([-+]?\d*\.?\d+)", re.IGNORECASE
            ),
            "hartree_convergence_parameter": re.compile(
                r"^\s*(hartree_convergence_parameter)\s+([-+]?\d*\.?\d+)",
                re.IGNORECASE,
            ),
            "harmonic_length_scale": re.compile(
                r"^\s*(harmonic_length_scale)\s+([-+]?\d*\.?\d+)",
                re.IGNORECASE,
            ),
            "ini_linear_mix_param": re.compile(
                r"^\s*(ini_linear_mix_param)\s+([-+]?\d*\.?\d+)", re.IGNORECASE
            ),
            "ini_spin_mix_parma": re.compile(
                r"^\s*(ini_spin_mix_parma)\s+([-+]?\d*\.?\d+)", re.IGNORECASE
            ),
            "initial_moment": re.compile(
                r"^\s*(initial_moment)\s+([-+]?\d*\.?\d+)", re.IGNORECASE
            ),
            "MD_MB_init": re.compile(
                r"^\s*(MD_MB_init)\s+([-+]?\d*\.?\d+)", re.IGNORECASE
            ),
            "MD_time_step": re.compile(
                r"^\s*(MD_time_step)\s+([-+]?\d*\.?\d+)", re.IGNORECASE
            ),
            "prec_mix_param": re.compile(
                r"^\s*(prec_mix_param)\s+([-+]?\d*\.?\d+)", re.IGNORECASE
            ),
            "set_vacuum_level": re.compile(
                r"^\s*(set_vacuum_level)\s+([-+]?\d*\.?\d+)", re.IGNORECASE
            ),
            "spin_mix_param": re.compile(
                r"^\s*(spin_mix_param)\s+([-+]?\d*\.?\d+)", re.IGNORECASE
            ),
        }

        self.exp_patterns = {
            "basis_threshold": re.compile(
                r"^\s*(basis_threshold)\s+([-+]?\d*\.?\d*([eEdD][-+]?\d+)?)",
                re.IGNORECASE,
            ),
            "occupation_thr": re.compile(
                r"^\s*(occupation_thr)\s+([-+]?\d*\.?\d*([eEdD][-+]?\d+)?)",
                re.IGNORECASE,
            ),
            "sc_accuracy_eev": re.compile(
                r"^\s*(sc_accuracy_eev)\s+([-+]?\d*\.?\d*([eEdD][-+]?\d+)?)",
                re.IGNORECASE,
            ),
            "sc_accuracy_etot": re.compile(
                r"^\s*(sc_accuracy_etot)\s+([-+]?\d*\.?\d*([eEdD][-+]?\d+)?)",
                re.IGNORECASE,
            ),
            "sc_accuracy_forces": re.compile(
                r"^\s*(sc_accuracy_forces)\s+([-+]?\d*\.?\d*([eEdD][-+]?\d+)?)",
                re.IGNORECASE,
            ),
            "sc_accuracy_rho": re.compile(
                r"^\s*(sc_accuracy_rho)\s+([-+]?\d*\.?\d*([eEdD][-+]?\d+)?)",
                re.IGNORECASE,
            ),
            "sc_accuracy_stress": re.compile(
                r"^\s*(sc_accuracy_stress)\s+([-+]?\d*\.?\d*([eEdD][-+]?\d+)?)",
                re.IGNORECASE,
            ),
        }

        self.int_patterns = {
            "empty_states": re.compile(
                r"^\s*(empty_states)\s+(\d+)", re.IGNORECASE
            ),
            "ini_linear_mixing": re.compile(
                r"^\s*(ini_linear_mixing)\s+(\d+)", re.IGNORECASE
            ),
            "max_relaxation_steps": re.compile(
                r"^\s*(max_relaxation_steps)\s+(\d+)", re.IGNORECASE
            ),
            "max_zeroin": re.compile(
                r"^\s*(max_zeroin)\s+(\d+)", re.IGNORECASE
            ),
            "multiplicity": re.compile(
                r"^\s*(multiplicity)\s+(\d+)", re.IGNORECASE
            ),
            "n_max_pulay": re.compile(
                r"^\s*(n_max_pulay)\s+(\d+)", re.IGNORECASE
            ),
            "sc_iter_limit": re.compile(
                r"^\s*(sc_iter_limit)\s+(\d+)", re.IGNORECASE
            ),
            "walltime": re.compile(r"^\s*(walltime)\s+(\d+)", re.IGNORECASE),
        }
        # TH: some of them seem unnecessary for our purposes and are complicated
        #     to put into regex which is why i commented them out
        self.list_patterns = {
            #'init_hess',
            "k_grid": re.compile(
                r"^\s*(k_grid)\s+(\d+)\s+(\d+)\s+(\d+)", re.IGNORECASE
            ),
            "k_offset": re.compile(
                r"^\s*(k_offset)\s+(\d*\.?\d+)\s+(\d*\.?\d+)\s+(\d*\.?\d+)",
                re.IGNORECASE,
            ),
            #'MD_run',
            #'MD_schedule',
            #'MD_segment',
            #'mixer_threshold',
            "occupation_type": re.compile(
                r"^\s*(occupation_type)\s+(\S+)\s+(\d*\.?\d+)(?:\s+(\d+))?",
                re.IGNORECASE,
            ),
            #'output',
            #'cube',
            #'preconditioner',
            "relativistic": re.compile(
                r"^\s*(relativistic)\s+(\S+)\s+(\S+)(?:\s+(\d+))?",
                re.IGNORECASE,
            ),
            #'relax_geometry',
        }

        self.special_patterns = {
            #'many_body_dispersion': re.compile(r'^\s*(many_body_dispersion)\s', re.IGNORECASE)
            "many_body_dispersion": re.compile(
                r"""
                ^\s*                                               
                (many_body_dispersion)\b                              
                (?:                                                
                    \s+beta=(?P<beta>-?\d+(\.\d+)?)                
                    |\s+k_grid=(?P<k_grid>\d+:\d+:\d+)             
                    |\s+freq_grid=(?P<freq_grid>\d+)               
                    |\s+self_consistent=(?P<self_consistent>\.true\.|\.false\.) 
                    |\s+vdw_params_kind=(?P<vdw_params_kind>[^\s]+)
                )*                                                 
            """,
                re.IGNORECASE | re.VERBOSE,
            ),
            "many_body_dispersion_nl": re.compile(
                r"""
                ^\s*                                               
                (many_body_dispersion_nl)\b                               
                (?:                                                
                    \s+beta=(?P<beta>-?\d+(\.\d+)?)                
                    |\s+k_grid=(?P<k_grid>\d+:\d+:\d+)             
                    |\s+freq_grid=(?P<freq_grid>\d+)               
                    |\s+self_consistent=(?P<self_consistent>\.true\.|\.false\.) 
                    |\s+vdw_params_kind=(?P<vdw_params_kind>[^\s]+)
                )*                                                 
            """,
                re.IGNORECASE | re.VERBOSE,
            ),
        }

    def f90_bool_to_py_bool(self, f90_bool: str) -> bool:

        if f90_bool.lower() == ".true.":
            return True
        elif f90_bool.lower() == ".false.":
            return False

    def __call__(
        self,
        path_to_control: str,
    ) -> dict:
        aims_settings = {}
        with open(path_to_control, "r") as input_file:
            for line in input_file:

                if "#" in line:
                    line = line.split("#")[0]

                for key, pattern in self.string_patterns.items():
                    match = pattern.match(line)
                    if match:
                        aims_settings[match.group(1)] = match.group(2)

                for key, pattern in self.bool_patterns.items():
                    match = pattern.match(line)
                    if match:
                        aims_settings[match.group(1)] = (
                            self.f90_bool_to_py_bool(match.group(2))
                        )

                for key, pattern in self.float_patterns.items():
                    match = pattern.match(line)
                    if match:
                        aims_settings[match.group(1)] = float(match.group(2))

                for key, pattern in self.exp_patterns.items():
                    match = pattern.match(line)
                    if match:
                        matched_value = (
                            match.group(2).replace("d", "e").replace("D", "E")
                        )
                        aims_settings[match.group(1)] = float(matched_value)

                for key, pattern in self.int_patterns.items():
                    match = pattern.match(line)
                    if match:
                        aims_settings[match.group(1)] = int(match.group(2))

                for key, pattern in self.list_patterns.items():
                    match = pattern.match(line)
                    if match:
                        if key == "k_grid":
                            aims_settings[match.group(1)] = [
                                int(match.group(2)),
                                int(match.group(3)),
                                int(match.group(4)),
                            ]
                        if key == "k_offset":
                            aims_settings[match.group(1)] = [
                                float(match.group(2)),
                                float(match.group(3)),
                                float(match.group(4)),
                            ]
                        if key == "occupation_type":
                            if match.group(4) is not None:
                                aims_settings[match.group(1)] = [
                                    match.group(2),
                                    float(match.group(3)),
                                    int(match.group(4)),
                                ]
                            else:
                                aims_settings[match.group(1)] = [
                                    match.group(2),
                                    float(match.group(3)),
                                ]
                        if key == "relativistic":
                            if match.group(4) is not None:
                                aims_settings[match.group(1)] = [
                                    match.group(2),
                                    match.group(3),
                                    int(match.group(4)),
                                ]
                            else:
                                aims_settings[match.group(1)] = [
                                    match.group(2),
                                    match.group(3),
                                ]

                for key, pattern in self.special_patterns.items():
                    match = pattern.match(line)
                    if match:
                        if key == "many_body_dispersion":
                            if any(match.groupdict().values()):
                                # If parameters are found, store them in a dictionary
                                aims_settings[key] = ""
                                for param, value in match.groupdict().items():
                                    if value is not None:
                                        aims_settings[
                                            key
                                        ] += f"{param}={value} "
                            else:
                                # If no parameters are found, store an empty string
                                aims_settings[key] = ""

                        if key == "many_body_dispersion_nl":
                            if any(match.groupdict().values()):
                                # If parameters are found, store them in a dictionary
                                aims_settings[key] = ""
                                for param, value in match.groupdict().items():
                                    if value is not None:
                                        aims_settings[
                                            key
                                        ] += f"{param}={value} "
                            else:
                                # If no parameters are found, store an empty string
                                aims_settings[key] = ""
        return aims_settings


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

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add filter for rank
    logger.addFilter(lambda _: rank == 0)

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
