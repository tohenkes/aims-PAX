"""
The module with Initial Dataset generation algorithm rewritten for
existing models.
"""
import argparse

import numpy as np
from ase import Atoms
from ase.io import read

from aims_PAX.atomate2.atomic_energies import AtomicEnergies
from aims_PAX.atomate2.msonable.ensemble import Ensemble, Stage
from aims_PAX.atomate2.utils import create_restart_point
from aims_PAX.tools.utilities.input_utils import read_input_files, read_geometry
from aims_PAX.tools.utilities.utilities import get_seeds, create_seeds_tags_dict, save_models, Z_from_geometry, \
    create_ztable


def get_species(dataset: dict) -> set:
    """Extract the set of all chemical species from a dictionary of Atoms lists."""
    return {
        symbol
        for atoms_list in dataset.values()
        for atoms in atoms_list
        for symbol in atoms.get_chemical_symbols()
    }

def idg(path_to_aimspax_settings: str, path_to_model_settings: str):
    """
    Main function for initializing, training, and managing a machine learning ensemble
    workflow. This function handles reading configuration files, generating or loading
    seeds and datasets, and facilitating ensemble training and model checkpointing.

    Args:
        path_to_aimspax_settings (str): Path to the AIMS-pax settings file.
        path_to_model_settings (str): Path to the model settings file.
    """
    # read config files
    (model_settings, project_settings, _, path_to_geometry) = (
        read_input_files(
            path_to_model_settings,
            path_to_aimspax_settings,
            procedure="full",
        )
    )
    # read seeds and related tags
    seeds_tags_dict_file = project_settings.MISC.dataset_dir / "seeds_tags_dict.npz"
    if seeds_tags_dict_file.exists():
        seeds_tags_dict = dict(np.load(seeds_tags_dict_file, allow_pickle=True))
    else:
        ensemble_seeds = get_seeds(model_settings.GENERAL.seed,
                                   project_settings.INITIAL_DATASET_GENERATION.ensemble_size)
        seeds_tags_dict = create_seeds_tags_dict(
            seeds=ensemble_seeds,
            model_settings=model_settings,
            dataset_dir=project_settings.MISC.dataset_dir,
        )
    tags: list[str] = list(seeds_tags_dict.keys())

    # get trajectories and model-dependent inputs:
    # - get the datasets from files
    train_data_dir = project_settings.MISC.dataset_dir / "datasets" / "initial" / "training"
    valid_data_dir = project_settings.MISC.dataset_dir / "datasets" / "initial" / "validation"
    # - get ase sets from files (list constructors are used to avoid linter warnings)
    training_sets = {tag: list(read(train_data_dir / f"initial_train_set_{tag}.xyz",
                               format="extxyz",
                               index=":")) for tag in tags}
    valid_sets = {tag: list(read(valid_data_dir / f"initial_valid_set_{tag}.xyz",
                               format="extxyz",
                               index=":")) for tag in tags}
    train_species = get_species(training_sets)
    valid_species = get_species(valid_sets)
    all_species = sorted(train_species | valid_species)
    model_atoms = Atoms(symbols=all_species)

    model_inputs = {}
    if model_settings.GENERAL.model_choice == "mace":
        model_inputs["z"] = Z_from_geometry(model_atoms)
    elif model_settings.GENERAL.model_choice in ("so3lr", "so3krates"):
        model_inputs["z"] = np.arange(1, 119)
    model_inputs["z_table"] = create_ztable(model_inputs["z"])

    # get atomic energies -- the last bit
    atomic_energies = {
        tag: AtomicEnergies.from_z(model_inputs["z"], need_updating=True)
        for tag in tags
    }

    # create the ensemble
    ensemble = Ensemble.from_scratch(
        Stage.IDG,
        tags,
        project_settings.MISC,
        model_settings,
        atomic_energies,
        model_inputs,
    )

    ensemble.update_datasets(training_sets, valid_sets)
    analysis = project_settings.INITIAL_DATASET_GENERATION.analysis

    train_settings = dict(
        n_epochs=project_settings.INITIAL_DATASET_GENERATION.intermediate_epochs_idg,
        valid_skip=project_settings.INITIAL_DATASET_GENERATION.valid_skip,
        analysis=analysis,
        desired_accuracy=(project_settings.INITIAL_DATASET_GENERATION.desired_acc *
                          project_settings.INITIAL_DATASET_GENERATION.desired_acc_scale_idg)
    )
    while not ensemble.done:
        ensemble.train(**train_settings)
    if project_settings.MISC.create_restart:
        trajectories = read_geometry(path_to_geometry, log=True)
        create_restart_point(trajectories, ensemble, analysis=analysis)

    save_models(
        ensemble=ensemble.ensemble,
        training_setups=ensemble.training_setups,
        model_dir=model_settings.GENERAL.model_dir,
        current_epoch=ensemble.epoch,
        model_settings=model_settings.ARCHITECTURE,
        model_choice=model_settings.GENERAL.model_choice
    )


def main():
    """The main function for the command line execution."""
    parser = argparse.ArgumentParser(
        description="Create initial dataset for aims-PAX."
    )
    parser.add_argument(
        "--model-settings",
        type=str,
        default="./model.yaml",
        help="Path to model settings file",
    )
    parser.add_argument(
        "--aimsPAX-settings",
        type=str,
        default="./aimsPAX.yaml",
        help="Path to aimsPAX settings file",
    )
    args = parser.parse_args()
    idg(args.aimsPAX_settings, args.model_settings)


if __name__ == "__main__":
    main()