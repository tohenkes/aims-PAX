"""
This module contains tests for the `msonable.Ensemble` module.
"""
from ase.io import read

from aims_PAX.atomate2.atomic_energies import AtomicEnergies
from aims_PAX.atomate2.msonable.ensemble import Ensemble
from aims_PAX.atomate2.utils import get_model_dependent_inputs
from aims_PAX.tools.model_tools import training_tools
from aims_PAX.tools.utilities.input_utils import read_input_files, read_geometry
from aims_PAX.tools.utilities.utilities import get_seeds, create_seeds_tags_dict, create_keyspec


def test_ensemble(data_dir, clean_dir, si):
    """Test the Ensemble class"""
    project_settings_file = data_dir / "project_settings" / "aimsPAX.yaml"
    model_settings_file = data_dir / "project_settings" / "model.yaml"
    # read config files
    (model_settings, project_settings, path_to_control, path_to_geometry) = (
        read_input_files(
            model_settings_file,
            project_settings_file,
            procedure="full",
        )
    )
    # patch the paths to make sure the outputs go to the clean_dir
    project_settings.MISC.output_dir = clean_dir
    # get seeds and related tags
    ensemble_seeds = get_seeds(model_settings.GENERAL.seed,
                               project_settings.INITIAL_DATASET_GENERATION.ensemble_size)
    assert 270 in ensemble_seeds
    seeds_tags_dict = create_seeds_tags_dict(
        seeds=ensemble_seeds,
        model_settings=model_settings,
        dataset_dir=project_settings.MISC.dataset_dir,
    )
    tags = list(seeds_tags_dict.keys())
    assert "exp-270" in tags
    # get trajectories and model-dependent inputs
    trajectories = read_geometry(si, log=True)
    model_inputs = get_model_dependent_inputs(model_settings.GENERAL.model_choice,
                                              trajectories=trajectories)
    key_specification = create_keyspec(
        energy_key=project_settings.MISC.energy_key,
        forces_key=project_settings.MISC.forces_key,
        stress_key=project_settings.MISC.stress_key,
        dipole_key=project_settings.MISC.dipole_key,
        polarizability_key=project_settings.MISC.polarizability_key,
        head_key=project_settings.MISC.head_key,
        charges_key=project_settings.MISC.charges_key,
        total_charge_key=project_settings.MISC.total_charge_key,
        total_spin_key=project_settings.MISC.total_spin_key,
    )
    assert "z_table" in model_inputs
    # get atomic energies -- the last bit
    atomic_energies = AtomicEnergies.from_z(tags, model_inputs["z"], need_updating=True)
    assert atomic_energies.get(tags[0])[14] == 0.0
    # create the ensemble
    ensemble = Ensemble.from_scratch(
        tags,
        model_settings,
        atomic_energies,
        model_inputs,
        key_specification
    )
    # get the datasets from files
    train_data_dir = data_dir / "datasets" / "initial" / "training"
    valid_data_dir = data_dir / "datasets" / "initial" / "validation"
    # get ase sets from files
    training_sets = {tag: read(train_data_dir / f"initial_train_set_{tag}.xyz",
                               format="extxyz",
                               index=":") for tag in tags}
    valid_sets = {tag: read(valid_data_dir / f"initial_valid_set_{tag}.xyz",
                               format="extxyz",
                               index=":") for tag in tags}



