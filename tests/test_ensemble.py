"""
This module contains tests for the `msonable.Ensemble` module.
"""
import logging

from ase.io import read

from aims_PAX.atomate2.atomic_energies import AtomicEnergies
from aims_PAX.atomate2.msonable.ensemble import Stage, Ensemble
from aims_PAX.atomate2.utils import get_model_dependent_inputs, create_restart_point
from aims_PAX.tools.utilities.input_utils import read_input_files, read_geometry
from aims_PAX.tools.utilities.utilities import get_seeds, create_seeds_tags_dict, create_keyspec, setup_logger


def test_ensemble(data_dir, clean_dir, si):
    """Test the Ensemble class"""
    setup_logger(
        level=logging.INFO,
        tag="test_ensemble",
        directory=clean_dir.as_posix(),
    )
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
    # get seeds and related tags
    ensemble_seeds = get_seeds(model_settings.GENERAL.seed,
                               project_settings.INITIAL_DATASET_GENERATION.ensemble_size)
    assert 102 in ensemble_seeds
    seeds_tags_dict = create_seeds_tags_dict(
        seeds=ensemble_seeds,
        model_settings=model_settings,
        dataset_dir=project_settings.MISC.dataset_dir,
    )
    tags: list[str] = list(seeds_tags_dict.keys())
    assert "exp-102" in tags
    # get trajectories and model-dependent inputs

    # a bad idea generally to get model_inputs like this: they should be constructed
    # from training and validation sets. Alhthough we know that the test data is si only
    trajectories = read_geometry(si, log=True)
    model_inputs = get_model_dependent_inputs(model_settings.GENERAL.model_choice,
                                              trajectories=trajectories)

    assert "z_table" in model_inputs
    # get atomic energies -- the last bit
    atomic_energies = {
        tag: AtomicEnergies.from_z(model_inputs["z"], need_updating=True)
        for tag in tags
    }

    assert atomic_energies[tags[0]].as_dict()[14] == 0.0
    # create the ensemble
    ensemble = Ensemble.from_scratch(
        Stage.IDG,
        tags,
        project_settings.MISC,
        model_settings,
        atomic_energies,
        model_inputs,
    )
    # get the datasets from files
    train_data_dir = data_dir / "datasets" / "initial" / "training"
    valid_data_dir = data_dir / "datasets" / "initial" / "validation"
    # get ase sets from files (list constructors are used to avoid linter warnings)
    training_sets = {tag: list(read(train_data_dir / f"initial_train_set_{tag}.xyz",
                               format="extxyz",
                               index=":")) for tag in tags}
    valid_sets = {tag: list(read(valid_data_dir / f"initial_valid_set_{tag}.xyz",
                               format="extxyz",
                               index=":")) for tag in tags}
    ensemble.update_datasets(training_sets, valid_sets)
    analysis = project_settings.INITIAL_DATASET_GENERATION.analysis

    train_settings = dict(
        n_epochs=project_settings.INITIAL_DATASET_GENERATION.intermediate_epochs_idg,
        valid_skip=project_settings.INITIAL_DATASET_GENERATION.valid_skip,
        analysis=analysis,
        desired_accuracy=(project_settings.INITIAL_DATASET_GENERATION.desired_acc *
                          project_settings.INITIAL_DATASET_GENERATION.desired_acc_scale_idg)
    )
    ensemble.train(**train_settings)
    if project_settings.MISC.create_restart:
        create_restart_point(trajectories, ensemble, analysis=analysis)


