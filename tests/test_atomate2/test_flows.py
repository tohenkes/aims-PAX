"""
Tests for atomate2 part of aims-PAX.
"""

from jobflow import run_locally

from aims_PAX.settings import ModelSettings
from aims_PAX.atomate2.flows.idg import InitialDatasetGenerator


def test_idg_maker(data_dir, clean_dir, project_settings, control_periodic, si):
    """
    Ensures that the InitialDatasetGenerator maker job can be run.
    """
    settings = project_settings(control_periodic, si, clean_dir)
    model_settings_file = data_dir / "project_settings" / "model.yaml"
    model_settings = ModelSettings.from_file(model_settings_file)

    job = InitialDatasetGenerator(
        settings=settings.INITIAL_DATASET_GENERATION,
        md_settings=settings.MD,
        misc_settings=settings.MISC,
        model_settings=model_settings
    ).make()
    
    response = run_locally(job, create_folders=True)
    assert response
    