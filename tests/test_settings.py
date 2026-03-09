"""
Tests yaml settings for aims-PAX project
"""
from aims_PAX.settings import AimsPAXSettings


def test_settings(data_dir):
    """Tests loading and validating settings file"""
    settings_file = data_dir / "project_settings" / "aimsPAX.yaml"
    settings = AimsPAXSettings.from_file(settings_file)
    assert(settings.INITIAL_DATASET_GENERATION.foundational_model_settings.mace_model == "medium")
