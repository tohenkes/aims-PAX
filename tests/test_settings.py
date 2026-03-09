"""
Tests yaml settings for aims-PAX project
"""
from aims_PAX.settings import AimsPAXSettings, ModelSettings


def test_settings(data_dir):
    """Tests loading and validating settings file"""
    settings_file = data_dir / "project_settings" / "aimsPAX.yaml"
    settings = AimsPAXSettings.from_file(settings_file)
    assert(settings.ACTIVE_LEARNING.desired_acc == 0.1)


def test_model(tmp_path, monkeypatch, data_dir):
    """Tests loading and validating model file"""
    # run this test in temp dir so that the directories are created there
    monkeypatch.chdir(tmp_path)
    settings_file = data_dir / "project_settings" / "model.yaml"
    settings = ModelSettings.from_file(settings_file)
    assert(settings.GENERAL.seed == 42)
    assert (tmp_path / "checkpoints").is_dir()
