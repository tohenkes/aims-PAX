"""
Tests yaml settings for aims-PAX project
"""

import pytest
import yaml
from pydantic import ValidationError

from aims_PAX.settings import AimsPAXSettings, ModelSettings


def test_settings(data_dir):
    """Tests loading and validating settings file"""
    settings_file = data_dir / "project_settings" / "aimsPAX.yaml"
    settings = AimsPAXSettings.from_file(settings_file)
    assert settings.ACTIVE_LEARNING.desired_acc == 0.1
    assert settings.MD is not None
    assert settings.ACTIVE_LEARNING.num_trajectories == 8


def test_model(tmp_path, monkeypatch, data_dir):
    """Tests loading and validating model file"""
    # run this test in temp dir so that the directories are created there
    monkeypatch.chdir(tmp_path)
    settings_file = data_dir / "project_settings" / "model.yaml"
    settings = ModelSettings.from_file(settings_file)
    assert settings.GENERAL.seed == 42
    assert settings.GENERAL.model_choice == "mace"
    assert (tmp_path / "checkpoints").is_dir()


# ── isomtk barostat ──────────────────────────────────────────────────────────


def test_isomtk_barostat(tmp_path, monkeypatch):
    """barostat: isomtk should validate as IsotropicMTKNPT"""
    from aims_PAX.settings.project import IsotropicMTKNPT

    monkeypatch.chdir(tmp_path)
    data = yaml.safe_load("""
ACTIVE_LEARNING:
  desired_acc: 0.1
  num_trajectories: 1
  species_dir: "."
MD:
  stat_ensemble: NPT
  barostat: isomtk
  temperature: 300
  timestep: 1.0
""")
    s = AimsPAXSettings.model_validate(data)
    assert isinstance(s.MD.root, IsotropicMTKNPT)


# ── IDG teacher reference ─────────────────────────────────────────────────────


def test_idg_teacher_ref_requires_foundational(tmp_path, monkeypatch):
    """use_teacher_reference=True in IDG requires initial_sampling: foundational"""
    monkeypatch.chdir(tmp_path)
    data = yaml.safe_load("""
INITIAL_DATASET_GENERATION:
  desired_acc: 0.1
  n_points_per_sampling_step_idg: 5
  use_teacher_reference: true
  initial_sampling: aimd
  teacher_reference_settings:
    model_type: mace-mp
MD:
  stat_ensemble: NVT
  thermostat: Langevin
  temperature: 300
  timestep: 1.0
CLUSTER:
  type: local
  max_workers: 1
""")
    with pytest.raises(
        ValidationError, match="initial_sampling: foundational"
    ):
        AimsPAXSettings.model_validate(data)


def test_idg_teacher_ref_requires_model_type(tmp_path, monkeypatch):
    """use_teacher_reference=True in IDG requires model_type in teacher_reference_settings"""
    monkeypatch.chdir(tmp_path)
    data = yaml.safe_load("""
INITIAL_DATASET_GENERATION:
  desired_acc: 0.1
  n_points_per_sampling_step_idg: 5
  use_teacher_reference: true
  initial_sampling: foundational
  teacher_reference_settings:
    mace_model: small
MD:
  stat_ensemble: NVT
  thermostat: Langevin
  temperature: 300
  timestep: 1.0
CLUSTER:
  type: local
  max_workers: 1
""")
    with pytest.raises(ValidationError, match="model_type"):
        AimsPAXSettings.model_validate(data)


# ── AL teacher reference ──────────────────────────────────────────────────────


def test_al_teacher_ref_requires_model_type(tmp_path, monkeypatch):
    """use_teacher_reference=True in AL requires model_type in teacher_reference_settings"""
    monkeypatch.chdir(tmp_path)
    data = yaml.safe_load("""
ACTIVE_LEARNING:
  desired_acc: 0.1
  num_trajectories: 1
  use_teacher_reference: true
  teacher_reference_settings:
    mace_model: small
MD:
  stat_ensemble: NVT
  thermostat: Langevin
  temperature: 300
  timestep: 1.0
CLUSTER:
  type: local
  max_workers: 1
""")
    with pytest.raises(ValidationError, match="model_type"):
        AimsPAXSettings.model_validate(data)


def test_al_teacher_ref_disables_analysis(tmp_path, monkeypatch):
    """use_teacher_reference=True with analysis=True silently disables analysis"""
    monkeypatch.chdir(tmp_path)
    data = yaml.safe_load("""
ACTIVE_LEARNING:
  desired_acc: 0.1
  num_trajectories: 1
  use_teacher_reference: true
  analysis: true
  teacher_reference_settings:
    model_type: mace-mp
MD:
  stat_ensemble: NVT
  thermostat: Langevin
  temperature: 300
  timestep: 1.0
CLUSTER:
  type: local
  max_workers: 1
""")
    s = AimsPAXSettings.model_validate(data)
    assert s.ACTIVE_LEARNING.analysis is False


# ── CLUSTER cross-field ───────────────────────────────────────────────────────


def test_teacher_ref_requires_cluster(tmp_path, monkeypatch):
    """use_teacher_reference=True requires CLUSTER to be present"""
    monkeypatch.chdir(tmp_path)
    data = yaml.safe_load("""
ACTIVE_LEARNING:
  desired_acc: 0.1
  num_trajectories: 1
  use_teacher_reference: true
  teacher_reference_settings:
    model_type: mace-mp
MD:
  stat_ensemble: NVT
  thermostat: Langevin
  temperature: 300
  timestep: 1.0
""")
    with pytest.raises(ValidationError, match="CLUSTER settings are required"):
        AimsPAXSettings.model_validate(data)


def test_dft_with_cluster_requires_launch_str(tmp_path, monkeypatch):
    """DFT with CLUSTER present but no launch_str should raise"""
    monkeypatch.chdir(tmp_path)
    data = yaml.safe_load("""
ACTIVE_LEARNING:
  desired_acc: 0.1
  num_trajectories: 1
  species_dir: "."
MD:
  stat_ensemble: NVT
  thermostat: Langevin
  temperature: 300
  timestep: 1.0
CLUSTER:
  type: slurm
  parsl_options:
    nodes_per_block: 1
    init_blocks: 1
    min_blocks: 1
    max_blocks: 1
    label: test
  slurm_str: |
    #SBATCH --partition=debug
  worker_str: "export OMP_NUM_THREADS=1"
""")
    with pytest.raises(ValidationError, match="launch_str is required"):
        AimsPAXSettings.model_validate(data)


def test_dft_with_cluster_valid(tmp_path, monkeypatch):
    """DFT with CLUSTER present and launch_str set should validate."""
    monkeypatch.chdir(tmp_path)
    data = yaml.safe_load("""
ACTIVE_LEARNING:
  desired_acc: 0.1
  num_trajectories: 1
  species_dir: "."
MD:
  stat_ensemble: NVT
  thermostat: Langevin
  temperature: 300
  timestep: 1.0
CLUSTER:
  type: slurm
  launch_str: "srun aims.x"
  parsl_options:
    nodes_per_block: 1
    init_blocks: 1
    min_blocks: 1
    max_blocks: 1
    label: test
  slurm_str: |
    #SBATCH --partition=debug
  worker_str: "export OMP_NUM_THREADS=1"
""")
    s = AimsPAXSettings.model_validate(data)
    assert s.CLUSTER is not None
    assert s.CLUSTER.launch_str == "srun aims.x"
