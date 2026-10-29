"""
Tests yaml settings for aims-PAX project
"""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aims_PAX.settings import AimsPAXSettings, ModelSettings


def test_settings(data_dir, monkeypatch):
    """Tests loading and validating settings file"""
    monkeypatch.chdir(
        data_dir.parent.parent
    )  # project root, so relative species_dir resolves
    settings_file = data_dir / "project_settings" / "aimsPAX.yaml"
    settings = AimsPAXSettings.from_file(settings_file)
    assert settings.ACTIVE_LEARNING.desired_acc == 0.1
    assert settings.MD.root.timestep == 1.0
    assert settings.ACTIVE_LEARNING.num_trajectories == 8


def test_model(tmp_path, monkeypatch, data_dir):
    """Tests loading and validating model file"""
    # run this test in temp dir so that the directories are created there
    monkeypatch.chdir(tmp_path)
    settings_file = data_dir / "project_settings" / "model.yaml"
    settings = ModelSettings.from_file(settings_file)
    settings.resolve_dirs(tmp_path)
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
    with pytest.raises(ValidationError, match="foundational"):
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
    with pytest.raises(ValidationError, match="CLUSTER"):
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
    with pytest.raises(ValidationError, match="launch_str"):
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


# ── CLUSTER.calc_dir must be isolated from user files ─────────────────────────

_BASE_SETTINGS = """
ACTIVE_LEARNING:
  desired_acc: 0.1
  num_trajectories: 1
  species_dir: "."
MD:
  stat_ensemble: NVT
  thermostat: Langevin
  temperature: 300
  timestep: 1.0
"""


def _local_cluster(**kwargs) -> str:
    lines = ["CLUSTER:", "  type: local", "  launch_str: aims.x"]
    lines += [f"  {key}: {json.dumps(value)}" for key, value in kwargs.items()]
    return "\n".join(lines) + "\n"


def _misc(**kwargs) -> str:
    lines = ["MISC:"]
    lines += [f"  {key}: {json.dumps(value)}" for key, value in kwargs.items()]
    return "\n".join(lines) + "\n"


def _settings_data(cluster_yaml: str = "", misc_yaml: str = "") -> dict:
    return yaml.safe_load(_BASE_SETTINGS + cluster_yaml + misc_yaml)


def _make_inputs(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("geometry.in", "control.in"):
        (directory / name).write_text("")
    return directory / "control.in", directory / "geometry.in"


@pytest.mark.parametrize("kind", ["dot", "dotdot", "absolute"])
def test_calc_dir_refused_when_it_contains_workdir(
    tmp_path, monkeypatch, kind
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    calc_dir = {"dot": ".", "dotdot": "..", "absolute": str(workdir)}[kind]
    data = _settings_data(_local_cluster(calc_dir=calc_dir))
    with pytest.raises(
        ValidationError,
        match=r"calc_dir .* contains the current working directory",
    ):
        AimsPAXSettings.model_validate(data)


def test_calc_dir_refused_when_it_contains_output_dir(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    shared = tmp_path / "shared"
    data = _settings_data(
        _local_cluster(calc_dir=str(shared)),
        _misc(output_dir=str(shared / "out")),
    )
    with pytest.raises(ValidationError, match="contains MISC.output_dir"):
        AimsPAXSettings.model_validate(data)


def test_calc_dir_refused_when_it_equals_dataset_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = _settings_data(_local_cluster(calc_dir="data"))
    with pytest.raises(ValidationError, match="MISC.dataset_dir"):
        AimsPAXSettings.model_validate(data)


def test_calc_dir_refused_when_it_contains_input_file_dir(
    tmp_path, monkeypatch
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    _, geometry = _make_inputs(tmp_path / "inputs")
    data = _settings_data(
        _local_cluster(calc_dir=str(tmp_path / "inputs")),
        _misc(
            output_dir=str(tmp_path / "out"), path_to_geometry=str(geometry)
        ),
    )
    with pytest.raises(ValidationError, match="MISC.path_to_geometry"):
        AimsPAXSettings.model_validate(data)


def test_calc_dir_refused_when_dict_control_file_inside(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    control, geometry = _make_inputs(tmp_path / "inputs")
    data = _settings_data(
        _local_cluster(calc_dir=str(tmp_path / "inputs")),
        _misc(
            output_dir=str(tmp_path / "out"),
            path_to_control={0: str(control)},
            path_to_geometry={0: str(geometry)},
        ),
    )
    with pytest.raises(ValidationError, match="MISC.path_to_control"):
        AimsPAXSettings.model_validate(data)


def test_calc_dir_refused_via_symlink(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    data = _settings_data(_local_cluster(calc_dir="link"))
    with pytest.raises(
        ValidationError, match="contains the current working directory"
    ):
        AimsPAXSettings.model_validate(data)


def test_calc_dir_refused_even_with_clean_dirs_false(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = _settings_data(_local_cluster(calc_dir=".", clean_dirs=False))
    with pytest.raises(ValidationError, match="calc_dir"):
        AimsPAXSettings.model_validate(data)


def test_slurm_calc_dir_equal_to_workdir_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cluster = f"""
CLUSTER:
  type: slurm
  launch_str: "srun aims.x"
  calc_dir: {json.dumps(str(tmp_path))}
  parsl_options:
    nodes_per_block: 1
    init_blocks: 1
    min_blocks: 1
    max_blocks: 1
    label: test
  slurm_str: |
    #SBATCH --partition=debug
  worker_str: "export OMP_NUM_THREADS=1"
"""
    with pytest.raises(ValidationError, match="calc_dir"):
        AimsPAXSettings.model_validate(_settings_data(cluster))


@pytest.mark.parametrize(
    "calc_dir",
    ["calc", "./calc", "TMP/calc", "TMP_calc"],
    ids=["relative", "dot-relative", "absolute-subdir", "absolute-sibling"],
)
def test_calc_dir_dedicated_subdirectory_accepted(
    tmp_path, monkeypatch, calc_dir
):
    monkeypatch.chdir(tmp_path)
    calc_dir = calc_dir.replace("TMP", str(tmp_path))
    data = _settings_data(_local_cluster(calc_dir=calc_dir))
    s = AimsPAXSettings.model_validate(data)
    assert s.CLUSTER.calc_dir == Path(calc_dir)


def test_default_calc_dir_accepted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = AimsPAXSettings.model_validate(_settings_data(_local_cluster()))
    assert s.CLUSTER is not None


def test_calc_dir_check_skipped_without_cluster(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = AimsPAXSettings.model_validate(_settings_data())
    assert s.CLUSTER is None
