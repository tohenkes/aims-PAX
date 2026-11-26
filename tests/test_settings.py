"""
Tests yaml settings for aims-PAX project
"""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aims_PAX.settings import AimsPAXSettings, ModelSettings
from tests.helpers import mace_cfg, so3lr_cfg


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


def test_calc_dir_refused_when_it_equals_dataset_dir_under_other_output_dir(
    tmp_path, monkeypatch
):
    """dataset_dir is resolved against output_dir, not the cwd."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    output_dir = tmp_path / "out"
    data = _settings_data(
        _local_cluster(calc_dir=str(output_dir / "data")),
        _misc(output_dir=str(output_dir)),
    )
    with pytest.raises(ValidationError, match="MISC.dataset_dir"):
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


# -- SO3LR-tauri architecture options ---------------------------------------


def test_unknown_so3lr_architecture_key_rejected():
    with pytest.raises(ValidationError, match="use_rms_nrom"):
        ModelSettings.model_validate(so3lr_cfg(use_rms_nrom=True))


def test_unknown_so3krates_architecture_key_rejected():
    cfg = so3lr_cfg(use_rms_nrom=True)
    cfg["ARCHITECTURE"]["model_choice"] = "so3krates"
    with pytest.raises(ValidationError, match="use_rms_nrom"):
        ModelSettings.model_validate(cfg)


def test_mace_extra_keys_still_ignored():
    ModelSettings.model_validate(mace_cfg(some_old_key=1))  # no error


def test_tauri_defaults_match_so3krates_torch():
    s = ModelSettings.model_validate(so3lr_cfg()).ARCHITECTURE
    assert s.use_rms_norm is False
    assert s.qk_norm is False
    assert s.use_residual_scalars is False
    assert s.final_layer_bias is True
    assert s.nhl_repulsion_bool is False
    assert s.c6_ratios_bool is False
    assert s.use_simple_hirshfeld is False
    assert s.legacy_dispersion_bool is True
    assert s.use_pme is False
    assert s.pme_smearing is None
    assert s.pme_mesh_spacing is None
    assert s.use_pme_dispersion is False
    assert s.pme_dispersion_smearing is None
    assert s.pme_dispersion_mesh_spacing is None
    assert s.pme_mesh_rule == "jax"


def test_so3krates_has_core_tauri_flags():
    cfg = so3lr_cfg(use_rms_norm=True, final_layer_bias=False)
    cfg["ARCHITECTURE"]["model_choice"] = "so3krates"
    s = ModelSettings.model_validate(cfg).ARCHITECTURE
    assert s.use_rms_norm is True
    assert s.final_layer_bias is False


@pytest.mark.parametrize(
    "flag",
    ["c6_ratios_bool", "use_simple_hirshfeld", "use_pme_dispersion"],
)
def test_dispersion_flags_need_dispersion(flag):
    with pytest.raises(ValidationError, match=flag):
        ModelSettings.model_validate(
            so3lr_cfg(**{flag: True, "dispersion_energy_bool": False})
        )


def test_use_pme_needs_electrostatics():
    with pytest.raises(ValidationError, match="use_pme"):
        ModelSettings.model_validate(
            so3lr_cfg(use_pme=True, electrostatic_energy_bool=False)
        )


def test_dispersion_flags_valid_with_dispersion():
    s = ModelSettings.model_validate(
        so3lr_cfg(
            c6_ratios_bool=True,
            use_simple_hirshfeld=True,
            use_pme_dispersion=True,
            use_pme=True,
        )
    ).ARCHITECTURE
    assert s.c6_ratios_bool and s.use_pme and s.use_pme_dispersion


def test_nhl_and_zbl_together_only_warn(caplog):
    with caplog.at_level("WARNING"):
        s = ModelSettings.model_validate(
            so3lr_cfg(nhl_repulsion_bool=True, zbl_repulsion_bool=True)
        ).ARCHITECTURE
    assert s.nhl_repulsion_bool and s.zbl_repulsion_bool
    assert "NHL" in caplog.text


def test_invalid_pme_mesh_rule_rejected():
    with pytest.raises(ValidationError, match="pme_mesh_rule"):
        ModelSettings.model_validate(so3lr_cfg(pme_mesh_rule="other"))


def test_training_tauri_defaults():
    t = ModelSettings.model_validate(so3lr_cfg()).TRAINING
    assert t.pretrained_theory_level is None
    assert t.freeze_c6_ratios is True


def test_pretrained_theory_level_must_be_non_negative():
    cfg = so3lr_cfg()
    cfg["TRAINING"] = {"pretrained_theory_level": -1}
    with pytest.raises(ValidationError, match="pretrained_theory_level"):
        ModelSettings.model_validate(cfg)
    cfg["TRAINING"] = {"pretrained_theory_level": 0}
    s = ModelSettings.model_validate(cfg)
    assert s.TRAINING.pretrained_theory_level == 0


def test_num_theory_levels_not_a_setting():
    cfg = so3lr_cfg()
    cfg["TRAINING"] = {"num_theory_levels": 2}
    s = ModelSettings.model_validate(cfg)
    assert not hasattr(s.TRAINING, "num_theory_levels")


def test_example_configs_still_validate():
    example = Path(__file__).parent.parent / "example"
    checked = 0
    for p in example.rglob("*.yaml"):
        data = yaml.safe_load(p.read_text())
        if isinstance(data, dict) and "ARCHITECTURE" in data:
            ModelSettings.model_validate(data)
            checked += 1
    assert checked >= 2


# ── per-system indices must agree between geometry, control and MD ───────────


def test_control_dict_with_extra_index_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    control, geometry = _make_inputs(tmp_path / "inputs")
    data = _settings_data(
        _local_cluster(calc_dir=str(tmp_path / "calc")),
        _misc(
            output_dir=str(tmp_path / "out"),
            path_to_control={0: str(control), 1: str(control)},
            path_to_geometry={0: str(geometry)},
        ),
    )
    with pytest.raises(ValidationError, match="indices"):
        AimsPAXSettings.model_validate(data)


def test_md_dict_must_match_control_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    control, geometry = _make_inputs(tmp_path / "inputs")
    data = _settings_data(
        _local_cluster(calc_dir=str(tmp_path / "calc")),
        _misc(
            output_dir=str(tmp_path / "out"),
            path_to_control={0: str(control), 1: str(control)},
            path_to_geometry=str(geometry),
        ),
    )
    md = {"stat_ensemble": "NVT", "thermostat": "Langevin", "timestep": 1.0}
    data["MD"] = {i: {**md, "temperature": 300} for i in range(3)}
    with pytest.raises(ValidationError, match="MD"):
        AimsPAXSettings.model_validate(data)
