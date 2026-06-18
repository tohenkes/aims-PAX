"""
End-to-end teacher-student pipeline tests — mace-mp variant.

Requires network on first run; MACE downloads 'small' weights to its own
cache (~/.cache/mace) and subsequent runs are fully offline.

Variant A (SO3LR, hermetic) lives in test_teacher_student_e2e.py.
Runtime: ~5-10 min on CPU. Guarded by @pytest.mark.slow + @pytest.mark.network.
PARSL DFK: managed by procedures; no fixture-level teardown needed.
"""

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import ase.io
import pytest

EXAMPLE_LOCAL = Path(__file__).parent.parent / "example" / "local"

_AIMSPAX_TEMPLATE_MACEMP = """\
INITIAL_DATASET_GENERATION:
  desired_acc: 0.0
  desired_acc_scale_idg: 1.0
  valid_ratio: 0.1
  n_points_per_sampling_step_idg: 2
  valid_skip: 1
  intermediate_epochs_idg: 1
  ensemble_size: 2
  skip_step_initial: 1
  initial_sampling: foundational
  foundational_model: mace-mp
  foundational_model_settings:
    mace_model: small
  use_teacher_reference: true
  teacher_reference_settings:
    model_type: mace-mp
    mace_model: small
  max_initial_epochs: 2
  max_initial_set_size: 8
  converge_initial: false
  save_trajectories: false

ACTIVE_LEARNING:
  desired_acc: 0.0
  valid_ratio: 0.1
  epochs_per_worker: 1
  intermediate_epochs_al: 1
  valid_skip: 1
  ensemble_size: 2
  c_x: -1.0
  skip_step_mlff: 2
  num_trajectories: 1
  max_train_set_size: 4
  max_MD_steps: 6
  converge_al: false
  use_teacher_reference: true
  teacher_reference_settings:
    model_type: mace-mp
    mace_model: small
  save_trajectories: false

MD:
  stat_ensemble: NVT
  thermostat: Langevin
  temperature: 300.0
  timestep: 1.0
  friction: 0.001
  MD_seed: 42

CLUSTER:
  type: local
  max_workers: 1
"""


@pytest.fixture(scope="module")
def pipeline_ws_macemp(tmp_path_factory):
    """Shared module workspace: setup -> IDG run -> yield for AL test."""
    parsl = pytest.importorskip("parsl")  # noqa: F841
    tmp = tmp_path_factory.mktemp("ts_macemp_e2e")

    # copy geometry and model config only — no local .model file needed
    for name in ("geometry.in", "model.yaml"):
        shutil.copy(EXAMPLE_LOCAL / name, tmp / name)
    (tmp / "control.in").write_text("")  # placeholder — unused in teacher mode
    (tmp / "aimsPAX.yaml").write_text(_AIMSPAX_TEMPLATE_MACEMP)

    orig = os.getcwd()
    os.chdir(tmp)

    try:
        from aims_PAX.tools.utilities.input_utils import read_input_files
        from aims_PAX.procedures.initial_dataset import (
            InitialDatasetPARSLTeacher,
        )

        model_settings, aimspax_settings, _, _ = read_input_files(
            path_to_model_settings=str(tmp / "model.yaml"),
            path_to_aimsPAX_settings=str(tmp / "aimsPAX.yaml"),
            procedure="initial-ds",
        )
        initial_ds = InitialDatasetPARSLTeacher(
            model_settings=model_settings,
            aimsPAX_settings=aimspax_settings,
            path_to_control=str(tmp / "control.in"),
            path_to_geometry=str(tmp / "geometry.in"),
            close_parsl=True,
        )
        initial_ds.run()

        yield SimpleNamespace(
            tmp=tmp,
            initial_ds=initial_ds,
            model_settings=model_settings,
            aimspax_settings=aimspax_settings,
        )
    finally:
        os.chdir(orig)


@pytest.mark.slow
@pytest.mark.network
def test_idg_teacher_student_macemp(pipeline_ws_macemp):
    """IDG run: teacher-labeled XYZ files and model checkpoint written."""
    ds = pipeline_ws_macemp.initial_ds

    train_dir = Path(ds.dataset_dir) / "initial" / "training"
    valid_dir = Path(ds.dataset_dir) / "initial" / "validation"
    train_files = list(train_dir.glob("initial_train_set_*.xyz"))
    valid_files = list(valid_dir.glob("initial_valid_set_*.xyz"))
    assert len(train_files) >= 1, "No training XYZ files written"
    assert len(valid_files) >= 1, "No validation XYZ files written"

    for xyz_path in train_files + valid_files:
        atoms_list = ase.io.read(str(xyz_path), index=":")
        for atoms in atoms_list:
            assert (
                "REF_energy" in atoms.info
            ), f"REF_energy missing in {xyz_path}"
            assert (
                "REF_forces" in atoms.arrays
            ), f"REF_forces missing in {xyz_path}"

    model_files = list(
        Path(ds.model_settings.GENERAL.model_dir).glob("*.model")
    )
    assert len(model_files) >= 1, "No model checkpoint files written"


@pytest.mark.slow
@pytest.mark.network
def test_al_teacher_student_macemp(pipeline_ws_macemp):
    """AL run after IDG; uses IDG models + datasets from same workspace."""
    ws = pipeline_ws_macemp

    from aims_PAX.tools.utilities.input_utils import read_input_files
    from aims_PAX.procedures.active_learning import ALProcedurePARSL

    model_settings, aimspax_settings, _, _ = read_input_files(
        path_to_model_settings=str(ws.tmp / "model.yaml"),
        path_to_aimsPAX_settings=str(ws.tmp / "aimsPAX.yaml"),
        procedure="al",
    )
    al = ALProcedurePARSL(
        model_settings=model_settings,
        aimsPAX_settings=aimspax_settings,
        path_to_control=str(ws.tmp / "control.in"),
        path_to_geometry=str(ws.tmp / "geometry.in"),
    )
    al.run()

    cfg = al.config  # ALConfiguration — has dataset_dir, model_settings

    final_train_dir = Path(cfg.dataset_dir) / "final" / "training"
    assert final_train_dir.exists(), "AL final training dir not created"
    final_files = list(final_train_dir.glob("*.xyz"))
    assert len(final_files) >= 1, "No final training XYZ files written"

    assert al.state_manager.total_points_added >= 1, (
        "Expected at least 1 teacher-labeled point; "
        f"got {al.state_manager.total_points_added}"
    )

    model_files = list(
        Path(cfg.model_settings.GENERAL.model_dir).glob("*.model")
    )
    assert len(model_files) >= 1, "No model files saved after AL"
