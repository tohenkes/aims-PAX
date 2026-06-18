"""
End-to-end teacher-student pipeline tests.

Variant A (hermetic, implemented here):
  foundational sampler: SO3LRCalculator from so3krates_torch (pre-installed)
  teacher reference:   TorchkratesCalculator + local so3_zbl_r6_des370K.model
  no network access required.

Variant B (documented, not implemented):
  foundational_model: mace-mp
  teacher_reference_settings.model_type: mace-mp, mace_model: small
  downloads MACE-MP weights on first use — requires network.

Runtime: ~2-5 min on CPU depending on system. Guarded by @pytest.mark.slow.
PARSL DFK: managed by procedures; each procedure calls parsl.dfk().cleanup()
at the end of run(). No fixture-level teardown needed.
"""

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import ase.io
import pytest

EXAMPLE_LOCAL = Path(__file__).parent.parent / "example" / "local"

_AIMSPAX_TEMPLATE = """\
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
  foundational_model: so3lr
  foundational_model_settings:
    r_max_lr: null
  use_teacher_reference: true
  teacher_reference_settings:
    model_type: so3lr
    model_path: {model_path}
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
    model_type: so3lr
    model_path: {model_path}
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
def pipeline_ws(tmp_path_factory):
    """Shared module workspace: setup -> IDG run -> yield for AL test."""
    parsl = pytest.importorskip("parsl")  # noqa: F841
    tmp = tmp_path_factory.mktemp("ts_e2e")

    # --- copy example assets ---
    for name in (
        "geometry.in",
        "model.yaml",
        "so3_zbl_r6_des370K.model",
    ):
        shutil.copy(EXAMPLE_LOCAL / name, tmp / name)
    # placeholder — unused in teacher mode
    (tmp / "control.in").write_text("")

    model_path_abs = str(tmp / "so3_zbl_r6_des370K.model")

    # --- write aimsPAX.yaml with absolute model path ---
    aimspax_text = _AIMSPAX_TEMPLATE.format(model_path=model_path_abs)
    (tmp / "aimsPAX.yaml").write_text(aimspax_text)

    # --- change CWD so relative paths in model.yaml resolve to tmp ---
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
            close_parsl=True,  # IDG owns its PARSL DFK
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
def test_idg_teacher_student(pipeline_ws):
    ws = pipeline_ws
    ds = ws.initial_ds

    # --- training/validation XYZ files written ---
    train_dir = Path(ds.dataset_dir) / "initial" / "training"
    valid_dir = Path(ds.dataset_dir) / "initial" / "validation"
    train_files = list(train_dir.glob("initial_train_set_*.xyz"))
    valid_files = list(valid_dir.glob("initial_valid_set_*.xyz"))
    assert len(train_files) >= 1, "No training XYZ files written"
    assert len(valid_files) >= 1, "No validation XYZ files written"

    # --- every structure carries REF_energy and REF_forces ---
    for xyz_path in train_files + valid_files:
        atoms_list = ase.io.read(str(xyz_path), index=":")
        for atoms in atoms_list:
            assert (
                "REF_energy" in atoms.info
            ), f"REF_energy missing in {xyz_path}"
            assert (
                "REF_forces" in atoms.arrays
            ), f"REF_forces missing in {xyz_path}"

    # --- at least one ensemble model file saved ---
    model_files = list(
        Path(ds.model_settings.GENERAL.model_dir).glob("*.model")
    )
    assert len(model_files) >= 1, "No model checkpoint files written"


@pytest.mark.slow
def test_al_teacher_student(pipeline_ws):
    """AL run after IDG; uses IDG models + datasets from same workspace."""
    ws = pipeline_ws

    # IDG must have left model files; AL will load these for its ensemble.
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
    al.run()  # terminates when max_MD_steps reached for the 1 trajectory

    cfg = al.config

    # --- AL terminated: final training dataset has points ---
    final_train_dir = Path(cfg.dataset_dir) / "final" / "training"
    assert final_train_dir.exists(), "AL final training dir not created"
    final_files = list(final_train_dir.glob("*.xyz"))
    assert len(final_files) >= 1, "No final training XYZ files written"

    # --- at least one teacher-labeled point added ---
    # With ensemble_size=2 and c_x=-1.0, threshold ~= 0 so any non-zero
    # uncertainty triggers labeling. Count via state_manager.
    assert al.state_manager.total_points_added >= 1, (
        "Expected at least 1 teacher-labeled point; "
        f"got {al.state_manager.total_points_added}"
    )

    # --- ensemble models were saved after AL ---
    model_dir = Path(cfg.model_settings.GENERAL.model_dir)
    model_files = list(model_dir.glob("*.model"))
    assert len(model_files) >= 1, "No model files saved after AL"
