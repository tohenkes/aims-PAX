import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.io import write

from aims_PAX.procedures.active_learning import ALProcedure
from aims_PAX.procedures.preparation import ALEnsemble

# ===========================================================================
# Stub helpers
# ===========================================================================


def make_al_loop_stub(
    *,
    train_dataset_len=0,
    max_train_set_size=99,
    num_trajectories=1,
    num_MD_limits_reached=0,
    current_valid_error=1.0,
    desired_accuracy=0.0,
    status="waiting",
):
    stub = SimpleNamespace(
        config=SimpleNamespace(
            num_trajectories=num_trajectories,
            max_train_set_size=max_train_set_size,
            desired_accuracy=desired_accuracy,
        ),
        state_manager=SimpleNamespace(
            trajectory_status={i: status for i in range(num_trajectories)},
            num_MD_limits_reached=num_MD_limits_reached,
            current_valid_error=current_valid_error,
        ),
        ensemble_manager=SimpleNamespace(train_dataset_len=train_dataset_len),
        _waiting_task=MagicMock(return_value=None),
        _training_task=MagicMock(),
        _running_task=MagicMock(),
        dft_manager=MagicMock(),
    )
    return stub


def make_training_task_stub(*, trajectory_total_epochs=5, epochs_per_worker=5):
    stub = SimpleNamespace(
        config=SimpleNamespace(epochs_per_worker=epochs_per_worker),
        state_manager=SimpleNamespace(
            trajectory_status={0: "training"},
            trajectory_total_epochs={0: trajectory_total_epochs},
            num_workers_training=1,
        ),
        ensemble_manager=SimpleNamespace(ensemble_model_sets={}),
        train_manager=MagicMock(),
        _assign_models_to_trajectories=MagicMock(),
    )
    stub.train_manager.prepare_training.return_value = {}
    return stub


def make_running_task_stub():
    point = MagicMock()
    stub = SimpleNamespace(
        state_manager=SimpleNamespace(
            trajectory_MD_steps={0: 10},
        ),
        run_manager=MagicMock(),
        md_manager=MagicMock(),
        restart_manager=MagicMock(),
        trajectories=MagicMock(),
        config=SimpleNamespace(analysis=False),
        get_uncertainty=MagicMock(),
    )
    stub.run_manager.should_terminate_worker.return_value = False
    stub.run_manager.update_md_step.return_value = 11
    stub.run_manager.calculate_uncertainty_data.return_value = (
        point,
        MagicMock(),
        0.1,
    )
    return stub, point


# ===========================================================================
# §1 — _al_loop exit conditions
# ===========================================================================


def test_al_loop_exits_on_max_train_set_size(caplog):
    stub = make_al_loop_stub(
        train_dataset_len=4,
        max_train_set_size=4,
        status="waiting",
    )
    with caplog.at_level(logging.INFO):
        ALProcedure._al_loop(stub)
    assert stub.dft_manager.finalize_dft.call_count == 1
    assert "Maximum size of training set reached" in caplog.text


def test_al_loop_exits_on_all_md_limits_reached(caplog):
    stub = make_al_loop_stub(
        num_MD_limits_reached=1,
        num_trajectories=1,
        status="running",
    )
    with caplog.at_level(logging.INFO):
        ALProcedure._al_loop(stub)
    assert stub.dft_manager.finalize_dft.call_count == 1
    assert "All trajectories reached maximum MD steps" in caplog.text


def test_al_loop_exits_on_desired_accuracy(caplog):
    stub = make_al_loop_stub(
        current_valid_error=0.001,
        desired_accuracy=0.01,
        status="waiting",
    )
    with caplog.at_level(logging.INFO):
        ALProcedure._al_loop(stub)
    assert stub.dft_manager.finalize_dft.call_count == 1
    assert "Desired accuracy reached" in caplog.text


# ===========================================================================
# §2 — _waiting_task routing
# ===========================================================================


def test_waiting_task_base_routes_to_data_manager():
    stub = SimpleNamespace(
        config=SimpleNamespace(restart=False),
        point=None,
        data_manager=MagicMock(),
    )
    ALProcedure._waiting_task(stub, idx=0)
    assert stub.data_manager.handle_received_point.called is True


# ===========================================================================
# §3 — _training_task status transitions
# ===========================================================================


def test_training_task_transitions_to_running_at_epoch_limit():
    stub = make_training_task_stub(
        trajectory_total_epochs=5, epochs_per_worker=5
    )
    ALProcedure._training_task(stub, idx=0)
    assert stub.state_manager.trajectory_status[0] == "running"
    assert stub.state_manager.trajectory_total_epochs[0] == 0
    assert stub.state_manager.num_workers_training == 0


def test_training_task_remains_in_training_below_limit():
    stub = make_training_task_stub(
        trajectory_total_epochs=2, epochs_per_worker=5
    )
    ALProcedure._training_task(stub, idx=0)
    assert stub.state_manager.trajectory_status[0] == "training"


def test_training_task_calls_prepare_and_train():
    stub = make_training_task_stub(
        trajectory_total_epochs=5, epochs_per_worker=5
    )
    ALProcedure._training_task(stub, idx=0)
    assert stub.train_manager.prepare_training.called
    assert stub.train_manager.perform_training.called


# ===========================================================================
# §4 — _assign_models_to_trajectories
# ===========================================================================


def test_assign_models_updates_trajectory_calc_models():
    model = MagicMock()
    traj = SimpleNamespace(calc=SimpleNamespace(models=None))
    stub = SimpleNamespace(
        config=SimpleNamespace(use_foundational=False),
        ensemble={"m0": model},
        trajectories={"t0": traj},
    )
    ALProcedure._assign_models_to_trajectories(stub)
    assert traj.calc.models == [model]


def test_assign_models_updates_ensemble_calc_for_foundational():
    model = MagicMock()
    stub = SimpleNamespace(
        config=SimpleNamespace(use_foundational=True),
        ensemble={"m0": model},
        mlff_manager=SimpleNamespace(
            mlff_calc_ensemble=SimpleNamespace(models=None)
        ),
    )
    ALProcedure._assign_models_to_trajectories(stub)
    assert stub.mlff_manager.mlff_calc_ensemble.models == [model]


# ===========================================================================
# §5 — _running_task
# ===========================================================================


def test_running_task_returns_killed_when_terminate_true():
    stub, _point = make_running_task_stub()
    stub.run_manager.should_terminate_worker.return_value = True
    result = ALProcedure._running_task(stub, idx=0)
    assert result == "killed"
    assert stub.run_manager.execute_md_step.called is False


def test_running_task_calls_execute_md_step_on_normal_path():
    stub, _point = make_running_task_stub()
    ALProcedure._running_task(stub, idx=0)
    assert stub.run_manager.execute_md_step.call_count == 1


def test_running_task_stores_point_from_uncertainty_data():
    stub, point = make_running_task_stub()
    ALProcedure._running_task(stub, idx=0)
    assert stub.point is point
    assert stub.run_manager.process_uncertainty_decision.call_count == 1


# ===========================================================================
# §6 — ALEnsemble._setup_datasets_from_provided_files
# ===========================================================================

_PATCH_TARGET = "aims_PAX.procedures.preparation.ase_to_model_ensemble_sets"


def _make_single_h2_extxyz(path: Path) -> None:
    """Write a minimal extxyz file with one H2 Atoms object."""
    atoms = [Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])]
    write(str(path), atoms)


def _make_ensemble_stub(tmp_path, train_src, valid_src):
    """Build a minimal SimpleNamespace that satisfies _setup_datasets_from_provided_files."""
    al_settings = SimpleNamespace(
        initial_train_dataset=train_src,
        initial_valid_dataset=valid_src,
    )
    config = SimpleNamespace(
        al_settings=al_settings,
        dataset_dir=tmp_path / "dataset",
        r_max=5.0,
        r_max_lr=None,
        seed=42,
        key_specification=None,
        all_heads=None,
    )
    stub = SimpleNamespace(
        config=config,
        ensemble={"tag_a": MagicMock(), "tag_b": MagicMock()},
        z_table=MagicMock(),
    )
    return stub


def test_single_dataset_replicated_to_all_members(tmp_path):
    """A single-path dataset is replicated to every ensemble member."""
    train_path = tmp_path / "train.extxyz"
    valid_path = tmp_path / "valid.extxyz"
    _make_single_h2_extxyz(train_path)
    _make_single_h2_extxyz(valid_path)

    stub = _make_ensemble_stub(tmp_path, train_path, valid_path)

    with patch(_PATCH_TARGET, return_value={}) as mock_convert:
        ALEnsemble._setup_datasets_from_provided_files(stub)

    assert "tag_a" in stub.ensemble_ase_sets
    assert "tag_b" in stub.ensemble_ase_sets
    assert len(stub.ensemble_ase_sets["tag_a"]["train"]) > 0
    assert len(stub.ensemble_ase_sets["tag_b"]["train"]) > 0
    mock_convert.assert_called_once()


def test_per_member_dict_assigns_correctly(tmp_path):
    """Per-member dict datasets are assigned to the correct tag."""
    train_a = tmp_path / "train_a.extxyz"
    train_b = tmp_path / "train_b.extxyz"
    valid_a = tmp_path / "valid_a.extxyz"
    valid_b = tmp_path / "valid_b.extxyz"

    # tag_a gets 1 structure, tag_b gets 2 structures
    write(str(train_a), [Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])])
    write(
        str(train_b),
        [
            Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]),
            Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.80]]),
        ],
    )
    write(str(valid_a), [Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])])
    write(str(valid_b), [Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])])

    train_src = {"tag_a": train_a, "tag_b": train_b}
    valid_src = {"tag_a": valid_a, "tag_b": valid_b}
    stub = _make_ensemble_stub(tmp_path, train_src, valid_src)

    with patch(_PATCH_TARGET, return_value={}):
        ALEnsemble._setup_datasets_from_provided_files(stub)

    assert len(stub.ensemble_ase_sets["tag_a"]["train"]) == 1
    assert len(stub.ensemble_ase_sets["tag_b"]["train"]) == 2


def test_per_member_dict_wrong_tags_raises(tmp_path):
    """Dict with keys that don't match ensemble tags raises ValueError."""
    train_x = tmp_path / "train_x.extxyz"
    valid_x = tmp_path / "valid_x.extxyz"
    _make_single_h2_extxyz(train_x)
    _make_single_h2_extxyz(valid_x)

    # ensemble has tag_a/tag_b but dict provides tag_x
    train_src = {"tag_x": train_x}
    valid_src = {"tag_x": valid_x}
    stub = _make_ensemble_stub(tmp_path, train_src, valid_src)

    with pytest.raises(ValueError, match="tag"):
        ALEnsemble._setup_datasets_from_provided_files(stub)


# ===========================================================================
# §7 — ALEnsemble._load_seeds_tags_dict fallback
# ===========================================================================


def test_seeds_tags_dict_fallback(tmp_path):
    """When seeds_tags_dict.npz is absent, dict is derived from ensemble keys."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    config = SimpleNamespace(
        seeds_tags_dict=None,
        dataset_dir=dataset_dir,
    )
    stub = SimpleNamespace(
        config=config,
        ensemble={"tag_a": MagicMock(), "tag_b": MagicMock()},
    )

    # No seeds_tags_dict.npz file present — fallback must trigger
    ALEnsemble._load_seeds_tags_dict(stub)

    assert stub.config.seeds_tags_dict is not None
    assert set(stub.config.seeds_tags_dict.keys()) == {"tag_a", "tag_b"}
