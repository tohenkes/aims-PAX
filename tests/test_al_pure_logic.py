"""
Pure-logic unit tests for ALRunningManager, ALDataManager, and
TrainingOrchestrator.  No real constructors are invoked — stubs are
assembled with __new__ + attribute injection.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aims_PAX.procedures.al_managers import (
    ALDataManager,
    ALRunningManager,
    TrainingOrchestrator,
)

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


def make_point():
    """Minimal received_point stub (needs mutable info dict)."""
    return SimpleNamespace(info={}, arrays={})


def make_running_manager(**overrides):
    config = SimpleNamespace(
        num_trajectories=4,
        max_train_set_size=100,
        desired_accuracy=0.01,
        max_MD_steps=1000,
    )
    state = SimpleNamespace(
        num_MD_limits_reached=0,
        current_valid_error=1.0,
        trajectory_status={0: "running", 1: "running"},
    )
    ensemble = SimpleNamespace(train_dataset_len=0)
    mgr = ALRunningManager.__new__(ALRunningManager)
    mgr.config = config
    mgr.state_manager = state
    mgr.ensemble_manager = ensemble
    for k, v in overrides.items():
        if k.startswith("cfg_"):
            setattr(config, k[4:], v)
        elif k.startswith("state_"):
            setattr(state, k[6:], v)
        elif k.startswith("ens_"):
            setattr(ensemble, k[4:], v)
    return mgr


def make_data_manager(train_contents=None, max_size=10):
    train = list(train_contents) if train_contents is not None else []
    mgr = ALDataManager.__new__(ALDataManager)
    mgr.config = SimpleNamespace(
        max_train_set_size=max_size,
        valid_ratio=0.1,
        use_multihead_model=False,
        all_heads=None,
        r_max=5.0,
        key_specification=None,
        r_max_lr=None,
        replay_strategy="full_dataset",
    )
    mgr.ensemble_manager = SimpleNamespace(
        z_table=None,
        train_dataset_len=len(train),
        ensemble_ase_sets={"m0": {"train": train, "valid": []}},
        ensemble_model_sets={"m0": {"train": [], "valid": {"Default": []}}},
    )
    mgr.state_manager = SimpleNamespace(
        trajectory_status={0: "running"},
        num_workers_training=0,
        train_points_added=0,
        valid_points_added=0,
        total_points_added=0,
        last_point_added={0: None},
    )
    mgr.current_head_name = "Default"
    mgr._log_dataset_sizes = lambda tag: None
    return mgr


def make_orchestrator(
    valid_skip=3, intermediate_epochs_al=10, traj_epochs=None
):
    mgr = TrainingOrchestrator.__new__(TrainingOrchestrator)
    mgr.config = SimpleNamespace(
        valid_skip=valid_skip,
        intermediate_epochs_al=intermediate_epochs_al,
    )
    mgr.state_manager = SimpleNamespace(
        trajectory_intermediate_epochs={0: traj_epochs or 0}
    )
    return mgr


def make_session(current_epoch=0, max_epochs=10, is_convergence=True):
    return SimpleNamespace(
        current_epoch=current_epoch,
        max_epochs=max_epochs,
        is_convergence=is_convergence,
    )


# ---------------------------------------------------------------------------
# §1 — ALRunningManager stopping conditions (11 tests)
# ---------------------------------------------------------------------------

# --- check_all_trajectories_reached_limit ---


def test_trajectories_not_all_reached():
    mgr = make_running_manager(
        state_num_MD_limits_reached=3, cfg_num_trajectories=4
    )
    assert mgr.check_all_trajectories_reached_limit() is False


def test_all_trajectories_reached():
    mgr = make_running_manager(
        state_num_MD_limits_reached=4, cfg_num_trajectories=4
    )
    assert mgr.check_all_trajectories_reached_limit() is True


def test_trajectories_over_count_returns_false():
    # implementation uses ==, not >=, so exceeding count returns False
    mgr = make_running_manager(
        state_num_MD_limits_reached=5, cfg_num_trajectories=4
    )
    assert mgr.check_all_trajectories_reached_limit() is False


# --- check_max_training_set_size_reached ---


def test_train_size_below_limit():
    mgr = make_running_manager(
        ens_train_dataset_len=50, cfg_max_train_set_size=100
    )
    assert mgr.check_max_training_set_size_reached() is False


def test_train_size_exactly_at_limit():
    mgr = make_running_manager(
        ens_train_dataset_len=100, cfg_max_train_set_size=100
    )
    assert mgr.check_max_training_set_size_reached() is True


def test_train_size_above_limit():
    mgr = make_running_manager(
        ens_train_dataset_len=150, cfg_max_train_set_size=100
    )
    assert mgr.check_max_training_set_size_reached() is True


# --- check_desired_accuracy_reached ---


def test_accuracy_not_reached():
    mgr = make_running_manager(
        state_current_valid_error=0.1, cfg_desired_accuracy=0.01
    )
    assert mgr.check_desired_accuracy_reached() is False


def test_accuracy_reached():
    mgr = make_running_manager(
        state_current_valid_error=0.005, cfg_desired_accuracy=0.01
    )
    assert mgr.check_desired_accuracy_reached() is True


def test_accuracy_at_threshold_not_reached():
    # strict <: equal-to-threshold is not "reached"
    mgr = make_running_manager(
        state_current_valid_error=0.01, cfg_desired_accuracy=0.01
    )
    assert mgr.check_desired_accuracy_reached() is False


# --- should_terminate_worker ---
# Real signature: should_terminate_worker(self, current_MD_step, idx)


def test_terminate_over_limit_running():
    mgr = make_running_manager(cfg_max_MD_steps=1000)
    mgr.state_manager.num_MD_limits_reached = 0
    mgr.state_manager.trajectory_status = {0: "running"}
    result = mgr.should_terminate_worker(1001, 0)
    assert result is True
    assert mgr.state_manager.num_MD_limits_reached == 1
    assert mgr.state_manager.trajectory_status[0] == "killed"


def test_terminate_at_limit_not_over():
    # > not >=: step == max_MD_steps does NOT terminate
    mgr = make_running_manager(cfg_max_MD_steps=1000)
    mgr.state_manager.trajectory_status = {0: "running"}
    result = mgr.should_terminate_worker(1000, 0)
    assert result is False


def test_terminate_non_running_status():
    # AND condition: even over limit, non-"running" status is not terminated
    mgr = make_running_manager(cfg_max_MD_steps=1000)
    mgr.state_manager.trajectory_status = {0: "training"}
    result = mgr.should_terminate_worker(1001, 0)
    assert result is False


# ---------------------------------------------------------------------------
# §2a — _add_to_training_set direct (6 tests)
# ---------------------------------------------------------------------------

PATCH_CMD = "aims_PAX.procedures.al_managers.create_model_dataset"


def test_add_training_sets_status():
    mgr = make_data_manager()
    ALDataManager._add_to_training_set(mgr, 0, make_point(), [1])
    assert mgr.state_manager.trajectory_status[0] == "training"


def test_add_training_updates_dataset_len():
    mgr = make_data_manager(train_contents=["a", "b"])
    ALDataManager._add_to_training_set(mgr, 0, make_point(), [1])
    assert mgr.ensemble_manager.train_dataset_len == 3


def test_add_training_under_limit_returns_false():
    mgr = make_data_manager(train_contents=list(range(5)), max_size=10)
    result = ALDataManager._add_to_training_set(mgr, 0, make_point(), [1])
    assert result is False


def test_add_training_exactly_at_limit_returns_false():
    # 9 items + 1 appended = 10; 10 > 10 is False (uses >, not >=)
    mgr = make_data_manager(train_contents=list(range(9)), max_size=10)
    result = ALDataManager._add_to_training_set(mgr, 0, make_point(), [1])
    assert result is False


def test_add_training_over_limit_returns_true():
    # 10 items + 1 appended = 11; 11 > 10 is True
    mgr = make_data_manager(train_contents=list(range(10)), max_size=10)
    result = ALDataManager._add_to_training_set(mgr, 0, make_point(), [1])
    assert result is True


def test_add_training_increments_train_points_added():
    mgr = make_data_manager()
    ALDataManager._add_to_training_set(mgr, 0, make_point(), [1])
    assert mgr.state_manager.train_points_added == 1


# ---------------------------------------------------------------------------
# §2b — handle_received_point (3 tests, patch create_model_dataset)
# ---------------------------------------------------------------------------


def test_routes_to_validation_when_quota_not_met():
    mgr = make_data_manager()
    mgr.state_manager.total_points_added = 10
    mgr.state_manager.valid_points_added = 0  # 0 < 0.1*10=1.0 → validation
    with patch(PATCH_CMD, return_value=[SimpleNamespace()]):
        result = mgr.handle_received_point(0, make_point())
    assert result is False
    assert mgr.state_manager.valid_points_added == 1


def test_routes_to_training_when_quota_met():
    mgr = make_data_manager()
    mgr.state_manager.total_points_added = 10
    mgr.state_manager.valid_points_added = 1  # 1 < 1.0 is False → training
    with patch(PATCH_CMD, return_value=[SimpleNamespace()]):
        result = mgr.handle_received_point(0, make_point())
    assert result is False
    assert mgr.state_manager.train_points_added == 1


def test_handle_returns_true_when_max_size_reached():
    mgr = make_data_manager(train_contents=list(range(10)), max_size=10)
    mgr.state_manager.total_points_added = 10
    mgr.state_manager.valid_points_added = 1  # force training path
    with patch(PATCH_CMD, return_value=[SimpleNamespace()]):
        result = mgr.handle_received_point(0, make_point())
    assert result is True


# ---------------------------------------------------------------------------
# §3 — TrainingOrchestrator._should_validate (7 tests)
# ---------------------------------------------------------------------------

# --- Convergence path (is_convergence=True) ---


def test_validate_convergence_at_skip():
    orch = make_orchestrator(valid_skip=3)
    session = make_session(current_epoch=3, max_epochs=10, is_convergence=True)
    assert orch._should_validate(session, 0) is True


def test_validate_convergence_not_at_skip():
    orch = make_orchestrator(valid_skip=3)
    session = make_session(current_epoch=1, max_epochs=10, is_convergence=True)
    assert orch._should_validate(session, 0) is False


def test_validate_convergence_zero_epoch():
    # epoch 0 always validates (0 % any_positive == 0)
    orch = make_orchestrator(valid_skip=3)
    session = make_session(current_epoch=0, max_epochs=10, is_convergence=True)
    assert orch._should_validate(session, 0) is True


def test_validate_convergence_last_epoch():
    # last epoch (max_epochs - 1) always validates regardless of skip
    orch = make_orchestrator(valid_skip=3)
    session = make_session(current_epoch=9, max_epochs=10, is_convergence=True)
    assert orch._should_validate(session, 0) is True


# --- Intermediate path (is_convergence=False) ---


def test_validate_intermediate_at_skip():
    orch = make_orchestrator(
        valid_skip=3, intermediate_epochs_al=10, traj_epochs=3
    )
    session = make_session(is_convergence=False)
    assert orch._should_validate(session, 0) is True


def test_validate_intermediate_not_at_skip():
    orch = make_orchestrator(
        valid_skip=3, intermediate_epochs_al=10, traj_epochs=1
    )
    session = make_session(is_convergence=False)
    assert orch._should_validate(session, 0) is False


def test_validate_intermediate_last_epoch():
    # intermediate_epochs_al - 1 = 9 triggers last-epoch path
    orch = make_orchestrator(
        valid_skip=3, intermediate_epochs_al=10, traj_epochs=9
    )
    session = make_session(is_convergence=False)
    assert orch._should_validate(session, 0) is True
