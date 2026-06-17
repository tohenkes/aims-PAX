import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aims_PAX.procedures.al_managers import (
    ALTrainingManager,
    TrainingOrchestrator,
    TrainingSession,
)

# ===========================================================================
# Stub helpers
# ===========================================================================


def make_conv_validation_stub(
    *, patience=2, margin=0.01, no_improvement=0, best_loss=1.0
):
    orch = TrainingOrchestrator.__new__(TrainingOrchestrator)
    orch.config = SimpleNamespace(
        convergence_patience=patience,
        margin=margin,
    )
    session = TrainingSession.__new__(TrainingSession)
    session.best_valid_loss = best_loss
    session.best_epoch = 0
    session.no_improvement = no_improvement
    session.current_epoch = 1
    session.training_setups = {"m0": {"checkpoint_handler": None}}
    ensemble = {"m0": MagicMock()}
    return orch, session, ensemble


def make_training_manager(*, intermediate_epochs=3, ensemble_size=1):
    tags = [f"m{i}" for i in range(ensemble_size)]
    mgr = ALTrainingManager.__new__(ALTrainingManager)
    mgr.config = SimpleNamespace(
        intermediate_epochs_al=intermediate_epochs,
        use_multihead_model=False,
        create_restart=False,
    )
    mgr.state_manager = SimpleNamespace(
        trajectory_intermediate_epochs={0: 0},
        trajectory_total_epochs={0: 0},
        total_epoch=0,
    )
    mgr.ensemble_manager = SimpleNamespace(
        training_setups={tag: {} for tag in tags},
        ensemble={tag: MagicMock() for tag in tags},
        ensemble_model_sets={},
    )
    mgr.restart_manager = MagicMock()
    orchestrator = MagicMock()
    orchestrator.save_restart = False
    orchestrator.validate_and_update_state.return_value = False
    orchestrator.best_member = None
    mgr.orchestrator = orchestrator
    return mgr


def make_converge_manager(*, max_epochs=3, ensemble_size=1):
    tags = [f"m{i}" for i in range(ensemble_size)]
    mgr = ALTrainingManager.__new__(ALTrainingManager)
    mgr.config = SimpleNamespace(
        max_convergence_epochs=max_epochs,
        use_multihead_model=False,
        convergence_patience=2,
    )
    mgr.state_manager = SimpleNamespace(total_epoch=0)
    mgr.ensemble_manager = SimpleNamespace(
        ensemble={tag: MagicMock() for tag in tags},
        ensemble_model_sets={},
    )
    mgr.training_setups_convergence = {tag: {} for tag in tags}
    mgr._setup_convergence = lambda: None
    mgr._get_initial_epoch = lambda: 0
    mgr._final_save = MagicMock()
    orchestrator = MagicMock()
    orchestrator.validate_and_update_state.return_value = False
    mgr.orchestrator = orchestrator
    return mgr


# ===========================================================================
# §1 — TrainingSession initialization
# ===========================================================================


def test_session_defaults():
    session = TrainingSession({}, {}, max_epochs=5)
    assert session.current_epoch == 0
    assert session.best_valid_loss == np.inf
    assert session.no_improvement == 0
    assert session.best_epoch == 0
    assert session.is_convergence is False


def test_session_initial_epoch_offset():
    session = TrainingSession({}, {}, 5, initial_epoch=10)
    assert session.current_epoch == 10


# ===========================================================================
# §2 — _handle_convergence_validation
# ===========================================================================


@patch("aims_PAX.procedures.al_managers.save_checkpoint")
def test_conv_val_improvement_updates_best_loss(mock_save):
    orch, session, ensemble = make_conv_validation_stub(
        best_loss=1.0, margin=0.01
    )
    orch._handle_convergence_validation(session, 0.5, ensemble)
    assert session.best_valid_loss == 0.5
    assert session.no_improvement == 0


@patch("aims_PAX.procedures.al_managers.save_checkpoint")
def test_conv_val_improvement_resets_no_improvement(mock_save):
    orch, session, ensemble = make_conv_validation_stub(
        no_improvement=2, best_loss=1.0
    )
    orch._handle_convergence_validation(session, 0.5, ensemble)
    assert session.no_improvement == 0


@patch("aims_PAX.procedures.al_managers.save_checkpoint")
def test_conv_val_no_improvement_increments_counter(mock_save):
    orch, session, ensemble = make_conv_validation_stub(
        no_improvement=0, best_loss=1.0
    )
    orch._handle_convergence_validation(session, 1.0, ensemble)
    assert session.no_improvement == 1


@patch("aims_PAX.procedures.al_managers.save_checkpoint")
def test_conv_val_returns_true_when_patience_exceeded(mock_save):
    orch, session, ensemble = make_conv_validation_stub(
        no_improvement=3, patience=2
    )
    result = orch._handle_convergence_validation(session, 1.0, ensemble)
    assert result is True


@patch("aims_PAX.procedures.al_managers.save_checkpoint")
def test_conv_val_returns_false_below_patience(mock_save):
    orch, session, ensemble = make_conv_validation_stub(
        no_improvement=0, patience=5
    )
    result = orch._handle_convergence_validation(session, 1.0, ensemble)
    assert result is False


# ===========================================================================
# §3 — Utility methods
# ===========================================================================


def test_update_epoch_counters():
    mgr = make_training_manager()
    mgr._update_epoch_counters(0)
    assert mgr.state_manager.trajectory_total_epochs[0] == 1
    assert mgr.state_manager.trajectory_intermediate_epochs[0] == 1
    assert mgr.state_manager.total_epoch == 1


def test_check_batch_size_returns_set_size():
    mgr = make_training_manager()
    mgr.ensemble_manager.ensemble_model_sets = {
        "m0": {"train": list(range(10))}
    }
    result = mgr._check_batch_size(4, "m0")
    assert result == 4


def test_check_batch_size_falls_back_to_one():
    mgr = make_training_manager()
    mgr.ensemble_manager.ensemble_model_sets = {
        "m0": {"train": list(range(2))}
    }
    result = mgr._check_batch_size(4, "m0")
    assert result == 1


# ===========================================================================
# §4 — perform_training loop control
# ===========================================================================


def test_perform_training_train_count_single_member():
    mgr = make_training_manager(intermediate_epochs=3, ensemble_size=1)
    mgr._finalize_training = lambda idx: None
    mgr.perform_training(idx=0)
    assert mgr.orchestrator.train_single_epoch.call_count == 3


def test_perform_training_train_count_two_members():
    mgr = make_training_manager(intermediate_epochs=2, ensemble_size=2)
    mgr._finalize_training = lambda idx: None
    mgr.perform_training(idx=0)
    assert mgr.orchestrator.train_single_epoch.call_count == 4


def test_perform_training_validate_called_per_epoch():
    mgr = make_training_manager(intermediate_epochs=4, ensemble_size=1)
    mgr._finalize_training = lambda idx: None
    mgr.perform_training(idx=0)
    assert mgr.orchestrator.validate_and_update_state.call_count == 4


def test_perform_training_runs_all_epochs_regardless_of_validation():
    mgr = make_training_manager(intermediate_epochs=5, ensemble_size=1)
    mgr.orchestrator.validate_and_update_state.side_effect = [
        False,
        True,
        False,
        False,
        False,
    ]
    mgr._finalize_training = lambda idx: None
    mgr.perform_training(idx=0)
    assert mgr.orchestrator.train_single_epoch.call_count == 5


# ===========================================================================
# §5 — converge loop control
# ===========================================================================


def test_converge_calls_train_per_epoch_per_member():
    mgr = make_converge_manager(max_epochs=3, ensemble_size=2)
    mgr.converge()
    assert mgr.orchestrator.train_single_epoch.call_count == 6


def test_converge_stops_early_and_calls_final_save():
    mgr = make_converge_manager(max_epochs=5, ensemble_size=1)
    mgr.orchestrator.validate_and_update_state.side_effect = [False, True]
    mgr.converge()
    assert mgr.orchestrator.train_single_epoch.call_count == 2
    assert mgr._final_save.call_count == 1


def test_converge_calls_final_save_at_max_epochs():
    mgr = make_converge_manager(max_epochs=3, ensemble_size=1)
    mgr.converge()
    assert mgr._final_save.call_count == 1


def test_converge_best_trains_only_best_member():
    mgr = make_converge_manager(max_epochs=3, ensemble_size=2)
    mgr.config.converge_best = True

    def _reduce_to_best():
        # Simulate _setup_sh_convergence: keep only the best member
        mgr.ensemble_manager.ensemble.pop("m1", None)

    mgr._setup_convergence = _reduce_to_best
    mgr.converge()
    # 2-member ensemble reduced to 1 → 3 train calls, not 6
    assert mgr.orchestrator.train_single_epoch.call_count == 3
