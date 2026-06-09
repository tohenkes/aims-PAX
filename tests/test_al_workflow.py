from types import SimpleNamespace
from unittest.mock import MagicMock

from aims_PAX.procedures.active_learning import (
    ALProcedure,
    ALProcedureSerial,
)

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


def test_al_loop_exits_on_max_train_set_size():
    stub = make_al_loop_stub(
        train_dataset_len=4,
        max_train_set_size=4,
        status="waiting",
    )
    ALProcedure._al_loop(stub)
    assert stub.dft_manager.finalize_dft.call_count == 1


def test_al_loop_exits_on_all_md_limits_reached():
    stub = make_al_loop_stub(
        num_MD_limits_reached=1,
        num_trajectories=1,
        status="running",
    )
    ALProcedure._al_loop(stub)
    assert stub.dft_manager.finalize_dft.call_count == 1


def test_al_loop_exits_on_desired_accuracy():
    stub = make_al_loop_stub(
        current_valid_error=0.001,
        desired_accuracy=0.01,
        status="waiting",
    )
    ALProcedure._al_loop(stub)
    assert stub.dft_manager.finalize_dft.call_count == 1


# ===========================================================================
# §2 — _waiting_task routing
# ===========================================================================


def test_waiting_task_serial_does_nothing():
    stub = SimpleNamespace(data_manager=MagicMock())
    ALProcedureSerial._waiting_task(stub, idx=0)
    assert stub.data_manager.mock_calls == []


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
