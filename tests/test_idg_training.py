from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aims_PAX.procedures.initial_dataset import InitialDatasetProcedure
from aims_PAX.procedures.preparation import PrepareInitialDatasetProcedure

# ===========================================================================
# Module-level patch targets
# ===========================================================================

TRAIN_MOD = "aims_PAX.procedures.initial_dataset"
PREP_MOD = "aims_PAX.procedures.preparation"

# ===========================================================================
# Stub factories
# ===========================================================================

_UPDATE_DATASETS_RETURN = (
    [],
    {
        "train": [None] * 5,
        "valid": {"Default": [None] * 2},
        "train_loader": None,
        "valid_loader": None,
    },
)

_VALIDATE_EPOCH_RETURN = ({"m0": 0.5}, 0.5, {"mae_f": 0.5}, None)


def make_train_stub():
    training_setup = {
        "loss_fn": MagicMock(),
        "optimizer": MagicMock(),
        "lr_scheduler": MagicMock(),
        "device": "cpu",
        "max_grad_norm": 1.0,
        "output_args": {},
        "ema": None,
        "checkpoint_handler": MagicMock(),
    }
    stub = SimpleNamespace(
        sampled_points=list(range(20)),
        ensemble={"m0": MagicMock()},
        ensemble_ase_sets={"m0": []},
        ensemble_model_sets={
            "m0": {
                "train": [None] * 5,
                "valid": {"Default": [None] * 2},
                "train_loader": None,
                "valid_loader": None,
            }
        },
        training_setups={"m0": training_setup},
        ensemble_atomic_energies={"m0": []},
        ensemble_atomic_energies_dict={"m0": {}},
        valid_ratio=0.1,
        z_table=None,
        seed=42,
        r_max=5.0,
        r_max_lr=None,
        all_heads=["Default"],
        key_specification=None,
        set_batch_size=4,
        set_valid_batch_size=4,
        model_choice="mace",
        scaling=None,
        update_atomic_energies=False,
        update_avg_num_neighbors=False,
        dtype="float64",
        device="cpu",
        model_settings=SimpleNamespace(
            GENERAL=SimpleNamespace(loss_dir=Path("/tmp/loss")),
            MISC=SimpleNamespace(error_table=False),
        ),
        intermediate_epochs_idg=3,
        valid_skip=1,
        analysis=False,
        epoch=0,
        max_initial_epochs=100,
        create_restart=False,
        dataset_dir=Path("/tmp/dataset"),
        desired_acc=0.0,
        desired_acc_scale_idg=10.0,
        step=0,
        current_valid=float("inf"),
        use_multihead_model=False,
    )
    stub._get_member_points = lambda number: []
    return stub


def make_train_stub_2members():
    stub = make_train_stub()
    ts = {
        "loss_fn": MagicMock(),
        "optimizer": MagicMock(),
        "lr_scheduler": MagicMock(),
        "device": "cpu",
        "max_grad_norm": 1.0,
        "output_args": {},
        "ema": None,
        "checkpoint_handler": MagicMock(),
    }
    stub.ensemble["m1"] = MagicMock()
    stub.ensemble_ase_sets["m1"] = []
    stub.ensemble_model_sets["m1"] = {
        "train": [None] * 5,
        "valid": {"Default": [None] * 2},
        "train_loader": None,
        "valid_loader": None,
    }
    stub.training_setups["m1"] = ts
    stub.ensemble_atomic_energies["m1"] = []
    stub.ensemble_atomic_energies_dict["m1"] = {}
    return stub


def make_converge_stub(tmp_path):
    return SimpleNamespace(
        ensemble={"m0": MagicMock()},
        ensemble_model_sets={
            "m0": {
                "train": [None] * 5,
                "valid": {"Default": [None] * 2},
                "train_loader": None,
                "valid_loader": None,
            }
        },
        ensemble_ase_sets={"m0": []},
        ensemble_atomic_energies={"m0": []},
        ensemble_atomic_energies_dict={"m0": {}},
        set_batch_size=4,
        set_valid_batch_size=4,
        model_choice="mace",
        scaling=None,
        update_atomic_energies=False,
        z_table=None,
        dtype="float64",
        device="cpu",
        config=SimpleNamespace(update_avg_num_neighbors=False),
        model_settings=SimpleNamespace(
            GENERAL=SimpleNamespace(
                loss_dir=tmp_path,
                model_dir=tmp_path,
            ),
            MISC=SimpleNamespace(error_table=False),
        ),
        restart=False,
        checkpoints_dir=tmp_path,
        mol_idxs=None,
        valid_skip=1,
        max_convergence_epochs=10,
        idg_settings=SimpleNamespace(convergence_patience=5),
        margin=0.001,
    )


def _make_setup_model_training_return():
    return {
        "loss_fn": MagicMock(),
        "optimizer": MagicMock(),
        "lr_scheduler": MagicMock(),
        "device": "cpu",
        "max_grad_norm": 1.0,
        "output_args": {},
        "ema": None,
        "checkpoint_handler": MagicMock(),
        "epoch": 0,
    }


# ===========================================================================
# §5.1 — _train iteration counts
# ===========================================================================


@patch(f"{TRAIN_MOD}.save_datasets")
@patch(f"{TRAIN_MOD}.save_checkpoint")
@patch(f"{TRAIN_MOD}.validate_epoch")
@patch(f"{TRAIN_MOD}.train_epoch")
@patch(f"{TRAIN_MOD}.tools")
@patch(f"{TRAIN_MOD}.update_model_auxiliaries")
@patch(
    f"{TRAIN_MOD}.create_dataloader", return_value=(MagicMock(), MagicMock())
)
@patch(f"{TRAIN_MOD}.update_datasets", return_value=_UPDATE_DATASETS_RETURN)
def test_train_calls_train_epoch_per_epoch_and_member(
    mock_update_datasets,
    mock_create_dataloader,
    mock_update_model_auxiliaries,
    mock_tools,
    mock_train_epoch,
    mock_validate_epoch,
    mock_save_checkpoint,
    mock_save_datasets,
):
    # 3 epochs, 1 member → train_epoch called 3 times
    stub = make_train_stub()
    stub.intermediate_epochs_idg = 3
    stub.valid_skip = 1000  # skips most validation; provide return anyway
    mock_validate_epoch.return_value = _VALIDATE_EPOCH_RETURN
    InitialDatasetProcedure._train(stub)
    assert mock_train_epoch.call_count == 3


@patch(f"{TRAIN_MOD}.save_datasets")
@patch(f"{TRAIN_MOD}.save_checkpoint")
@patch(f"{TRAIN_MOD}.validate_epoch")
@patch(f"{TRAIN_MOD}.train_epoch")
@patch(f"{TRAIN_MOD}.tools")
@patch(f"{TRAIN_MOD}.update_model_auxiliaries")
@patch(
    f"{TRAIN_MOD}.create_dataloader", return_value=(MagicMock(), MagicMock())
)
@patch(f"{TRAIN_MOD}.update_datasets", return_value=_UPDATE_DATASETS_RETURN)
def test_train_calls_train_epoch_for_each_ensemble_member(
    mock_update_datasets,
    mock_create_dataloader,
    mock_update_model_auxiliaries,
    mock_tools,
    mock_train_epoch,
    mock_validate_epoch,
    mock_save_checkpoint,
    mock_save_datasets,
):
    # 2 epochs × 2 members → train_epoch called 4 times
    stub = make_train_stub_2members()
    stub.intermediate_epochs_idg = 2
    stub.valid_skip = 1000
    mock_validate_epoch.return_value = (
        {"m0": 0.5, "m1": 0.5},
        0.5,
        {"mae_f": 0.5},
        None,
    )
    InitialDatasetProcedure._train(stub)
    assert mock_train_epoch.call_count == 4


@patch(f"{TRAIN_MOD}.save_datasets")
@patch(f"{TRAIN_MOD}.save_checkpoint")
@patch(f"{TRAIN_MOD}.validate_epoch")
@patch(f"{TRAIN_MOD}.train_epoch")
@patch(f"{TRAIN_MOD}.tools")
@patch(f"{TRAIN_MOD}.update_model_auxiliaries")
@patch(
    f"{TRAIN_MOD}.create_dataloader", return_value=(MagicMock(), MagicMock())
)
@patch(f"{TRAIN_MOD}.update_datasets", return_value=_UPDATE_DATASETS_RETURN)
def test_train_calls_update_datasets_once_per_member(
    mock_update_datasets,
    mock_create_dataloader,
    mock_update_model_auxiliaries,
    mock_tools,
    mock_train_epoch,
    mock_validate_epoch,
    mock_save_checkpoint,
    mock_save_datasets,
):
    # update_datasets called once per member before the epoch loop
    stub = make_train_stub_2members()
    stub.intermediate_epochs_idg = 1
    mock_validate_epoch.return_value = (
        {"m0": 0.5, "m1": 0.5},
        0.5,
        {"mae_f": 0.5},
        None,
    )
    InitialDatasetProcedure._train(stub)
    assert mock_update_datasets.call_count == 2


@patch(f"{TRAIN_MOD}.save_datasets")
@patch(f"{TRAIN_MOD}.save_checkpoint")
@patch(f"{TRAIN_MOD}.validate_epoch")
@patch(f"{TRAIN_MOD}.train_epoch")
@patch(f"{TRAIN_MOD}.tools")
@patch(f"{TRAIN_MOD}.update_model_auxiliaries")
@patch(
    f"{TRAIN_MOD}.create_dataloader", return_value=(MagicMock(), MagicMock())
)
@patch(f"{TRAIN_MOD}.update_datasets", return_value=_UPDATE_DATASETS_RETURN)
def test_train_validates_at_valid_skip_intervals(
    mock_update_datasets,
    mock_create_dataloader,
    mock_update_model_auxiliaries,
    mock_tools,
    mock_train_epoch,
    mock_validate_epoch,
    mock_save_checkpoint,
    mock_save_datasets,
):
    # valid_skip=2, epochs 0-3: validates only at even epochs (0, 2) → 2 calls
    stub = make_train_stub()
    stub.intermediate_epochs_idg = 4
    stub.valid_skip = 2
    stub.epoch = 0
    stub.desired_acc = 0.0
    mock_validate_epoch.return_value = _VALIDATE_EPOCH_RETURN
    InitialDatasetProcedure._train(stub)
    assert mock_validate_epoch.call_count == 2


# ===========================================================================
# §5.2 — _train stopping criteria
# ===========================================================================


@patch(f"{TRAIN_MOD}.save_datasets")
@patch(f"{TRAIN_MOD}.save_checkpoint")
@patch(f"{TRAIN_MOD}.validate_epoch")
@patch(f"{TRAIN_MOD}.train_epoch")
@patch(f"{TRAIN_MOD}.tools")
@patch(f"{TRAIN_MOD}.update_model_auxiliaries")
@patch(
    f"{TRAIN_MOD}.create_dataloader", return_value=(MagicMock(), MagicMock())
)
@patch(f"{TRAIN_MOD}.update_datasets", return_value=_UPDATE_DATASETS_RETURN)
def test_train_returns_true_when_max_epochs_reached(
    mock_update_datasets,
    mock_create_dataloader,
    mock_update_model_auxiliaries,
    mock_tools,
    mock_train_epoch,
    mock_validate_epoch,
    mock_save_checkpoint,
    mock_save_datasets,
):
    # epoch starts at max_initial_epochs-1, after 1 epoch it equals max
    stub = make_train_stub()
    stub.intermediate_epochs_idg = 1
    stub.valid_skip = 1000
    stub.max_initial_epochs = 100
    stub.epoch = stub.max_initial_epochs - 1
    mock_validate_epoch.return_value = _VALIDATE_EPOCH_RETURN
    result = InitialDatasetProcedure._train(stub)
    assert result is True


@patch(f"{TRAIN_MOD}.save_datasets")
@patch(f"{TRAIN_MOD}.save_checkpoint")
@patch(f"{TRAIN_MOD}.validate_epoch")
@patch(f"{TRAIN_MOD}.train_epoch")
@patch(f"{TRAIN_MOD}.tools")
@patch(f"{TRAIN_MOD}.update_model_auxiliaries")
@patch(
    f"{TRAIN_MOD}.create_dataloader", return_value=(MagicMock(), MagicMock())
)
@patch(f"{TRAIN_MOD}.update_datasets", return_value=_UPDATE_DATASETS_RETURN)
def test_train_returns_none_when_below_max_epochs(
    mock_update_datasets,
    mock_create_dataloader,
    mock_update_model_auxiliaries,
    mock_tools,
    mock_train_epoch,
    mock_validate_epoch,
    mock_save_checkpoint,
    mock_save_datasets,
):
    # epoch 0 → 3 after training, far below max_initial_epochs=100
    stub = make_train_stub()
    stub.epoch = 0
    stub.max_initial_epochs = 100
    stub.intermediate_epochs_idg = 3
    stub.valid_skip = 1000
    mock_validate_epoch.return_value = _VALIDATE_EPOCH_RETURN
    result = InitialDatasetProcedure._train(stub)
    assert result is None


@patch(f"{TRAIN_MOD}.save_datasets")
@patch(f"{TRAIN_MOD}.save_checkpoint")
@patch(f"{TRAIN_MOD}.validate_epoch")
@patch(f"{TRAIN_MOD}.train_epoch")
@patch(f"{TRAIN_MOD}.tools")
@patch(f"{TRAIN_MOD}.update_model_auxiliaries")
@patch(
    f"{TRAIN_MOD}.create_dataloader", return_value=(MagicMock(), MagicMock())
)
@patch(f"{TRAIN_MOD}.update_datasets", return_value=_UPDATE_DATASETS_RETURN)
def test_train_breaks_early_when_desired_acc_met(
    mock_update_datasets,
    mock_create_dataloader,
    mock_update_model_auxiliaries,
    mock_tools,
    mock_train_epoch,
    mock_validate_epoch,
    mock_save_checkpoint,
    mock_save_datasets,
):
    # desired_acc=0.5, scale=1.0 → threshold=0.5; mae_f=0.4 < 0.5 → break
    stub = make_train_stub()
    stub.desired_acc = 0.5
    stub.desired_acc_scale_idg = 1.0
    stub.valid_skip = 1
    stub.intermediate_epochs_idg = 10
    mock_validate_epoch.return_value = (
        {"m0": 0.4},
        0.4,
        {"mae_f": 0.4},
        None,
    )
    InitialDatasetProcedure._train(stub)
    assert mock_train_epoch.call_count == 1


# ===========================================================================
# §5.3 — converge training loop
# ===========================================================================


@patch(f"{PREP_MOD}.save_checkpoint")
@patch(f"{PREP_MOD}.torch")
@patch(f"{PREP_MOD}.validate_epoch_ensemble")
@patch(f"{PREP_MOD}.train_epoch")
@patch(f"{PREP_MOD}.tools")
@patch(f"{PREP_MOD}.setup_model_training")
@patch(f"{PREP_MOD}.update_model_auxiliaries")
@patch(
    f"{PREP_MOD}.create_dataloader", return_value=(MagicMock(), MagicMock())
)
def test_converge_calls_setup_model_training_once_per_member(
    mock_create_dataloader,
    mock_update_model_auxiliaries,
    mock_setup_model_training,
    mock_tools,
    mock_train_epoch,
    mock_validate_epoch_ensemble,
    mock_torch,
    mock_save_checkpoint,
    tmp_path,
):
    # setup_model_training called once per ensemble member
    stub = make_converge_stub(tmp_path)
    stub.ensemble["m1"] = MagicMock()
    stub.ensemble_model_sets["m1"] = {
        "train": [None] * 5,
        "valid": {"Default": [None] * 2},
        "train_loader": None,
        "valid_loader": None,
    }
    stub.ensemble_ase_sets["m1"] = []
    stub.ensemble_atomic_energies["m1"] = []
    stub.ensemble_atomic_energies_dict["m1"] = {}
    mock_setup_model_training.side_effect = [
        _make_setup_model_training_return(),
        _make_setup_model_training_return(),
    ]
    mock_validate_epoch_ensemble.return_value = (
        {"m0": 1.0, "m1": 1.0},
        1.0,
        None,
    )
    PrepareInitialDatasetProcedure.converge(stub)
    assert mock_setup_model_training.call_count == 2


@patch(f"{PREP_MOD}.save_checkpoint")
@patch(f"{PREP_MOD}.torch")
@patch(f"{PREP_MOD}.validate_epoch_ensemble")
@patch(f"{PREP_MOD}.train_epoch")
@patch(f"{PREP_MOD}.tools")
@patch(f"{PREP_MOD}.setup_model_training")
@patch(f"{PREP_MOD}.update_model_auxiliaries")
@patch(
    f"{PREP_MOD}.create_dataloader", return_value=(MagicMock(), MagicMock())
)
def test_converge_runs_all_max_epochs_with_continuous_improvement(
    mock_create_dataloader,
    mock_update_model_auxiliaries,
    mock_setup_model_training,
    mock_tools,
    mock_train_epoch,
    mock_validate_epoch_ensemble,
    mock_torch,
    mock_save_checkpoint,
    tmp_path,
):
    # decreasing losses → continuous improvement → all 5 epochs run
    stub = make_converge_stub(tmp_path)
    stub.max_convergence_epochs = 5
    stub.valid_skip = 1
    stub.margin = 0.001
    stub.idg_settings = SimpleNamespace(convergence_patience=1000)
    mock_setup_model_training.return_value = (
        _make_setup_model_training_return()
    )
    losses = [1.0, 0.9, 0.8, 0.7, 0.6]
    mock_validate_epoch_ensemble.side_effect = [
        ({"m0": v}, v, None) for v in losses
    ]
    PrepareInitialDatasetProcedure.converge(stub)
    assert mock_train_epoch.call_count == 5


# ===========================================================================
# §5.4 — converge patience / early stopping
# ===========================================================================


@patch(f"{PREP_MOD}.save_checkpoint")
@patch(f"{PREP_MOD}.torch")
@patch(f"{PREP_MOD}.validate_epoch_ensemble")
@patch(f"{PREP_MOD}.train_epoch")
@patch(f"{PREP_MOD}.tools")
@patch(f"{PREP_MOD}.setup_model_training")
@patch(f"{PREP_MOD}.update_model_auxiliaries")
@patch(
    f"{PREP_MOD}.create_dataloader", return_value=(MagicMock(), MagicMock())
)
def test_converge_stops_early_after_patience_exceeded(
    mock_create_dataloader,
    mock_update_model_auxiliaries,
    mock_setup_model_training,
    mock_tools,
    mock_train_epoch,
    mock_validate_epoch_ensemble,
    mock_torch,
    mock_save_checkpoint,
    tmp_path,
):
    # patience=2; constant loss 1.0 → saves once at j=0, then
    # no_improvement reaches 3 > 2 at j=3 → stops; 4 train_epoch calls
    stub = make_converge_stub(tmp_path)
    stub.idg_settings = SimpleNamespace(convergence_patience=2)
    stub.max_convergence_epochs = 100
    stub.valid_skip = 1
    stub.margin = 0.001
    mock_setup_model_training.return_value = (
        _make_setup_model_training_return()
    )
    mock_validate_epoch_ensemble.return_value = ({"m0": 1.0}, 1.0, None)
    PrepareInitialDatasetProcedure.converge(stub)
    assert mock_train_epoch.call_count == 4
    assert mock_torch.save.call_count == 1


@patch(f"{PREP_MOD}.save_checkpoint")
@patch(f"{PREP_MOD}.torch")
@patch(f"{PREP_MOD}.validate_epoch_ensemble")
@patch(f"{PREP_MOD}.train_epoch")
@patch(f"{PREP_MOD}.tools")
@patch(f"{PREP_MOD}.setup_model_training")
@patch(f"{PREP_MOD}.update_model_auxiliaries")
@patch(
    f"{PREP_MOD}.create_dataloader", return_value=(MagicMock(), MagicMock())
)
def test_converge_does_not_stop_early_with_large_patience(
    mock_create_dataloader,
    mock_update_model_auxiliaries,
    mock_setup_model_training,
    mock_tools,
    mock_train_epoch,
    mock_validate_epoch_ensemble,
    mock_torch,
    mock_save_checkpoint,
    tmp_path,
):
    # patience=1000, max_epochs=3, constant loss → all 3 epochs run
    stub = make_converge_stub(tmp_path)
    stub.idg_settings = SimpleNamespace(convergence_patience=1000)
    stub.max_convergence_epochs = 3
    stub.valid_skip = 1
    mock_setup_model_training.return_value = (
        _make_setup_model_training_return()
    )
    mock_validate_epoch_ensemble.side_effect = [
        ({"m0": 1.0}, 1.0, None),
        ({"m0": 1.0}, 1.0, None),
        ({"m0": 1.0}, 1.0, None),
    ]
    PrepareInitialDatasetProcedure.converge(stub)
    assert mock_train_epoch.call_count == 3


# ===========================================================================
# §5.5 — converge checkpoint saving
# ===========================================================================


@patch(f"{PREP_MOD}.save_checkpoint")
@patch(f"{PREP_MOD}.torch")
@patch(f"{PREP_MOD}.validate_epoch_ensemble")
@patch(f"{PREP_MOD}.train_epoch")
@patch(f"{PREP_MOD}.tools")
@patch(f"{PREP_MOD}.setup_model_training")
@patch(f"{PREP_MOD}.update_model_auxiliaries")
@patch(
    f"{PREP_MOD}.create_dataloader", return_value=(MagicMock(), MagicMock())
)
def test_converge_saves_model_on_improvement(
    mock_create_dataloader,
    mock_update_model_auxiliaries,
    mock_setup_model_training,
    mock_tools,
    mock_train_epoch,
    mock_validate_epoch_ensemble,
    mock_torch,
    mock_save_checkpoint,
    tmp_path,
):
    # losses [0.5, 0.4] → 2 improvements → 2 torch.save and 2 save_checkpoint
    stub = make_converge_stub(tmp_path)
    stub.max_convergence_epochs = 2
    stub.valid_skip = 1
    stub.margin = 0.001
    stub.idg_settings = SimpleNamespace(convergence_patience=1000)
    mock_setup_model_training.return_value = (
        _make_setup_model_training_return()
    )
    mock_validate_epoch_ensemble.side_effect = [
        ({"m0": 0.5}, 0.5, None),
        ({"m0": 0.4}, 0.4, None),
    ]
    PrepareInitialDatasetProcedure.converge(stub)
    assert mock_torch.save.call_count == 2
    assert mock_save_checkpoint.call_count == 2


@patch(f"{PREP_MOD}.save_checkpoint")
@patch(f"{PREP_MOD}.torch")
@patch(f"{PREP_MOD}.validate_epoch_ensemble")
@patch(f"{PREP_MOD}.train_epoch")
@patch(f"{PREP_MOD}.tools")
@patch(f"{PREP_MOD}.setup_model_training")
@patch(f"{PREP_MOD}.update_model_auxiliaries")
@patch(
    f"{PREP_MOD}.create_dataloader", return_value=(MagicMock(), MagicMock())
)
def test_converge_skips_save_when_improvement_below_margin(
    mock_create_dataloader,
    mock_update_model_auxiliaries,
    mock_setup_model_training,
    mock_tools,
    mock_train_epoch,
    mock_validate_epoch_ensemble,
    mock_torch,
    mock_save_checkpoint,
    tmp_path,
):
    # margin=0.1; second delta=0.0001 < 0.1 → not saved; only 1 torch.save
    stub = make_converge_stub(tmp_path)
    stub.margin = 0.1
    stub.max_convergence_epochs = 2
    stub.valid_skip = 1
    stub.idg_settings = SimpleNamespace(convergence_patience=1000)
    mock_setup_model_training.return_value = (
        _make_setup_model_training_return()
    )
    mock_validate_epoch_ensemble.side_effect = [
        ({"m0": 1.0}, 1.0, None),
        ({"m0": 0.9999}, 0.9999, None),
    ]
    PrepareInitialDatasetProcedure.converge(stub)
    assert mock_torch.save.call_count == 1


@patch(f"{PREP_MOD}.save_checkpoint")
@patch(f"{PREP_MOD}.torch")
@patch(f"{PREP_MOD}.validate_epoch_ensemble")
@patch(f"{PREP_MOD}.train_epoch")
@patch(f"{PREP_MOD}.tools")
@patch(f"{PREP_MOD}.setup_model_training")
@patch(f"{PREP_MOD}.update_model_auxiliaries")
@patch(
    f"{PREP_MOD}.create_dataloader", return_value=(MagicMock(), MagicMock())
)
def test_converge_save_uses_model_dir_path(
    mock_create_dataloader,
    mock_update_model_auxiliaries,
    mock_setup_model_training,
    mock_tools,
    mock_train_epoch,
    mock_validate_epoch_ensemble,
    mock_torch,
    mock_save_checkpoint,
    tmp_path,
):
    # torch.save path parent should equal model_dir (tmp_path)
    stub = make_converge_stub(tmp_path)
    stub.max_convergence_epochs = 1
    stub.valid_skip = 1
    stub.margin = 0.001
    stub.idg_settings = SimpleNamespace(convergence_patience=1000)
    mock_setup_model_training.return_value = (
        _make_setup_model_training_return()
    )
    mock_validate_epoch_ensemble.return_value = ({"m0": 0.5}, 0.5, None)
    PrepareInitialDatasetProcedure.converge(stub)
    assert mock_torch.save.call_count >= 1
    saved_path = mock_torch.save.call_args_list[0][0][1]
    assert saved_path.parent == tmp_path
