from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import ase.build
import numpy as np

from aims_PAX.procedures.initial_dataset import (
    InitialDatasetFoundational,
    InitialDatasetProcedure,
)
from aims_PAX.procedures.preparation import PrepareInitialDatasetProcedure

TRAIN_MOD = "aims_PAX.procedures.initial_dataset"
PREP_MOD = "aims_PAX.procedures.preparation"


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


# ===========================================================================
# §6.1 — _sample_and_train
# ===========================================================================


def test_sample_and_train_increments_step():
    stub = SimpleNamespace(
        step=0,
        sampled_points=[],
        _sample_points=MagicMock(return_value=["point"]),
        _train=MagicMock(),
    )
    InitialDatasetProcedure._sample_and_train(stub)
    assert stub.step == 1


def test_sample_and_train_retries_on_empty_sampled_points():
    call_results = [[], [], ["point"]]
    stub = SimpleNamespace(
        step=0,
        sampled_points=[],
        _sample_points=MagicMock(side_effect=call_results),
        _train=MagicMock(),
    )
    InitialDatasetProcedure._sample_and_train(stub)
    assert stub._sample_points.call_count == 3


def test_sample_and_train_calls_train_with_result():
    stub = SimpleNamespace(
        step=0,
        sampled_points=[],
        _sample_points=MagicMock(return_value=["point"]),
        _train=MagicMock(),
    )
    InitialDatasetProcedure._sample_and_train(stub)
    stub._train.assert_called_once()


# ===========================================================================
# §6.2 — _handle_analysis
# ===========================================================================


@patch(f"{TRAIN_MOD}.np.savez")
def test_handle_analysis_appends_loss_entry(mock_savez, tmp_path):
    stub = SimpleNamespace(
        collect_losses={"epoch": [], "avg_losses": [], "ensemble_losses": []},
        epoch=3,
        output_dir=tmp_path,
    )
    InitialDatasetProcedure._handle_analysis(stub, 0.42, {"m0": 0.42})
    assert stub.collect_losses["epoch"] == [3]
    assert stub.collect_losses["avg_losses"] == [0.42]
    assert stub.collect_losses["ensemble_losses"] == [{"m0": 0.42}]


@patch(f"{TRAIN_MOD}.np.savez")
def test_handle_analysis_saves_npz_file(mock_savez, tmp_path):
    stub = SimpleNamespace(
        collect_losses={"epoch": [], "avg_losses": [], "ensemble_losses": []},
        epoch=0,
        output_dir=tmp_path,
    )
    InitialDatasetProcedure._handle_analysis(stub, 1.0, {"m0": 1.0})
    mock_savez.assert_called_once()


@patch(f"{TRAIN_MOD}.np.savez")
def test_handle_analysis_no_save_path_does_not_crash(mock_savez, tmp_path):
    stub = SimpleNamespace(
        collect_losses={"epoch": [], "avg_losses": [], "ensemble_losses": []},
        epoch=0,
        output_dir=tmp_path,
    )
    InitialDatasetProcedure._handle_analysis(stub, 0.5, {"m0": 0.5})
    mock_savez.assert_called_once()
    call_path = mock_savez.call_args[0][0]
    assert "initial_losses.npz" in str(call_path)


# ===========================================================================
# §6.3 — check_initial_ds_done
# ===========================================================================


def test_check_done_returns_true_when_flag_set():
    stub = SimpleNamespace(
        create_restart=True,
        init_ds_restart_dict={"initial_ds_done": True},
        logger=SimpleNamespace(handlers=[MagicMock()]),
    )
    result = PrepareInitialDatasetProcedure.check_initial_ds_done(stub)
    assert result is True
    assert stub.logger.handlers == []


def test_check_done_returns_false_when_not_set():
    stub = SimpleNamespace(
        create_restart=False,
    )
    result = PrepareInitialDatasetProcedure.check_initial_ds_done(stub)
    assert result is False


# ===========================================================================
# §6.4 — _collect_restart_points
# ===========================================================================


def test_collect_restart_copies_atoms():
    atoms = ase.build.bulk("Cu")
    stub = SimpleNamespace(last_points={})
    PrepareInitialDatasetProcedure._collect_restart_points(stub, {0: atoms})
    assert 0 in stub.last_points
    assert stub.last_points[0] is not atoms


def test_collect_restart_preserves_velocities():
    atoms = ase.build.bulk("Cu")
    velocities = np.ones((len(atoms), 3)) * 0.5
    atoms.set_velocities(velocities)
    stub = SimpleNamespace(last_points={})
    PrepareInitialDatasetProcedure._collect_restart_points(stub, {0: atoms})
    np.testing.assert_array_almost_equal(
        stub.last_points[0].get_velocities(), velocities
    )


# ===========================================================================
# §6.5 — converge no_improvement reset
# ===========================================================================


@patch(f"{PREP_MOD}.save_checkpoint")
@patch(f"{PREP_MOD}.torch")
@patch(f"{PREP_MOD}.validate_epoch_ensemble")
@patch(f"{PREP_MOD}.train_epoch")
@patch(f"{PREP_MOD}.tools")
@patch(
    f"{PREP_MOD}.setup_model_training",
    return_value={
        "loss_fn": MagicMock(),
        "optimizer": MagicMock(),
        "lr_scheduler": MagicMock(),
        "device": "cpu",
        "max_grad_norm": 1.0,
        "output_args": {},
        "ema": None,
        "checkpoint_handler": MagicMock(),
        "epoch": 0,
    },
)
@patch(f"{PREP_MOD}.update_model_auxiliaries")
@patch(
    f"{PREP_MOD}.create_dataloader", return_value=(MagicMock(), MagicMock())
)
def test_converge_resets_no_improvement_counter_on_new_best(
    mock_create_dl,
    mock_update_aux,
    mock_setup_training,
    mock_tools,
    mock_train_epoch,
    mock_validate,
    mock_torch,
    mock_save_checkpoint,
    tmp_path,
):
    # patience=1, epochs=3: 1.0 → 1.0 → 0.5
    # epoch 0: inf→1.0, improves, no_improvement=0, saved
    # epoch 1: 1.0→1.0, no improvement, no_improvement=1 (==patience, NOT >)
    # epoch 2: 1.0→0.5, improves, no_improvement reset to 0, saved
    stub = make_converge_stub(tmp_path)
    stub.idg_settings = SimpleNamespace(convergence_patience=1)
    stub.max_convergence_epochs = 3
    stub.valid_skip = 1
    stub.margin = 0.001
    mock_validate.side_effect = [
        ({"m0": 1.0}, 1.0, None),
        ({"m0": 1.0}, 1.0, None),
        ({"m0": 0.5}, 0.5, None),
    ]
    PrepareInitialDatasetProcedure.converge(stub)
    assert mock_train_epoch.call_count == 3
    assert mock_torch.save.call_count == 2


# ===========================================================================
# §6.6 — InitialDatasetFoundational._recalc_dft
# ===========================================================================


def test_recalc_dft_converged_stores_energy_and_forces():
    atoms = ase.build.bulk("Cu")
    forces = np.zeros((len(atoms), 3))
    calc_mock = MagicMock()
    calc_mock.asi.is_scf_converged = True
    calc_mock.results = {
        "energy": -5.0,
        "forces": forces,
        "stress": np.zeros(6),
    }
    stub = SimpleNamespace(
        aims_calc=calc_mock,
        compute_stress=False,
        properties=["energy", "forces"],
    )
    result = InitialDatasetFoundational._recalc_dft(stub, atoms)
    assert result is atoms
    assert result.info["REF_energy"] == -5.0
    np.testing.assert_array_equal(result.arrays["REF_forces"], forces)


def test_recalc_dft_unconverged_returns_none():
    atoms = ase.build.bulk("Cu")
    calc_mock = MagicMock()
    calc_mock.asi.is_scf_converged = False
    stub = SimpleNamespace(
        aims_calc=calc_mock,
        compute_stress=False,
        properties=["energy", "forces"],
    )
    result = InitialDatasetFoundational._recalc_dft(stub, atoms)
    assert result is None


def test_recalc_dft_stores_stress_when_compute_stress():
    atoms = ase.build.bulk("Cu")
    stress = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    calc_mock = MagicMock()
    calc_mock.asi.is_scf_converged = True
    calc_mock.results = {
        "energy": -5.0,
        "forces": np.zeros((len(atoms), 3)),
        "stress": stress,
    }
    stub = SimpleNamespace(
        aims_calc=calc_mock,
        compute_stress=True,
        properties=["energy", "forces", "stress"],
    )
    result = InitialDatasetFoundational._recalc_dft(stub, atoms)
    np.testing.assert_array_equal(result.info["REF_stress"], stress)


# ===========================================================================
# §6.7 — InitialDatasetFoundational._sample_points
# ===========================================================================


def test_foundational_sample_points_filters_unconverged():
    atoms1 = ase.build.bulk("Cu")
    atoms2 = ase.build.bulk("Cu")
    atoms3 = ase.build.bulk("Cu")
    stub = SimpleNamespace(
        trajectories={0: MagicMock()},
        sampled_points={0: [atoms1, atoms2, atoms3]},
    )
    stub._md_w_foundational = MagicMock()
    stub._recalc_dft = MagicMock(side_effect=[atoms1, None, atoms3])
    result = InitialDatasetFoundational._sample_points(stub)
    assert len(result) == 2
    assert atoms1 in result
    assert atoms3 in result


def test_foundational_sample_points_calls_md_once():
    atoms = ase.build.bulk("Cu")
    stub = SimpleNamespace(
        trajectories={0: MagicMock()},
        sampled_points={0: [atoms]},
    )
    stub._md_w_foundational = MagicMock()
    stub._recalc_dft = MagicMock(return_value=atoms)
    InitialDatasetFoundational._sample_points(stub)
    stub._md_w_foundational.assert_called_once()


def test_foundational_sample_points_empty_after_all_fail():
    atoms1 = ase.build.bulk("Cu")
    atoms2 = ase.build.bulk("Cu")
    stub = SimpleNamespace(
        trajectories={0: MagicMock()},
        sampled_points={0: [atoms1, atoms2]},
    )
    stub._md_w_foundational = MagicMock()
    stub._recalc_dft = MagicMock(return_value=None)
    result = InitialDatasetFoundational._sample_points(stub)
    assert result == []
