"""
Checkpoint / restart round-trip tests for ALRestart (Part A) and
model serialisation via ensemble_from_folder (Part B).

Part A (tests 1–4) are fast and have no @pytest.mark.slow decoration.
Part B (test 5) builds a real MACE model and is marked @pytest.mark.slow.
"""

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from aims_PAX.procedures.preparation import ALRestart
from aims_PAX.tools.utilities.utilities import ensemble_from_folder
from tests.helpers import build_si_model

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


def _make_saver(tmp_path, analysis=False):
    """ALRestart stub pre-populated with known values for saving."""
    obj = ALRestart.__new__(ALRestart)
    obj.config = SimpleNamespace(
        analysis=analysis,
        mol_idxs=None,
        al_restart_path=str(tmp_path / "al_state.npy"),
        num_trajectories=2,
        replay_strategy="full_dataset",
    )
    obj.state_manager = SimpleNamespace(
        trajectories=None,
        MD_checkpoints=None,
        trajectory_status={0: "running", 1: "paused"},
        trajectory_MD_steps={0: 100, 1: 200},
        trajectory_total_epochs=None,
        trajectory_intermediate_epochs=None,
        ensemble_reset_opt=None,
        ensemble_no_improvement=None,
        ensemble_best_valid=None,
        current_valid_error=0.05,
        threshold=0.15,
        total_points_added=42,
        train_points_added=38,
        valid_points_added=4,
        num_MD_limits_reached=1,
        num_workers_training=2,
        num_workers_waiting=0,
        total_epoch=100,
        check=None,
        uncertainties=[np.array([0.1, 0.2, 0.3])],
        uncert_not_crossed=None,
        last_point_added={0: None},
    )
    if analysis:
        obj.state_manager.t_intervals = [0, 10, 20]
        obj.state_manager.analysis_checks = [True, False]
        obj.state_manager.collect_losses = [0.1, 0.08, 0.06]
        obj.state_manager.collect_thresholds = [0.15, 0.13, 0.12]
    obj.al_restart_dict = {}  # update_restart_dict calls .update() on this
    obj.md_manager = SimpleNamespace()
    obj.ensemble_manager = SimpleNamespace()
    return obj


def _make_loader(tmp_path, analysis=False):
    """Fresh ALRestart stub with all watched attrs set to None for loading."""
    obj = ALRestart.__new__(ALRestart)
    obj.config = SimpleNamespace(
        analysis=analysis,
        mol_idxs=None,
        al_restart_path=str(tmp_path / "al_state.npy"),
        num_trajectories=2,
        replay_strategy="full_dataset",
    )
    # All base attributes present but zeroed — load must overwrite them.
    obj.state_manager = SimpleNamespace(
        trajectories=None,
        MD_checkpoints=None,
        trajectory_status=None,
        trajectory_MD_steps=None,
        trajectory_total_epochs=None,
        trajectory_intermediate_epochs=None,
        ensemble_reset_opt=None,
        ensemble_no_improvement=None,
        ensemble_best_valid=None,
        current_valid_error=None,
        threshold=None,
        total_points_added=None,
        train_points_added=None,
        valid_points_added=None,
        num_MD_limits_reached=None,
        num_workers_training=None,
        num_workers_waiting=None,
        total_epoch=None,
        check=None,
        uncertainties=None,
        uncert_not_crossed=None,
        last_point_added=None,
    )
    if analysis:
        obj.state_manager.t_intervals = None
        obj.state_manager.analysis_checks = None
        obj.state_manager.collect_losses = None
        obj.state_manager.collect_thresholds = None
    obj.al_restart_dict = {}
    obj.md_manager = SimpleNamespace()  # no md_manager attrs in our dict
    obj.ensemble_manager = SimpleNamespace()
    return obj


# ---------------------------------------------------------------------------
# Part A — AL state round-trip (no @pytest.mark.slow)
# ---------------------------------------------------------------------------


def test_al_state_roundtrip_scalars(tmp_path):
    """Scalar fields survive a save → load cycle."""
    save_path = str(tmp_path / "al_state.npy")
    saver = _make_saver(tmp_path)
    saver.update_restart_dict(None, None, save_restart=save_path)

    loader = _make_loader(tmp_path)
    loader._load_restart_checkpoint()

    assert loader.state_manager.threshold == 0.15
    assert loader.state_manager.current_valid_error == 0.05
    assert loader.state_manager.total_points_added == 42
    assert loader.state_manager.train_points_added == 38
    assert loader.state_manager.valid_points_added == 4
    assert loader.state_manager.total_epoch == 100


def test_al_state_roundtrip_uncertainties(tmp_path):
    """Uncertainty arrays survive a save → load cycle (mutation-sensitive)."""
    save_path = str(tmp_path / "al_state.npy")
    saver = _make_saver(tmp_path)
    saver.update_restart_dict(None, None, save_restart=save_path)

    loader = _make_loader(tmp_path)
    loader._load_restart_checkpoint()

    assert np.array_equal(
        loader.state_manager.uncertainties[0],
        np.array([0.1, 0.2, 0.3]),
    )

    # Mutation test: different uncertainties → different loaded result.
    saver2 = _make_saver(tmp_path)
    saver2.state_manager.uncertainties = [np.array([0.9, 0.8, 0.7])]
    save_path2 = str(tmp_path / "al_state2.npy")
    saver2.config.al_restart_path = save_path2
    saver2.update_restart_dict(None, None, save_restart=save_path2)

    loader2 = _make_loader(tmp_path)
    loader2.config.al_restart_path = save_path2
    loader2._load_restart_checkpoint()

    assert not np.array_equal(
        loader2.state_manager.uncertainties[0],
        np.array([0.1, 0.2, 0.3]),
    )


def test_al_state_roundtrip_trajectory_dicts(tmp_path):
    """Trajectory status and MD-steps dicts survive a save → load cycle."""
    save_path = str(tmp_path / "al_state.npy")
    saver = _make_saver(tmp_path)
    saver.update_restart_dict(None, None, save_restart=save_path)

    loader = _make_loader(tmp_path)
    loader._load_restart_checkpoint()

    assert loader.state_manager.trajectory_status == {
        0: "running",
        1: "paused",
    }
    assert loader.state_manager.trajectory_MD_steps == {0: 100, 1: 200}


def test_al_state_roundtrip_analysis_fields(tmp_path):
    """Analysis fields survive a save → load cycle when analysis=True."""
    save_path = str(tmp_path / "al_state.npy")
    saver = _make_saver(tmp_path, analysis=True)
    saver.update_restart_dict(None, None, save_restart=save_path)

    loader = _make_loader(tmp_path, analysis=True)
    loader._load_restart_checkpoint()

    assert loader.state_manager.collect_losses == [0.1, 0.08, 0.06]
    assert loader.state_manager.collect_thresholds == [0.15, 0.13, 0.12]
    assert loader.state_manager.t_intervals == [0, 10, 20]
    assert loader.state_manager.analysis_checks == [True, False]


# ---------------------------------------------------------------------------
# Part B — model checkpoint round-trip (@pytest.mark.slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_model_roundtrip(tmp_path):
    """MACE model parameters and atomic energies survive torch.save/load."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = build_si_model()
    torch.save(model, model_dir / "m0.model")

    loaded = ensemble_from_folder(
        str(model_dir), device="cpu", dtype=torch.float64
    )

    assert "m0" in loaded
    loaded_model = loaded["m0"]

    # Parameter tensors must match exactly
    for p_orig, p_load in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(
            p_orig, p_load
        ), "parameter mismatch after reload"

    # Atomic energies must survive
    ae_orig = model.atomic_energies_fn.atomic_energies
    ae_load = loaded_model.atomic_energies_fn.atomic_energies
    assert torch.allclose(
        ae_orig, ae_load
    ), "atomic energies mismatch after reload"
