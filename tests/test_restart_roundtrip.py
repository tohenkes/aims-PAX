"""
Checkpoint / restart round-trip tests for ALRestart (Part A) and
model serialisation via ensemble_from_folder (Part B).

Part A (tests 1–4) are fast and have no @pytest.mark.slow decoration.
Part B (test 5) builds a real MACE model and is marked @pytest.mark.slow.
"""

import logging

import numpy as np
import pytest
from ase import Atoms
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
        trajectories={0: Atoms("H"), 1: Atoms("H")},
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


def _random_subset_restart(tmp_path, yaml_num_trajectories):
    """Save a 2-trajectory checkpoint, then load it with random_subset
    replay and the given yaml num_trajectories. Returns the loader and
    the trajectory indices whose training subset was rebuilt."""
    saver = _make_saver(tmp_path)
    saver.state_manager.last_point_added = {0: Atoms("H"), 1: Atoms("H")}
    saver.update_restart_dict(
        None, None, save_restart=str(tmp_path / "al_state.npy")
    )

    loader = _make_loader(tmp_path)
    loader.config.replay_strategy = "random_subset"
    loader.config.num_trajectories = yaml_num_trajectories
    subset_calls = []
    loader.ensemble_manager = SimpleNamespace(
        create_training_subset=lambda model_point, idx: subset_calls.append(
            idx
        )
    )
    loader._load_restart_checkpoint()
    return loader, subset_calls


@pytest.mark.parametrize("yaml_value", [1, 3])
def test_restart_overrides_num_trajectories_with_warning(
    tmp_path, caplog, yaml_value
):
    """The checkpoint's trajectory count wins over aimsPAX.yaml, is
    applied before the random_subset rebuild, and is announced."""
    with caplog.at_level(logging.WARNING):
        loader, subset_calls = _random_subset_restart(tmp_path, yaml_value)

    assert loader.config.num_trajectories == 2
    assert subset_calls == [0, 1]
    assert "num_trajectories" in caplog.text
    assert str(yaml_value) in caplog.text and "2" in caplog.text


def test_restart_keeps_matching_num_trajectories_silently(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        loader, subset_calls = _random_subset_restart(tmp_path, 2)

    assert loader.config.num_trajectories == 2
    assert subset_calls == [0, 1]
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_restart_recomputes_trajectory_system(tmp_path):
    """The trajectory -> system map is rebuilt on restart, and the
    restored trajectories get their system's charge/spin again."""
    from aims_PAX.procedures.preparation import ALStateManager
    from aims_PAX.tools.utilities.systems import resolve_systems
    from aims_PAX.settings.project import MDSettings

    geometry = tmp_path / "geometry.in"
    geometry.write_text("atom 0.0 0.0 0.0 O\natom 0.0 0.0 1.21 O\n")
    singlet = tmp_path / "singlet.in"
    singlet.write_text("xc pbe\n")
    triplet = tmp_path / "triplet.in"
    triplet.write_text("xc pbe\nspin collinear\nfixed_spin_moment 2\n")
    systems = resolve_systems(
        geometry,
        {3: singlet, 7: triplet},
        MDSettings.model_validate(
            {"stat_ensemble": "NVT", "thermostat": "Langevin"}
        ),
    )
    config = SimpleNamespace(
        systems=systems,
        charge_spin={3: (0.0, 0.0), 7: (0.0, 1.0)},
        misc=SimpleNamespace(
            total_charge_key="total_charge", total_spin_key="total_spin"
        ),
        num_trajectories=3,
        seeds_tags_dict={},
        analysis=False,
        mol_idxs=None,
        create_restart=True,
        al_restart_path=str(tmp_path / "al_restart.npy"),
        replay_strategy="full_dataset",
    )

    fresh = SimpleNamespace(state_manager=ALStateManager(config))
    fresh.state_manager.initialize_fresh_state()
    # a checkpoint written before charge/spin were annotated
    for atoms in fresh.state_manager.trajectories.values():
        atoms.info.clear()
    saver = ALRestart(
        config, fresh.state_manager, SimpleNamespace(), SimpleNamespace()
    )
    saver.update_restart_dict(None, None, save_restart=config.al_restart_path)

    restarted = SimpleNamespace(state_manager=ALStateManager(config))
    ALRestart(
        config, restarted.state_manager, SimpleNamespace(), SimpleNamespace()
    ).handle_restart()

    assert (
        restarted.state_manager.trajectory_system
        == fresh.state_manager.trajectory_system
        == {0: 3, 1: 7, 2: 3}
    )
    spins = [
        restarted.state_manager.trajectories[i].info["total_spin"]
        for i in range(3)
    ]
    assert spins == [0.0, 1.0, 0.0]


def test_restart_keeps_raised_trajectory_count(tmp_path):
    """num_trajectories raised to the number of systems in the first run
    is raised again on restart (the settings still say 1)."""
    from aims_PAX.procedures.preparation import ALStateManager
    from aims_PAX.tools.utilities.systems import resolve_systems
    from aims_PAX.settings.project import MDSettings

    geometry = tmp_path / "geometry.in"
    geometry.write_text("atom 0.0 0.0 0.0 O\natom 0.0 0.0 1.21 O\n")
    controls = {}
    for i in (0, 1):
        controls[i] = tmp_path / f"control{i}.in"
        controls[i].write_text("xc pbe\n")
    config = SimpleNamespace(
        systems=resolve_systems(
            geometry,
            controls,
            MDSettings.model_validate(
                {"stat_ensemble": "NVT", "thermostat": "Langevin"}
            ),
        ),
        charge_spin={},
        num_trajectories=1,
        seeds_tags_dict={},
        analysis=False,
        mol_idxs=None,
        create_restart=True,
        al_restart_path=str(tmp_path / "al_restart.npy"),
        replay_strategy="full_dataset",
    )
    fresh = ALStateManager(config)
    fresh.initialize_fresh_state()
    assert config.num_trajectories == 2
    ALRestart(
        config, fresh, SimpleNamespace(), SimpleNamespace()
    ).update_restart_dict(None, None, save_restart=config.al_restart_path)

    config.num_trajectories = 1  # as read from aimsPAX.yaml again
    restarted = ALStateManager(config)
    ALRestart(
        config, restarted, SimpleNamespace(), SimpleNamespace()
    ).handle_restart()
    assert config.num_trajectories == 2


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
