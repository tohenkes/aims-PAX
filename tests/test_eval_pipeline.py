"""
Integration (slow) tests for eval_utils.py pipeline functions:
    - evaluate_model
    - ensemble_prediction
    - test_model
    - test_ensemble

All tests require a real MACE model and run under @pytest.mark.slow.
"""

from pathlib import Path

import math

import ase.io
import numpy as np
import pytest
import torch
from mace.tools import AtomicNumberTable

from aims_PAX.tools.utilities.eval_utils import (
    ensemble_prediction,
    evaluate_model,
    test_ensemble as eval_test_ensemble,
    test_model as eval_test_model,
)
from aims_PAX.tools.utilities.utilities import create_keyspec

from tests.helpers import build_si_model, make_loader

_TEST_DATA = Path(__file__).parent / "test_data"
_TRAIN_XYZ = (
    _TEST_DATA
    / "datasets/initial/training/combined_initial_train_set.xyz"
)
_VALID_XYZ = (
    _TEST_DATA
    / "datasets/initial/validation/combined_initial_valid_set.xyz"
)


@pytest.fixture(scope="module")
def si_model_and_data():
    """Build Si model and load atoms once for the entire module."""
    z_table = AtomicNumberTable([14])
    train_atoms = ase.io.read(_TRAIN_XYZ, index=":")
    valid_atoms = ase.io.read(_VALID_XYZ, index=":")
    r_max = 5.0
    model = build_si_model(z_table=z_table, r_max=r_max)
    keyspec = create_keyspec()
    return model, train_atoms, valid_atoms, z_table, r_max, keyspec


# ---------------------------------------------------------------------------
# 1. evaluate_model — shapes and finiteness
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_evaluate_model_shapes_and_finite(si_model_and_data):
    """evaluate_model returns finite energies/forces with correct shapes."""
    model, train_atoms, _, _z, _r, _ = si_model_and_data
    atoms = train_atoms[:4]
    energies, forces = evaluate_model(
        atoms_list=atoms,
        model=model,
        batch_size=2,
        device="cpu",
    )
    n = len(atoms)
    assert energies.shape == (n,), (
        f"Expected energies shape ({n},), got {energies.shape}"
    )
    n_atoms_per_struct = atoms[0].get_global_number_of_atoms()
    assert forces.shape == (n, n_atoms_per_struct, 3), (
        f"Unexpected forces shape {forces.shape}"
    )
    assert np.all(np.isfinite(energies)), "Energies contain non-finite values"
    assert np.all(np.isfinite(forces)), "Forces contain non-finite values"


# ---------------------------------------------------------------------------
# 2. ensemble_prediction — shapes and return_energies flag
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ensemble_prediction_shapes(si_model_and_data):
    """ensemble_prediction returns [n_models, n_mols, n_atoms, 3] forces."""
    model, train_atoms, _, _z, _r, _ = si_model_and_data
    atoms = train_atoms[:3]
    n_mols = len(atoms)
    n_atoms = atoms[0].get_global_number_of_atoms()

    # ensemble_prediction takes a LIST of models (not a dict)
    ensemble_list = [model, model]
    n_models = len(ensemble_list)

    forces = ensemble_prediction(
        models=ensemble_list,
        atoms_list=atoms,
        device="cpu",
        batch_size=1,
        return_energies=False,
    )
    assert forces.shape == (n_models, n_mols, n_atoms, 3), (
        f"Unexpected forces shape {forces.shape}"
    )

    energies, forces_e = ensemble_prediction(
        models=ensemble_list,
        atoms_list=atoms,
        device="cpu",
        batch_size=1,
        return_energies=True,
    )
    assert energies.shape == (n_models, n_mols), (
        f"Unexpected energies shape {energies.shape}"
    )
    assert forces_e.shape == (n_models, n_mols, n_atoms, 3), (
        f"Unexpected forces shape {forces_e.shape}"
    )


# ---------------------------------------------------------------------------
# 3. test_model — metrics dict with finite mae_e and mae_f
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_test_model_metrics_finite(si_model_and_data):
    """eval_test_model returns a metrics dict with finite mae_e and mae_f."""
    model, _, valid_atoms, z_table, r_max, keyspec = si_model_and_data
    loader = make_loader(
        atoms_list=valid_atoms,
        keyspec=keyspec,
        z_table=z_table,
        r_max=r_max,
        batch_size=2,
        shuffle=False,
    )
    output_args = {"forces": True, "virials": False, "stress": False}
    result = eval_test_model(
        model=model,
        data_loader=loader,
        output_args=output_args,
        device=torch.device("cpu"),
        return_predictions=False,
    )
    assert isinstance(result, dict), "Expected a dict from eval_test_model"
    assert "mae_e" in result, "Missing key 'mae_e' in metrics"
    assert "mae_f" in result, "Missing key 'mae_f' in metrics"
    assert math.isfinite(result["mae_e"]), (
        f"mae_e is not finite: {result['mae_e']}"
    )
    assert math.isfinite(result["mae_f"]), (
        f"mae_f is not finite: {result['mae_f']}"
    )


# ---------------------------------------------------------------------------
# 4. test_model — return_predictions=True includes predictions sub-dict
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_test_model_return_predictions(si_model_and_data):
    """eval_test_model with return_predictions=True includes a predictions dict."""
    model, _, valid_atoms, z_table, r_max, keyspec = si_model_and_data
    loader = make_loader(
        atoms_list=valid_atoms,
        keyspec=keyspec,
        z_table=z_table,
        r_max=r_max,
        batch_size=2,
        shuffle=False,
    )
    output_args = {"forces": True, "virials": False, "stress": False}
    result = eval_test_model(
        model=model,
        data_loader=loader,
        output_args=output_args,
        device=torch.device("cpu"),
        return_predictions=True,
    )
    assert isinstance(result, dict), "Expected a dict from eval_test_model"
    assert "predictions" in result, (
        "Missing 'predictions' key when return_predictions=True"
    )
    preds = result["predictions"]
    assert isinstance(preds, dict), "predictions should be a dict"
    # When forces=True, predictions must contain 'forces' key
    assert "forces" in preds, (
        "predictions dict missing 'forces' key (output_args forces=True)"
    )
    assert isinstance(preds["forces"], torch.Tensor), (
        "predictions['forces'] should be a Tensor"
    )
    # energy key should be absent: output_args has no 'energy' key
    assert "energy" not in preds, (
        "predictions['energy'] should be absent when output_args energy=False"
    )


# ---------------------------------------------------------------------------
# 5. test_ensemble — structure and per-model keys
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_test_ensemble_structure(si_model_and_data):
    """eval_test_ensemble returns (avg_metrics, per_model_metrics) with finite values."""
    model, _, valid_atoms, _z, _r, keyspec = si_model_and_data
    # eval_test_ensemble takes a dict ensemble
    ensemble = {"m0": model, "m1": model}
    output_args = {"forces": True, "virials": False, "stress": False}

    result = eval_test_ensemble(
        ensemble=ensemble,
        batch_size=2,
        output_args=output_args,
        device="cpu",
        atoms_list=valid_atoms,
        key_specification=keyspec,
    )

    assert isinstance(result, tuple), (
        "Expected a 2-tuple from eval_test_ensemble"
    )
    assert len(result) == 2, (
        f"Expected tuple of length 2, got {len(result)}"
    )
    avg_metrics, per_model_metrics = result

    assert isinstance(avg_metrics, dict), "avg_metrics should be a dict"
    assert "mae_e" in avg_metrics, "Missing 'mae_e' in avg_metrics"
    assert "mae_f" in avg_metrics, "Missing 'mae_f' in avg_metrics"
    assert math.isfinite(avg_metrics["mae_e"]), (
        f"avg mae_e is not finite: {avg_metrics['mae_e']}"
    )
    assert math.isfinite(avg_metrics["mae_f"]), (
        f"avg mae_f is not finite: {avg_metrics['mae_f']}"
    )

    assert isinstance(per_model_metrics, dict), (
        "per_model_metrics should be a dict"
    )
    assert set(per_model_metrics.keys()) == {"m0", "m1"}, (
        f"Expected keys {{'m0', 'm1'}}, "
        f"got {set(per_model_metrics.keys())}"
    )
    for tag in ("m0", "m1"):
        m = per_model_metrics[tag]
        assert "mae_e" in m, (
            f"Missing 'mae_e' in per_model_metrics['{tag}']"
        )
        assert "mae_f" in m, (
            f"Missing 'mae_f' in per_model_metrics['{tag}']"
        )
        assert math.isfinite(m["mae_e"]), (
            f"per_model mae_e for '{tag}' is not finite"
        )
        assert math.isfinite(m["mae_f"]), (
            f"per_model mae_f for '{tag}' is not finite"
        )
