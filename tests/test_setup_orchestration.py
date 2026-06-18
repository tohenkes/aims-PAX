"""
Unit and integration tests for setup/orchestration helpers:
  - §1  ALEnsemble.create_training_subset  (preparation.py ~1281)
  - §2  ALCalculatorMLFF.handle_atomic_energies  (preparation.py ~1401)
  - §3  utilities.get_atomic_energies_from_ensemble  (utilities.py ~642)
  - §4  utilities.setup_ensemble_dicts  (utilities.py ~432)

§1 and §2 use __new__+SimpleNamespace stubs (no real constructors).
§3 and §4 build real MACE models and are marked @pytest.mark.slow.

Investigation notes
-------------------
* The `"valid"` key in ensemble_model_sets[tag] is a dict of
  {head_name: list}, NOT a flat list.  DataLoaders returned by
  create_dataloader are (train_loader, {head_name: valid_loader}).
* create_training_subset is only called when
  replay_strategy == "random_subset".  There is no "full_dataset"
  branch inside the method; that replay mode bypasses it entirely.
  Test 3 therefore verifies the subset-size clamping behaviour: when
  train_subset_size >= full set length the DataLoader dataset length
  equals the full set length.
* ALCalculatorMLFF dispatches via handle_atomic_energies →
  _load_atomic_energies_from_source (when atomic_energies_dict is None)
  or → _use_specified_atomic_energies (when not None).  The private
  helpers are _load_from_checkpoint, _load_from_ensemble, and
  _use_specified_atomic_energies.
* setup_ensemble_dicts expects seeds_tags_dict keys to be
  pre-formed as "{name_exp}-{seed}" (matching create_seeds_tags_dict
  output).  It mutates the dict in-place, replacing old keys with
  new ones of the same form.  Passing "m0"/"m1" style keys causes a
  RuntimeError (dict changes size during iteration).
"""

import math
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from mace.modules.models import MACE
from mace.tools import AtomicNumberTable

from aims_PAX.procedures.preparation import ALCalculatorMLFF, ALEnsemble
from aims_PAX.settings import ModelSettings
from aims_PAX.tools.utilities.utilities import (
    get_atomic_energies_from_ensemble,
    setup_ensemble_dicts,
)
from tests.helpers import build_si_model

# ---------------------------------------------------------------------------
# §1 — ALEnsemble.create_training_subset
# ---------------------------------------------------------------------------

_TRAIN_SIZE = 20
_VALID_SIZE = 10
_SUBSET_TRAIN = 8
_SUBSET_VALID = 5
_BATCH = 4
_VALID_BATCH = 2


def _make_al_ensemble(
    train_size=_TRAIN_SIZE,
    valid_size=_VALID_SIZE,
    train_subset_size=_SUBSET_TRAIN,
    valid_subset_size=_SUBSET_VALID,
):
    """Build a minimal ALEnsemble stub for create_training_subset tests."""
    obj = ALEnsemble.__new__(ALEnsemble)
    obj.config = SimpleNamespace(
        replay_strategy="random_subset",
        train_subset_size=train_subset_size,
        valid_subset_size=valid_subset_size,
        set_batch_size=_BATCH,
        set_valid_batch_size=_VALID_BATCH,
    )
    # ensemble_ase_sets is used only for its keys
    obj.ensemble_ase_sets = {"m0": {}}
    obj.ensemble_model_sets = {
        "m0": {
            "train": list(range(train_size)),
            "valid": {"Default": list(range(valid_size))},
            "train_subset": {},
            "valid_subset": {},
        }
    }
    return obj


def test_random_subset_sizes():
    """create_training_subset populates train_subset/valid_subset DataLoaders
    whose dataset lengths are <= the full set sizes and equal the clamped
    subset sizes."""
    obj = _make_al_ensemble()
    model_point = [999]  # single new point included in train set

    obj.create_training_subset(model_point, idx=0)

    train_loader = obj.ensemble_model_sets["m0"]["train_subset"][0]
    valid_loaders = obj.ensemble_model_sets["m0"]["valid_subset"][0]

    assert train_loader is not None, "train_subset[0] must be set"
    assert "Default" in valid_loaders, "valid_subset[0] must have 'Default'"

    train_ds_len = len(
        train_loader.dataset
    )  # .dataset gives full size before drop_last
    valid_ds_len = len(valid_loaders["Default"].dataset)

    # Subset sizes should be clamped: min(requested, full_size)
    expected_train = min(_SUBSET_TRAIN, _TRAIN_SIZE)
    expected_valid = min(_SUBSET_VALID, _VALID_SIZE)

    assert (
        train_ds_len == expected_train
    ), f"train dataset length: got {train_ds_len}, want {expected_train}"
    assert (
        valid_ds_len == expected_valid
    ), f"valid dataset length: got {valid_ds_len}, want {expected_valid}"

    # Subset invariant: subset ≤ full set
    assert train_ds_len <= _TRAIN_SIZE
    assert valid_ds_len <= _VALID_SIZE


def test_random_subset_deterministic():
    """Two calls with the same model_point produce the same sampled indices.

    Note: the seed is threaded implicitly via Python's random module.  The
    method calls random.sample without an explicit per-call seed, so results
    are only reproducible when the global random state is fixed externally.
    We fix it here via random.seed before each call.
    """
    obj = _make_al_ensemble()
    model_point = [999]

    random.seed(0)
    obj.create_training_subset(model_point, idx=0)
    train_a = list(obj.ensemble_model_sets["m0"]["train_subset"][0].dataset)
    valid_a = list(
        obj.ensemble_model_sets["m0"]["valid_subset"][0]["Default"].dataset
    )

    # Reset state and re-build to avoid any leftover effects
    obj2 = _make_al_ensemble()
    random.seed(0)
    obj2.create_training_subset(model_point, idx=0)
    train_b = list(obj2.ensemble_model_sets["m0"]["train_subset"][0].dataset)
    valid_b = list(
        obj2.ensemble_model_sets["m0"]["valid_subset"][0]["Default"].dataset
    )

    assert train_a == train_b, "train subset must be deterministic"
    assert valid_a == valid_b, "valid subset must be deterministic"


def test_subset_size_clamped_to_full_set():
    """When train_subset_size >= full train length the DataLoader dataset
    has the same length as the full training set.

    Note: create_training_subset is only invoked for replay_strategy
    "random_subset".  This test verifies the clamping behaviour when the
    requested subset size equals the full set size, i.e. the "effective
    full-dataset" case within the random_subset code path.
    """
    obj = _make_al_ensemble(
        train_size=_TRAIN_SIZE,
        valid_size=_VALID_SIZE,
        train_subset_size=_TRAIN_SIZE,  # request the whole set
        valid_subset_size=_VALID_SIZE,
    )
    model_point = [999]

    obj.create_training_subset(model_point, idx=0)

    train_loader = obj.ensemble_model_sets["m0"]["train_subset"][0]
    valid_loaders = obj.ensemble_model_sets["m0"]["valid_subset"][0]

    # train dataset = all items from train + model_point = full set + 1 new
    # but clamping gives _TRAIN_SIZE (since train_subset_size == _TRAIN_SIZE
    # and _check_subset_size returns min(len, requested))
    assert len(train_loader.dataset) == _TRAIN_SIZE
    assert len(valid_loaders["Default"].dataset) == _VALID_SIZE


# ---------------------------------------------------------------------------
# §2 — ALCalculatorMLFF.handle_atomic_energies / branch routing
# ---------------------------------------------------------------------------


def _make_al_calc(atomic_energies_dict, restart):
    """Build a minimal ALCalculatorMLFF stub with mocked private helpers."""
    obj = ALCalculatorMLFF.__new__(ALCalculatorMLFF)
    obj.config = SimpleNamespace(
        atomic_energies_dict=atomic_energies_dict,
        restart=restart,
        seeds_tags_dict={"test-0": 0},
    )
    obj.ensemble_manager = SimpleNamespace()
    obj.ensemble_atomic_energies = None
    obj.ensemble_atomic_energies_dict = None
    obj.update_atomic_energies = False

    # Replace private helpers with Mocks
    obj._load_from_checkpoint = Mock()
    obj._load_from_ensemble = Mock()
    obj._use_specified_atomic_energies = Mock()
    # Silence the logging helper
    obj._log_atomic_energies = Mock()
    return obj


@pytest.mark.parametrize(
    "atomic_energies_dict, restart, expected_called, " "expected_not_called",
    [
        (
            {"Si": -1.234},
            False,
            "_use_specified_atomic_energies",
            ["_load_from_checkpoint", "_load_from_ensemble"],
        ),
        (
            None,
            False,
            "_load_from_ensemble",
            ["_load_from_checkpoint", "_use_specified_atomic_energies"],
        ),
        (
            None,
            True,
            "_load_from_checkpoint",
            ["_load_from_ensemble", "_use_specified_atomic_energies"],
        ),
    ],
    ids=[
        "test_handle_atomic_energies_specified_branch",
        "test_handle_atomic_energies_from_ensemble_branch",
        "test_handle_atomic_energies_from_checkpoint_branch",
    ],
)
def test_handle_atomic_energies_branch_routing(
    atomic_energies_dict, restart, expected_called, expected_not_called
):
    """handle_atomic_energies routes to exactly one private helper."""
    obj = _make_al_calc(atomic_energies_dict, restart)
    obj.handle_atomic_energies()

    assert (
        getattr(obj, expected_called).call_count == 1
    ), f"{expected_called} should be called exactly once"
    for name in expected_not_called:
        assert (
            getattr(obj, name).call_count == 0
        ), f"{name} should NOT be called"


# ---------------------------------------------------------------------------
# §3 — utilities.get_atomic_energies_from_ensemble
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_get_atomic_energies_from_ensemble():
    """get_atomic_energies_from_ensemble returns two dicts keyed by tag
    with finite numeric values for Si (Z=14)."""
    z_table = AtomicNumberTable([14])
    model = build_si_model(z_table=z_table)
    ensemble = {"m0": model}
    z = np.array([14])

    ae, ae_dict = get_atomic_energies_from_ensemble(
        ensemble=ensemble,
        z=z,
        model_choice="mace",
        dtype="float64",
    )

    # Both outputs are dicts with the correct key
    assert isinstance(ae, dict), "ensemble_atomic_energies must be a dict"
    assert isinstance(
        ae_dict, dict
    ), "ensemble_atomic_energies_dict must be a dict"
    assert "m0" in ae, "ensemble_atomic_energies must have key 'm0'"
    assert "m0" in ae_dict, "ensemble_atomic_energies_dict must have key 'm0'"

    # ae_dict["m0"] maps atomic number 14 → a finite float
    assert 14 in ae_dict["m0"], "ae_dict['m0'] must contain key 14 (Si)"
    val = float(ae_dict["m0"][14])
    assert math.isfinite(
        val
    ), f"atomic energy for Si must be finite, got {val}"

    # ae["m0"] is a numpy array of finite values
    assert isinstance(ae["m0"], np.ndarray), "ae['m0'] must be a numpy array"
    assert all(
        math.isfinite(float(v)) for v in ae["m0"]
    ), "all atomic energies must be finite"


# ---------------------------------------------------------------------------
# §4 — utilities.setup_ensemble_dicts
# ---------------------------------------------------------------------------


def _mace_settings():
    return ModelSettings(
        **{
            "GENERAL": {
                "name_exp": "test",
                "seed": 0,
                "default_dtype": "float64",
            },
            "ARCHITECTURE": {
                "model_choice": "mace",
                "num_channels": 8,
                "num_interactions": 1,
                "max_L": 0,
            },
            "MISC": {"device": "cpu"},
        }
    )


@pytest.mark.slow
def test_setup_ensemble_dicts_two_seeds():
    """setup_ensemble_dicts returns two distinct MACE model objects,
    one per seed, keyed by '{name_exp}-{seed}'."""
    z_table = AtomicNumberTable([14])
    model_settings = _mace_settings()

    # seeds_tags_dict must use pre-formed name_exp-seed keys
    # (matching create_seeds_tags_dict output) because setup_ensemble_dicts
    # reformats them in-place by constructing name_exp + "-" + str(seed).
    seeds_tags_dict = {"test-0": 0, "test-1": 1}
    ensemble_atomic_energies_dict = {
        "test-0": {14: -1.0},
        "test-1": {14: -1.0},
    }

    result = setup_ensemble_dicts(
        seeds_tags_dict=seeds_tags_dict,
        z_table=z_table,
        model_settings=model_settings,
        ensemble_atomic_energies_dict=ensemble_atomic_energies_dict,
        num_elements=118,
        device="cpu",
    )

    assert "test-0" in result, "result must contain key 'test-0'"
    assert "test-1" in result, "result must contain key 'test-1'"

    assert isinstance(
        result["test-0"], MACE
    ), "result['test-0'] must be a MACE instance"
    assert isinstance(
        result["test-1"], MACE
    ), "result['test-1'] must be a MACE instance"

    assert (
        result["test-0"] is not result["test-1"]
    ), "the two models must be distinct objects"
