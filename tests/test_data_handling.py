"""
Phase 2 — data-pipeline tests for `aims_PAX.tools.utilities.data_handling`.

§1 random_train_valid_split — seeded, deterministic (pure, plain lists)
§2 split_data — local-RNG regression guards (pure, plain lists)
§3 ase_to_model_ensemble_sets / update_datasets — member-sizes and append
§4 KeySpecification / Configuration — dataclass construction

Convention focus:
  #2 (invariants, not hardcoded internals)
  #5 (mutation-sensitive — §2 regression tests must fail if local-RNG fix is
      reverted to global random.shuffle)
"""

import random

import ase.build
import numpy as np
import pytest
from mace import tools

from aims_PAX.tools.utilities.data_handling import (
    Configuration,
    KeySpecification,
    ase_to_model_ensemble_sets,
    random_train_valid_split,
    split_data,
    update_datasets,
)
from aims_PAX.tools.utilities.utilities import create_keyspec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _items(n=10):
    """Stable list of distinct plain objects for split tests."""
    return list(range(n))


def _make_atoms():
    """Cu 1-atom fcc primitive cell with the DFT info keys expected by
    load_from_atoms (using the REF_* convention from create_keyspec defaults).
    """
    atoms = ase.build.bulk("Cu", "fcc", a=3.6)
    atoms.info["REF_energy"] = -3.0
    atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3))
    return atoms


@pytest.fixture(scope="module")
def keyspec():
    return create_keyspec()


@pytest.fixture(scope="module")
def cu_z_table():
    return tools.get_atomic_number_table_from_zs([29])  # Cu


# ---------------------------------------------------------------------------
# §1 — random_train_valid_split
# ---------------------------------------------------------------------------


def test_random_split_partitions_input():
    items = _items(10)
    train, valid = random_train_valid_split(items, valid_fraction=0.2, seed=0)
    assert len(train) + len(valid) == 10
    assert set(train) | set(valid) == set(items)
    assert not set(train) & set(valid)


def test_random_split_valid_fraction_size():
    items = _items(10)
    train, valid = random_train_valid_split(items, valid_fraction=0.3, seed=0)
    assert len(valid) == 3
    assert len(train) == 7


def test_random_split_seed_reproducible():
    items = _items(20)
    t1, v1 = random_train_valid_split(items, 0.2, seed=42)
    t2, v2 = random_train_valid_split(items, 0.2, seed=42)
    assert t1 == t2
    assert v1 == v2


def test_random_split_seed_varies():
    items = _items(20)
    t1, _ = random_train_valid_split(items, 0.2, seed=0)
    t2, _ = random_train_valid_split(items, 0.2, seed=1)
    assert t1 != t2


def test_random_split_rejects_bad_fraction():
    items = _items(10)
    with pytest.raises(AssertionError):
        random_train_valid_split(items, 0.0, seed=0)
    with pytest.raises(AssertionError):
        random_train_valid_split(items, 1.0, seed=0)


# ---------------------------------------------------------------------------
# §2 — split_data (local-RNG regression guards)
# ---------------------------------------------------------------------------


def test_split_data_sizes():
    train, valid = split_data(_items(10), 0.2, seed=0)
    assert len(valid) == 2
    assert len(train) == 8
    assert set(train) | set(valid) == set(_items(10))
    assert not set(train) & set(valid)


def test_split_data_does_not_mutate_input():
    items = _items(10)
    original = items.copy()
    split_data(items, 0.2, seed=0)
    assert items == original


def test_split_data_seed_reproducible():
    items = _items(20)
    t1, v1 = split_data(items, 0.2, seed=7)
    t2, v2 = split_data(items, 0.2, seed=7)
    assert t1 == t2
    assert v1 == v2


def test_split_data_seed_varies():
    items = _items(20)
    t1, _ = split_data(items, 0.2, seed=0)
    t2, _ = split_data(items, 0.2, seed=7)
    assert t1 != t2


def test_split_data_independent_of_global_rng():
    """Regression: split_data must use a local RNG, not global random.shuffle.

    Before the fix, global random.shuffle consumed the process-wide RNG stream,
    so the partition depended on every prior random.* call. This test proves the
    fix: perturbing the global state must not change the result for a given seed.
    If this test is reverted to use global random.shuffle, it will fail.
    """
    items = _items(20)
    random.seed(0)
    t1, v1 = split_data(items, 0.2, seed=99)

    random.seed(12345)
    for _ in range(500):
        random.random()
    t2, v2 = split_data(items, 0.2, seed=99)

    assert t1 == t2
    assert v1 == v2


# ---------------------------------------------------------------------------
# §3 — ase_to_model_ensemble_sets / update_datasets
# ---------------------------------------------------------------------------


def test_ase_to_ensemble_sets_member_sizes(keyspec, cu_z_table):
    n_train, n_valid = 4, 2
    ase_sets = {
        "m0": {
            "train": [_make_atoms() for _ in range(n_train)],
            "valid": [_make_atoms() for _ in range(n_valid)],
        },
        "m1": {
            "train": [_make_atoms() for _ in range(n_train)],
            "valid": [_make_atoms() for _ in range(n_valid)],
        },
    }
    result = ase_to_model_ensemble_sets(
        ase_sets, cu_z_table, r_max=5.0, key_specification=keyspec
    )
    assert set(result.keys()) == {"m0", "m1"}
    for tag in ("m0", "m1"):
        assert len(result[tag]["train"]) == n_train
        assert len(result[tag]["valid"]["Default"]) == n_valid


def test_update_datasets_appends_to_ase_sets(keyspec, cu_z_table):
    n_init_train, n_init_valid = 4, 2
    n_new = 5
    ase_set = {
        "train": [_make_atoms() for _ in range(n_init_train)],
        "valid": [_make_atoms() for _ in range(n_init_valid)],
    }
    model_set = {"train": [], "valid": {"Default": []}}
    new_points = [_make_atoms() for _ in range(n_new)]

    # split_data(5, 0.2) → n_valid = int(5 * 0.2) = 1, n_train = 4
    n_new_valid = int(n_new * 0.2)
    n_new_train = n_new - n_new_valid

    ase_set, model_set = update_datasets(
        new_points=new_points,
        model_set=model_set,
        ase_set=ase_set,
        valid_split=0.2,
        z_table=cu_z_table,
        seed=42,
        r_max=5.0,
        key_specification=keyspec,
    )

    assert len(ase_set["train"]) == n_init_train + n_new_train
    assert len(ase_set["valid"]) == n_init_valid + n_new_valid
    assert len(model_set["train"]) == n_new_train
    assert len(model_set["valid"]["Default"]) == n_new_valid


# ---------------------------------------------------------------------------
# §4 — KeySpecification / Configuration dataclasses
# ---------------------------------------------------------------------------


def test_keyspecification_defaults():
    ks = KeySpecification()
    assert isinstance(ks.info_keys, dict)
    assert isinstance(ks.arrays_keys, dict)


def test_keyspecification_update():
    ks = KeySpecification()
    ks.update(info_keys={"energy": "E"}, arrays_keys={"forces": "F"})
    assert ks.info_keys["energy"] == "E"
    assert ks.arrays_keys["forces"] == "F"


def test_create_keyspec_has_expected_keys(keyspec):
    assert "energy" in keyspec.info_keys
    assert "forces" in keyspec.arrays_keys
    assert keyspec.info_keys["energy"] == "REF_energy"
    assert keyspec.arrays_keys["forces"] == "REF_forces"


def test_configuration_construction():
    cfg = Configuration(
        atomic_numbers=np.array([29, 29]),
        positions=np.zeros((2, 3)),
        energy=-3.5,
    )
    assert cfg.energy == -3.5
    assert cfg.forces is None
    assert cfg.weight == 1.0
    assert cfg.config_type == "Default"
