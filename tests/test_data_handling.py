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

import ase
import ase.build
import numpy as np
import pytest
from mace import tools
from torch.utils.data import DataLoader

from aims_PAX.tools.utilities.data_handling import (
    Configuration,
    KeySpecification,
    ase_to_model_ensemble_sets,
    create_dataloader,
    create_model_dataset,
    random_train_valid_split,
    sort_ase_dataset_to_heads,
    split_data,
    split_data_heads_evenly,
    test_config_types as group_by_config_type,
    update_datasets,
    update_keyspec_from_kwargs,
)
from aims_PAX.tools.utilities.utilities import create_keyspec, create_ztable

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


# ---------------------------------------------------------------------------
# §5 — update_keyspec_from_kwargs
# ---------------------------------------------------------------------------


def _fresh_keyspec():
    """Return a KeySpecification with default REF_* keys."""
    return create_keyspec()


def test_update_keyspec_info_key():
    ks = _fresh_keyspec()
    original_forces = ks.arrays_keys["forces"]
    update_keyspec_from_kwargs(ks, {"energy_key": "MY_energy"})
    assert ks.info_keys["energy"] == "MY_energy"
    # other keys must be untouched
    assert ks.arrays_keys["forces"] == original_forces


def test_update_keyspec_arrays_key():
    ks = _fresh_keyspec()
    original_energy = ks.info_keys["energy"]
    update_keyspec_from_kwargs(ks, {"forces_key": "MY_forces"})
    assert ks.arrays_keys["forces"] == "MY_forces"
    assert ks.info_keys["energy"] == original_energy


def test_update_keyspec_unknown_key_ignored():
    ks = _fresh_keyspec()
    snapshot_info = dict(ks.info_keys)
    snapshot_arrays = dict(ks.arrays_keys)
    update_keyspec_from_kwargs(ks, {"unknown_key": "x"})
    assert ks.info_keys == snapshot_info
    assert ks.arrays_keys == snapshot_arrays


# ---------------------------------------------------------------------------
# §6 — test_config_types
# ---------------------------------------------------------------------------


def _make_config(config_type: str, energy: float = -1.0) -> Configuration:
    return Configuration(
        atomic_numbers=np.array([14]),
        positions=np.array([[0.0, 0.0, 0.0]]),
        energy=energy,
        forces=np.zeros((1, 3)),
        config_type=config_type,
    )


def test_test_config_types_correct_grouping():
    c1 = _make_config("typeA", -1.0)
    c2 = _make_config("typeB", -2.0)
    c3 = _make_config("typeA", -3.0)
    result = group_by_config_type([c1, c2, c3])
    groups = {ct: confs for ct, confs in result}
    assert set(groups.keys()) == {"typeA", "typeB"}
    assert groups["typeA"] == [c1, c3]
    assert groups["typeB"] == [c2]


def test_test_config_types_insertion_order():
    c1 = _make_config("typeA")
    c2 = _make_config("typeB")
    c3 = _make_config("typeA")
    result = group_by_config_type([c1, c2, c3])
    # typeA appeared first, so it must be first in the output
    assert result[0][0] == "typeA"
    assert result[1][0] == "typeB"


def test_test_config_types_single_type():
    configs = [_make_config("only", float(-i)) for i in range(4)]
    result = group_by_config_type(configs)
    assert len(result) == 1
    assert result[0][0] == "only"
    assert result[0][1] == configs


# ---------------------------------------------------------------------------
# §7 — split_data_heads_evenly
# ---------------------------------------------------------------------------


def _make_atoms_list(n: int) -> list:
    """Minimal 1-atom Si Atoms objects (need .copy())."""
    return [ase.Atoms("Si", positions=[[0, 0, 0]]) for _ in range(n)]


@pytest.mark.parametrize(
    "n_data,num_heads,expected_total,all_non_empty",
    [
        (6, 3, 6, True),  # even split: 2 each
        (4, 3, 4, True),  # head 0 gets 2, heads 1 and 2 get 1
        (2, 3, 3, True),  # sparse: cycle until all 3 heads have ≥1
    ],
)
def test_split_data_heads_evenly_keys_and_non_empty(
    n_data, num_heads, expected_total, all_non_empty
):
    data = _make_atoms_list(n_data)
    result = split_data_heads_evenly(data, num_heads)
    assert set(result.keys()) == set(range(num_heads))
    assert sum(len(v) for v in result.values()) == expected_total
    if all_non_empty:
        assert all(len(v) > 0 for v in result.values())


def test_split_data_heads_evenly_total_sufficient():
    """Even split: total items == len(data), round-robin order."""
    data = _make_atoms_list(6)
    result = split_data_heads_evenly(data, 3)
    total = sum(len(v) for v in result.values())
    assert total == 6
    for h in range(3):
        assert len(result[h]) == 2


def test_split_data_heads_evenly_sparse_total():
    """Sparse case: total items == num_heads (each head gets exactly 1)."""
    data = _make_atoms_list(2)
    result = split_data_heads_evenly(data, 3)
    total = sum(len(v) for v in result.values())
    assert total == 3


def test_split_data_heads_evenly_uneven():
    """4 items, 3 heads: head 0 gets 2, heads 1 and 2 get 1."""
    data = _make_atoms_list(4)
    result = split_data_heads_evenly(data, 3)
    assert len(result[0]) == 2
    assert len(result[1]) == 1
    assert len(result[2]) == 1


# ---------------------------------------------------------------------------
# §8 — sort_ase_dataset_to_heads
# ---------------------------------------------------------------------------


def _atoms_with_head(head: str) -> ase.Atoms:
    a = ase.Atoms("Si", positions=[[0, 0, 0]])
    a.info["head"] = head
    return a


def _atoms_no_head() -> ase.Atoms:
    return ase.Atoms("Si", positions=[[0, 0, 0]])


def test_sort_ase_dataset_to_heads_two_heads():
    a1 = _atoms_with_head("headA")
    a2 = _atoms_with_head("headB")
    a3 = _atoms_with_head("headA")
    result = sort_ase_dataset_to_heads([a1, a2, a3])
    assert set(result.keys()) == {"headA", "headB"}
    assert result["headA"] == [a1, a3]
    assert result["headB"] == [a2]


def test_sort_ase_dataset_to_heads_no_head_key():
    a1 = _atoms_no_head()
    a2 = _atoms_no_head()
    result = sort_ase_dataset_to_heads([a1, a2])
    assert list(result.keys()) == ["Default"]
    assert len(result["Default"]) == 2


def test_sort_ase_dataset_to_heads_mixed():
    a_head = _atoms_with_head("headA")
    a_default = _atoms_no_head()
    result = sort_ase_dataset_to_heads([a_head, a_default])
    assert set(result.keys()) == {"headA", "Default"}
    assert result["headA"] == [a_head]
    assert result["Default"] == [a_default]


# ---------------------------------------------------------------------------
# §9 — create_dataloader
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def si_z_table():
    return create_ztable(np.array([14]))


@pytest.fixture(scope="module")
def si_model_datasets(data_dir, si_z_table):
    from ase.io import read

    ks = create_keyspec()
    train_data = read(
        str(
            data_dir
            / "datasets"
            / "initial"
            / "training"
            / "combined_initial_train_set.xyz"
        ),
        index=":",
    )
    valid_data = read(
        str(
            data_dir
            / "datasets"
            / "initial"
            / "validation"
            / "combined_initial_valid_set.xyz"
        ),
        index=":",
    )
    train_set = create_model_dataset(
        train_data,
        seed=0,
        z_table=si_z_table,
        r_max=5.0,
        key_specification=ks,
    )
    valid_set = create_model_dataset(
        valid_data,
        seed=0,
        z_table=si_z_table,
        r_max=5.0,
        key_specification=ks,
    )
    return train_set, {"Default": valid_set}


def test_create_dataloader_returns_tuple(si_model_datasets):
    train_set, valid_set = si_model_datasets
    result = create_dataloader(train_set, valid_set, 2, 2)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_create_dataloader_train_loader_is_dataloader(si_model_datasets):
    train_set, valid_set = si_model_datasets
    train_loader, _ = create_dataloader(train_set, valid_set, 2, 2)
    assert isinstance(train_loader, DataLoader)


def test_create_dataloader_valid_loaders_is_dict(si_model_datasets):
    train_set, valid_set = si_model_datasets
    _, valid_loaders = create_dataloader(train_set, valid_set, 2, 2)
    assert isinstance(valid_loaders, dict)
    assert "Default" in valid_loaders


def test_create_dataloader_train_yields_batches(si_model_datasets):
    train_set, valid_set = si_model_datasets
    train_loader, _ = create_dataloader(train_set, valid_set, 2, 2)
    batches = list(train_loader)
    assert len(batches) >= 1
