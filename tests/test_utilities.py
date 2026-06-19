import numpy as np
import pytest
import ase
import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from mace.tools import AtomicNumberTable
from aims_PAX.settings import ModelSettings
from aims_PAX.settings.project import MDSettings
from aims_PAX.tools.utilities.utilities import (
    select_best_member,
    get_seeds,
    Z_from_geometry,
    compute_average_E0s,
    create_seeds_tags_dict,
    compute_avg_num_neighbors,
    get_free_vols,
    get_hirshfeld_charges,
    compute_max_error,
    to_numpy,
    atoms_full_copy,
    create_ztable,
    create_keyspec,
    is_multi_trajectory_md,
    normalize_md_settings,
    dump_yaml_for_log,
)


def test_select_best_member():
    result = select_best_member({"a": 0.5, "b": 0.2, "c": 0.9})
    assert result == "b"


def test_get_seeds_deterministic():
    result1 = get_seeds(42, 3)
    result2 = get_seeds(42, 3)
    assert np.array_equal(result1, result2)
    assert all(0 <= val < 1000 for val in result1)


def test_Z_from_geometry():
    atoms = ase.Atoms("Si6")
    zs = Z_from_geometry(atoms)
    assert zs.shape == (6,)
    assert np.all(zs == 14)


def test_compute_average_E0s():
    energies_train = [-12.0, -18.0]
    zs_train = [
        np.array([14, 14, 14, 14, 14, 14]),
        np.array([14, 14, 14, 14, 14, 14]),
    ]
    z_table = AtomicNumberTable([14])

    atomic_energies_dict = compute_average_E0s(
        energies_train, zs_train, z_table
    )

    expected = np.linalg.lstsq([[6], [6]], [-12.0, -18.0], rcond=None)[0][0]
    assert atomic_energies_dict[14] == pytest.approx(expected)


def test_create_seeds_tags_dict():
    settings_dict = {
        "GENERAL": {"model_choice": "mace", "name_exp": "exp"},
        "ARCHITECTURE": {"model_choice": "mace"},
        "MISC": {"device": "cpu"},
    }
    settings = ModelSettings(**settings_dict)

    seeds = np.array([102, 435])
    result = create_seeds_tags_dict(
        seeds, settings, dataset_dir=None, save_seeds_tags_dict=None
    )

    assert set(result.keys()) == {"exp-102", "exp-435"}
    assert result["exp-102"] == 102
    assert result["exp-435"] == 435


def test_compute_avg_num_neighbors():
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    data = Data(edge_index=edge_index, num_nodes=3)
    loader = PyGDataLoader([data], batch_size=1)

    result = compute_avg_num_neighbors(loader)
    assert result == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Hirshfeld parsing helpers
# ---------------------------------------------------------------------------

HIRSHFELD_LINES = [
    "  Performing Hirshfeld analysis of fragment charges and moments\n",
    "  Free atom volume                    :      12.34\n",
    "  Hirshfeld charge                    :      -0.56\n",
    "  Free atom volume                    :       7.89\n",
    "  Hirshfeld charge                    :       0.12\n",
]


def test_get_free_vols():
    result = get_free_vols(HIRSHFELD_LINES)
    assert result == [12.34, 7.89]


def test_get_hirshfeld_charges():
    result = get_hirshfeld_charges(HIRSHFELD_LINES)
    assert result == [-0.56, 0.12]


# ---------------------------------------------------------------------------
# compute_max_error
# ---------------------------------------------------------------------------


def test_compute_max_error():
    delta = np.array([[1.0, -3.0], [2.0, -1.0]])
    assert compute_max_error(delta) == 3.0


# ---------------------------------------------------------------------------
# to_numpy
# ---------------------------------------------------------------------------


def test_to_numpy():
    t = torch.tensor([1.0, 2.0, 3.0])
    result = to_numpy(t)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# atoms_full_copy
# ---------------------------------------------------------------------------


def test_atoms_full_copy():
    atoms = ase.Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [1.36, 1.36, 1.36]],
        cell=[[2.72, 0.0, 0.0], [0.0, 2.72, 0.0], [0.0, 0.0, 2.72]],
        pbc=True,
    )
    atoms.set_momenta([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    atoms.info["foo"] = 42

    original_cell = atoms.get_cell().copy()
    expected_masses = atoms.get_masses().copy()

    copy = atoms_full_copy(atoms)

    # Mutate the original
    atoms.positions[0] = [99.0, 99.0, 99.0]
    atoms.info["foo"] = 999
    atoms.set_cell([[9.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]])

    assert np.allclose(copy.positions[0], [0.0, 0.0, 0.0])
    assert copy.info["foo"] == 42
    assert np.allclose(copy.get_cell(), original_cell)
    assert np.all(copy.get_pbc() == np.array([True, True, True]))
    assert np.allclose(copy.get_momenta(), [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    assert np.allclose(copy.get_masses(), expected_masses)


# ---------------------------------------------------------------------------
# create_ztable
# ---------------------------------------------------------------------------


def test_create_ztable():
    zs = np.array([14, 14, 8])
    z_table = create_ztable(zs)
    assert 8 in z_table.zs
    assert 14 in z_table.zs
    assert len(z_table.zs) == 2


# ---------------------------------------------------------------------------
# create_keyspec
# ---------------------------------------------------------------------------


def test_create_keyspec_defaults():
    keyspec = create_keyspec()
    assert keyspec.info_keys["energy"] == "REF_energy"
    assert keyspec.arrays_keys["forces"] == "REF_forces"
    assert keyspec.info_keys["head"] == "head"


def test_create_keyspec_custom_energy_key():
    keyspec = create_keyspec(energy_key="MY_energy")
    assert keyspec.info_keys["energy"] == "MY_energy"


# ---------------------------------------------------------------------------
# is_multi_trajectory_md
# ---------------------------------------------------------------------------

_NVT_DICT = {
    "stat_ensemble": "nvt",
    "thermostat": "langevin",
    "temperature": 300,
    "timestep": 1.0,
}


@pytest.mark.parametrize(
    "md_settings, expected",
    [
        (None, False),
        (MDSettings(_NVT_DICT), False),
        (MDSettings({0: _NVT_DICT, 1: _NVT_DICT}), True),
    ],
)
def test_is_multi_trajectory_md(md_settings, expected):
    assert is_multi_trajectory_md(md_settings) == expected


# ---------------------------------------------------------------------------
# normalize_md_settings
# ---------------------------------------------------------------------------


def test_normalize_md_settings_single():
    md = MDSettings(_NVT_DICT)
    result_dict, is_multi = normalize_md_settings(md, num_trajectories=2)
    assert set(result_dict.keys()) == {0, 1}
    assert is_multi is False
    assert "stat_ensemble" in result_dict[0]


def test_normalize_md_settings_multi():
    md = MDSettings({0: _NVT_DICT, 1: _NVT_DICT})
    result_dict, is_multi = normalize_md_settings(md, num_trajectories=2)
    assert set(result_dict.keys()) == {0, 1}
    assert is_multi is True


# ---------------------------------------------------------------------------
# dump_yaml_for_log
# ---------------------------------------------------------------------------


def test_dump_yaml_for_log_roundtrip():
    data = {"key": "value"}
    result = dump_yaml_for_log(data)
    assert yaml.safe_load(result) == data


def test_dump_yaml_for_log_multiline_preserves_content():
    # The function converts multiline values to _LiteralStr internally, but
    # make_serializable converts them back to plain str before safe_dump.
    # The round-trip content must still be correct.
    data = {"msg": "line1\nline2"}
    result = dump_yaml_for_log(data)
    loaded = yaml.safe_load(result)
    assert loaded["msg"] == "line1\nline2"


def test_dump_yaml_for_log_force_literal_key_present():
    # slurm_str is in force_literal_keys; value is emitted as plain scalar.
    data = {"slurm_str": "no newline"}
    result = dump_yaml_for_log(data)
    loaded = yaml.safe_load(result)
    assert loaded["slurm_str"] == "no newline"
