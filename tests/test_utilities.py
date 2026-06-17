import numpy as np
import pytest
import ase
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from mace.tools import AtomicNumberTable
from aims_PAX.settings import ModelSettings
from aims_PAX.tools.utilities.utilities import (
    select_best_member,
    get_seeds,
    Z_from_geometry,
    compute_average_E0s,
    create_seeds_tags_dict,
    compute_avg_num_neighbors,
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
        "ARCHITECTURE": {"model": "MACE"},
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
