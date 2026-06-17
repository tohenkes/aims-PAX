from pathlib import Path

import ase.io
import pytest
import torch
from mace.tools import AtomicNumberTable

import so3krates_torch.tools.torch_geometric as so3_torch_geometric

from aims_PAX.settings import ModelSettings
from aims_PAX.tools.model_tools.setup_MACE import setup_mace
from aims_PAX.tools.model_tools.setup_so3 import setup_so3krates, setup_so3lr
from aims_PAX.tools.utilities.data_handling import create_model_dataset
from aims_PAX.tools.utilities.utilities import (
    compute_average_E0s,
    create_keyspec,
)

_TEST_DATA = Path(__file__).parent / "test_data"
TRAIN_XYZ = (
    _TEST_DATA
    / "datasets/initial/training/combined_initial_train_set.xyz"
)


def mace_settings(seed=42):
    return ModelSettings(
        **{
            "GENERAL": {"name_exp": "test", "seed": seed},
            "ARCHITECTURE": {
                "model_choice": "mace",
                "num_channels": 8,
                "num_interactions": 1,
                "max_L": 0,
            },
            "MISC": {"device": "cpu"},
        }
    )


def so3krates_settings(seed=42):
    return ModelSettings(
        **{
            "GENERAL": {"name_exp": "test", "seed": seed},
            "ARCHITECTURE": {
                "model_choice": "so3krates",
                "num_features": 8,
                "num_layers": 1,
                "degrees": [1],
                "num_heads": 1,
                "energy_regression_dim": 8,
            },
            "MISC": {"device": "cpu"},
        }
    )


def so3lr_settings(seed=42):
    return ModelSettings(
        **{
            "GENERAL": {"name_exp": "test", "seed": seed},
            "ARCHITECTURE": {
                "model_choice": "so3lr",
                "num_features": 8,
                "num_layers": 1,
                "degrees": [1],
                "num_heads": 1,
                "energy_regression_dim": 8,
                "zbl_repulsion_bool": False,
                "electrostatic_energy_bool": False,
                "dispersion_energy_bool": False,
            },
            "MISC": {"device": "cpu"},
        }
    )


def load_si_fixtures():
    atoms_list = ase.io.read(TRAIN_XYZ, index=":")
    z_table = AtomicNumberTable([14])
    energies = [a.info["REF_energy"] for a in atoms_list]
    zs = [a.get_atomic_numbers() for a in atoms_list]
    atomic_energies_dict = compute_average_E0s(energies, zs, z_table)
    return atoms_list, z_table, atomic_energies_dict


def make_loader(atoms_list, z_table, r_max):
    keyspec = create_keyspec()
    dataset = create_model_dataset(
        data=atoms_list,
        seed=42,
        z_table=z_table,
        r_max=r_max,
        key_specification=keyspec,
    )
    loader = so3_torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    return loader


@pytest.mark.slow
def test_setup_mace_distinct_seeds():
    _, z_table, atomic_energies_dict = load_si_fixtures()
    model_a = setup_mace(mace_settings(seed=1), z_table, atomic_energies_dict)
    model_b = setup_mace(mace_settings(seed=2), z_table, atomic_energies_dict)
    assert isinstance(model_a, torch.nn.Module)
    assert isinstance(model_b, torch.nn.Module)
    params_a = list(model_a.parameters())[0].detach().flatten()
    params_b = list(model_b.parameters())[0].detach().flatten()
    assert not torch.allclose(params_a, params_b)


@pytest.mark.slow
def test_setup_mace_output_shapes():
    atoms_list, z_table, atomic_energies_dict = load_si_fixtures()
    settings = mace_settings()
    model = setup_mace(settings, z_table, atomic_energies_dict)
    loader = make_loader(atoms_list, z_table, r_max=5.0)
    for batch in loader:
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
        break
    assert output["energy"].shape == (1,)
    assert output["forces"].shape[1] == 3


@pytest.mark.slow
def test_setup_so3krates_distinct_seeds():
    _, z_table, atomic_energies_dict = load_si_fixtures()
    model_a = setup_so3krates(
        so3krates_settings(seed=1), atomic_energies_dict, z_table
    )
    model_b = setup_so3krates(
        so3krates_settings(seed=2), atomic_energies_dict, z_table
    )
    assert isinstance(model_a, torch.nn.Module)
    assert isinstance(model_b, torch.nn.Module)
    params_a = list(model_a.parameters())[0].detach().flatten()
    params_b = list(model_b.parameters())[0].detach().flatten()
    assert not torch.allclose(params_a, params_b)


@pytest.mark.slow
def test_setup_so3krates_output_shapes():
    atoms_list, z_table, atomic_energies_dict = load_si_fixtures()
    settings = so3krates_settings()
    model = setup_so3krates(settings, atomic_energies_dict, z_table)
    loader = make_loader(atoms_list, z_table, r_max=4.5)
    for batch in loader:
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
        break
    assert output["energy"].shape == (1,)
    assert output["forces"].shape[1] == 3


@pytest.mark.slow
def test_setup_so3lr_distinct_seeds():
    _, z_table, atomic_energies_dict = load_si_fixtures()
    model_a = setup_so3lr(
        so3lr_settings(seed=1), z_table, atomic_energies_dict
    )
    model_b = setup_so3lr(
        so3lr_settings(seed=2), z_table, atomic_energies_dict
    )
    assert isinstance(model_a, torch.nn.Module)
    assert isinstance(model_b, torch.nn.Module)
    params_a = list(model_a.parameters())[0].detach().flatten()
    params_b = list(model_b.parameters())[0].detach().flatten()
    assert not torch.allclose(params_a, params_b)


@pytest.mark.slow
def test_setup_so3lr_output_shapes():
    atoms_list, z_table, atomic_energies_dict = load_si_fixtures()
    settings = so3lr_settings()
    model = setup_so3lr(settings, z_table, atomic_energies_dict)
    loader = make_loader(atoms_list, z_table, r_max=4.5)
    for batch in loader:
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
        break
    assert output["energy"].shape == (1,)
    assert output["forces"].shape[1] == 3
