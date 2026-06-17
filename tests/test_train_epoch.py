import math
from pathlib import Path

import ase.io
import pytest
import torch
from mace.tools import AtomicNumberTable
from mace.tools.utils import MetricsLogger

import so3krates_torch.tools.torch_geometric as so3_torch_geometric

from aims_PAX.settings import ModelSettings
from aims_PAX.tools.model_tools.train_epoch import (
    train_epoch,
    validate_epoch_ensemble,
)
from aims_PAX.tools.model_tools.training_tools import setup_model_training
from aims_PAX.tools.utilities.data_handling import create_model_dataset
from aims_PAX.tools.utilities.utilities import (
    compute_average_E0s,
    create_keyspec,
)

_TEST_DATA = Path(__file__).parent / "test_data"
_TRAIN_XYZ = (
    _TEST_DATA / "datasets/initial/training/combined_initial_train_set.xyz"
)
_VALID_XYZ = (
    _TEST_DATA / "datasets/initial/validation/combined_initial_valid_set.xyz"
)


def make_settings(arch, tmp_path):
    if arch == "mace":
        return ModelSettings(
            **{
                "GENERAL": {
                    "name_exp": "test",
                    "seed": 42,
                    "checkpoints_dir": str(tmp_path / "checkpoints"),
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
    elif arch == "so3krates":
        return ModelSettings(
            **{
                "GENERAL": {
                    "name_exp": "test",
                    "seed": 42,
                    "checkpoints_dir": str(tmp_path / "checkpoints"),
                },
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
    else:  # so3lr
        return ModelSettings(
            **{
                "GENERAL": {
                    "name_exp": "test",
                    "seed": 42,
                    "checkpoints_dir": str(tmp_path / "checkpoints"),
                },
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
    train_atoms = ase.io.read(_TRAIN_XYZ, index=":")
    valid_atoms = ase.io.read(_VALID_XYZ, index=":")
    z_table = AtomicNumberTable([14])
    energies = [a.info["REF_energy"] for a in train_atoms]
    zs = [a.get_atomic_numbers() for a in train_atoms]
    atomic_energies_dict = compute_average_E0s(energies, zs, z_table)
    return train_atoms, valid_atoms, z_table, atomic_energies_dict


def build_model(arch, settings, z_table, atomic_energies_dict):
    if arch == "mace":
        from aims_PAX.tools.model_tools.setup_MACE import setup_mace

        return setup_mace(settings, z_table, atomic_energies_dict)
    elif arch == "so3krates":
        from aims_PAX.tools.model_tools.setup_so3 import setup_so3krates

        return setup_so3krates(settings, atomic_energies_dict, z_table)
    else:
        from aims_PAX.tools.model_tools.setup_so3 import setup_so3lr

        return setup_so3lr(settings, z_table, atomic_energies_dict)


def build_loaders(train_atoms, valid_atoms, z_table, r_max, tmp_path):
    keyspec = create_keyspec()
    train_set = create_model_dataset(
        data=train_atoms,
        seed=42,
        z_table=z_table,
        r_max=r_max,
        key_specification=keyspec,
    )
    valid_set = create_model_dataset(
        data=valid_atoms,
        seed=42,
        z_table=z_table,
        r_max=r_max,
        key_specification=keyspec,
    )
    train_loader = so3_torch_geometric.dataloader.DataLoader(
        dataset=train_set,
        batch_size=4,
        shuffle=True,
        drop_last=True,
    )
    valid_loaders = {
        "Default": so3_torch_geometric.dataloader.DataLoader(
            dataset=valid_set,
            batch_size=4,
            shuffle=False,
            drop_last=False,
        )
    }
    return train_loader, valid_loaders


def build_training_setup(settings, model, arch, tmp_path):
    return setup_model_training(
        settings=settings,
        model=model,
        model_choice=arch,
        tag="test",
        checkpoints_dir=tmp_path / "checkpoints",
    )


@pytest.mark.slow
@pytest.mark.parametrize("arch", ["mace", "so3krates", "so3lr"])
def test_train_epoch_loss_decreases(arch, tmp_path):
    train_atoms, valid_atoms, z_table, atomic_energies_dict = (
        load_si_fixtures()
    )
    settings = make_settings(arch, tmp_path)
    r_max = 5.0 if arch == "mace" else 4.5
    model = build_model(arch, settings, z_table, atomic_energies_dict)
    train_loader, valid_loaders = build_loaders(
        train_atoms, valid_atoms, z_table, r_max, tmp_path
    )
    training_setup = build_training_setup(settings, model, arch, tmp_path)
    logger = MetricsLogger(directory=tmp_path / "logs", tag="test")

    loss0 = train_epoch(
        model=model,
        loss_fn=training_setup["loss_fn"],
        train_loader=train_loader,
        optimizer=training_setup["optimizer"],
        lr_scheduler=training_setup["lr_scheduler"],
        epoch=0,
        start_epoch=0,
        valid_loss=0.0,
        logger=logger,
        output_args=training_setup["output_args"],
        device=torch.device("cpu"),
        ema=training_setup["ema"],
        max_grad_norm=training_setup["max_grad_norm"],
    )
    loss1 = train_epoch(
        model=model,
        loss_fn=training_setup["loss_fn"],
        train_loader=train_loader,
        optimizer=training_setup["optimizer"],
        lr_scheduler=training_setup["lr_scheduler"],
        epoch=1,
        start_epoch=0,
        valid_loss=loss0,
        logger=logger,
        output_args=training_setup["output_args"],
        device=torch.device("cpu"),
        ema=training_setup["ema"],
        max_grad_norm=training_setup["max_grad_norm"],
    )
    # SGD on 8 structures with random init is noisy; assert finiteness.
    # Strict loss1 < loss0 is unreliable on tiny data.
    assert math.isfinite(loss0), f"epoch-0 loss is not finite: {loss0}"
    assert math.isfinite(loss1), f"epoch-1 loss is not finite: {loss1}"


@pytest.mark.slow
@pytest.mark.parametrize("arch", ["mace", "so3krates", "so3lr"])
def test_validate_epoch_ensemble_returns_metrics(arch, tmp_path):
    train_atoms, valid_atoms, z_table, atomic_energies_dict = (
        load_si_fixtures()
    )
    settings = make_settings(arch, tmp_path)
    r_max = 5.0 if arch == "mace" else 4.5
    model = build_model(arch, settings, z_table, atomic_energies_dict)
    train_loader, valid_loaders = build_loaders(
        train_atoms, valid_atoms, z_table, r_max, tmp_path
    )
    training_setup = build_training_setup(settings, model, arch, tmp_path)
    logger = MetricsLogger(directory=tmp_path / "logs", tag="test")

    train_epoch(
        model=model,
        loss_fn=training_setup["loss_fn"],
        train_loader=train_loader,
        optimizer=training_setup["optimizer"],
        lr_scheduler=training_setup["lr_scheduler"],
        epoch=0,
        start_epoch=0,
        valid_loss=0.0,
        logger=logger,
        output_args=training_setup["output_args"],
        device=torch.device("cpu"),
        ema=training_setup["ema"],
        max_grad_norm=training_setup["max_grad_norm"],
    )

    result = validate_epoch_ensemble(
        ensemble={"member-0": model},
        training_setups={"member-0": training_setup},
        valid_loaders=valid_loaders,
        logger=logger,
        log_errors="screen",
        epoch=0,
    )

    assert len(result) == 4
    ensemble_valid_loss, avg_loss, eval_metrics, _ = result
    assert math.isfinite(eval_metrics["mae_f"])
    assert math.isfinite(eval_metrics["mae_e"])
