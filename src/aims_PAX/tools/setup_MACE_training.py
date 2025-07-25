import torch
from typing import Optional
from torch_ema import ExponentialMovingAverage
from mace import modules, tools
from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch
from mace.modules.loss import (
    weighted_mean_squared_error_energy,
    mean_squared_error_forces,
)
import aims_PAX
import numpy as np
import os

#############################################################################
############ This part is mostly taken from the MACE source code ############
############ with slight modifications to fit the needs of AL    ############
#############################################################################

# this function is basically from the MACE run_train script but only
# contains the parts important for training settings.


def setup_mace_training(
    settings: dict,
    model,
    tag: str,
    restart: bool = False,
    convergence: bool = False,
    checkpoints_dir: str = None,
    al_settings: dict = None,
) -> dict:
    """
    Setup the MACE training according to the settings and return it.

    Args:
        settings (dict): MACE model settings
        model: MACE model to train
        tag (str): Tag for identifying the model

    Raises:
        RuntimeError: If the scheduler is unknown.

    Returns:
        dict: Dictionary containing the training setup.
    """

    general_settings = settings["GENERAL"]
    training_settings = settings["TRAINING"]
    misc_settings = settings["MISC"]

    if checkpoints_dir is None:
        checkpoints_dir = general_settings["checkpoints_dir"]

    training_setup = {}
    loss_fn: torch.nn.Module
    if training_settings["loss"].lower() == "weighted":
        loss_fn = modules.WeightedEnergyForcesLoss(
            energy_weight=training_settings["energy_weight"],
            forces_weight=training_settings["forces_weight"],
        )

    elif training_settings["loss"].lower() == "forces_only":
        loss_fn = modules.WeightedForcesLoss(
            forces_weight=training_settings["forces_weight"]
        )

    elif training_settings["loss"].lower() == "weighted_stress":
        loss_fn = modules.WeightedEnergyForcesStressLoss(
            energy_weight=training_settings["energy_weight"],
            forces_weight=training_settings["forces_weight"],
            stress_weight=training_settings["stress_weight"],
        )
    elif training_settings["loss"].lower() == "intermol":
        mol_idxs = al_settings.get("mol_idxs")
        assert mol_idxs is not None, "mol_idxs must be provided"
        mol_idxs = np.load(mol_idxs, allow_pickle=True)["arr_0"].tolist()

        loss_fn = WeightedEnergyForceIntermolForceLoss(
            energy_weight=training_settings["energy_weight"],
            forces_weight=training_settings["forces_weight"],
            intermol_forces_weight=training_settings["intermol_forces_weight"],
            mol_idxs=mol_idxs,
        )
    else:
        loss_fn = modules.EnergyForcesLoss(
            energy_weight=training_settings["energy_weight"],
            forces_weight=training_settings["forces_weight"],
        )

    training_setup["loss_fn"] = loss_fn

    decay_interactions = {}
    no_decay_interactions = {}
    for name, param in model.interactions.named_parameters():
        if "linear.weight" in name or "skip_tp_full.weight" in name:
            decay_interactions[name] = param
        else:
            no_decay_interactions[name] = param

    param_options = dict(
        params=[
            {
                "name": "embedding",
                "params": model.node_embedding.parameters(),
                "weight_decay": 0.0,
            },
            {
                "name": "interactions_decay",
                "params": list(decay_interactions.values()),
                "weight_decay": training_settings["weight_decay"],
            },
            {
                "name": "interactions_no_decay",
                "params": list(no_decay_interactions.values()),
                "weight_decay": 0.0,
            },
            {
                "name": "products",
                "params": model.products.parameters(),
                "weight_decay": training_settings["weight_decay"],
            },
            {
                "name": "readouts",
                "params": model.readouts.parameters(),
                "weight_decay": 0.0,
            },
        ],
        lr=training_settings["lr"],
        amsgrad=training_settings["amsgrad"],
    )

    optimizer: torch.optim.Optimizer
    if training_settings["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(**param_options)
    else:
        optimizer = torch.optim.Adam(**param_options)

    training_setup["optimizer"] = optimizer

    if training_settings["scheduler"] == "ExponentialLR":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=training_settings["lr_scheduler_gamma"]
        )
    elif training_settings["scheduler"] == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=training_settings["lr_factor"],
            patience=training_settings["scheduler_patience"],
        )
    else:
        raise RuntimeError(
            f"Unknown scheduler: {training_settings['scheduler']}"
        )

    training_setup["lr_scheduler"] = lr_scheduler

    ema: Optional[ExponentialMovingAverage] = None
    if training_settings["ema"]:
        ema = ExponentialMovingAverage(
            model.parameters(), decay=training_settings["ema_decay"]
        )
        training_setup["ema"] = ema

    if not convergence:
        checkpoint_handler = tools.CheckpointHandler(
            directory=checkpoints_dir,
            tag=tag,
            keep=misc_settings["keep_checkpoints"],
            swa_start=training_settings.get("start_swa"),
        )
        if restart:
            epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(
                    model,
                    training_setup["optimizer"],
                    training_setup["lr_scheduler"],
                ),
                swa=False,
                device=misc_settings["device"],
            )
        else:
            epoch = 0
        training_setup["checkpoint_handler"] = checkpoint_handler
    else:
        checkpoint_handler_convergence = tools.CheckpointHandler(
            directory=checkpoints_dir + "/convergence",
            tag=tag + "_convergence",
            keep=misc_settings["keep_checkpoints"],
            swa_start=training_settings.get("start_swa"),
        )

        if restart and os.path.exists(checkpoints_dir + "/convergence"):
            epoch = checkpoint_handler_convergence.load_latest(
                state=tools.CheckpointState(
                    model,
                    training_setup["optimizer"],
                    training_setup["lr_scheduler"],
                ),
                swa=False,
                device=misc_settings["device"],
            )
        else:
            epoch = 0
        training_setup["checkpoint_handler"] = checkpoint_handler_convergence

    training_setup["eval_interval"] = training_settings["eval_interval"]
    training_setup["patience"] = training_settings["patience"]
    training_setup["device"] = misc_settings["device"]
    training_setup["max_grad_norm"] = training_settings["clip_grad"]
    training_setup["output_args"] = {
        "forces": True,
        "virials": False,  # TODO: Remove hardcoding
        "stress": general_settings.get("compute_stress", False),
    }
    training_setup["epoch"] = epoch
    return training_setup


def reset_optimizer(
    model: modules.MACE,
    training_setup: dict,
    training_settings: dict,
) -> dict:
    """
    Resets the optimizer state and adds the freshly initialized optimizer
    to the training setup (which is used to control the training in
    aims PAX).

    Args:
        model (modules.MACE): MACE model (to access the parameters)
        training_setup (dict): Current training setup containing the optimizer.
        training_settings (dict): Training settings containing optimizer
                                    options.
    Returns:
        dict: Training setup with the new optimizer.
    """
    decay_interactions = {}
    no_decay_interactions = {}
    for name, param in model.interactions.named_parameters():
        if "linear.weight" in name or "skip_tp_full.weight" in name:
            decay_interactions[name] = param
        else:
            no_decay_interactions[name] = param

    param_options = dict(
        params=[
            {
                "name": "embedding",
                "params": model.node_embedding.parameters(),
                "weight_decay": 0.0,
            },
            {
                "name": "interactions_decay",
                "params": list(decay_interactions.values()),
                "weight_decay": training_settings["weight_decay"],
            },
            {
                "name": "interactions_no_decay",
                "params": list(no_decay_interactions.values()),
                "weight_decay": 0.0,
            },
            {
                "name": "products",
                "params": model.products.parameters(),
                "weight_decay": training_settings["weight_decay"],
            },
            {
                "name": "readouts",
                "params": model.readouts.parameters(),
                "weight_decay": 0.0,
            },
        ],
        lr=training_settings["lr"],
        amsgrad=training_settings["amsgrad"],
    )

    optimizer: torch.optim.Optimizer
    if training_settings["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(**param_options)
    else:
        optimizer = torch.optim.Adam(**param_options)

    training_setup["optimizer"] = optimizer
    return training_setup


def mean_square_intermol_error(
    ref: Batch,
    pred: TensorDict,
    mol_idxs: list,
) -> torch.Tensor:

    intermol_forces_pred = compute_mol_forces(
        forces=pred["forces"],
        select_idxs=mol_idxs,
    )

    intermol_forces_ref = compute_mol_forces(
        forces=ref["forces"],
        select_idxs=mol_idxs,
    )

    return torch.mean(torch.square(intermol_forces_pred - intermol_forces_ref))


def compute_mol_forces(forces, select_idxs) -> torch.tensor:
    """
    Compute molecular forces by summing atomic forces for selected indices.
    Handles both batched and non-batched inputs.

    Args:
        forces (torch.tensor): Tensor of atomic forces. Shape can be
                            [n_atoms, 3] or [batch_size, n_atoms, 3].
        select_idxs (list): List of indices for molecules.

    Returns:
        torch.tensor: Tensor of molecular forces. Shape is
                        [len(select_idxs), 3] for non-batched input,
                      or [batch_size, len(select_idxs), 3] for batched input.
    """
    if forces.ndim == 2:  # Non-batched case
        mol_forces = torch.empty(
            (len(select_idxs), 3), dtype=forces.dtype, device=forces.device
        )
        for idx, mol in enumerate(select_idxs):
            mol_forces[idx, :] = forces[mol].sum(axis=0)
    elif forces.ndim == 3:  # Batched case
        batch_size = forces.shape[0]
        mol_forces = torch.empty(
            (batch_size, len(select_idxs), 3),
            dtype=forces.dtype,
            device=forces.device,
        )
        for idx, mol in enumerate(select_idxs):
            mol_forces[:, idx, :] = forces[:, mol, :].sum(axis=1)
    else:
        raise ValueError("Unexpected number of dimensions in forces tensor")

    return mol_forces


class WeightedEnergyForceIntermolForceLoss(torch.nn.Module):
    """
    Weighted loss function for energy, forces, and inter-molecular forces.
    """

    def __init__(
        self,
        energy_weight: float = 1.0,
        forces_weight: float = 1.0,
        intermol_forces_weight: float = 1.0,
        mol_idxs: list = None,
    ):
        super().__init__()
        assert mol_idxs is not None, "mol_idxs must be provided"
        self.mol_idxs = mol_idxs

        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.intermol_forces_weight = intermol_forces_weight

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:

        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.intermol_forces_weight
            * mean_square_intermol_error(ref, pred, self.mol_idxs)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}), "
            f"intermol_forces_weight={self.intermol_forces_weight:.3f})"
        )
