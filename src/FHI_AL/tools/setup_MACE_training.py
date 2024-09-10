import torch
from typing import Optional
import torch.nn.functional
from torch_ema import ExponentialMovingAverage
from mace import modules, tools
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
)-> dict:
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

    training_setup = {}
    loss_fn: torch.nn.Module
    if training_settings["loss"] == "weighted":
        loss_fn = modules.WeightedEnergyForcesLoss(
            energy_weight=training_settings["energy_weight"],
            forces_weight=training_settings["forces_weight"],
        )

    elif training_settings["loss"] == "forces_only":
        loss_fn = modules.WeightedForcesLoss(
            forces_weight=training_settings["forces_weight"]
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
            directory=general_settings["checkpoints_dir"],
            tag=tag,
            keep=misc_settings["keep_checkpoints"],
            swa_start=training_settings.get('start_swa'),
        )
        if restart:
            epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(
                    model, training_setup["optimizer"], training_setup["lr_scheduler"]
                ),
                swa=False,
                device=misc_settings["device"],
            )
        else:
            epoch = 0
        training_setup['checkpoint_handler'] = checkpoint_handler
    else:
        checkpoint_handler_convergence = tools.CheckpointHandler(
            directory=general_settings["checkpoints_dir"]+'/convergence',
            tag=tag+"_convergence",
            keep=misc_settings["keep_checkpoints"],
            swa_start=training_settings.get('start_swa'),
        )

        
        if restart and os.path.exists(
            general_settings["checkpoints_dir"]+'/convergence'
        ):
            epoch = checkpoint_handler_convergence.load_latest(
                state=tools.CheckpointState(
                    model, training_setup["optimizer"], training_setup["lr_scheduler"]
                ),
                swa=False,
                device=misc_settings["device"],
            )
        else:
            epoch = 0
        training_setup['checkpoint_handler'] = checkpoint_handler_convergence
    
    
    training_setup['eval_interval']=training_settings["eval_interval"]
    training_setup['patience']=training_settings["patience"]
    training_setup['device']=misc_settings["device"]
    training_setup['max_grad_norm']=training_settings["clip_grad"]
    training_setup['output_args'] = {
        "forces": True,
        "virials": False, #TODO: Remove hardcoding
        "stress": False, #TODO: Remove hardcoding
    }
    training_setup['epoch'] = epoch
    return training_setup
    
