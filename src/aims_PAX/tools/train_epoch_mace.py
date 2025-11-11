import torch
import numpy as np
from typing import Optional, Dict
from mace.tools.train import take_step, evaluate, valid_err_log
from mace.tools.utils import MetricsLogger
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

#############################################################################
############ This part is mostly taken from the MACE source code ############
############ with slight modifications to fit the needs of AL    ############
#############################################################################


def train_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    epoch: int,
    start_epoch: int,
    valid_loss: float,
    logger: MetricsLogger,
    output_args: Dict[str, bool],
    device: torch.device,
    ema: Optional[ExponentialMovingAverage] = None,
    max_grad_norm: Optional[float] = 10.0,
) -> float:
    """
    Trains a MACE model for a single epoch.

    Args:
        model (torch.nn.Module): MACE model to train.
        loss_fn (torch.nn.Module): Loss function to use.
        train_loader (DataLoader):  Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        lr_scheduler (torch.optim.lr_scheduler.ExponentialLR): Learning rate scheduler.
        epoch (int): Current epoch.
        start_epoch (int): First epoch.
        valid_loss (float): Current validation loss.
        logger (MetricsLogger): Logger for metrics.
        output_args (Dict[str, bool]): Output arguments.
        device (torch.device): Device to use. CPU or GPU.
        ema (Optional[ExponentialMovingAverage], optional): Exponential moving average. Defaults to None.
        max_grad_norm (Optional[float], optional): Gradient clipping. Defaults to 10.0.

    Returns:
        float: Total loss for the epoch.
    """
    # if max_grad_norm is not None:
    # logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")
    # Train
    if lr_scheduler is not None:
        if epoch > start_epoch:
            lr_scheduler.step(
                metrics=valid_loss
            )  # Can break if exponential LR, TODO fix that!
    total_loss = 0.0
    batches = 0
    for batch in train_loader:
        loss, opt_metrics = take_step(
            model=model,
            loss_fn=loss_fn,
            batch=batch,
            optimizer=optimizer,
            ema=ema,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            device=device,
        )
        opt_metrics["mode"] = "opt"
        opt_metrics["epoch"] = epoch
        logger.log(opt_metrics)
        total_loss += loss.item()
        batches += 1
    total_loss /= batches
    return total_loss


def validate_epoch_ensemble(
    ensemble: dict,
    training_setups: dict,
    ensemble_set: dict,
    # ema: Optional[ExponentialMovingAverage],
    # loss_fn: torch.nn.Module,
    # valid_loader: DataLoader,
    # output_args: Dict[str, bool],
    # device: torch.device,
    logger: MetricsLogger,
    log_errors: str,
    epoch: int,
    data_loader_key: str = "valid_loader",
) -> tuple[dict, float, dict]:
    """
    Evaluates an ensemble of models on the validation set and returns
    average loss and metrics.

    Args:
        ensemble (dict): Ensemble of MACE models.
        ema (Optional[ExponentialMovingAverage]): Exponential moving average.
        loss_fn (torch.nn.Module): Loss function.
        valid_loader (DataLoader): Validation data loader.
        output_args (Dict[str, bool]): Output arguments.
        device (torch.device): Device to use. CPU or GPU.
        logger (MetricsLogger): Logger for metrics.
        log_errors (str): Logging errors.
        epoch (int): Current epoch.

    Returns:
        tuple[dict, float, dict]: Ensemble validation loss, average loss, and metrics.
    """
    ensemble_valid_loss, ensemble_eval_metrics = {}, []
    for tag, model in ensemble.items():

        ema = training_setups[tag]["ema"]
        loss_fn = training_setups[tag]["loss_fn"]
        device = training_setups[tag]["device"]
        output_args = training_setups[tag]["output_args"]

        # can be different depending on replay strategy
        valid_loader = ensemble_set[tag][data_loader_key]

        if ema is not None:
            with ema.average_parameters():
                valid_loss, eval_metrics = evaluate(
                    model=model,
                    loss_fn=loss_fn,
                    data_loader=valid_loader,
                    output_args=output_args,
                    device=device,
                )
        else:
            valid_loss, eval_metrics = evaluate(
                model=model,
                loss_fn=loss_fn,
                data_loader=valid_loader,
                output_args=output_args,
                device=device,
            )
        ensemble_valid_loss[tag] = valid_loss
        ensemble_eval_metrics.append(eval_metrics)

    valid_loss = np.mean(list(ensemble_valid_loss.values()))
    eval_metrics = {}
    for key in ensemble_eval_metrics[0]:
        if key not in ["mode", "epoch"]:
            eval_metrics[key] = np.mean(
                [m[key] for m in ensemble_eval_metrics]
            )
        eval_metrics["mode"] = "eval"
        eval_metrics["epoch"] = epoch

    if logger is not None:

        valid_err_log(
            valid_loss=valid_loss,
            eval_metrics=eval_metrics,
            logger=logger,
            log_errors=log_errors,
            epoch=epoch,
        )

    return ensemble_valid_loss, valid_loss, eval_metrics
