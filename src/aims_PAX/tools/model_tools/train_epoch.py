import torch
from contextlib import contextmanager
import numpy as np
from typing import Optional, Dict
from torch.optim import LBFGS
from mace.tools.train import valid_err_log
from mace.tools.utils import MetricsLogger
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from mace.tools import torch_geometric
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import logging
from mace.tools.utils import to_numpy

# TODO: copy this into aims-PAX src code
from so3krates_torch.tools.eval import ModelEval

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
            step_method = lr_scheduler.step
            has_metrics = (
                hasattr(step_method, "__code__")
                and "metrics" in step_method.__code__.co_varnames
            )
            if has_metrics:
                # ReduceLROnPlateau scheduler needs metrics
                lr_scheduler.step(metrics=valid_loss)
            else:
                # ExponentialLR and other schedulers don't need metrics
                lr_scheduler.step()
                
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


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    ema: Optional[ExponentialMovingAverage],
    logger: MetricsLogger,
    device: torch.device,
    rank: Optional[int] = 0,
) -> None:

    if isinstance(optimizer, LBFGS):
        _, opt_metrics = take_step_lbfgs(
            model=model,
            loss_fn=loss_fn,
            data_loader=data_loader,
            optimizer=optimizer,
            ema=ema,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            device=device,
            rank=rank,
        )
        opt_metrics["mode"] = "opt"
        opt_metrics["epoch"] = epoch
        if rank == 0:
            logger.log(opt_metrics)
    else:
        for batch in data_loader:
            _, opt_metrics = take_step(
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
            if rank == 0:
                logger.log(opt_metrics)


def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    batch = batch.to(device)
    batch_dict = batch.to_dict()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch_dict,
            training=True,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        loss = loss_fn(pred=output, ref=batch)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm
            )

        return loss

    loss = closure()
    optimizer.step()

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


def take_step_lbfgs(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    logging.debug(
        f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )

    total_sample_count = 0
    for batch in data_loader:
        total_sample_count += batch.num_graphs

    def closure():

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)

        # Process each batch and then collect the results we pass to the optimizer
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=True,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            batch_loss = loss_fn(pred=output, ref=batch)
            batch_loss = batch_loss * (batch.num_graphs / total_sample_count)

            batch_loss.backward()
            total_loss += batch_loss

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm
            )

        return total_loss



    loss = optimizer.step(closure)

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


def validate_epoch_multihead(
    model_dict: dict,
    training_setups: dict,
    valid_loaders: dict,
    logger: MetricsLogger,
    log_errors: str,
    epoch: int,
) -> tuple[dict, float, dict]:
    """
    Evaluates an multihead model on the validation set and returns
    average loss and metrics (over heads).

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
    tag = list(model_dict.keys())[0]
    model = model_dict[tag]
    
    mh_valid_loss, mh_eval_metrics = {}, []

    ema = training_setups[tag]["ema"]
    loss_fn = training_setups[tag]["loss_fn"]
    device = training_setups[tag]["device"]
    output_args = training_setups[tag]["output_args"]

    for valid_loader_name, valid_loader in valid_loaders.items():
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
        
        mh_valid_loss[valid_loader_name] = valid_loss
        mh_eval_metrics.append(eval_metrics)
        if logger is not None:
            valid_err_log(
                valid_loss=valid_loss,
                eval_metrics=eval_metrics,
                logger=logger,
                log_errors=log_errors,
                epoch=epoch,
                valid_loader_name=valid_loader_name,
            )

    valid_loss = np.mean(list(mh_valid_loss.values()))
    eval_metrics = {}
    for key in mh_eval_metrics[0]:
        if key not in ["mode", "epoch", "head"]:
            eval_metrics[key] = np.mean(
                [m[key] for m in mh_eval_metrics]
            )
        eval_metrics["mode"] = "eval"
        eval_metrics["epoch"] = epoch
    average_mh_mae_f = eval_metrics['mae_f'] * 1000
    logging.info(f"Average multi-head MAE_F: {average_mh_mae_f:.2f}  meV / Å")
    return {tag: valid_loss}, valid_loss, eval_metrics, mh_valid_loss


def validate_epoch_ensemble(
    ensemble: dict,
    training_setups: dict,
    valid_loaders: dict,
    logger: MetricsLogger,
    log_errors: str,
    epoch: int,
) -> tuple[dict, float, dict]:
    """
    Evaluates an ensemble of models on the validation set and returns
    average loss and metrics (over members).

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

        if ema is not None:
            with ema.average_parameters():
                valid_loss, eval_metrics = evaluate(
                    model=model,
                    loss_fn=loss_fn,
                    data_loader=valid_loaders["Default"],
                    output_args=output_args,
                    device=device,
                )
        else:
            valid_loss, eval_metrics = evaluate(
                model=model,
                loss_fn=loss_fn,
                data_loader=valid_loaders["Default"],
                output_args=output_args,
                device=device,
            )
        ensemble_valid_loss[tag] = valid_loss
        ensemble_eval_metrics.append(eval_metrics)

    valid_loss = np.mean(list(ensemble_valid_loss.values()))
    eval_metrics = {}
    for key in ensemble_eval_metrics[0]:
        if key not in ["mode", "epoch", "head"]:
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

    return ensemble_valid_loss, valid_loss, eval_metrics, None


def validate_epoch(
    ensemble: dict,
    training_setups: dict,
    valid_loaders: dict,
    logger: MetricsLogger,
    log_errors: str,
    epoch: int,
    multihead: bool = False,
) -> tuple[dict, float, dict]:
    if multihead:
        return validate_epoch_multihead(
            model_dict=ensemble,
            training_setups=training_setups,
            valid_loaders=valid_loaders,
            logger=logger,
            log_errors=log_errors,
            epoch=epoch,
        )
    else:
        return validate_epoch_ensemble(
            ensemble=ensemble,
            training_setups=training_setups,
            valid_loaders=valid_loaders,
            logger=logger,
            log_errors=log_errors,
            epoch=epoch,
        )


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    output_args: Dict[str, bool],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:

    metrics = ModelEval(loss_fn=loss_fn).to(device)

    with preserve_grad_state(model):
        start_time = time.time()
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=False,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            avg_loss, aux = metrics(batch, output)

    avg_loss, aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    return avg_loss, aux

@contextmanager
def preserve_grad_state(model):
    """
    Taken from
    https://github.com/ACEsuit/mace/pull/830/
    https://doi.org/10.1038/s41524-025-01727-x
    """
    # save the original requires_grad state for all parameters
    requires_grad_backup = {
        param: param.requires_grad for param in model.parameters()
    }
    try:
        # temporarily disable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = False
        yield  # perform evaluation here
    finally:
        # restore the original requires_grad states
        for param, requires_grad in requires_grad_backup.items():
            param.requires_grad = requires_grad