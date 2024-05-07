import torch
import numpy as np
from typing import Optional, Dict
from mace.tools.train import take_step, evaluate
from mace.tools.utils import MetricsLogger
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
import logging

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
):
    #if max_grad_norm is not None:
        #logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")
# Train
    if lr_scheduler is not None:
        if epoch > start_epoch:
            lr_scheduler.step(
                metrics=valid_loss
            )  # Can break if exponential LR, TODO fix that!
    total_loss = 0.0
    batches = 0
    for batch in train_loader:
        loss , opt_metrics = take_step(
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
    ensemble,
    ema,
    loss_fn,
    valid_loader,
    output_args,
    device,
    logger,
    log_errors,
    epoch
):
    ensemble_valid_loss, ensemble_eval_metrics = {}, []
    for tag, model in ensemble.items():
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
            eval_metrics[key] = np.mean([m[key] for m in ensemble_eval_metrics])
        eval_metrics["mode"] = "eval"
        eval_metrics["epoch"] = epoch

    if logger is not None:
        logger.log(eval_metrics)
        if log_errors == "PerAtomRMSE":
            error_e = eval_metrics["rmse_e_per_atom"] * 1e3
            error_f = eval_metrics["rmse_f"] * 1e3
            logging.info(
                f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
            )
        elif (
            log_errors == "PerAtomRMSEstressvirials"
            and eval_metrics["rmse_stress_per_atom"] is not None
        ):
            error_e = eval_metrics["rmse_e_per_atom"] * 1e3
            error_f = eval_metrics["rmse_f"] * 1e3
            error_stress = eval_metrics["rmse_stress_per_atom"] * 1e3
            logging.info(
                f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_stress_per_atom={error_stress:.1f} meV / A^3"
            )
        elif (
            log_errors == "PerAtomRMSEstressvirials"
            and eval_metrics["rmse_virials_per_atom"] is not None
        ):
            error_e = eval_metrics["rmse_e_per_atom"] * 1e3
            error_f = eval_metrics["rmse_f"] * 1e3
            error_virials = eval_metrics["rmse_virials_per_atom"] * 1e3
            logging.info(
                f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_virials_per_atom={error_virials:.1f} meV"
            )
        elif log_errors == "TotalRMSE":
            error_e = eval_metrics["rmse_e"] * 1e3
            error_f = eval_metrics["rmse_f"] * 1e3
            logging.info(
                f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
            )
        elif log_errors == "PerAtomMAE":
            error_e = eval_metrics["mae_e_per_atom"] * 1e3
            error_f = eval_metrics["mae_f"] * 1e3
            logging.info(
                f"Epoch {epoch}: loss={valid_loss:.4f}, MAE_E_per_atom={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A"
            )
        elif log_errors == "TotalMAE":
            error_e = eval_metrics["mae_e"] * 1e3
            error_f = eval_metrics["mae_f"] * 1e3
            logging.info(
                f"Epoch {epoch}: loss={valid_loss:.4f}, MAE_E={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A"
            )
        elif log_errors == "DipoleRMSE":
            error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
            logging.info(
                f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_MU_per_atom={error_mu:.2f} mDebye"
            )
        elif log_errors == "EnergyDipoleRMSE":
            error_e = eval_metrics["rmse_e_per_atom"] * 1e3
            error_f = eval_metrics["rmse_f"] * 1e3
            error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
            logging.info(
                f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_Mu_per_atom={error_mu:.2f} mDebye"
            )

    return ensemble_valid_loss, valid_loss, eval_metrics

#    if valid_loss >= lowest_loss:
#        patience_counter += 1
#        patience_counter >= patience:
#            logging.info(
#                f"Stopping optimization after {patience_counter} epochs without improvement"
#            )
#            break
#    else:
#        lowest_loss = valid_loss
#        patience_counter = 0
#        if ema is not None:
#            with ema.average_parameters():
#                checkpoint_handler.save(
#                    state=CheckpointState(model, optimizer, lr_scheduler),
#                    epochs=epoch,
#                    keep_last=keep_last,
#                )
#                keep_last = False
#        else:
#            checkpoint_handler.save(
#                state=CheckpointState(model, optimizer, lr_scheduler),
#                epochs=epoch,
#                keep_last=keep_last,
#            )
#            keep_last = False