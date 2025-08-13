import os
import logging
import ase
from ase.io import read
from typing import Callable
import time
from pathlib import Path
import numpy as np
import threading
import queue
import shutil
from mace import tools
from .preparation import (
    ALCalculatorMLFFManager,
    ALEnsembleManager,
    ALStateManager,
    ALRestartManager,
    ALConfigurationManager,
    ALMDManager,
)
from aims_PAX.tools.utilities.data_handling import (
    create_dataloader,
    save_datasets,
    create_mace_dataset,
)
from aims_PAX.tools.utilities.utilities import (
    update_model_auxiliaries,
    save_checkpoint,
    save_models,
    select_best_member,
    atoms_full_copy,
    AIMSControlParser,
)
from aims_PAX.tools.utilities.mpi_utils import (
    send_points_non_blocking,
    CommHandler,
)
from aims_PAX.tools.setup_MACE_training import (
    setup_mace_training,
    reset_optimizer,
)
from aims_PAX.tools.train_epoch_mace import (
    train_epoch,
    validate_epoch_ensemble,
)
from aims_PAX.tools.utilities.eval_utils import (
    ensemble_prediction,
)
from aims_PAX.tools.utilities.parsl_utils import (
    prepare_parsl,
    recalc_dft_parsl,
    handle_parsl_logger,
)
from aims_PAX.tools.uncertainty import (
    get_threshold,
)

try:
    import parsl
except ImportError:
    parsl = None
try:
    import asi4py
except Exception as e:
    asi4py = None


class ALDataManager:
    """
    Tasked with handling all data relevant tasks i.e.
    putting points into training or validation sets
    and transforming them into MACE format.
    """

    def __init__(
        self,
        config: ALConfigurationManager,
        ensemble_manager: ALEnsembleManager,
        state_manager: ALStateManager,
        comm_handler: CommHandler,
        rank: int,
    ):

        self.config = config
        self.ensemble_manager = ensemble_manager
        self.state_manager = state_manager
        self.comm_handler = comm_handler
        self.rank = rank

    def handle_received_point(
        self, idx: int, received_point: np.ndarray
    ) -> bool:
        """
        Process a received point by adding it to either validation or training
        set.

        Args:
            idx: Index of the trajectory worker
            received_point: The data point received from DFT calculation

        Returns:
            True if max dataset size is reached, False otherwise
        """
        mace_point = create_mace_dataset(
            data=[received_point],
            z_table=self.ensemble_manager.z_table,
            seed=None,
            r_max=self.config.r_max,
        )

        # Determines if a point should go to validation or training set
        validation_quota = (
            self.config.valid_ratio * self.state_manager.total_points_added
        )
        needs_validation_data = (
            self.state_manager.valid_points_added < validation_quota
        )

        if needs_validation_data:
            self._add_to_validation_set(idx, received_point, mace_point)
        else:
            max_size_reached = self._add_to_training_set(
                idx, received_point, mace_point
            )
            if max_size_reached:
                return True

        self.state_manager.total_points_added += 1
        return False

    def _add_to_validation_set(
        self, idx: int, received_point: np.ndarray, mace_point
    ) -> None:
        """
        Simply adds a point to the validation set and
        changes the trajectory worker state to "running".

        Args:
            idx (int): Index of the trajectory worker
            received_point (np.ndarray): The data point received from DFT
                                            calculation
            mace_point: The MACE formatted data point
        """

        self.state_manager.trajectory_status[idx] = "running"

        if self.rank == 0:
            logging.info(
                f"Trajectory worker {idx} is adding a point to "
                "the validation set."
            )

            # Add to all ensemble member datasets
            for tag in self.ensemble_manager.ensemble_ase_sets.keys():
                self.ensemble_manager.ensemble_ase_sets[tag]["valid"].append(
                    received_point
                )
                self.ensemble_manager.ensemble_mace_sets[tag][
                    "valid"
                ] += mace_point

        if self.rank == 0:
            self._log_dataset_sizes(tag)
        self.state_manager.valid_points_added += 1

    def _add_to_training_set(
        self, idx: int, received_point: np.ndarray, mace_point
    ) -> bool:
        """
        Add point to training set and check if max size is reached.
        Changes the trajectory worker state to "training".

        Returns:
            True if max dataset size is reached, False otherwise
        """
        self.state_manager.trajectory_status[idx] = "training"
        self.state_manager.num_workers_training += 1

        if self.rank == 0:
            logging.info(
                f"Trajectory worker {idx} is adding a point to the "
                "training set."
            )

            # Add to all ensemble member datasets
            for tag in self.ensemble_manager.ensemble_ase_sets.keys():
                self.ensemble_manager.ensemble_ase_sets[tag]["train"].append(
                    received_point
                )
                self.ensemble_manager.ensemble_mace_sets[tag][
                    "train"
                ] += mace_point

            self.ensemble_manager.train_dataset_len = len(
                self.ensemble_manager.ensemble_ase_sets[tag]["train"]
            )

        # Synchronize dataset length across all processes
        self.comm_handler.barrier()
        self.ensemble_manager.train_dataset_len = self.comm_handler.bcast(
            self.ensemble_manager.train_dataset_len, root=0
        )
        self.comm_handler.barrier()

        # Check if maximum dataset size is reached
        if (
            self.ensemble_manager.train_dataset_len
            > self.config.max_train_set_size
        ):
            return True

        # Log dataset sizes
        if self.rank == 0:
            self._log_dataset_sizes(tag)

        self.state_manager.train_points_added += 1
        return False

    def _log_dataset_sizes(self, tag: str) -> None:
        """Log current training and validation set sizes."""
        valid_length = len(
            self.ensemble_manager.ensemble_ase_sets[tag]["valid"]
        )
        logging.info(
            f"Size of the training and validation set: "
            f"{self.ensemble_manager.train_dataset_len}, {valid_length}."
        )


class TrainingSession:
    """
    Encapsulates a single training session configuration and state.
    Used for both intermediate training and convergence training.
    """

    def __init__(
        self,
        training_setups: dict,
        ensemble_mace_sets: dict,
        max_epochs: int,
        is_convergence: bool = False,
        initial_epoch: int = 0,
    ):
        self.training_setups = training_setups
        self.ensemble_mace_sets = ensemble_mace_sets
        self.max_epochs = max_epochs
        self.is_convergence = is_convergence
        self.current_epoch = initial_epoch
        self.best_valid_loss = np.inf
        self.best_epoch = 0
        self.no_improvement = 0
        self.ensemble_valid_losses = {}


class TrainingOrchestrator:
    """
    Handles the common training logic for both intermediate (during AL)
    and convergence training.
    """

    def __init__(
        self,
        config: ALConfigurationManager,
        ensemble_manager: ALEnsembleManager,
        state_manager: ALStateManager,
        restart_manager: ALRestartManager,
        md_manager: ALMDManager,
    ):
        self.config = config
        self.ensemble_manager = ensemble_manager
        self.state_manager = state_manager
        self.restart_manager = restart_manager
        self.md_manager = md_manager

    def train_single_epoch(
        self, session: TrainingSession, tag: str, model, logger=None
    ):
        """
        Train a single epoch for a specific model in the ensemble.

        Args:
            session (TrainingSession): Training session containing setup and state.
            tag (str): Tag identifying the specific model in the ensemble.
            model: The model to be trained.
            logger: Logger for tracking training progress. Defaults to None.

        Returns:
            logger: The logger used for this training epoch.
        """
        training_setup = session.training_setups[tag]

        if (
            not session.is_convergence
            and self.state_manager.ensemble_reset_opt.get(tag, False)
        ):
            logging.info(f"Resetting optimizer for model {tag}.")
            session.training_setups[tag] = reset_optimizer(
                model, training_setup, self.config.mace_settings["TRAINING"]
            )
            self.state_manager.ensemble_reset_opt[tag] = False
            training_setup = session.training_setups[tag]

        if logger is None:
            logger = tools.MetricsLogger(
                directory=self.config.mace_settings["GENERAL"]["loss_dir"],
                tag=tag + "_train",
            )

        lr_scheduler = None
        if session.is_convergence:
            lr_scheduler = training_setup["lr_scheduler"]
        elif hasattr(self, "use_scheduler") and self.use_scheduler:
            lr_scheduler = training_setup["lr_scheduler"]

        train_epoch(
            model=model,
            train_loader=session.ensemble_mace_sets[tag]["train_loader"],
            loss_fn=training_setup["loss_fn"],
            optimizer=training_setup["optimizer"],
            lr_scheduler=lr_scheduler,
            epoch=session.current_epoch,
            start_epoch=(
                session.current_epoch if session.is_convergence else None
            ),
            valid_loss=(
                session.ensemble_valid_losses.get(tag)
                if session.is_convergence
                else None
            ),
            logger=logger,
            device=training_setup["device"],
            max_grad_norm=training_setup["max_grad_norm"],
            output_args=training_setup["output_args"],
            ema=training_setup["ema"],
        )

        return logger

    def validate_and_update_state(
        self,
        session: TrainingSession,
        logger: tools.MetricsLogger,
        trajectory_idx: int = None,
    ) -> bool:
        """
        Validate the ensemble and updates the current validation
        loss in the session state.
        Also uses validation results to determine if training should stop
        in the case of the convergence training.

        Args:
            session (TrainingSession): The training session to update.
            logger (tools.MetricsLogger): Logger for tracking validation.
            trajectory_idx (int, optional): Index of the trajectory.
                                            Defaults to None.

        Returns:
            bool: Whether to stop training based on validation results.
        """
        should_validate = self._should_validate(session, trajectory_idx)

        if not should_validate:
            return False

        # Perform validation
        ensemble_valid_loss, valid_loss, metrics = validate_epoch_ensemble(
            ensemble=self.ensemble_manager.ensemble,
            training_setups=session.training_setups,
            ensemble_set=session.ensemble_mace_sets,
            logger=logger,
            log_errors=self.config.mace_settings["MISC"]["error_table"],
            epoch=self._get_validation_epoch(session, trajectory_idx),
        )

        # Update session state
        session.ensemble_valid_losses = ensemble_valid_loss

        if session.is_convergence:
            return self._handle_convergence_validation(
                session,
                valid_loss,
                ensemble=self.ensemble_manager.ensemble,
            )
        else:
            # always returns False here because intermediate
            # training only stops based on number of epochs
            return self._handle_intermediate_validation(
                session,
                valid_loss,
                ensemble_valid_loss,
                metrics,
                trajectory_idx,
            )

    def _should_validate(
        self, session: TrainingSession, trajectory_idx: int = None
    ) -> bool:
        """
        Uses the number of epochs to determine if validation should be
        performed.

        Args:
            session (TrainingSession): The training session to check.
            trajectory_idx (int, optional): The index of the trajectory.
                                Defaults to None.

        Returns:
            bool: Whether validation should be performed.
        """

        if session.is_convergence:
            return (
                session.current_epoch % self.config.valid_skip == 0
                or session.current_epoch == session.max_epochs - 1
            )
        else:
            current_intermediate = (
                self.state_manager.trajectory_intermediate_epochs[
                    trajectory_idx
                ]
            )
            return (
                current_intermediate % self.config.valid_skip == 0
                or current_intermediate
                == self.config.intermediate_epochs_al - 1
            )

    def _get_validation_epoch(
        self, session: TrainingSession, trajectory_idx: int = None
    ) -> int:
        """Get the epoch number for validation logging."""
        if session.is_convergence:
            return session.current_epoch
        else:
            return self.state_manager.trajectory_total_epochs[trajectory_idx]

    def _handle_convergence_validation(
        self,
        session: TrainingSession,
        valid_loss: float,
        ensemble: dict,
    ) -> bool:
        """
        Determines if model training has converged i.e.
        the validation loss has not improved over the
        course of `self.config.convergence_patience` epochs.

        Args:
            session (TrainingSession): The currrent training session to check.
            valid_loss (float): The current validation loss.
            ensemble (dict): The ensemble of models being trained.

        Returns:
            bool: Whether the model has converged.
        """
        improvement = (
            session.best_valid_loss > valid_loss
            and (session.best_valid_loss - valid_loss) > self.config.margin
        )

        if improvement:
            session.best_valid_loss = valid_loss
            session.best_epoch = session.current_epoch
            session.no_improvement = 0
        else:
            session.no_improvement += 1
        for tag, model in ensemble.items():
            save_checkpoint(
                session.training_setups[tag]["checkpoint_handler"],
                session.training_setups[tag],
                model,
                session.current_epoch,
                keep_last=False,
            )

        return session.no_improvement > self.config.convergence_patience

    def _handle_intermediate_validation(
        self,
        session: TrainingSession,
        valid_loss: float,
        ensemble_valid_loss: dict,
        metrics: dict,
        trajectory_idx: int,
    ) -> bool:
        """
        Uses the validation results to update which ensemble member is the best
        and collects analysis data if analysis is enabled.
        Also calls functions to check if ensemble members are improving, to
        save checkpoints and datasets, and to handle restarts.

        Args:
            session (TrainingSession): Current training session.
            valid_loss (float): Current validation loss.
            ensemble_valid_loss (dict): Current validation loss per ensemble
                                        member.
            metrics (dict): Current metrics of the models.
            trajectory_idx (int): Index of the trajectory worker.

        Returns:
            bool: False, as intermediate training does not stop based on
                    validation performance but on number of epochs.
        """
        # Update best member
        best_member = select_best_member(ensemble_valid_loss)
        if hasattr(self, "best_member"):
            self.best_member = best_member

        # Collect analysis data
        if self.config.analysis:
            self._collect_analysis_data(valid_loss, ensemble_valid_loss)

        # Update current validation error
        self.state_manager.current_valid_error = metrics["mae_f"]

        # Check for improvement
        self._process_member_improvement(ensemble_valid_loss)

        # Save datasets and handle restarts
        self._save_training_artifacts()
        for tag in ensemble_valid_loss.keys():
            # Save checkpoint
            model = self.ensemble_manager.ensemble[tag]
            save_checkpoint(
                session.training_setups[tag]["checkpoint_handler"],
                session.training_setups[tag],
                model,
                self.state_manager.trajectory_intermediate_epochs[
                    trajectory_idx
                ],
                keep_last=False,
            )

        return False

    def _collect_analysis_data(self, valid_loss, ensemble_valid_loss):
        """Collect data for analysis."""
        self.state_manager.collect_losses["epoch"].append(
            self.state_manager.total_epoch
        )
        self.state_manager.collect_losses["avg_losses"].append(valid_loss)
        self.state_manager.collect_losses["ensemble_losses"].append(
            ensemble_valid_loss
        )

    def _process_member_improvement(
        self,
        ensemble_valid_loss: dict,
    ):
        """
        Check if the models are improving during training
        based on validation loss and if they don't improve
        for a certain number of epochs, the optimizer is reset.

        Args:
            session (TrainingSession): Current training session.
            ensemble_valid_loss (dict): Current validation loss per ensemble
            trajectory_idx (int): Index of the trajectory worker.
        """
        for tag in ensemble_valid_loss.keys():
            current_loss = ensemble_valid_loss[tag]
            best_loss = self.state_manager.ensemble_best_valid[tag]

            if current_loss < best_loss:
                self.state_manager.ensemble_best_valid[tag] = current_loss
            else:
                self.state_manager.ensemble_no_improvement[tag] += 1

            # Check if optimizer reset is needed
            if (
                self.state_manager.ensemble_no_improvement[tag]
                > self.config.max_epochs_worker
            ):
                logging.info(
                    f"No improvements for {self.config.max_epochs_worker} epochs "
                    f"at ensemble member {tag}. Scheduling optimizer reset."
                )
                self.state_manager.ensemble_reset_opt[tag] = True
                self.state_manager.ensemble_no_improvement[tag] = 0

    def _save_training_artifacts(self):
        """Save datasets and handle restart checkpoints."""
        save_datasets(
            self.ensemble_manager.ensemble,
            self.ensemble_manager.ensemble_ase_sets,
            path=self.config.dataset_dir / "final",
        )

        if self.config.create_restart and hasattr(self, "save_restart"):
            self.save_restart = True


class ALTrainingManager:
    """
    Tasked with handling the training logic during active learning
    and convergence.
    """

    def __init__(
        self,
        config: ALConfigurationManager,
        ensemble_manager: ALEnsembleManager,
        mlff_manager: ALCalculatorMLFFManager,
        state_manager: ALStateManager,
        md_manager: ALMDManager,
        restart_manager: ALRestartManager,
        rank: int,
    ):
        self.config = config
        self.ensemble_manager = ensemble_manager
        self.mlff_manager = mlff_manager
        self.state_manager = state_manager
        self.restart_manager = restart_manager
        self.md_manager = md_manager
        self.rank = rank

        self.best_member = None
        self.use_scheduler = False
        self.save_restart = False

        self.orchestrator = TrainingOrchestrator(
            config,
            ensemble_manager,
            state_manager,
            restart_manager,
            md_manager,
        )
        self.orchestrator.best_member = self.best_member
        self.orchestrator.save_restart = self.save_restart
        self.orchestrator.use_scheduler = self.use_scheduler

    def perform_training(self, idx: int = 0):
        """Perform intermediate training during active learning."""
        session = TrainingSession(
            training_setups=self.ensemble_manager.training_setups,
            ensemble_mace_sets=self.ensemble_manager.ensemble_mace_sets,
            max_epochs=self.config.intermediate_epochs_al,
            is_convergence=False,
        )

        while (
            self.state_manager.trajectory_intermediate_epochs[idx]
            < self.config.intermediate_epochs_al
        ):

            logger = None
            for tag, model in self.ensemble_manager.ensemble.items():
                logger = self.orchestrator.train_single_epoch(
                    session, tag, model, logger
                )

            self.orchestrator.validate_and_update_state(session, logger, idx)

            self._update_epoch_counters(idx)

            self._handle_restart_checkpoint()

        # Finalize training
        self._finalize_training(idx)

        # Sync state back
        self.best_member = self.orchestrator.best_member

    def converge(self):
        """Converge the ensemble on the acquired dataset."""
        if self.rank != 0:
            return

        self._setup_convergence()

        # Create training session for convergence
        session = TrainingSession(
            training_setups=self.training_setups_convergence,
            ensemble_mace_sets=self.ensemble_manager.ensemble_mace_sets,
            max_epochs=self.config.max_convergence_epochs,
            is_convergence=True,
            initial_epoch=self._get_initial_epoch(),
        )

        # Initialize ensemble valid losses
        session.ensemble_valid_losses = {
            tag: np.inf for tag in self.ensemble_manager.ensemble.keys()
        }

        # Training loop
        for j in range(self.config.max_convergence_epochs):
            session.current_epoch = j

            logger = None
            # Train all ensemble members
            for tag, model in self.ensemble_manager.ensemble.items():
                logger = self.orchestrator.train_single_epoch(
                    session, tag, model, logger
                )

            # Validate and check for convergence
            should_stop = self.orchestrator.validate_and_update_state(
                session, logger
            )

            if should_stop:
                logging.info(
                    f"No improvements for {self.config.convergence_patience} epochs. "
                    "Training converged. Best model(s) "
                    f"(Epoch {session.best_epoch}) saved."
                )
                self._final_save(session)
                break

            if j == self.config.max_convergence_epochs - 1:
                logging.info(
                    f"Maximum number of epochs reached. "
                    f"Best model (Epoch {session.best_epoch}) saved."
                )
                self._final_save(session)

    def _setup_convergence(self):
        """Setup convergence training configuration."""
        if self.config.converge_best:
            logging.info(
                f"Converging best model ({self.best_member}) on "
                f"acquired dataset."
            )
            self.ensemble_manager.ensemble = {
                self.best_member: self.ensemble_manager.ensemble[
                    self.best_member
                ]
            }
        else:
            logging.info("Converging ensemble on acquired dataset.")

        temp_mace_sets = self._create_convergence_datasets()
        self.ensemble_manager.ensemble_mace_sets = self.prepare_training(
            temp_mace_sets
        )

        # Reset training configurations
        self.training_setups_convergence = {}
        for tag in self.ensemble_manager.ensemble.keys():
            self.training_setups_convergence[tag] = setup_mace_training(
                settings=self.config.mace_settings,
                model=self.ensemble_manager.ensemble[tag],
                tag=tag,
                restart=self.config.restart,
                convergence=True,
                checkpoints_dir=self.config.checkpoints_dir,
                mol_idxs=self.config.mol_idxs,
            )

    def _create_convergence_datasets(self):
        """Create datasets for convergence training."""
        temp_mace_sets = {}
        for tag, _ in self.ensemble_manager.ensemble.items():
            train_set = create_mace_dataset(
                data=self.ensemble_manager.ensemble_ase_sets[tag]["train"],
                z_table=self.ensemble_manager.z_table,
                seed=self.config.seeds_tags_dict[tag],
                r_max=self.config.r_max,
            )
            valid_set = create_mace_dataset(
                data=self.ensemble_manager.ensemble_ase_sets[tag]["valid"],
                z_table=self.ensemble_manager.z_table,
                seed=self.config.seeds_tags_dict[tag],
                r_max=self.config.r_max,
            )
            temp_mace_sets[tag] = {"train": train_set, "valid": valid_set}
        return temp_mace_sets

    def _get_initial_epoch(self):
        """Get initial epoch for convergence training."""
        if self.config.restart:
            return self.training_setups_convergence[
                list(self.ensemble_manager.ensemble.keys())[0]
            ]["epoch"]
        return 0

    def _update_epoch_counters(self, idx: int):
        self.state_manager.trajectory_total_epochs[idx] += 1
        self.state_manager.trajectory_intermediate_epochs[idx] += 1
        self.state_manager.total_epoch += 1

    def _handle_restart_checkpoint(self):
        if self.orchestrator.save_restart and self.config.create_restart:
            self.restart_manager.update_restart_dict(
                trajectories_keys=self.state_manager.trajectories.keys(),
                md_drivers=self.md_manager.md_drivers,
                save_restart="restart/al/al_restart.npy",
            )
            self.orchestrator.save_restart = False

    def _finalize_training(self, idx: int):
        self.models = [
            self.ensemble_manager.ensemble[tag]
            for tag in self.ensemble_manager.ensemble.keys()
        ]
        self.state_manager.trajectory_intermediate_epochs[idx] = 0

    def _check_batch_size(self, set_batch_size, tag):
        batch_size = (
            1
            if len(self.ensemble_manager.ensemble_mace_sets[tag]["train"])
            < set_batch_size
            else set_batch_size
        )
        return batch_size

    def prepare_training(self, mace_sets: dict):
        for _, (tag, model) in enumerate(
            self.ensemble_manager.ensemble.items()
        ):
            train_batch_size = self._check_batch_size(
                self.config.set_batch_size, tag
            )
            valid_batch_size = self._check_batch_size(
                self.config.set_valid_batch_size, tag
            )

            (
                mace_sets[tag]["train_loader"],
                mace_sets[tag]["valid_loader"],
            ) = create_dataloader(
                mace_sets[tag]["train"],
                mace_sets[tag]["valid"],
                train_batch_size,
                valid_batch_size,
            )

            update_model_auxiliaries(
                model,
                mace_sets[tag],
                self.config.scaling,
                self.mlff_manager.ensemble_atomic_energies[tag],
                self.mlff_manager.update_atomic_energies,
                self.mlff_manager.ensemble_atomic_energies_dict[tag],
                self.ensemble_manager.z_table,
                self.config.dtype,
                self.config.device,
            )
        return mace_sets

    def _final_save(self, session: TrainingSession):
        save_models(
            ensemble=self.ensemble_manager.ensemble,
            training_setups=session.training_setups,
            model_dir=self.config.mace_settings["GENERAL"]["model_dir"],
            current_epoch=session.current_epoch,
        )
        # save model(s) and datasets in final results directory
        os.makedirs("results", exist_ok=True)
        save_datasets(
            ensemble=self.ensemble_manager.ensemble,
            ensemble_ase_sets=self.ensemble_manager.ensemble_ase_sets,
            path=Path("results"),
        )
        save_models(
            ensemble=self.ensemble_manager.ensemble,
            training_setups=self.ensemble_manager.training_setups,
            model_dir=Path("results"),
            current_epoch=self.state_manager.total_epoch,
        )


class ALDFTManager:
    """
    Tasked with handling DFT calculations and their
    preparation.
    """

    def __init__(
        self,
        path_to_control: str,
        config: ALConfigurationManager,
        ensemble_manager: ALEnsembleManager,
        state_manager: ALStateManager,
        comm_handler: CommHandler,
    ):
        self.config = config
        self.ensemble_manager = ensemble_manager
        self.state_manager = state_manager
        self.comm_handler = comm_handler
        self.rank = self.comm_handler.rank

        self.control_parser = AIMSControlParser()
        self._handle_aims_settings(path_to_control)

        self.aims_calculator = None

    def handle_dft_call(self, point: ase.Atoms, idx: int):
        """
        Calls DFT calculation and checks if the SCF
        has converged. If it has not converged, the point
        is discarded and the MD checkpoint is loaded. The state
        of the trajectory worker is set to "running".
        If it has, the data is collected and the state of the
        trajectory worker is set to "waiting". The MD checkpoint
        is updated with the acquired DFT data.

        Args:
            point (ase.Atoms): Point to be recalculated.
            idx (int): Index of trajectory worker.
        """
        if self.rank == 0:
            logging.info(f"Trajectory worker {idx} is running DFT.")

        self.comm_handler.barrier()
        self.point = self.recalc_dft(point)

        if not self.aims_calculator.asi.is_scf_converged:
            if self.rank == 0:
                logging.info(
                    f"SCF not converged at worker {idx}. Discarding point and "
                    "restarting MD from last checkpoint."
                )
                self.state_manager.trajectories[idx] = atoms_full_copy(
                    self.state_manager.MD_checkpoints[idx]
                )
            self.state_manager.trajectory_status[idx] = "running"

        else:
            # we are updating the MD checkpoint here because then
            # we make sure  that the MD is restarted from a point
            # that is inside the training set  so the MLFF should
            # be able to handle this and lead to a better trajectory
            # that does not lead to convergence issues
            if self.rank == 0:
                received_point = self.state_manager.trajectories[idx].copy()
                received_point.info["REF_energy"] = self.point.info[
                    "REF_energy"
                ]
                received_point.arrays["REF_forces"] = self.point.arrays[
                    "REF_forces"
                ]
                if self.config.compute_stress:
                    received_point.info["REF_stress"] = self.point.info[
                        "REF_stress"
                    ]

                self.state_manager.MD_checkpoints[idx] = atoms_full_copy(
                    received_point
                )
                self.state_manager.MD_checkpoints[idx].calc = (
                    self.state_manager.trajectories[idx].calc
                )
            self.state_manager.trajectory_status[idx] = "waiting"
            self.state_manager.num_workers_training += 1

            self.comm_handler.barrier()
            self.state_manager.trajectory_status[idx] = "waiting"
            if self.rank == 0:
                logging.info(
                    f"Trajectory worker {idx} is going to add point "
                    "to the dataset."
                )

    def finalize_dft(self):
        self.aims_calculator.asi.close()

    def recalc_dft(self, current_point: ase.Atoms) -> ase.Atoms:
        """
        Uses the DFT calculator to compute desired
        properties of the current point. Returns
        None if the SCF has not converged.

        Args:
            current_point (ase.Atoms): Point to be recalculated.

        Returns:
            ase.Atoms: Updated atoms object with DFT data.
        """

        self.aims_calculator.calculate(
            current_point, properties=self.config.properties
        )

        if self.aims_calculator.asi.is_scf_converged:
            current_point.info["REF_energy"] = self.aims_calculator.results[
                "energy"
            ]
            current_point.arrays["REF_forces"] = self.aims_calculator.results[
                "forces"
            ]

            if self.config.compute_stress:
                current_point.info["REF_stress"] = (
                    self.aims_calculator.results["stress"]
                )

            return current_point
        else:
            if self.rank == 0:
                logging.info("SCF not converged.")
            return None

    def _handle_aims_settings(self, path_to_control: str):
        """Load and parse AIMS control file."""
        self.aims_settings = self.control_parser(path_to_control)
        self.aims_settings["compute_forces"] = True
        self.aims_settings["species_dir"] = self.config.species_dir
        self.aims_settings["postprocess_anyway"] = True


class ALDFTManagerSerial(ALDFTManager):
    """
    A class to handle DFT calculations in a serial manner.
    This class is used when the DFT calculations are not parallelized.
    """

    def __init__(
        self,
        path_to_control: str,
        config: ALConfigurationManager,
        ensemble_manager: ALEnsembleManager,
        state_manager: ALStateManager,
        comm_handler: CommHandler,
        path_to_geometry: str = "geometry.in",
    ):
        super().__init__(
            path_to_control=path_to_control,
            config=config,
            ensemble_manager=ensemble_manager,
            state_manager=state_manager,
            comm_handler=comm_handler,
        )
        self.aims_calculator = self._setup_aims_calculator(
            atoms=read(path_to_geometry)
        )

    def _setup_aims_calculator(self, atoms: ase.Atoms):
        aims_settings = self.aims_settings.copy()

        def init_via_ase(asi):
            from ase.calculators.aims import Aims, AimsProfile

            aims_settings["profile"] = AimsProfile(
                command="asi-doesnt-need-command"
            )
            calc = Aims(**aims_settings)
            calc.write_inputfiles(asi.atoms, properties=self.config.properties)

        if asi4py is None:
            raise ImportError(
                "asi4py is not properly installed. "
                "Please install it to use the AIMS calculator."
            )

        calc = asi4py.asecalc.ASI_ASE_calculator(
            self.config.ASI_path, init_via_ase, self.comm_handler.comm, atoms
        )
        return calc


class ALDFTManagerParallel(ALDFTManager):

    def __init__(
        self,
        path_to_control: str,
        config: ALConfigurationManager,
        ensemble_manager: ALEnsembleManager,
        state_manager: ALStateManager,
        comm_handler: CommHandler,
        color: int,
        world_comm,
        path_to_geometry: str = "geometry.in",
    ):
        super().__init__(
            path_to_control=path_to_control,
            config=config,
            ensemble_manager=ensemble_manager,
            state_manager=state_manager,
            comm_handler=comm_handler,
        )

        self.color = color
        # Initialize AIMS calculator
        self.aims_calculator = self._setup_aims_calculator(
            atoms=read(path_to_geometry)
        )
        self.world_comm = world_comm

    def _setup_aims_calculator(
        self,
        atoms: ase.Atoms,
    ) -> ase.Atoms:
        """
        Attaches the AIMS calculator to the atoms object.

        Args:
            atoms (ase.Atoms): Atoms object to attach the calculator to.
            pbc (bool, optional): If periodic boundry conditions are required
                                    or not. Defaults to False.

        Returns:
            ase.Atoms: Atoms object with the calculator attached.
        """
        aims_settings = self.aims_settings.copy()
        # only one communictor initializes aims
        if self.color == 1:
            self.properties = ["energy", "forces"]
            if self.config.compute_stress:
                self.properties.append("stress")

            def init_via_ase(asi):
                from ase.calculators.aims import Aims, AimsProfile

                aims_settings["profile"] = AimsProfile(
                    command="asi-doesnt-need-command"
                )
                calc = Aims(**aims_settings)
                calc.write_inputfiles(asi.atoms, properties=self.properties)

            if asi4py is None:
                raise ImportError(
                    "asi4py is not properly installed. "
                    "Please install it to use the AIMS calculator."
                )
            calc = asi4py.asecalc.ASI_ASE_calculator(
                self.config.ASI_path,
                init_via_ase,
                self.comm_handler.comm,
                atoms,
            )
            return calc
        else:
            return None

    def handle_dft_call(self, point, idx):
        self.comm_handler.barrier()
        if self.rank == 0:
            logging.info(f"Trajectory worker {idx} is sending point to DFT.")
            send_points_non_blocking(
                idx=idx,
                point_data=point,
                tag=1234,
                world_comm=self.world_comm,
            )
        self.comm_handler.barrier()
        self.state_manager.trajectory_status[idx] = "waiting"
        self.state_manager.num_workers_training += 1
        self.comm_handler.barrier()

        if self.rank == 0:
            logging.info(
                f"Trajectory worker {idx} is waiting for job to finish."
            )


class ALDFTManagerPARSL(ALDFTManager):
    def __init__(
        self,
        path_to_control: str,
        config: ALConfigurationManager,
        ensemble_manager: ALEnsembleManager,
        state_manager: ALStateManager,
        comm_handler: CommHandler,
    ):
        super().__init__(
            path_to_control=path_to_control,
            config=config,
            ensemble_manager=ensemble_manager,
            state_manager=state_manager,
            comm_handler=comm_handler,
        )
        parsl_setup_dict = prepare_parsl(
            cluster_settings=self.config.cluster_settings
        )
        self.parsl_config = parsl_setup_dict["config"]
        self.calc_dir = parsl_setup_dict["calc_dir"]
        self.clean_dirs = parsl_setup_dict["clean_dirs"]
        self.launch_str = parsl_setup_dict["launch_str"]
        try:
            parsl.dfk()
            logging.info(
                "PARSL is already initialized. Using existing PARSL context."
            )
        except parsl.errors.NoDataFlowKernelError:
            parsl_log_dir = Path(self.config.log_dir)
            handle_parsl_logger(log_dir=parsl_log_dir / "parsl_al.log")
            parsl.load(self.parsl_config)

        logging.info("Launching DFT manager thread for PARSL.")
        self.ab_initio_queue = queue.Queue()
        self.ab_intio_results = {}
        self.ab_initio_counter = {
            idx: 0 for idx in range(self.config.num_trajectories)
        }
        self.results_lock = threading.Lock()
        self.kill_thread = False
        threading.Thread(target=self._dft_thread, daemon=True).start()

    def handle_dft_call(self, point, idx: int):
        logging.info(f"Trajectory worker {idx} is sending point to DFT.")
        self.state_manager.trajectory_status[idx] = "waiting"
        self.state_manager.num_workers_training += 1
        self.ab_initio_queue.put((idx, point))
        logging.info(f"Trajectory worker {idx} is waiting for job to finish.")

    def _dft_thread(self):
        """
        Thread that constantly checks the queue for new jobs
        and submits them to PARSL for DFT calculations.
        It collects the results and stores them in a dictionary
        that is accessible from the main thread.
        It also cleans up the directories after the jobs are done
        if self.clean_dirs is True.
        """
        # collect parsl futures
        futures = {}
        for idx in range(self.config.num_trajectories):
            futures[idx] = {}
        # constantly check the queue for new jobs
        while True:
            if self.kill_thread:
                logging.info("Ab initio manager thread is stopping.")
                break
            try:
                idx, data = self.ab_initio_queue.get(timeout=1)
                with self.results_lock:
                    curr_job_no = self.ab_initio_counter[idx]
                futures[idx][curr_job_no] = recalc_dft_parsl(
                    positions=data.get_positions(),
                    species=data.get_chemical_symbols(),
                    cell=data.get_cell(),
                    pbc=data.pbc,
                    aims_settings=self.aims_settings,
                    directory=self.calc_dir / f"worker_{idx}_no_{curr_job_no}",
                    properties=self.config.properties,
                    ase_aims_command=self.launch_str,
                )
                with self.results_lock:
                    self.ab_initio_counter[idx] += 1
                self.ab_initio_queue.task_done()

            # is raised when the queue is empty after the timeout
            except queue.Empty:
                pass

            # goes through the futures and checks if they are done
            done_jobs = []
            for job_idx in futures.keys():
                for job_no, future in futures[job_idx].items():
                    if future.done():
                        done_jobs.append((job_idx, job_no))

            # if there are done jobs, get the results and store them in dict
            for job_idx, job_no in done_jobs:
                with self.results_lock:
                    temp_result = futures[job_idx][job_no].result()
                    if temp_result is None:
                        # if the result is None, it means the DFT calculation
                        # did not converge
                        self.ab_intio_results[job_idx] = False
                    else:
                        # the DFT calculation converged
                        self.ab_intio_results[job_idx] = temp_result
                        logging.info(
                            f"DFT calculation number {job_no} "
                            f"for worker {job_idx} finished."
                        )
                    # remove the job from the futures dict
                    # to avoid double counting
                    del futures[job_idx][job_no]
                    # remove folder with results
                    if self.clean_dirs:
                        try:
                            shutil.rmtree(
                                self.calc_dir / f"worker_{job_idx}_no_{job_no}"
                            )
                        except FileNotFoundError:
                            logging.warning(
                                f"Directory {self.calc_dir / f'worker_{job_idx}_{job_no}'}"
                                "not found. Skipping removal."
                            )

    def _setup_aims_calculator(self, atoms):
        pass

    def _recalc_dft(self, current_point):
        pass

    def finalize_dft(self):
        with threading.Lock():
            self.kill_thread = True
        time.sleep(5)
        # clean up done in analysis class if
        # self.config.create_restart is True
        if not self.config.analysis:
            parsl.dfk().cleanup()


class ALRunningManager:
    """
    Manages the "running" state of the active learning
    procedure.
    """

    def __init__(
        self,
        config: ALConfigurationManager,
        state_manager: ALStateManager,
        ensemble_manager: ALEnsembleManager,
        comm_handler: CommHandler,
        dft_manager: ALDFTManager,
        rank: int,
    ):
        self.config = config
        self.state_manager = state_manager
        self.ensemble_manager = ensemble_manager
        self.comm_handler = comm_handler
        self.rank = rank
        self.dft_manager = dft_manager

    def check_all_trajectories_reached_limit(self) -> bool:
        """
        Check if all trajectories reached maximum MD steps.
        """
        if (
            self.state_manager.num_MD_limits_reached
            == self.config.num_trajectories
        ):
            if self.rank == 0:
                logging.info("All trajectories reached maximum MD steps.")
            return True
        return False

    def check_max_training_set_size_reached(self) -> bool:
        """
        Check if maximum training set size is reached.
        """
        if (
            self.ensemble_manager.train_dataset_len
            >= self.config.max_train_set_size
        ):
            if self.rank == 0:
                logging.info("Maximum size of training set reached.")
            return True
        return False

    def check_desired_accuracy_reached(self) -> bool:
        """
        Check if desired accuracy is reached.
        """
        if (
            self.state_manager.current_valid_error
            < self.config.desired_accuracy
        ):
            if self.rank == 0:
                logging.info("Desired accuracy reached.")
            return True
        return False

    def should_terminate_worker(self, current_MD_step: int, idx: int) -> bool:
        """
        Check if worker should be terminated due to reaching max MD steps.
        """
        if (
            current_MD_step > self.config.max_MD_steps
            and self.state_manager.trajectory_status[idx] == "running"
        ):

            if self.rank == 0:
                logging.info(
                    f"Trajectory worker {idx} reached maximum MD steps "
                    "and is killed."
                )

            self.state_manager.num_MD_limits_reached += 1
            self.state_manager.trajectory_status[idx] = "killed"
            return True
        return False

    def execute_md_step(
        self,
        idx: int,
        md_manager: ALMDManager,
        md_drivers: dict,
        restart_manager: ALRestartManager,
        trajectories: dict,
    ):
        """
        Execute MD modifications and run MD step.

        Args:
            idx (int): Index of the trajectory worker.
            md_manager (ALMDManager): Manager for MD operations.
            md_drivers (dict): Dictionary of MD drivers.
            restart_manager (ALRestartManager): Manager for restart
                                            checkpoints.
            trajectories (dict): Dictionary of trajectories.
        """
        # Handle MD modifications if enabled
        if md_manager.mod_md and self.rank == 0:
            modified = md_manager.md_modifier(
                driver=md_drivers[idx],
                metric=md_manager.get_md_mod_metric(),
                idx=idx,
            )
            if modified and self.config.create_restart:
                self._update_restart_checkpoint(
                    restart_manager, trajectories, md_drivers
                )

        # Run MD step
        if self.rank == 0:
            md_drivers[idx].run(self.config.skip_step)

    def update_md_step(self, idx: int, current_MD_step: int) -> int:
        """Update MD step counters."""
        self.state_manager.trajectory_MD_steps[idx] += self.config.skip_step
        return current_MD_step + self.config.skip_step

    def handle_periodic_checkpoint(
        self, current_MD_step: int, restart_manager, trajectories, md_drivers
    ):
        """Handle periodic checkpointing during long MD runs."""
        if self.rank == 0:
            checkpoint_interval = (
                self.config.skip_step * 100
            )  # TODO: make configurable
            if current_MD_step % checkpoint_interval == 0:
                self._update_restart_checkpoint(
                    restart_manager, trajectories, md_drivers
                )

    def _update_restart_checkpoint(
        self, restart_manager, trajectories, md_drivers
    ):
        """Update restart checkpoint files."""
        restart_manager.update_restart_dict(
            trajectories_keys=trajectories.keys(),
            md_drivers=md_drivers,
            save_restart="restart/al/al_restart.npy",
        )

    def calculate_uncertainty_data(
        self,
        idx: int,
        current_MD_step: int,
        trajectories: dict,
        get_uncertainty_func: Callable,
    ) -> tuple:
        """
        Calculate uncertainty and update threshold on rank 0.

        Args:
            idx (int): Index of the trajectory worker.
            current_MD_step (int): Current MD step.
            trajectories (dict): Dictionary of trajectories.
            get_uncertainty_func (Callable): Function to calculate uncertainty.

        Returns:
            tuple: Contains point, prediction, and uncertainty.
        """
        if self.rank == 0:
            logging.info(
                f"Trajectory worker {idx} at MD step {current_MD_step}."
            )

            point = trajectories[idx].copy()
            prediction = trajectories[idx].calc.results["forces_comm"]
            uncertainty = get_uncertainty_func(prediction)

            self.state_manager.uncertainties.append(uncertainty)
            self._update_threshold_if_needed(idx)

            return point, prediction, uncertainty
        else:
            return None, None, None

    def _update_threshold_if_needed(self, idx: int):
        """Update uncertainty threshold based on collected data."""
        min_uncertainty_count = 10  # TODO: make configurable

        if len(self.state_manager.uncertainties) <= min_uncertainty_count:
            return

        # Check if threshold should be frozen
        if self._should_freeze_threshold():
            return

        # Update threshold if not frozen
        if not self.config.freeze_threshold:
            max_uncertainty_history = 400  # TODO: make configurable
            self.state_manager.threshold = get_threshold(
                uncertainties=self.state_manager.uncertainties,
                c_x=self.config.c_x,
                max_len=max_uncertainty_history,
            )

        # Collect threshold for analysis
        if self.config.analysis:
            self.state_manager.collect_thresholds[idx].append(
                self.state_manager.threshold
            )

    def _should_freeze_threshold(self) -> bool:
        """Determine if threshold should be frozen based on dataset size."""
        should_freeze = (
            self.ensemble_manager.train_dataset_len
            >= self.config.freeze_threshold_dataset
            and not self.config.freeze_threshold
        )

        if should_freeze:
            if self.rank == 0:
                logging.info(
                    f"Train data has reached size {self.ensemble_manager.train_dataset_len}: "
                    f"freezing threshold at {self.state_manager.threshold:.3f}."
                )
            self.config.freeze_threshold = True

        return should_freeze

    def synchronize_mpi_data(
        self,
        point: ase.Atoms,
        prediction: np.ndarray,
        uncertainty: float,
        current_MD_step: int,
    ) -> tuple:
        """
        Synchronize data across MPI ranks.

        Args:
            point (ase.Atoms): Current point.
            prediction (np.ndarray): Prediction data.
            uncertainty (float): Uncertainty value.
            current_MD_step (int): Current MD step.

        Returns:
            tuple: Contains point, prediction, uncertainty,
                    and current MD step.
        """
        # Initialize data on non-root ranks
        if self.rank != 0:
            uncertainty = None
            prediction = None
            point = None
            self.state_manager.threshold = None
            current_MD_step = None

        # Broadcast data from root to all ranks
        self.comm_handler.barrier()
        self.state_manager.threshold = self.comm_handler.bcast(
            self.state_manager.threshold, root=0
        )
        point = self.comm_handler.bcast(point, root=0)
        uncertainty = self.comm_handler.bcast(uncertainty, root=0)
        prediction = self.comm_handler.bcast(prediction, root=0)
        current_MD_step = self.comm_handler.bcast(current_MD_step, root=0)
        self.comm_handler.barrier()

        return point, prediction, uncertainty, current_MD_step

    def process_uncertainty_decision(
        self, idx: int, uncertainty: np.ndarray, point: ase.Atoms
    ):
        """
        Process uncertainty and decide whether to trigger DFT calculation.

        Args:
            idx (int): Index of the trajectory worker.
            uncertainty (np.ndarray): Uncertainty data.
            point (ase.Atoms): Current point.
        """
        uncertainty_exceeded = self._check_uncertainty_threshold(uncertainty)
        timeout_exceeded = self._check_uncertainty_timeout(idx)

        if uncertainty_exceeded or timeout_exceeded:
            self.state_manager.uncert_not_crossed[idx] = 0

            if self.rank == 0 and uncertainty_exceeded:
                logging.info(
                    f"Uncertainty of point is beyond threshold "
                    f"{self.state_manager.threshold:.4f} "
                    f"at worker {idx}: "
                    f"{uncertainty:.4f}."
                )
            if self.rank == 0 and timeout_exceeded:
                logging.info(
                    f"Uncertainty not exceeded for "
                    f"{self.config.uncert_not_crossed_limit} steps "
                    f"at worker {idx}."
                )

            # Handle intermolecular uncertainty if configured
            if self.config.mol_idxs is not None:
                self._handle_intermolecular_uncertainty(uncertainty)

            # Trigger DFT calculation
            self.dft_manager.handle_dft_call(point=point, idx=idx)
        else:
            self.state_manager.uncert_not_crossed[idx] += 1

    def _check_uncertainty_threshold(self, uncertainty) -> bool:
        """Check if uncertainty exceeds threshold."""
        return (uncertainty > self.state_manager.threshold).any()

    def _check_uncertainty_timeout(self, idx: int) -> bool:
        """Check if uncertainty timeout has been exceeded."""
        timeout_limit = (
            self.config.skip_step * self.config.uncert_not_crossed_limit
        )
        return self.state_manager.uncert_not_crossed[idx] > timeout_limit

    def _handle_intermolecular_uncertainty(self, uncertainty: np.ndarray):
        """
        Checks if the uncertainty threshold crossing is caused
        by inter- or intramolecular forces.
        """
        crossings = uncertainty > self.state_manager.threshold
        cross_global = crossings[0]
        cross_inter = crossings[1]

        # Update intermolecular crossing counter
        if cross_inter and not cross_global:
            self.config.intermol_crossed += 1
        elif cross_global:
            self.config.intermol_crossed = 0

        # Log intermolecular crossings
        if self.config.intermol_crossed != 0 and self.rank == 0:
            logging.info(
                f"Intermolecular uncertainty crossed {self.config.intermol_crossed} consecutive times."
            )

        # Enable intermolecular loss if threshold reached
        if self._should_enable_intermol_loss():
            self._enable_intermolecular_loss()

    def _should_enable_intermol_loss(self) -> bool:
        """Check if intermolecular loss should be enabled."""
        return (
            self.config.intermol_crossed >= self.config.intermol_crossed_limit
            and not self.config.switched_on_intermol
            and self.config.using_intermol_loss
        )

    def _enable_intermolecular_loss(self, ensemble, ensemble_manager):
        """Enable intermolecular loss for all ensemble members."""
        if self.rank == 0:

            logging.info(
                f"Intermolecular uncertainty crossed {self.config.intermol_crossed_limit} "
                f"consecutive times. Turning intermol_loss weight to "
                f"{self.config.intermol_forces_weight}."
            )

            for tag in ensemble.keys():
                ensemble_manager.training_setups[tag][
                    "loss_fn"
                ].intermol_forces_weight = self.config.intermol_forces_weight

            self.config.switched_on_intermol = True


class ALAnalysisManager:
    """
    Base class to handle analysis of DFT calculations.
    In aims PAX, analysis means that at defined intervals
    DFT reference calculations are performed. Together with
    the ML predictions, uncertainties and other metrics,
    they are collected and saved in a directory.
    They can then be used to analyze the behavior of the
    active learning procedure.
    """

    def __init__(
        self,
        config: ALConfigurationManager,
        ensemble_manager: ALEnsembleManager,
        dft_manager: ALDFTManager,
        state_manager: ALStateManager,
        md_manager: ALMDManager,
        comm_handler: CommHandler,
        rank: int = 0,
    ):
        self.config = config
        self.ensemble_manager = ensemble_manager
        self.dft_manager = dft_manager
        self.state_manager = state_manager
        self.comm_handler = comm_handler
        self.md_manager = md_manager
        self.rank = rank

        self.aims_calculator = self.dft_manager.aims_calculator

    def _analysis_dft_call(self, point: ase.Atoms, idx: int = None) -> bool:
        """
        Base DFT call for analysis.

        Args:
            point (ase.Atoms): Geometry to be analyzed
            idx (int, optional): Index of the trajectory. Defaults to None.

        Returns:
            bool: True if the SCF cycle converged, False otherwise.
        """
        self.aims_calculator.calculate(
            point, properties=self.config.properties
        )
        return self.aims_calculator.asi.is_scf_converged

    def save_analysis(self):
        """
        Saves the analysis data to files.
        """
        np.savez(
            "analysis/analysis_checks.npz", self.state_manager.analysis_checks
        )
        np.savez("analysis/t_intervals.npz", self.state_manager.t_intervals)
        np.savez("analysis/al_losses.npz", **self.state_manager.collect_losses)
        np.savez(
            "analysis/thresholds.npz", self.state_manager.collect_thresholds
        )
        if self.config.mol_idxs is not None:
            np.savez(
                "analysis/uncertainty_checks.npz",
                self.state_manager.uncertainty_checks,
            )

    def _process_analysis(
        self, idx: int, converged: bool, analysis_prediction: np.ndarray
    ):
        """
        Processes the results of the analysis after DFT calculation.
        Collects the uncertainty threshold. Discards point if
        SCF not converged.

        Args:
            idx (int): Index of trajectory.
            converged (bool): Whether the SCF cycles converged.
            analysis_prediction (np.ndarray): MLFF prediction.
        """
        if converged:
            check_results = self.analysis_check(
                current_md_step=self.state_manager.trajectory_MD_steps[idx],
                analysis_prediction=analysis_prediction,
                true_forces=self.aims_calculator.results["forces"],
            )
            for key in check_results.keys():
                self.state_manager.analysis_checks[idx][key].append(
                    check_results[key]
                )
            self.state_manager.analysis_checks[idx]["threshold"].append(
                self.state_manager.threshold
            )

            self.state_manager.collect_thresholds[idx].append(
                self.state_manager.threshold
            )

            self.state_manager.check += 1

            self.save_analysis()
            if self.rank == 0:
                logging.info(f"SCF converged at worker {idx} for analysis.")
        else:
            if self.rank == 0:
                logging.info(
                    f"SCF not converged at worker {idx} for analysis. "
                    "Discarding point."
                )

    def perform_analysis(
        self,
        point: ase.Atoms,
        idx: int,
        prediction: np.ndarray,
        current_MD_step: int,
        uncertainty: np.ndarray,
    ):
        """
        Performs analysis on a given point.
        Runs MLFF, DFT , processes the results.

        Args:
            point (ase.Atoms): The atomic structure to analyze.
            idx (int): Index of the trajectory.
            prediction (np.ndarray): MLFF prediction.
            current_MD_step (int): MD step at analysis.
            uncertainty (np.ndarray): Uncertainty of the prediction.
        """

        self.state_manager.t_intervals[idx].append(current_MD_step)
        if self.config.mol_idxs is not None:
            self.state_manager.uncertainty_checks.append(
                uncertainty > self.state_manager.threshold
            )
        if current_MD_step % self.config.analysis_skip == 0:
            if self.rank == 0:
                logging.info(
                    f"Trajectory worker {idx} is sending a point to"
                    " DFT for analysis."
                )

            if current_MD_step % self.config.skip_step == 0:
                self.state_manager.trajectories_analysis_prediction[idx] = (
                    prediction
                )
            else:
                if self.rank == 0:
                    self.state_manager.trajectories_analysis_prediction[
                        idx
                    ] = ensemble_prediction(
                        models=list(self.ensemble_manager.ensemble.values()),
                        atoms_list=[point],
                        device=self.config.device,
                        dtype=self.config.mace_settings["GENERAL"][
                            "default_dtype"
                        ],
                    )
                self.comm_handler.barrier()
                self.state_manager.trajectories_analysis_prediction[idx] = (
                    self.comm_handler.bcast(
                        self.state_manager.trajectories_analysis_prediction[
                            idx
                        ],
                        root=0,
                    )
                )
                self.comm_handler.barrier()

            self.comm_handler.barrier()
            send_point = atoms_full_copy(point)
            send_point.arrays["forces_comm"] = (
                self.state_manager.trajectories_analysis_prediction[idx]
            )
            send_point.info["current_MD_step"] = current_MD_step
            converged = self._analysis_dft_call(point=send_point, idx=idx)
            self.comm_handler.barrier()

            self._process_analysis(
                idx,
                converged,
                self.state_manager.trajectories_analysis_prediction[idx],
            )

    def analysis_check(
        self,
        analysis_prediction: np.ndarray,
        true_forces: np.ndarray,
        current_md_step: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the force uncertainty and the maximum
        force error for a given prediction and true forces.

        Args:
            analysis_prediction (np.ndarray): Ensemble prediction.
                             [n_members, n_points, n_atoms, 3]
            true_forces (np.ndarray): True forces.
                                [n_points, n_atoms, 3]

        Returns:
            tuple[np.ndarray, np.ndarray]: force uncertainty, true force error
        """

        check_results = {}
        atom_wise_uncertainty = self.md_manager.get_uncertainty.ensemble_sd(
            analysis_prediction
        )
        uncertainty_via_max = self.md_manager.get_uncertainty.max_atomic_sd(
            atom_wise_uncertainty
        )
        uncertainty_via_mean = self.md_manager.get_uncertainty.mean_atomic_sd(
            atom_wise_uncertainty
        )

        mean_analysis_prediction = analysis_prediction.mean(0).squeeze()

        diff_sq_mean = np.mean(
            (true_forces - mean_analysis_prediction) ** 2, axis=-1
        )

        max_error = np.max(np.sqrt(diff_sq_mean), axis=-1)
        mean_error = np.mean(np.sqrt(diff_sq_mean), axis=-1)
        atom_wise_error = np.sqrt(diff_sq_mean)

        check_results["prediction"] = mean_analysis_prediction
        check_results["true_forces"] = true_forces
        check_results["atom_wise_uncertainty"] = atom_wise_uncertainty
        check_results["uncertainty_via_max"] = uncertainty_via_max
        check_results["uncertainty_via_mean"] = uncertainty_via_mean
        check_results["atom_wise_error"] = atom_wise_error
        check_results["max_error"] = max_error
        check_results["mean_error"] = mean_error
        check_results["train_set_length"] = (
            self.ensemble_manager.train_dataset_len
        )
        check_results["current_md_step"] = current_md_step

        if self.config.mol_idxs is not None:
            total_certainty = self.md_manager.get_uncertainty(
                analysis_prediction
            )
            mol_forces_uncertainty = (
                self.md_manager.get_uncertainty.get_intermol_uncertainty(
                    analysis_prediction
                )
            )
            mol_forces_prediction = (
                self.md_manager.get_uncertainty.compute_mol_forces_ensemble(
                    analysis_prediction, self.config.mol_idxs
                )
                .mean(0)
                .squeeze()
            )
            mol_forces_true = (
                self.md_manager.get_uncertainty.compute_mol_forces_ensemble(
                    true_forces.reshape(1, *true_forces.shape),
                    self.config.mol_idxs,
                ).squeeze()
            )
            mol_diff_sq_mean = np.mean(
                (mol_forces_true - mol_forces_prediction) ** 2, axis=-1
            )
            max_mol_error = np.max(np.sqrt(mol_diff_sq_mean), axis=-1)
            mean_mol_error = np.mean(np.sqrt(mol_diff_sq_mean), axis=-1)
            mol_wise_error = np.sqrt(mol_diff_sq_mean)

            check_results["mol_forces_prediction"] = mol_forces_prediction
            check_results["mol_forces_true"] = mol_forces_true
            check_results["total_uncertainty"] = total_certainty
            check_results["mol_forces_uncertainty"] = mol_forces_uncertainty
            check_results["mol_wise_error"] = mol_wise_error
            check_results["max_mol_error"] = max_mol_error
            check_results["mean_mol_error"] = mean_mol_error

        return check_results


class ALAnalysisManagerParallel(ALAnalysisManager):
    """
    Analysis class for the parallel procedure. Handles
    all the MPI communications.
    """

    def __init__(
        self,
        config: ALConfigurationManager,
        ensemble_manager: ALEnsembleManager,
        dft_manager: ALDFTManagerParallel,
        state_manager: ALStateManager,
        md_manager: ALMDManager,
        comm_handler: CommHandler,
        rank: int,
        color: int,
        world_comm,
    ):
        super().__init__(
            config=config,
            ensemble_manager=ensemble_manager,
            dft_manager=dft_manager,
            state_manager=state_manager,
            md_manager=md_manager,
            comm_handler=comm_handler,
            rank=rank,
        )
        self.color = color
        self.world_comm = world_comm
        self.req_sys_info_analysis = None
        self.req_geo_info_analysis = None
        self.current_num_atoms_analysis = None
        self.received_analysis = None
        self.analysis_worker_reqs = {
            idx: None for idx in range(self.config.num_trajectories)
        }
        self.worker_reqs_analysis = {
            "energy": {
                idx: None for idx in range(self.config.num_trajectories)
            },
            "forces": {
                idx: None for idx in range(self.config.num_trajectories)
            },
            "stress": {
                idx: None for idx in range(self.config.num_trajectories)
            },
        }
        self.worker_reqs_analysis_bufs = {
            "energy": {
                idx: None for idx in range(self.config.num_trajectories)
            },
            "forces": {
                idx: None for idx in range(self.config.num_trajectories)
            },
            "stress": {
                idx: None for idx in range(self.config.num_trajectories)
            },
        }

    def analysis_waiting_task(self, idx: int):
        """
        Waits for the DFT analysis to finish for a given worker.
        TODO: refactor

        Args:
            idx (int): The index of the worker.
        """
        if (
            self.config.restart
            and self.state_manager.first_wait_after_restart[idx]
        ):
            # if the worker is waiting and we just restarted the
            # procedure, we have to relaunch the dft job and then
            # leave the function
            logging.info(
                f"Trajectory worker {idx} is restarting DFT analysis."
            )
            self._analysis_dft_call(idx, self.state_manager.trajectories[idx])
            self.state_manager.first_wait_after_restart[idx] = False
            return None

        if self.worker_reqs_analysis["energy"][idx] is None:
            self.worker_reqs_analysis_bufs["energy"][idx] = np.empty(
                shape=(2,), dtype=np.float64
            )
            self.worker_reqs_analysis["energy"][idx] = self.world_comm.Irecv(
                buf=self.worker_reqs_analysis_bufs["energy"][idx],
                source=1,
                tag=idx,
            )
        status, _ = self.worker_reqs_analysis["energy"][idx].test()

        if status:
            scf_failed = np.isnan(
                self.worker_reqs_analysis_bufs["energy"][idx][0]
            )
            # check if the energy is NaN
            if not scf_failed:
                if self.worker_reqs_analysis["forces"][idx] is None:
                    self.worker_reqs_analysis_bufs["forces"][idx] = np.empty(
                        shape=(
                            int(
                                self.worker_reqs_analysis_bufs["energy"][idx][
                                    1
                                ]
                            ),
                            3,
                        ),
                        dtype=np.float64,
                    )
                    self.worker_reqs_analysis["forces"][idx] = (
                        self.world_comm.Irecv(
                            buf=self.worker_reqs_analysis_bufs["forces"][idx],
                            source=1,
                            tag=idx + 10000,
                        )
                    )
                    if self.config.compute_stress:
                        self.worker_reqs_analysis_bufs["stress"][idx] = (
                            np.empty(shape=(6,), dtype=np.float64)
                        )
                        self.worker_reqs_analysis["stress"][idx] = (
                            self.world_comm.Irecv(
                                buf=self.worker_reqs_analysis_bufs["stress"][
                                    idx
                                ],
                                source=1,
                                tag=idx + 20000,
                            )
                        )
                status_forces = self.worker_reqs_analysis["forces"][idx].Wait()
                if self.config.compute_stress:
                    status_stress = self.worker_reqs_analysis["stress"][
                        idx
                    ].Wait()
                if status_forces or (
                    self.config.compute_stress and status_stress
                ):
                    self.worker_reqs_analysis["energy"][idx] = None
                    self.worker_reqs_analysis["forces"][idx] = None
                    if self.config.compute_stress:
                        self.worker_reqs_analysis["stress"][idx] = None

                    if self.rank == 0:
                        logging.info(
                            f"Worker {idx} received a point from DFT for "
                            "analysis."
                        )

                    analysis_forces = self.worker_reqs_analysis_bufs["forces"][
                        idx
                    ]
                    analysis_predicted_forces = (
                        self.state_manager.trajectories_analysis_prediction[
                            idx
                        ]
                    )
                    check_results = self.analysis_check(
                        analysis_prediction=analysis_predicted_forces,
                        true_forces=analysis_forces,
                        current_md_step=self.state_manager.trajectory_MD_steps[
                            idx
                        ],
                    )

                    for key in check_results.keys():
                        self.state_manager.analysis_checks[idx][key].append(
                            check_results[key]
                        )

                    self.state_manager.analysis_checks[idx][
                        "threshold"
                    ].append(self.state_manager.threshold)
                    self.state_manager.collect_thresholds[idx].append(
                        self.state_manager.threshold
                    )
                    self.state_manager.check += 1
                    self.save_analysis()
                    self.state_manager.trajectory_status[idx] = "running"

    def _analysis_dft_call(self, idx: int, point: ase.Atoms) -> None:
        """
        Sends point to DFT process asynchronously.

        Args:
            idx (int): The index of the worker.
            point (ase.Atoms): The atomic structure to send.

        Returns:
            None
        """
        self.comm_handler.barrier()
        if self.rank == 0:
            send_points_non_blocking(
                idx=idx,
                point_data=self.state_manager.trajectories[idx],
                tag=80545,
                world_comm=self.world_comm,
            )
        self.comm_handler.barrier()
        self.state_manager.trajectory_status[idx] = "analysis_waiting"
        self.comm_handler.barrier()
        if self.rank == 0:
            logging.info(
                f"Trajectory worker {idx} is waiting for analysis job to finish."
            )
        return None

    def analysis_listening_task(self):
        """
        Listens for incoming analysis data.

        Returns:
            bool: True if data was received, False otherwise.
        """
        received = False
        if self.req_sys_info_analysis is None:
            self.sys_info_buf_analysis = np.empty(
                shape=(14,), dtype=np.float64
            )
            self.req_sys_info_analysis = self.world_comm.Irecv(
                buf=self.sys_info_buf_analysis, source=0, tag=80545
            )

        status, _ = self.req_sys_info_analysis.test()
        if status:
            self.current_num_atoms_analysis = int(
                self.sys_info_buf_analysis[1]
            )
            self.req_sys_info_analysis = None
            if self.req_geo_info_analysis is None:
                self.geo_info_buf_analysis = np.empty(
                    shape=(2, self.current_num_atoms_analysis, 3),
                    dtype=np.float64,
                )

                self.req_geo_info_analysis = self.world_comm.Irecv(
                    buf=self.geo_info_buf_analysis, source=0, tag=80546
                )
            status_pos_spec = self.req_geo_info_analysis.Wait()
            if status_pos_spec:
                self.req_geo_info_analysis = None
                received = True

        return received

    def analysis_calculate_received(self) -> ase.Atoms:
        """
        Performs DFT calculation on the received point.

        Returns:
            ase.Atoms or None: The atoms object with DFT results if SCF
            converged, otherwise None.
        """
        current_pbc = (
            self.sys_info_buf_analysis[2:5].astype(np.bool_).reshape((3,))
        )
        current_species = self.geo_info_buf_analysis[0].astype(np.int32)
        # transform current_species to a list of species
        current_species = current_species[:, 0].tolist()
        current_positions = self.geo_info_buf_analysis[1]
        current_cell = self.sys_info_buf_analysis[5:14].reshape((3, 3))
        point = ase.Atoms(
            positions=current_positions,
            numbers=current_species,
            pbc=current_pbc,
            cell=current_cell,
        )
        self.comm_handler.barrier()
        dft_result = self.dft_manager.recalc_dft(point)
        # dft_result may be None if SCF did not converge
        return dft_result

    def _process_analysis(self, idx, converged, analysis_prediction):
        """
        Overwriting parent class. Processing is done in
        the waiting task.
        """
        return None


class ALAnalysisManagerPARSL(ALAnalysisManager):

    def __init__(
        self,
        config: ALConfigurationManager,
        ensemble_manager: ALEnsembleManager,
        dft_manager: ALDFTManagerPARSL,
        state_manager: ALStateManager,
        md_manager: ALMDManager,
        comm_handler: CommHandler,
    ):
        super().__init__(
            config=config,
            ensemble_manager=ensemble_manager,
            dft_manager=dft_manager,
            state_manager=state_manager,
            md_manager=md_manager,
            comm_handler=comm_handler,
        )
        self.analysis_queue = queue.Queue()
        self.analysis_counter = {
            idx: 0 for idx in range(self.config.num_trajectories)
        }
        self.analysis_kill_thread = False
        self.analysis_done = False
        self.results_lock = threading.Lock()
        logging.info("Launching analysis manager thread for PARSL.")
        threading.Thread(target=self._analysis_manager, daemon=True).start()

    def _analysis_manager(self):
        """
        Thread that handles analysis calculations with PARSL.
        Runs concurrently to the main and DFT thread. Shares the
        same PARSL process as the DFT thread i.e. the same queue
        for resources.
        Same working principle as DFT thread. Directly processes
        the analysis data.
        Note: The order of calculations is not the same as the
        order in which the calculations pop up during AL. This
        is why we keep track of the current MD step and count
        when they appear. Once the main process is stopped this
        thread is kept alive until all analysis jobs in the queue
        are done.
        TODO: refactor
        """
        futures = {}
        predicted_forces = {}
        current_md_steps = {}

        for idx in range(self.config.num_trajectories):
            predicted_forces[idx] = {}
            futures[idx] = {}
            current_md_steps[idx] = {}

        kill_requested = False

        while True:
            if self.analysis_kill_thread and not kill_requested:
                logging.info(
                    "Analysis manager kill switch triggered. "
                    "Waiting for pending analysis jobs..."
                )
                kill_requested = True

            # Try to get new data unless kill has been requested
            if not kill_requested:
                try:
                    idx, data = self.analysis_queue.get(timeout=1)
                    self.analysis_counter[idx] += 1
                    current_idx = self.analysis_counter[idx]
                    predicted_forces[idx][current_idx] = data.arrays[
                        "forces_comm"
                    ]
                    current_md_steps[idx][current_idx] = data.info[
                        "current_MD_step"
                    ]
                    futures[idx][current_idx] = recalc_dft_parsl(
                        positions=data.get_positions(),
                        species=data.get_chemical_symbols(),
                        cell=data.get_cell(),
                        pbc=data.pbc,
                        aims_settings=self.dft_manager.aims_settings,
                        directory=self.dft_manager.calc_dir
                        / f"worker_analysis{idx}_no_{current_idx}",
                        properties=self.config.properties,
                        ase_aims_command=self.dft_manager.launch_str,
                    )
                    self.analysis_queue.task_done()
                except queue.Empty:
                    pass

            # Check all futures for completion
            done_jobs = []
            for job_idx in list(futures.keys()):
                for job_no, future in list(futures[job_idx].items()):
                    if future.done():
                        done_jobs.append((job_idx, job_no))

            # Process done jobs
            for job_idx, job_no in done_jobs:
                with self.results_lock:
                    temp_result = futures[job_idx][job_no].result()
                if temp_result is None:
                    logging.info(
                        f"SCF during analysis for worker {job_idx} no {job_no}"
                        " failed. Discarding point."
                    )
                else:
                    # process analysis data directly and saves it
                    analysis_forces = predicted_forces[job_idx][job_no]
                    true_forces = temp_result["forces"]
                    check_results = self.analysis_check(
                        analysis_prediction=analysis_forces,
                        true_forces=true_forces,
                        current_md_step=current_md_steps[job_idx][job_no],
                    )
                    with self.results_lock:
                        for key in check_results:
                            self.state_manager.analysis_checks[job_idx][
                                key
                            ].append(check_results[key])
                        self.state_manager.analysis_checks[job_idx][
                            "threshold"
                        ].append(self.state_manager.threshold)
                        self.state_manager.collect_thresholds[job_idx].append(
                            self.state_manager.threshold
                        )
                        self.state_manager.check += 1
                        self.save_analysis()

                # Remove only the completed job
                del futures[job_idx][job_no]
                del predicted_forces[job_idx][job_no]
                del current_md_steps[job_idx][job_no]

                if self.dft_manager.clean_dirs:
                    try:
                        shutil.rmtree(
                            self.dft_manager.calc_dir
                            / f"worker_analysis{job_idx}_no_{job_no}"
                        )
                    except FileNotFoundError:
                        temp_path = f"worker_analysis{job_idx}_no_{job_no}"
                        temp_path = self.dft_manager.calc_dir / temp_path
                        logging.warning(
                            f"Directory {temp_path} not found. "
                            "Skipping removal."
                        )

            # Check for final shutdown
            if kill_requested and not any(futures.values()):
                logging.info(
                    "All pending analysis jobs completed. Shutting down thread."
                )
                self.analysis_done = True
                break

            time.sleep(0.1)

    def _analysis_dft_call(self, idx: int, point: ase.Atoms):
        # we can go back to running here as the analysis is not blocking
        self.state_manager.trajectory_status[idx] = "running"
        self.analysis_queue.put((idx, point))

    def _process_analysis(
        self, idx: int, converged: bool, analysis_prediction: np.ndarray
    ):
        """
        Dummy to overwrite the parent method
        processing is done in the analysis manager thread
        """
        return

    def finalize_analysis(self):
        """
        Finalizes the analysis by ensuring all analysis jobs are completed
        and cleans up any resources used by the analysis manager.
        """
        with self.results_lock:
            self.analysis_kill_thread = True
        # Wait for the analysis manager thread to finish
        while not self.analysis_done:
            time.sleep(0.1)
        parsl.dfk().cleanup()
