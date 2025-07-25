from pathlib import Path
import torch
import numpy as np
from mace import tools
from .preparation import (
    ALCalculatorManager,
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
    select_best_member,
    atoms_full_copy,
    AIMSControlParser,
)
from aims_PAX.tools.utilities.mpi_utils import (
    send_points_non_blocking,
    CommHandler
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
import logging
from contextlib import nullcontext
import ase
try:
    import asi4py
except Exception as e:
    asi4py = None


class ALDataManager:
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
        Process a received point by adding it to either validation or training set.

        Args:
            idx: Index of the trajectory worker
            received_point: The data point received from DFT calculation

        Returns:
            True if max dataset size is reached, False otherwise
        """
        # Convert to MACE format for neural network training
        mace_point = create_mace_dataset(
            data=[received_point],
            z_table=self.ensemble_manager.z_table,
            seed=None,
            r_max=self.config.r_max,
        )

        # Determine if point should go to validation or training set
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
        """Add point to validation set and update worker status."""
        self.state_manager.trajectory_status[idx] = "running"
        self.state_manager.num_workers_training -= 1

        if self.rank == 0:
            logging.info(
                f"Trajectory worker {idx} is adding a point to the validation set."
            )

            # Add to all ensemble member datasets
            for tag in self.ensemble_manager.ensemble_ase_sets.keys():
                self.ensemble_manager.ensemble_ase_sets[tag]["valid"].append(
                    received_point
                )
                self.ensemble_manager.ensemble_mace_sets[tag][
                    "valid"
                ] += mace_point

        self.state_manager.valid_points_added += 1

    def _add_to_training_set(
        self, idx: int, received_point: np.ndarray, mace_point
    ) -> bool:
        """
        Add point to training set and check if max size is reached.

        Returns:
            True if max dataset size is reached, False otherwise
        """
        self.state_manager.trajectory_status[idx] = "training"
        # Note: The increment/decrement pattern suggests this might be a bug
        # Consider reviewing this logic
        self.state_manager.num_workers_training += 1
        self.state_manager.num_workers_training -= 1

        if self.rank == 0:
            logging.info(
                f"Trajectory worker {idx} is adding a point to the training set."
            )

            # Add to all ensemble member datasets
            for tag in self.ensemble_manager.ensemble_ase_sets.keys():
                self.ensemble_manager.ensemble_ase_sets[tag]["train"].append(
                    received_point
                )
                self.ensemble_manager.ensemble_mace_sets[tag][
                    "train"
                ] += mace_point

            # Update training dataset length
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
        if self.ensemble_manager.train_dataset_len > self.config.max_set_size:
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


class ALTrainingManager:

    def __init__(
        self,
        config: ALConfigurationManager,
        ensemble_manager: ALEnsembleManager,
        calc_manager: ALCalculatorManager,
        state_manager: ALStateManager,
        md_manager: ALMDManager,
        restart_manager: ALRestartManager,
        rank: int
    ):
        self.config = config
        self.ensemble_manager = ensemble_manager
        self.calc_manager = calc_manager
        self.state_manager = state_manager
        self.restart_manager = restart_manager
        self.md_manager = md_manager
        self.rank = rank

        self.best_member = None
        self.use_scheduler = False  # TODO: remove hardcode

    def _check_batch_size(self, set_batch_size, tag):
        batch_size = (
            1
            if len(self.ensemble_manager.ensemble_mace_sets[tag]["train"])
            < set_batch_size
            else set_batch_size
        )
        return batch_size

    def prepare_training(
        self,
        mace_sets: dict,
    ):
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
            # because the dataset size is dynamically changing
            # we have to update the average number of neighbors,
            # shifts and the scaling factor for the models
            # usually they converge pretty fast
            update_model_auxiliaries(
                model,
                mace_sets[tag],
                self.config.scaling,
                self.calc_manager.ensemble_atomic_energies[tag],
                self.calc_manager.update_atomic_energies,
                self.calc_manager.ensemble_atomic_energies_dict[tag],
                self.ensemble_manager.z_table,
                self.config.dtype,
                self.config.device,
            )
        return mace_sets

    def perform_training(self, idx: int = 0):
        while (
            self.state_manager.trajectory_intermediate_epochs[idx]
            < self.config.intermediate_epochs
        ):
            for tag, model in self.ensemble_manager.ensemble.items():

                if self.state_manager.ensemble_reset_opt[tag]:
                    logging.info(f"Resetting optimizer for model {tag}.")
                    self.ensemble_manager.training_setups[tag] = (
                        reset_optimizer(
                            self.ensemble_manager.ensemble[tag],
                            self.ensemble_manager.training_setups[tag],
                            self.config.mace_settings["TRAINING"],
                        )
                    )
                    self.state_manager.ensemble_reset_opt[tag] = False

                logger = tools.MetricsLogger(
                    directory=self.config.mace_settings["GENERAL"][
                        "results_dir"
                    ],
                    tag=tag + "_train",
                )
                train_epoch(
                    model=model,
                    train_loader=self.ensemble_manager.ensemble_mace_sets[tag][
                        "train_loader"
                    ],
                    loss_fn=self.ensemble_manager.training_setups[tag][
                        "loss_fn"
                    ],
                    optimizer=self.ensemble_manager.training_setups[tag][
                        "optimizer"
                    ],
                    lr_scheduler=(
                        self.ensemble_manager.training_setups[tag][
                            "lr_scheduler"
                        ]
                        if self.use_scheduler
                        else None
                    ),  # no scheduler used here
                    epoch=self.state_manager.trajectory_intermediate_epochs[
                        idx
                    ],
                    start_epoch=None,
                    valid_loss=None,
                    logger=logger,
                    device=self.ensemble_manager.training_setups[tag][
                        "device"
                    ],
                    max_grad_norm=self.ensemble_manager.training_setups[tag][
                        "max_grad_norm"
                    ],
                    output_args=self.ensemble_manager.training_setups[tag][
                        "output_args"
                    ],
                    ema=self.ensemble_manager.training_setups[tag]["ema"],
                )
            if (
                self.state_manager.trajectory_intermediate_epochs[idx]
                % self.config.valid_skip
                == 0
                or self.state_manager.trajectory_intermediate_epochs[idx]
                == self.config.intermediate_epochs - 1
            ):
                ensemble_valid_loss, valid_loss, metrics = (
                    validate_epoch_ensemble(
                        ensemble=self.ensemble_manager.ensemble,
                        training_setups=self.ensemble_manager.training_setups,
                        ensemble_set=self.ensemble_manager.ensemble_mace_sets,
                        logger=logger,
                        log_errors=self.config.mace_settings["MISC"][
                            "error_table"
                        ],
                        epoch=self.state_manager.trajectory_total_epochs[idx],
                    )
                )
                self.best_member = select_best_member(ensemble_valid_loss)
                if self.config.analysis:
                    self.state_manager.collect_losses["epoch"].append(
                        self.state_manager.total_epoch
                    )
                    self.state_manager.collect_losses["avg_losses"].append(
                        valid_loss
                    )
                    self.state_manager.collect_losses[
                        "ensemble_losses"
                    ].append(ensemble_valid_loss)

                self.state_manager.current_valid_error = metrics["mae_f"]

                for tag in ensemble_valid_loss.keys():
                    if (
                        ensemble_valid_loss[tag]
                        < self.state_manager.ensemble_best_valid[tag]
                    ):
                        self.state_manager.ensemble_best_valid[tag] = (
                            ensemble_valid_loss[tag]
                        )
                    else:
                        self.state_manager.ensemble_no_improvement[tag] += 1

                    if (
                        self.state_manager.ensemble_no_improvement[tag]
                        > self.config.max_epochs_worker
                    ):
                        logging.info(
                            f"No improvements for {self.config.max_epochs_worker} epochs "
                            f"(maximum epochs per worker) at ensemble member {tag}."
                        )
                        self.state_manager.ensemble_reset_opt[tag] = True
                        self.state_manager.ensemble_no_improvement[tag] = 0

                    save_checkpoint(
                        self.ensemble_manager.training_setups[tag][
                            "checkpoint_handler"
                        ],
                        self.ensemble_manager.training_setups[tag],
                        model,
                        self.state_manager.trajectory_intermediate_epochs[
                            idx
                        ],
                        keep_last=False,
                    )

                    save_datasets(
                        self.ensemble_manager.ensemble,
                        self.ensemble_manager.ensemble_ase_sets,
                        path=self.config.dataset_dir / "final",
                    )
                    if self.config.create_restart:
                        self.save_restart = True

            self.state_manager.trajectory_total_epochs[idx] += 1
            self.state_manager.trajectory_intermediate_epochs[idx] += 1
            self.state_manager.total_epoch += 1

            if self.save_restart and self.config.create_restart:
                self.restart_manager.update_restart_dict(
                    trajectories_keys=self.state_manager.trajectories.keys(),
                    md_drivers=self.md_manager.md_drivers,
                    save_restart="restart/al/al_restart.npy",
                )
                self.save_restart = False
        self.models = [self.ensemble_manager.ensemble[tag] for tag in self.ensemble_manager.ensemble.keys()]
        self.state_manager.trajectory_intermediate_epochs[idx] = 0

    def converge(self):
        """
        Converges the ensemble on the acquired dataset. Trains the ensemble members
        until the validation loss does not improve anymore.
        """
        if self.rank == 0:
            if self.config.converge_best:
                logging.info(
                    f"Converging best model ({self.best_member}) on acquired dataset."
                )
                self.ensemble_manager.ensemble = {
                    self.best_member: self.ensemble_manager.ensemble[self.best_member]
                }
            else:
                logging.info("Converging ensemble on acquired dataset.")

            temp_mace_sets = {}
            for _, (tag, model) in enumerate(self.ensemble_manager.ensemble.items()):
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

            self.ensemble_manager.ensemble_mace_sets = self.prepare_training(
                mace_sets=temp_mace_sets
            )

            # resetting optimizer and scheduler
            self.training_setups_convergence = {}
            for tag in self.ensemble_manager.ensemble.keys():
                self.training_setups_convergence[tag] = setup_mace_training(
                    settings=self.config.mace_settings,
                    model=self.ensemble_manager.ensemble[tag],
                    tag=tag,
                    restart=self.config.restart,
                    convergence=True,
                    checkpoints_dir=self.config.checkpoints_dir,
                    al_settings=self.config.al,
                )
            best_valid_loss = np.inf
            epoch = 0
            if self.config.restart:
                epoch = self.training_setups_convergence[
                    list(self.ensemble_manager.ensemble.keys())[0]
                ]["epoch"]
            no_improvement = 0
            ensemble_valid_losses = {
                tag: np.inf for tag in self.ensemble_manager.ensemble.keys()
            }
            for j in range(self.config.max_final_epochs):
                # ensemble_loss = 0
                for tag, model in self.ensemble_manager.ensemble.items():
                    logger = tools.MetricsLogger(
                        directory=self.config.mace_settings["GENERAL"][
                            "results_dir"
                        ],
                        tag=tag + "_train",
                    )
                    train_epoch(
                        model=model,
                        train_loader=self.ensemble_manager.ensemble_mace_sets[
                            tag
                        ]["train_loader"],
                        loss_fn=self.training_setups_convergence[tag][
                            "loss_fn"
                        ],
                        optimizer=self.training_setups_convergence[tag][
                            "optimizer"
                        ],
                        lr_scheduler=self.training_setups_convergence[tag][
                            "lr_scheduler"
                        ],
                        valid_loss=ensemble_valid_losses[tag],
                        epoch=epoch,
                        start_epoch=epoch,
                        logger=logger,
                        device=self.training_setups_convergence[tag]["device"],
                        max_grad_norm=self.training_setups_convergence[tag][
                            "max_grad_norm"
                        ],
                        output_args=self.training_setups_convergence[tag][
                            "output_args"
                        ],
                        ema=self.training_setups_convergence[tag]["ema"],
                    )
                    # ensemble_loss += loss
                # ensemble_loss /= len(ensemble)

                if (
                    epoch % self.config.valid_skip == 0
                    or epoch == self.config.max_final_epochs - 1
                ):
                    (
                        ensemble_valid_losses,
                        valid_loss,
                        _,
                    ) = validate_epoch_ensemble(
                        ensemble=self.ensemble_manager.ensemble,
                        training_setups=self.training_setups_convergence,
                        ensemble_set=self.ensemble_manager.ensemble_mace_sets,
                        logger=logger,
                        log_errors=self.config.mace_settings["MISC"][
                            "error_table"
                        ],
                        epoch=epoch,
                    )

                    if (
                        best_valid_loss > valid_loss
                        and (best_valid_loss - valid_loss) > self.config.margin
                    ):
                        best_valid_loss = valid_loss
                        best_epoch = epoch
                        no_improvement = 0
                        for tag, model in self.ensemble_manager.ensemble.items():
                            param_context = (
                                self.training_setups_convergence[tag][
                                    "ema"
                                ].average_parameters()
                                if self.training_setups_convergence[tag]["ema"]
                                is not None
                                else nullcontext()
                            )
                            with param_context:
                                torch.save(
                                    model,
                                    Path(
                                        self.config.mace_settings["GENERAL"][
                                            "model_dir"
                                        ]
                                    )
                                    / (tag + ".model"),
                                )
                            save_checkpoint(
                                checkpoint_handler=self.training_setups_convergence[
                                    tag
                                ][
                                    "checkpoint_handler"
                                ],
                                training_setup=self.training_setups_convergence[
                                    tag
                                ],
                                model=model,
                                epoch=epoch,
                                keep_last=False,
                            )
                    else:
                        no_improvement += 1

                epoch += 1
                if no_improvement > self.config.patience:
                    logging.info(
                        f"No improvements for {self.config.patience} epochs. Training converged. Best model(s) (Epoch {best_epoch}) based on validation loss saved."
                    )
                    break
                if j == self.config.max_final_epochs - 1:
                    logging.info(
                        f"Maximum number of epochs reached. Best model (Epoch {best_epoch}) based on validation loss saved."
                    )


class ALDFTManager:
    
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
        
        # AIMS settings
        self.control_parser = AIMSControlParser()
        self._handle_aims_settings(path_to_control)
        
        self.aims_calculator = None

    def handle_dft_call(self, point, idx):
        if self.rank == 0:
            logging.info(f"Trajectory worker {idx} is running DFT.")

        self.comm_handler.barrier()
        self.point = self.recalc_aims(point)

        if not self.aims_calculator.asi.is_scf_converged:
            if self.rank == 0:
                logging.info(
                    f"SCF not converged at worker {idx}. Discarding point and restarting MD from last checkpoint."
                )
                self.state_manager.trajectories[idx] = atoms_full_copy(
                    self.state_manager.MD_checkpoints[idx]
                )
            self.state_manager.trajectory_status[idx] = "running"
        else:
            # we are updating the MD checkpoint here because then we make sure
            # that the MD is restarted from a point that is inside the training set
            # so the MLFF should be able to handle this and lead to a better trajectory
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
                    f"Trajectory worker {idx} is going to add point to the dataset."
                )
                
    def finalize_ab_initio(self):
        self.aims_calculator.asi.close()

    def recalc_aims(self, current_point: ase.Atoms) -> ase.Atoms:
        """Recalculate with AIMS and return updated atoms object."""
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
    ):
        super().__init__(
            path_to_control=path_to_control,
            config=config,
            ensemble_manager=ensemble_manager,
            state_manager=state_manager,
            comm_handler=comm_handler,
        )
        self.aims_calculator = self._setup_aims_calculator(
                atoms=self.state_manager.trajectories[0]
            )

    def _setup_aims_calculator(self, atoms: ase.Atoms):
        """Setup AIMS calculator."""
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
        world_comm
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
            atoms=self.state_manager.trajectories[0]
        )
        self.world_comm = world_comm
    
    def _setup_aims_calculator(
        self,
        atoms: ase.Atoms,
    ) -> ase.Atoms:
        """
        Attaches the AIMS calculator to the atoms object. Uses the AIMS settings
        from the control.in to set up the calculator.

        Args:
            atoms (ase.Atoms): Atoms object to attach the calculator to.
            pbc (bool, optional): If periodic boundry conditions are required or not.
            Defaults to False.

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
                self.config.ASI_path, init_via_ase, self.comm_handler.comm, atoms
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


class ALAnalysisManager:

    def __init__(
        self,
        config: ALConfigurationManager,
        ensemble_manager: ALEnsembleManager,
        dft_manager: ALDFTManager,
        state_manager: ALStateManager,
        md_manager: ALMDManager,
        comm_handler: CommHandler,
        rank: int,
    ):
        self.config = config
        self.ensemble_manager = ensemble_manager
        self.dft_manager = dft_manager
        self.state_manager = state_manager
        self.comm_handler = comm_handler
        self.md_manager = md_manager
        self.rank = rank

        self.aims_calculator = self.dft_manager.aims_calculator

    def _analysis_dft_call(self, point: ase.Atoms, idx: int = None):
        self.aims_calculator.calculate(
            point, properties=self.config.properties
        )
        return self.aims_calculator.asi.is_scf_converged

    def save_analysis(self):
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

        else:
            if self.rank == 0:
                logging.info(
                    f"SCF not converged at worker {idx} for analysis. Discarding point."
                )

    def perform_analysis(
        self,
        point,
        idx,
        prediction,
        current_MD_step,
        uncertainty,
    ):

        self.state_manager.t_intervals[idx].append(current_MD_step)
        if self.config.mol_idxs is not None:
            self.state_manager.uncertainty_checks.append(
                uncertainty > self.state_manager.threshold
            )
        if current_MD_step % self.config.analysis_skip == 0:
            if self.rank == 0:
                logging.info(
                    f"Trajectory worker {idx} is sending a point to DFT for analysis."
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

            # TODO: sometimes already calculated above so we should not calculate it again
            self.comm_handler.barrier()
            send_point = atoms_full_copy(point)
            send_point.arrays["forces_comm"] = (
                self.state_manager.trajectories_analysis_prediction[idx]
            )
            send_point.info["current_MD_step"] = current_MD_step
            converged = self._analysis_dft_call(point=send_point, idx=idx)
            self.comm_handler.barrier()

            self._process_analysis(
                idx=idx,
                converged=converged,
                analysis_prediction=self.state_manager.trajectories_analysis_prediction[
                    idx
                ],
            )

    def analysis_check(
        # TODO: put this somewhere else and its ugly
        # TODO: update docstring
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
            total_certainty = self.md_manager.get_uncertainty(analysis_prediction)
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
            mol_forces_true = self.md_manager.get_uncertainty.compute_mol_forces_ensemble(
                true_forces.reshape(1, *true_forces.shape),
                self.config.mol_idxs,
            ).squeeze()
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
        world_comm
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
    
    def _process_analysis(
        self, idx: int, converged: bool, analysis_prediction: np.ndarray
    ):
        # Dummy to overwrite the parent method
        return

    def _analysis_waiting_task(self, idx: int):

        if self.config.restart and self.state_manager.first_wait_after_restart[idx]:
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
                            f"Worker {idx} received a point from DFT for analysis."
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

    def _analysis_dft_call(self, idx, point):

        self.comm_handler.barrier()
        if self.rank == 0:
            send_points_non_blocking(
                idx=idx, 
                point_data=self.state_manager.trajectories[idx], 
                tag=80545
            )
        self.comm_handler.barrier()
        self.state_manager.trajectory_status[idx] = "analysis_waiting"
        self.comm_handler.barrier()
        if self.rank == 0:
            logging.info(
                f"Trajectory worker {idx} is waiting for analysis job to finish."
            )
        return None

    def _analysis_listening_task(self):
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
            idx = int(self.sys_info_buf_analysis[0])
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

    def _analysis_calculate_received(self):
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
        dft_result = self.dft_manager.recalc_aims(point)
        return dft_result





class ALAnalysisManagerPARSL:
    None

