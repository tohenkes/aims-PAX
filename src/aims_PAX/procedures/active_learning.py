from typing import Optional
import numpy as np
from .preparation import PrepareALProcedure
from .al_helpers import (
    ALDataManager,
    ALTrainingManager,
    ALAnalysisManager,
    ALAnalysisManagerParallel,
    ALDFTManagerSerial,
    ALDFTManagerParallel,
    ALDFTManagerPARSL,
    ALAnalysisManagerPARSL,
)
from aims_PAX.tools.uncertainty import (
    get_threshold,
)
from aims_PAX.tools.utilities.data_handling import (
    save_datasets,
)
from aims_PAX.tools.utilities.utilities import (
    atoms_full_copy,
)
from aims_PAX.tools.utilities.mpi_utils import CommHandler
from aims_PAX.tools.utilities.parsl_utils import (
    prepare_parsl,
    recalc_aims_parsl,
    handle_parsl_logger,
)
import shutil
import ase
import logging
import threading
import queue
import time

try:
    import parsl
except ImportError:
    parsl = None
try:
    import asi4py
except Exception as e:
    asi4py = None


class ALProcedure(PrepareALProcedure):
    """
    Base class for the active learning procedure. It handles the training
    of the ensemble members, the molecular dynamics simulations, the
    sampling of points and the saving of the datasets.
    """

    def __init__(
        self,
        mace_settings: dict,
        al_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
        use_mpi: bool = True,
        comm_handler: CommHandler = None,
    ):

        super().__init__(
            mace_settings=mace_settings,
            al_settings=al_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
            use_mpi=use_mpi,
            comm_handler=comm_handler,
        )
        self.data_manager = None
        self.train_manager = None
        self.converge = None
        self.dft_manager = None
        self.analysis_manager = None

    def _al_loop(self):
        while True:
            for trajectory_idx in range(self.config.num_trajectories):

                if (
                    self.state_manager.trajectory_status[trajectory_idx]
                    == "waiting"
                ):

                    set_limit = self._waiting_task(trajectory_idx)
                    if set_limit:  # stops the process if the maximum dataset
                        # size is reached
                        break

                if (
                    self.state_manager.trajectory_status[trajectory_idx]
                    == "training"
                ):  # and training_job: the idea is to let a worker train
                    # only if new points have been added. e.g. it can happen
                    # that one worker is beyond its MD limit but there is no
                    # new point that has been added

                    self._training_task(trajectory_idx)

                if (
                    self.state_manager.trajectory_status[trajectory_idx]
                    == "running"
                ):

                    self._running_task(trajectory_idx)

            if (
                self.state_manager.num_MD_limits_reached
                == self.config.num_trajectories
            ):
                if self.rank == 0:
                    logging.info("All trajectories reached maximum MD steps.")
                break

            if (
                self.ensemble_manager.train_dataset_len
                >= self.config.max_set_size
            ):
                if self.rank == 0:
                    logging.info("Maximum size of training set reached.")
                break

            if (
                self.state_manager.current_valid_error
                < self.config.desired_accuracy
            ):
                if self.rank == 0:
                    logging.info("Desired accuracy reached.")
                break
        self.dft_manager.finalize_dft()

    def _waiting_task(self, idx: int):
        """
        Only adds the current point to the training or validation set
        in the base class.

        Args:
            idx (int): Index of the trajectory worker.

        """

        self.data_manager.handle_received_point(
            idx,
            received_point=self.point,  # TODO: change self.point name
        )

    def _training_task(self, idx: int):
        """
        Creates the dataloader of the updated dataset, updates
        the average number of neighbors, shifts and scaling factor
        and trains the ensemble members. Saves the models and checkpoints.

        Args:
            idx (int): Index of the trajectory worker.
        """
        if self.rank == 0:
            self.ensemble_manager.ensemble_mace_sets = (
                self.train_manager.prepare_training(
                    mace_sets=self.ensemble_manager.ensemble_mace_sets
                )
            )

            logging.info(f"Trajectory worker {idx} is training.")
            self.train_manager.perform_training(idx)

        # update calculators with the new models
        self.comm_handler.barrier()
        self.state_manager.current_valid_error = self.comm_handler.bcast(
            self.state_manager.current_valid_error, root=0
        )
        self.comm_handler.barrier()
        if self.rank == 0:
            for trajectory in self.trajectories.values():
                trajectory.calc.models = [
                    self.ensemble[tag] for tag in self.ensemble.keys()
                ]

        self.comm_handler.barrier()
        self.state_manager.total_epoch = self.comm_handler.bcast(
            self.state_manager.total_epoch, root=0
        )
        self.state_manager.trajectory_total_epochs[idx] = (
            self.comm_handler.bcast(
                self.state_manager.trajectory_total_epochs[idx], root=0
            )
        )
        self.comm_handler.barrier()

        if (
            self.state_manager.trajectory_total_epochs[idx]
            >= self.config.max_epochs_worker
        ):
            self.state_manager.trajectory_status[idx] = "running"
            self.state_manager.num_workers_training -= 1
            self.state_manager.trajectory_total_epochs[idx] = 0
            if self.rank == 0:
                logging.info(f"Trajectory worker {idx} finished training.")

    def _running_task(self, idx: int):
        """
        Runs the molecular dynamics simulation using the MLFF and
        checks the uncertainty. If the uncertainty is above the threshold
        the point is calculated using FHI aims and sent to the waiting task.

        Args:
            idx (int): Index of the trajectory worker.

        """

        current_MD_step = self.state_manager.trajectory_MD_steps[idx]

        # kill the worker if the maximum number of MD steps is reached
        if (
            current_MD_step > self.config.max_MD_steps
            and self.state_manager.trajectory_status[idx] == "running"
        ):
            if self.rank == 0:
                logging.info(
                    f"Trajectory worker {idx} reached maximum MD steps and is killed."
                )
            self.state_manager.num_MD_limits_reached += 1
            self.state_manager.trajectory_status[idx] = "killed"
            return "killed"

        else:
            # TODO:
            # ideally we would first check the uncertainty, then optionally
            # calculate the aims forces and use them to propagate
            # currently the mace forces are used even if the uncertainty is too high
            # but ase is weird and i don't want to change it so whatever. when we have our own
            # MD engine we can adress this.

            if self.md_manager.mod_md:
                if self.rank == 0:
                    modified = self.md_manager.md_modifier(
                        driver=self.md_drivers[idx],
                        metric=self.md_manager.get_md_mod_metric(),
                        idx=idx,
                    )
                    if modified and self.config.create_restart:
                        self.restart_manager.update_restart_dict(
                            trajectories_keys=self.trajectories.keys(),
                            md_drivers=self.md_drivers,
                            save_restart="restart/al/al_restart.npy",
                        )

            if self.rank == 0:
                self.md_drivers[idx].run(self.config.skip_step)
            self.state_manager.trajectory_MD_steps[
                idx
            ] += self.config.skip_step
            current_MD_step += self.config.skip_step

            # somewhat arbitrary; i just want to save checkpoints if the MD phase
            # is super long
            if self.rank == 0:
                if current_MD_step % (self.config.skip_step * 100) == 0:
                    self.restart_manager.update_restart_dict(
                        trajectories_keys=self.trajectories.keys(),
                        md_drivers=self.md_drivers,
                        save_restart="restart/al/al_restart.npy",
                    )

                logging.info(
                    f"Trajectory worker {idx} at MD step {current_MD_step}."
                )

                self.point = self.trajectories[idx].copy()
                prediction = self.trajectories[idx].calc.results["forces_comm"]
                uncertainty = self.get_uncertainty(prediction)

                self.state_manager.uncertainties.append(uncertainty)

                if (
                    len(self.state_manager.uncertainties) > 10
                ):  # TODO: remove hardcode
                    if (
                        self.ensemble_manager.train_dataset_len
                        >= self.config.freeze_threshold_dataset
                    ) and not self.config.freeze_threshold:
                        if self.rank == 0:
                            logging.info(
                                f"Train data has reached size {self.ensemble_manager.train_dataset_len}: freezing threshold at {self.state_manager.threshold :.3f}."
                            )
                        self.config.freeze_threshold = True

                    if not self.config.freeze_threshold:
                        self.state_manager.threshold = get_threshold(
                            uncertainties=self.state_manager.uncertainties,
                            c_x=self.config.c_x,
                            max_len=400,  # TODO: remove hardcode
                        )

                    if self.config.analysis:
                        self.state_manager.collect_thresholds[idx].append(
                            self.state_manager.threshold
                        )

            if self.rank != 0:
                uncertainty = None
                prediction = None
                self.point = None
                self.state_manager.threshold = None
                current_MD_step = None

            self.comm_handler.barrier()
            self.state_manager.threshold = self.comm_handler.bcast(
                self.state_manager.threshold, root=0
            )
            self.point = self.comm_handler.bcast(self.point, root=0)
            uncertainty = self.comm_handler.bcast(uncertainty, root=0)
            prediction = self.comm_handler.bcast(prediction, root=0)
            current_MD_step = self.comm_handler.bcast(current_MD_step, root=0)
            self.comm_handler.barrier()

            if (
                uncertainty > self.state_manager.threshold
            ).any() or self.state_manager.uncert_not_crossed[
                idx
            ] > self.config.skip_step * self.config.uncert_not_crossed_limit:
                self.state_manager.uncert_not_crossed[idx] = 0
                if self.rank == 0:
                    if (uncertainty > self.state_manager.threshold).any():
                        logging.info(
                            f"Uncertainty of point is beyond threshold {np.round(self.state_manager.threshold ,3)} at worker {idx}: {np.round(uncertainty,3)}."
                        )
                if self.config.mol_idxs is not None:
                    crossings = uncertainty > self.state_manager.threshold
                    cross_global = crossings[0]
                    cross_inter = crossings[1]

                    if cross_inter and not cross_global:
                        self.config.intermol_crossed += 1

                    if cross_global:
                        self.config.intermol_crossed = 0

                    if self.config.intermol_crossed != 0:
                        if self.rank == 0:
                            logging.info(
                                f"Intermolecular uncertainty crossed {self.config.intermol_crossed} consecutive times."
                            )

                    if (
                        self.config.intermol_crossed
                        >= self.config.intermol_crossed_limit
                        and not self.config.switched_on_intermol
                        and self.config.using_intermol_loss
                    ):
                        if self.rank == 0:
                            logging.info(
                                f"Intermolecular uncertainty crossed "
                                f"{self.config.intermol_crossed_limit} consecutive "
                                "times. Turning intermol_loss weight to "
                                f"{self.config.intermol_forces_weight}."
                            )
                            for tag in self.ensemble.keys():
                                self.ensemble_manager.training_setups[tag][
                                    "loss_fn"
                                ].intermol_forces_weight = (
                                    self.config.intermol_forces_weight
                                )
                            self.config.switched_on_intermol = True

                self.dft_manager.handle_dft_call(point=self.point, idx=idx)

            else:
                self.state_manager.uncert_not_crossed[idx] += 1

            if self.config.analysis:
                self.analysis_manager.perform_analysis(
                    point=self.point,
                    idx=idx,
                    prediction=prediction,
                    current_MD_step=current_MD_step,
                    uncertainty=uncertainty,
                )

    def run(self):
        """
        Main function to run the active learning procedure. Initializes variables and
        controls the workers tasks.
        """

        if self.rank == 0:
            logging.info("Starting active learning procedure.")

        self.comm_handler.barrier()
        self._al_loop()
        # turn keys which are ints into strings
        # save the datasets and the intervals for analysis
        if self.rank == 0:
            logging.info(
                f"Active learning procedure finished. The best ensemble member based on validation loss is {self.train_manager.best_member}."
            )
            save_datasets(
                ensemble=self.ensemble,
                ensemble_ase_sets=self.ensemble_manager.ensemble_ase_sets,
                path=self.config.dataset_dir / "final",
            )

            if self.config.analysis:
                self.analysis_manager.save_analysis()

            if self.config.create_restart:
                self.restart_manager.update_restart_dict(
                    trajectories_keys=self.trajectories.keys(),
                    md_drivers=self.md_drivers,
                    save_restart="restart/al/al_restart.npy",
                )
                self.restart_manager.al_restart_dict["al_done"] = True


class ALProcedureSerial(ALProcedure):
    """
    Serial implementation of the active learning procedure.
    DFT, MLFF sampling and training is done on the same rank
    and one after the other.
    """

    def __init__(
        self,
        mace_settings: dict,
        al_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
    ):

        super().__init__(
            mace_settings=mace_settings,
            al_settings=al_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )
        self.data_manager = ALDataManager(
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            state_manager=self.state_manager,
            comm_handler=self.comm_handler,
            rank=self.rank,
        )

        self.train_manager = ALTrainingManager(
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            calc_manager=self.calc_manager,
            state_manager=self.state_manager,
            md_manager=self.md_manager,
            restart_manager=self.restart_manager,
            rank=self.rank,
        )
        self.converge = self.train_manager.converge

        self.dft_manager = ALDFTManagerSerial(
            path_to_control=path_to_control,
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            state_manager=self.state_manager,
            comm_handler=self.comm_handler,
        )

        self.analysis_manager = ALAnalysisManager(
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            dft_manager=self.dft_manager,
            state_manager=self.state_manager,
            md_manager=self.md_manager,
            comm_handler=self.comm_handler,
            rank=self.rank,
        )


class ALProcedureParallel(ALProcedure):
    """
    Parallel implementation of the active learning procedure.
    MPI is used to run DFT and ML tasks (training, sampling)
    in parallel.
    """

    def __init__(
        self,
        mace_settings: dict,
        al_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
    ):

        self.comm_handler = CommHandler()
        self.world_comm = self.comm_handler.comm
        self.rank = self.comm_handler.get_rank()
        world_size = self.comm_handler.get_size()

        # one for ML and one for DFT
        if self.rank == 0:
            self.color = 0
        else:
            self.color = 1

        self.comm = self.world_comm.Split(color=self.color, key=self.rank)
        self.comm_handler = CommHandler()
        self.comm_handler.rank = self.rank
        self.comm_handler.size = self.comm.Get_size()
        self.comm_handler.comm = self.comm
        super().__init__(
            mace_settings=mace_settings,
            al_settings=al_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
            comm_handler=self.comm_handler,
        )
        self.world_size = world_size
        self.world_comm.barrier()

        if self.rank == 0:
            logging.info(f"Procedure runs on {self.world_size} workers.")
        setattr(
            self.state_manager,
            "first_wait_after_restart",
            {idx: True for idx in range(self.config.num_trajectories)},
        )
        self.data_manager = ALDataManager(
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            state_manager=self.state_manager,
            comm_handler=self.comm_handler,
            rank=self.rank,
        )

        self.train_manager = ALTrainingManager(
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            calc_manager=self.calc_manager,
            state_manager=self.state_manager,
            md_manager=self.md_manager,
            restart_manager=self.restart_manager,
            rank=self.rank,
        )
        self.converge = self.train_manager.converge

        self.dft_manager = ALDFTManagerParallel(
            path_to_control=path_to_control,
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            state_manager=self.state_manager,
            comm_handler=self.comm_handler,
            color=self.color,
            world_comm=self.world_comm,
        )
        if self.config.analysis:
            self.analysis_manager = ALAnalysisManagerParallel(
                config=self.config,
                ensemble_manager=self.ensemble_manager,
                dft_manager=self.dft_manager,
                state_manager=self.state_manager,
                md_manager=self.md_manager,
                comm_handler=self.comm_handler,
                rank=self.rank,
                color=self.color,
                world_comm=self.world_comm,
            )

    def _al_loop(self):
        self.worker_reqs = {
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

        self.worker_reqs_bufs = {
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

        self.req_sys_info = None
        self.req_geo_info = None
        self.current_num_atoms = None

        if self.color == 1:
            self.req_kill = self.world_comm.irecv(source=0, tag=422)

        while True:
            if self.color == 0:
                for trajectory_idx in range(self.config.num_trajectories):

                    if (
                        self.state_manager.trajectory_status[trajectory_idx]
                        == "running"
                    ):

                        self._running_task(trajectory_idx)

                    if (
                        self.state_manager.trajectory_status[trajectory_idx]
                        == "training"
                    ):

                        self._training_task(trajectory_idx)

                    if (
                        self.state_manager.trajectory_status[trajectory_idx]
                        == "waiting"
                    ):

                        set_limit = self._waiting_task(trajectory_idx)
                        if set_limit:
                            self._send_kill()
                            break

                    if self.config.analysis:
                        if (
                            self.state_manager.trajectory_status[
                                trajectory_idx
                            ]
                            == "analysis_waiting"
                        ):
                            self.analysis_manager.analysis_waiting_task(
                                trajectory_idx
                            )

                if (
                    self.state_manager.num_MD_limits_reached
                    == self.config.num_trajectories
                ):
                    if self.rank == 0:
                        logging.info(
                            "All trajectories reached maximum MD steps."
                        )
                        self._send_kill()
                    break

                if (
                    self.ensemble_manager.train_dataset_len
                    >= self.config.max_set_size
                ):
                    if self.rank == 0:
                        logging.info("Maximum size of training set reached.")
                        self._send_kill()
                    break

                if (
                    self.state_manager.current_valid_error
                    < self.config.desired_accuracy
                ):
                    if self.rank == 0:
                        logging.info("Desired accuracy reached.")
                        self._send_kill()
                    break

            # this color handles DFT
            # TODO: put into a separate method
            if self.color == 1:

                self.comm_handler.barrier()
                # if any rank gets a kill signal,
                # all ranks will stop
                local_kill_int = self.req_kill.Test()
                global_kill_int = self.comm_handler.comm.allreduce(
                    local_kill_int, op=self.comm_handler.mpi.LOR
                )
                kill_signal = bool(global_kill_int)

                if kill_signal:
                    return None

                if self.rank != 1:
                    self.geo_info_buf = None
                    self.sys_info_buf = None
                    received = False

                if self.rank == 1:
                    received = self._listening_task()

                self.comm_handler.barrier()
                received = self.comm_handler.bcast(
                    received, root=0
                )  # global rank 1 is 0 of split comm
                self.current_num_atoms = self.comm_handler.bcast(
                    self.current_num_atoms, root=0
                )
                self.comm_handler.barrier()

                if received:
                    if self.rank != 1:
                        self.sys_info_buf = np.empty(
                            shape=(14,), dtype=np.float64
                        )
                        self.geo_info_buf = np.empty(
                            shape=(2, self.current_num_atoms, 3),
                            dtype=np.float64,
                        )
                    self.comm_handler.barrier()
                    self.comm_handler.comm.Bcast(buf=self.sys_info_buf, root=0)
                    self.comm_handler.comm.Bcast(buf=self.geo_info_buf, root=0)
                    self.comm_handler.barrier()

                    self.comm_handler.barrier()
                    dft_result = self._calculate_received()

                    if self.rank == 1:
                        self._send_result_back(
                            idx=int(self.sys_info_buf[0]),
                            dft_result=dft_result,
                            num_atoms=self.current_num_atoms,
                        )

                if self.config.analysis:

                    if self.rank != 1:
                        self.analysis_manager.geo_info_buf_analysis = None
                        self.analysis_manager.sys_info_buf_analysis = None
                        received_analysis = False

                    if self.rank == 1:
                        received_analysis = (
                            self.analysis_manager.analysis_listening_task()
                        )

                    self.comm_handler.barrier()
                    received_analysis = self.comm_handler.bcast(
                        received_analysis, root=0
                    )  # global rank 1 is 0 of split comm
                    self.analysis_manager.current_num_atoms_analysis = (
                        self.comm_handler.bcast(
                            self.analysis_manager.current_num_atoms_analysis,
                            root=0,
                        )
                    )
                    self.comm_handler.barrier()

                    if received_analysis:
                        if self.rank != 1:
                            self.analysis_manager.sys_info_buf_analysis = (
                                np.empty(shape=(14,), dtype=np.float64)
                            )
                            self.analysis_manager.geo_info_buf_analysis = np.empty(
                                shape=(
                                    2,
                                    self.analysis_manager.current_num_atoms_analysis,
                                    3,
                                ),
                                dtype=np.float64,
                            )
                        self.comm_handler.barrier()
                        self.comm_handler.comm.Bcast(
                            buf=self.analysis_manager.sys_info_buf_analysis,
                            root=0,
                        )
                        self.comm_handler.comm.Bcast(
                            buf=self.analysis_manager.geo_info_buf_analysis,
                            root=0,
                        )
                        self.comm_handler.barrier()

                        dft_result_analysis = (
                            self.analysis_manager.analysis_calculate_received()
                        )

                        if self.rank == 1:
                            current_num_atoms_analysis = (
                                self.analysis_manager.current_num_atoms_analysis
                            )
                            self._send_result_back(
                                idx=int(
                                    self.analysis_manager.sys_info_buf_analysis[
                                        0
                                    ]
                                ),
                                dft_result=dft_result_analysis,
                                num_atoms=current_num_atoms_analysis,
                            )

        if self.color == 1:
            self.dft_manager.aims_calculator.asi.close()
        logging.info(f"RANK {self.rank} finished AL procedure.")

    def _waiting_task(self, idx: int):
        """
        Handles waiting for DFT calculation results from workers.

        This method manages the MPI communication to receive DFT results including
        energy, forces, and optionally stress tensor from worker processes.

        Args:
            idx (int): Index of the trajectory worker.
        """

        # Handle restart case - relaunch DFT job if needed
        if self.config.restart and self.first_wait_after_restart[idx]:
            self.dft_manager.handle_dft_call(
                point=self.trajectories[idx], idx=idx
            )
            self.first_wait_after_restart[idx] = False
            return None

        # Initialize energy request if not already done
        if self.worker_reqs["energy"][idx] is None:
            self._setup_energy_request(idx)

        # Check if energy data has arrived
        status, _ = self.worker_reqs["energy"][idx].test()
        if not status:
            return None

        # Process the received energy data
        energy_value = self.worker_reqs_bufs["energy"][idx][0]
        scf_failed = np.isnan(energy_value)

        if scf_failed:
            self._handle_scf_failure(idx)
        else:
            self._process_successful_calculation(idx)

    def _setup_energy_request(self, idx: int):
        """Setup MPI request to receive energy data."""
        self.worker_reqs_bufs["energy"][idx] = np.empty(
            shape=(2,), dtype=np.float64
        )
        self.worker_reqs["energy"][idx] = self.world_comm.Irecv(
            buf=self.worker_reqs_bufs["energy"][idx], source=1, tag=idx
        )

    def _handle_scf_failure(self, idx: int):
        """Handle SCF convergence failure by resetting worker and restarting MD."""
        if self.rank == 0:
            logging.info(
                f"SCF not converged at worker {idx}. "
                "Discarding point and restarting MD from last checkpoint."
            )
            self._reset_worker_requests(idx)
            self.trajectories[idx] = atoms_full_copy(
                self.state_manager.MD_checkpoints[idx]
            )
        self.state_manager.trajectory_status[idx] = "running"

    def _process_successful_calculation(self, idx: int):
        """Process successful DFT calculation by receiving forces and stress."""
        # Setup and receive forces
        if self.worker_reqs["forces"][idx] is None:
            num_atoms = int(self.worker_reqs_bufs["energy"][idx][1])
            self._setup_forces_request(idx, num_atoms)

            # Setup stress request if needed
            if self.config.compute_stress:
                self._setup_stress_request(idx)

        # Wait for all data to arrive
        self.worker_reqs["forces"][idx].Wait()
        if self.config.compute_stress:
            self.worker_reqs["stress"][idx].Wait()

        # Process the received point
        self._finalize_received_point(idx)

    def _setup_forces_request(self, idx: int, num_atoms: int):
        """Setup MPI request to receive forces data."""
        self.worker_reqs_bufs["forces"][idx] = np.empty(
            shape=(num_atoms, 3), dtype=np.float64
        )
        self.worker_reqs["forces"][idx] = self.world_comm.Irecv(
            buf=self.worker_reqs_bufs["forces"][idx],
            source=1,
            tag=idx + 10000,
        )

    def _setup_stress_request(self, idx: int):
        """Setup MPI request to receive stress data."""
        self.worker_reqs_bufs["stress"][idx] = np.empty(
            shape=(6,), dtype=np.float64
        )
        self.worker_reqs["stress"][idx] = self.world_comm.Irecv(
            buf=self.worker_reqs_bufs["stress"][idx],
            source=1,
            tag=idx + 20000,
        )

    def _finalize_received_point(self, idx: int):
        """Finalize processing of successfully received DFT point."""
        # Reset request trackers
        self._reset_worker_requests(idx)

        if self.rank == 0:
            logging.info(f"Worker {idx} received a point from DFT.")

            # Update MD checkpoint for better trajectory restart
            self.state_manager.MD_checkpoints[idx] = atoms_full_copy(
                self.trajectories[idx]
            )

        # Create received point with DFT results
        received_point = self.trajectories[idx].copy()
        received_point.info["REF_energy"] = self.worker_reqs_bufs["energy"][
            idx
        ][0]
        received_point.arrays["REF_forces"] = self.worker_reqs_bufs["forces"][
            idx
        ]

        if self.config.compute_stress:
            received_point.info["REF_stress"] = self.worker_reqs_bufs[
                "stress"
            ][idx]

        # Hand off to data manager
        self.data_manager.handle_received_point(
            idx=idx, received_point=received_point
        )

    def _reset_worker_requests(self, idx: int):
        """Reset all worker request trackers for the given index."""
        self.worker_reqs["energy"][idx] = None
        self.worker_reqs["forces"][idx] = None
        if self.config.compute_stress:
            self.worker_reqs["stress"][idx] = None

    def _listening_task(
        self,
    ) -> bool:
        """
        Listens for incoming data from the DFT worker.

        Returns:
            bool: True if data was received, False otherwise.
        """
        received = False
        if self.req_sys_info is None:
            self.sys_info_buf = np.empty(shape=(14,), dtype=np.float64)
            self.req_sys_info = self.world_comm.Irecv(
                buf=self.sys_info_buf, source=0, tag=1234
            )

        status, _ = self.req_sys_info.test()
        if status:
            self.current_num_atoms = int(self.sys_info_buf[1])
            self.req_sys_info = None
            if self.req_geo_info is None:
                self.geo_info_buf = np.empty(
                    shape=(2, self.current_num_atoms, 3), dtype=np.float64
                )

                self.req_geo_info = self.world_comm.Irecv(
                    buf=self.geo_info_buf, source=0, tag=1235
                )
            status_pos_spec = self.req_geo_info.Wait()
            if status_pos_spec:
                self.req_geo_info = None
                received = True

        return received

    def _calculate_received(self) -> Optional[ase.Atoms]:
        """
        Takes the required info from the received buffers and
        calculates the DFT result using the DFT manager.

        Returns:
            Optional[ase.Atoms]: The DFT result if the calculation was
            successful, None if the calculation failed.
        """

        current_pbc = self.sys_info_buf[2:5].astype(np.bool_).reshape((3,))
        current_species = self.geo_info_buf[0].astype(np.int32)
        # transform current_species to a list of species
        current_species = current_species[:, 0].tolist()
        current_positions = self.geo_info_buf[1]
        current_cell = self.sys_info_buf[5:14].reshape((3, 3))

        point = ase.Atoms(
            positions=current_positions,
            numbers=current_species,
            pbc=current_pbc,
            cell=current_cell,
        )
        self.comm_handler.barrier()
        dft_result = self.dft_manager.recalc_aims(point)
        return dft_result

    # TODO: put this to tools/utilities/mpi_utils.py
    def _send_result_back(
        self, idx: int, dft_result: Optional[ase.Atoms], num_atoms: int
    ):
        """
        Sends the DFT result back to the main worker.
        If the SCF is not converged, NaN values are sent back.

        Args:
            idx (int): The index of the worker.
            dft_result (Optional[ase.Atoms]): The DFT result from the worker.
            num_atoms (int): The number of atoms in the system.
        """

        if dft_result is not None:
            logging.info(
                f"DFT calculation for worker {idx} finished and sending point back."
            )
            dft_energies = dft_result.info["REF_energy"]
            dft_energies_num_atoms = np.array(
                [dft_energies, num_atoms], dtype=np.float64
            )
            dft_forces = dft_result.arrays["REF_forces"]
            if self.config.compute_stress:
                dft_stress = dft_result.info["REF_stress"]
        else:
            dft_energies_num_atoms = np.array(
                [np.nan, np.nan], dtype=np.float64
            )
            dft_forces = np.empty(shape=(num_atoms, 3), dtype=np.float64).fill(
                np.nan
            )
            if self.config.compute_stress:
                dft_stress = np.empty(shape=(6,), dtype=np.float64).fill(
                    np.nan
                )
            logging.info(
                f"DFT calculation for worker {idx} failed. Sending NaN values back."
            )

        self.world_comm.Isend(buf=dft_energies_num_atoms, dest=0, tag=idx)
        self.world_comm.Isend(
            buf=np.asarray(dft_forces, dtype=np.float64),
            dest=0,
            tag=idx + 10000,
        )
        if self.config.compute_stress:
            self.world_comm.Isend(
                buf=np.asarray(dft_stress, dtype=np.float64),
                dest=0,
                tag=idx + 20000,
            )

    def _send_kill(self):
        """
        Sends a kill signal to all other workers from global rank 0.
        """
        for dest in range(1, self.world_size):
            self.kill_send = self.world_comm.isend(True, dest=dest, tag=422)
            self.kill_send.Wait()


class ALProcedurePARSL(ALProcedure):
    """
    Implementation of the active learning procedure using PARSL.
    This class handles the DFT calculations using PARSL via
    threads. Multiple DFT calculations can be run in parallel
    without the need for MPI and ML tasks run in parallel to
    DFT.
    """

    def __init__(
        self,
        mace_settings: dict,
        al_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
    ):

        if parsl is None:
            raise ImportError(
                "PARSL is not installed. Please install PARSL"
                " to use this feature."
            )
        super().__init__(
            mace_settings=mace_settings,
            al_settings=al_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
            use_mpi=False,
        )
        self.data_manager = ALDataManager(
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            state_manager=self.state_manager,
            comm_handler=self.comm_handler,
            rank=self.rank,
        )

        self.train_manager = ALTrainingManager(
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            calc_manager=self.calc_manager,
            state_manager=self.state_manager,
            md_manager=self.md_manager,
            restart_manager=self.restart_manager,
            rank=self.rank,
        )
        self.converge = self.train_manager.converge

        self.dft_manager = ALDFTManagerPARSL(
            path_to_control=path_to_control,
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            state_manager=self.state_manager,
            comm_handler=self.comm_handler,
        )

        self.first_wait_after_restart = {
            idx: True for idx in range(self.config.num_trajectories)
        }

        if self.config.analysis:
            self.analysis_manager = ALAnalysisManagerPARSL(
                config=self.config,
                ensemble_manager=self.ensemble_manager,
                dft_manager=self.dft_manager,
                state_manager=self.state_manager,
                md_manager=self.md_manager,
                comm_handler=self.comm_handler,
            )

    def _waiting_task(self, idx):

        if self.config.restart and self.first_wait_after_restart[idx]:
            # if the worker is waiting and we just restarted the
            # procedure, we have to relaunch the dft job and then
            # leave the function
            logging.info(f"Worker {idx} is restarting DFT job after restart.")
            self.dft_manager.handle_dft_call(idx)
            self.first_wait_after_restart[idx] = False
            return None

        # with self.results_lock:
        job_result = self.dft_manager.ab_intio_results.get(idx, "not_done")

        if job_result == "not_done":
            # if the job is not done, we return None and wait for the next iteration
            return None
        else:
            if not job_result:
                logging.info(
                    f"SCF not converged at worker {idx}. Discarding point and restarting MD from last checkpoint."
                )
                self.trajectories[idx] = atoms_full_copy(
                    self.state_manager.MD_checkpoints[idx]
                )
                self.state_manager.trajectory_status[idx] = "running"

            else:
                logging.info(f"Worker {idx} received a point.")
                received_point = self.trajectories[idx].copy()
                received_point.info["REF_energy"] = job_result["energy"]
                received_point.arrays["REF_forces"] = job_result["forces"]
                if self.config.compute_stress:
                    received_point.info["REF_stress"] = job_result["stress"]

                self.state_manager.MD_checkpoints[idx] = atoms_full_copy(
                    received_point
                )
                self.state_manager.MD_checkpoints[idx].calc = (
                    self.trajectories[idx].calc
                )

                self.data_manager.handle_received_point(
                    idx=idx, received_point=received_point
                )
            with self.dft_manager.results_lock:
                # remove the job from the results dict to avoid double counting
                del self.dft_manager.ab_intio_results[idx]

    def _al_loop(self):
        super()._al_loop()
        # because for PARSL the analysis runs on a separate thread,
        # we have to finalize the analysis here explictly
        if self.config.analysis:
            self.analysis_manager.finalize_analysis()
