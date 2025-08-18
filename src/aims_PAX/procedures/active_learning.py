import os
import logging
from typing import Optional
import numpy as np
import ase
from pathlib import Path
from .preparation import PrepareALProcedure
from .al_managers import (
    ALRunningManager,
    ALDataManager,
    ALTrainingManager,
    ALAnalysisManager,
    ALAnalysisManagerParallel,
    ALDFTManagerSerial,
    ALDFTManagerParallel,
    ALDFTManagerPARSL,
    ALAnalysisManagerPARSL,
)
from aims_PAX.tools.utilities.data_handling import (
    save_datasets,
)
from aims_PAX.tools.utilities.utilities import (
    atoms_full_copy,
    save_models,
    log_yaml_block,
)
from aims_PAX.tools.utilities.mpi_utils import CommHandler

try:
    import parsl
except ImportError:
    parsl = None


class ALProcedure(PrepareALProcedure):
    """
    Base class for the active learning procedure. It handles the training
    of the ensemble members, the molecular dynamics simulations, the
    sampling of points and the saving of the datasets.
    """

    def __init__(
        self,
        mace_settings: dict,
        aimsPAX_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
        use_mpi: bool = True,
        comm_handler: CommHandler = None,
    ):

        super().__init__(
            mace_settings=mace_settings,
            aimsPAX_settings=aimsPAX_settings,
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
        self.run_manager = ALRunningManager(
            config=self.config,
            state_manager=self.state_manager,
            ensemble_manager=self.ensemble_manager,
            comm_handler=self.comm_handler,
            rank=self.rank,
            dft_manager=self.dft_manager,
        )

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
                >= self.config.max_train_set_size
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
        # Handle restart case - relaunch DFT job if needed

        if self.config.restart and self.first_wait_after_restart[idx]:
            self.point = None
            if self.rank == 0:
                self.point = self.trajectories[idx].copy()

            self.comm_handler.barrier()
            self.point = self.comm_handler.bcast(self.point, root=0)
            self.comm_handler.barrier()

            self.dft_manager.handle_dft_call(point=self.point, idx=idx)
            self.first_wait_after_restart[idx] = False

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
            >= self.config.epochs_per_worker
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

        if self.run_manager.should_terminate_worker(current_MD_step, idx):
            return "killed"

        self.run_manager.execute_md_step(
            idx,
            self.md_manager,
            self.md_manager.md_drivers,
            self.restart_manager,
            self.trajectories,
        )
        current_MD_step = self.run_manager.update_md_step(idx, current_MD_step)

        self.run_manager.handle_periodic_checkpoint(
            current_MD_step,
            self.restart_manager,
            self.trajectories,
            self.md_manager.md_drivers,
        )

        point, prediction, uncertainty = (
            self.run_manager.calculate_uncertainty_data(
                idx, current_MD_step, self.trajectories, self.get_uncertainty
            )
        )

        self.point, prediction, uncertainty, current_MD_step = (
            self.run_manager.synchronize_mpi_data(
                point, prediction, uncertainty, current_MD_step
            )
        )

        self.run_manager.process_uncertainty_decision(
            idx, uncertainty, self.point
        )

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
        Main function to run the active learning procedure.
        Saves the datasets, restart information and analysis results.
        """

        if self.rank == 0:
            logging.info("Starting active learning procedure.")

        self.comm_handler.barrier()
        self._al_loop()

        if self.rank == 0:
            logging.info(
                "Active learning procedure finished. The best ensemble member"
                " based on validation loss is "
                f"{self.train_manager.best_member}."
            )
            save_datasets(
                ensemble=self.ensemble,
                ensemble_ase_sets=self.ensemble_manager.ensemble_ase_sets,
                path=self.config.dataset_dir / "final",
            )
            save_models(
                ensemble=self.ensemble,
                training_setups=self.ensemble_manager.training_setups,
                model_dir=self.config.mace_settings["GENERAL"]["model_dir"],
                current_epoch=self.state_manager.total_epoch,
            )

            # save final results in new directory called results:
            if not self.config.converge_al:
                os.makedirs("results", exist_ok=True)
                save_datasets(
                    ensemble=self.ensemble,
                    ensemble_ase_sets=self.ensemble_manager.ensemble_ase_sets,
                    path=Path("results"),
                )
                save_models(
                    ensemble=self.ensemble,
                    training_setups=self.ensemble_manager.training_setups,
                    model_dir=Path("results"),
                    current_epoch=self.state_manager.total_epoch,
                )
            #  else: handled in convergence call

            if self.config.analysis:
                self.analysis_manager.save_analysis()

            if self.config.create_restart:
                self.restart_manager.update_restart_dict(
                    trajectories_keys=self.trajectories.keys(),
                    md_drivers=self.md_manager.md_drivers,
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
        aimsPAX_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
    ):

        super().__init__(
            mace_settings=mace_settings,
            aimsPAX_settings=aimsPAX_settings,
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
            mlff_manager=self.mlff_manager,
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
            path_to_geometry=path_to_geometry,
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

        self.run_manager = ALRunningManager(
            config=self.config,
            state_manager=self.state_manager,
            ensemble_manager=self.ensemble_manager,
            comm_handler=self.comm_handler,
            rank=self.rank,
            dft_manager=self.dft_manager,
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
        aimsPAX_settings: dict,
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
            aimsPAX_settings=aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
            comm_handler=self.comm_handler,
        )
        self.world_size = world_size
        self.world_comm.barrier()

        if self.rank == 0:
            logging.info(f"Procedure runs on {self.world_size} workers.")

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
            mlff_manager=self.mlff_manager,
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
            path_to_geometry=path_to_geometry,
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

        self.run_manager = ALRunningManager(
            config=self.config,
            state_manager=self.state_manager,
            ensemble_manager=self.ensemble_manager,
            comm_handler=self.comm_handler,
            rank=self.rank,
            dft_manager=self.dft_manager,
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
                    >= self.config.max_train_set_size
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
            if self.color == 1:
                killed = self._dft_parallel_tasks()
                if killed:
                    break

        if self.color == 1:
            self.dft_manager.aims_calculator.asi.close()

    def _dft_parallel_tasks(
        self,
    ) -> bool:
        """
        Handles DFT management for the parallel AL procedure.
        It listens for incoming data asynchronously, calculates
        DFT and sends the results back. Continuously listens for
        a signal to stop and then killing the processes.

        Returns:
            bool: True if a kill signal was received, False otherwise.
        """

        self.comm_handler.barrier()
        # if any rank gets a kill signal,
        # all ranks will stop
        local_kill_int = self.req_kill.Test()
        global_kill_int = self.comm_handler.comm.allreduce(
            local_kill_int, op=self.comm_handler.mpi.LOR
        )
        kill_signal = bool(global_kill_int)

        if kill_signal:
            return True

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
                self.sys_info_buf = np.empty(shape=(14,), dtype=np.float64)
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
            self._dft_parallel_analysis_tasks()
        return False

    def _dft_parallel_analysis_tasks(self):
        """
        Handles the analysis tasks in parallel DFT procedure.
        Basicially does the same as self._dft_parallel_tasks but
        for the analysis calculations.
        """

        if self.rank != 1:
            self.analysis_manager.geo_info_buf_analysis = None
            self.analysis_manager.sys_info_buf_analysis = None
            received_analysis = False

        if self.rank == 1:
            received_analysis = self.analysis_manager.analysis_listening_task()

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
                self.analysis_manager.sys_info_buf_analysis = np.empty(
                    shape=(14,), dtype=np.float64
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
                    idx=int(self.analysis_manager.sys_info_buf_analysis[0]),
                    dft_result=dft_result_analysis,
                    num_atoms=current_num_atoms_analysis,
                )

    def _waiting_task(self, idx: int):
        """
        Handles waiting for DFT calculation results from workers.

        This method manages the MPI communication to receive DFT results
        including energy, forces, and optionally stress tensor from
        worker processes.

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
        """
        Handle SCF convergence failure by resetting
        worker and restarting MD.
        """
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
        """
        Process successful DFT calculation by
        receiving forces and stress.
        """
        if self.worker_reqs["forces"][idx] is None:
            num_atoms = int(self.worker_reqs_bufs["energy"][idx][1])
            self._setup_forces_request(idx, num_atoms)

            if self.config.compute_stress:
                self._setup_stress_request(idx)

        self.worker_reqs["forces"][idx].Wait()
        if self.config.compute_stress:
            self.worker_reqs["stress"][idx].Wait()

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
        self._reset_worker_requests(idx)

        if self.rank == 0:
            logging.info(f"Worker {idx} received a point from DFT.")

            self.state_manager.MD_checkpoints[idx] = atoms_full_copy(
                self.trajectories[idx]
            )

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
        dft_result = self.dft_manager.recalc_dft(point)
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
        aimsPAX_settings: dict,
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
            aimsPAX_settings=aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
            use_mpi=False,
        )

        logging.info("Using followng settings for the HPC environment:")
        log_yaml_block("CLUSTER:", self.config.aimsPAX_settings["CLUSTER"])
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
            mlff_manager=self.mlff_manager,
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

        if self.config.analysis:
            self.analysis_manager = ALAnalysisManagerPARSL(
                config=self.config,
                ensemble_manager=self.ensemble_manager,
                dft_manager=self.dft_manager,
                state_manager=self.state_manager,
                md_manager=self.md_manager,
                comm_handler=self.comm_handler,
            )
        self.run_manager = ALRunningManager(
            config=self.config,
            state_manager=self.state_manager,
            ensemble_manager=self.ensemble_manager,
            comm_handler=self.comm_handler,
            rank=self.rank,
            dft_manager=self.dft_manager,
        )

    def _waiting_task(self, idx):

        if self.config.restart and self.first_wait_after_restart[idx]:
            # if the worker is waiting and we just restarted the
            # procedure, we have to relaunch the dft job and then
            # leave the function
            logging.info(f"Worker {idx} is restarting DFT job after restart.")
            self.dft_manager.handle_dft_call(
                point=self.trajectories[idx], idx=idx
            )
            self.first_wait_after_restart[idx] = False
            return None

        # with self.results_lock:
        job_result = self.dft_manager.ab_intio_results.get(idx, "not_done")

        if job_result == "not_done":
            # if the job is not done, we return None and
            # wait for the next iteration
            return None
        else:
            if not job_result:
                logging.info(
                    f"SCF not converged at worker {idx}. Discarding point and"
                    " restarting MD from last checkpoint."
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
