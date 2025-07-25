import numpy as np
from .preparation import PrepareALProcedure
from .al_helpers import (
    ALDataManager,
    ALTrainingManager,
    ALAnalysisManager,
    ALDFTManagerSerial,
    ALDFTManagerParallel
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
    Class for the active learning procedure. It handles the training
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
                    if (
                        set_limit
                    ):  # stops the process if the maximum dataset
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
        self.dft_manager.finalize_ab_initio()

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
            self.ensemble_manager.ensemble_mace_sets = self.train_manager.prepare_training(
                mace_sets=self.ensemble_manager.ensemble_mace_sets
            )

            logging.info(f"Trajectory worker {idx} is training.")
            # we train only for some epochs before we move to the next worker which may be running MD
            # all workers train on the same models with the respective training settings for
            # each ensemble member

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
            # calculate true error and uncertainty on validation set

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

                self.dft_manager.handle_dft_call(
                    point=self.point,
                    idx=idx
                )

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
        self.world_size = self.comm_handler.get_size()

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
        if self.rank == 0:
            logging.info(f"Procedure runs on {self.world_size} workers.")
        setattr(
            self.state_manager,
            "first_wait_after_restart",
            {idx: True for idx in range(self.config.num_trajectories)}
            
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
            self.analysis_manager = ALAnalysisManager(
                config=self.config,
                ensemble_manager=self.ensemble_manager,
                dft_manager=self.dft_manager,
                state_manager=self.state_manager,
                md_manager=self.md_manager,
                comm_handler=self.comm_handler,
                rank=self.rank,
            )

    def _send_kill(self):
        """
        Sends a kill signal
        """
        logging.info(f"RANK {self.rank} sending kill signal to all workers.")
        for dest in range(1, self.world_size):
            self.kill_send = self.world_comm.isend(True, dest=dest, tag=422)
            self.kill_send.Wait()

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

        if self.config.analysis:
            self.req_sys_info_analysis = None
            self.req_geo_info_analysis = None
            self.current_num_atoms_analysis = None
            self.received_analysis = None
            self.analysis_worker_reqs = {
                idx: None for idx in range(self.config.num_trajectories)
            }

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
                            self._analysis_waiting_task(trajectory_idx)

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

            if self.color == 1:
                kill_signal = self.req_kill.Test()
                if kill_signal:
                    if self.rank == 0:
                        logging.info("KILL RECEIVED.")
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
                        self.geo_info_buf_analysis = None
                        self.sys_info_buf_analysis = None
                        received_analysis = False

                    if self.rank == 1:
                        received_analysis = self._analysis_listening_task()

                    self.comm_handler.barrier()
                    received_analysis = self.comm_handler.bcast(
                        received_analysis, root=0
                    )  # global rank 1 is 0 of split comm
                    self.current_num_atoms_analysis = self.comm_handler.bcast(
                        self.current_num_atoms_analysis, root=0
                    )
                    self.comm_handler.barrier()

                    if received_analysis:
                        if self.rank != 1:
                            self.sys_info_buf_analysis = np.empty(
                                shape=(14,), dtype=np.float64
                            )
                            self.geo_info_buf_analysis = np.empty(
                                shape=(2, self.current_num_atoms_analysis, 3),
                                dtype=np.float64,
                            )
                        self.comm_handler.barrier()
                        self.comm_handler.comm.Bcast(
                            buf=self.sys_info_buf_analysis, root=0
                        )
                        self.comm_handler.comm.Bcast(
                            buf=self.geo_info_buf_analysis, root=0
                        )
                        self.comm_handler.barrier()

                        dft_result_analysis = (
                            self._analysis_calculate_received()
                        )

                        if self.rank == 1:
                            self._send_result_back(
                                idx=int(self.sys_info_buf_analysis[0]),
                                dft_result=dft_result_analysis,
                                num_atoms=self.current_num_atoms_analysis,
                            )

        if self.color == 1:
            self.dft_manager.aims_calculator.asi.close()
        logging.info(f"RANK {self.rank} finished AL procedure.")

    def _waiting_task(self, idx: int):
        """
        TODO

        Args:
            idx (int): Index of the trajectory worker.

        """

        if self.config.restart and self.first_wait_after_restart[idx]:
            # if the worker is waiting and we just restarted the
            # procedure, we have to relaunch the dft job and then
            # leave the function
            self.dft_manager.handle_dft_call(
                point=self.trajectories[idx],
                idx=idx)
            self.first_wait_after_restart[idx] = False
            return None

        if self.worker_reqs["energy"][idx] is None:
            self.worker_reqs_bufs["energy"][idx] = np.empty(
                shape=(2,), dtype=np.float64
            )
            self.worker_reqs["energy"][idx] = self.world_comm.Irecv(
                buf=self.worker_reqs_bufs["energy"][idx], source=1, tag=idx
            )
        status, _ = self.worker_reqs["energy"][idx].test()

        if status:
            scf_failed = np.isnan(self.worker_reqs_bufs["energy"][idx][0])
            # check if the energy is NaN
            if not scf_failed:
                if self.worker_reqs["forces"][idx] is None:
                    self.worker_reqs_bufs["forces"][idx] = np.empty(
                        shape=(
                            int(self.worker_reqs_bufs["energy"][idx][1]),
                            3,
                        ),
                        dtype=np.float64,
                    )
                    self.worker_reqs["forces"][idx] = self.world_comm.Irecv(
                        buf=self.worker_reqs_bufs["forces"][idx],
                        source=1,
                        tag=idx + 10000,
                    )
                    if self.config.compute_stress:
                        self.worker_reqs_bufs["stress"][idx] = np.empty(
                            shape=(6,), dtype=np.float64
                        )
                        self.worker_reqs["stress"][idx] = (
                            self.world_comm.Irecv(
                                buf=self.worker_reqs_bufs["stress"][idx],
                                source=1,
                                tag=idx + 20000,
                            )
                        )
                status_forces = self.worker_reqs["forces"][idx].Wait()
                if self.config.compute_stress:
                    status_stress = self.worker_reqs["stress"][idx].Wait()
                if status_forces or (
                    self.config.compute_stress and status_stress
                ):
                    self.worker_reqs["energy"][idx] = None
                    self.worker_reqs["forces"][idx] = None
                    if self.config.compute_stress:
                        self.worker_reqs["stress"][idx] = None

                    if self.rank == 0:
                        logging.info(
                            f"Worker {idx} received a point from DFT."
                        )
                    # we are updating the MD checkpoint here because then we make sure
                    # that the MD is restarted from a point that is inside the training set
                    # so the MLFF should be able to handle this and lead to a better trajectory
                    if self.rank == 0:
                        self.state_manager.MD_checkpoints[idx] = (
                            atoms_full_copy(self.trajectories[idx])
                        )

                    received_point = self.trajectories[idx].copy()
                    received_point.info["REF_energy"] = self.worker_reqs_bufs[
                        "energy"
                    ][idx][0]
                    received_point.arrays["REF_forces"] = (
                        self.worker_reqs_bufs["forces"][idx]
                    )
                    if self.config.compute_stress:
                        received_point.info["REF_stress"] = (
                            self.worker_reqs_bufs["stress"][idx]
                        )

                    self.data_manager.handle_received_point(
                        idx=idx, received_point=received_point
                    )
            else:
                if self.rank == 0:
                    logging.info(
                        f"SCF not converged at worker {idx}. Discarding point and restarting MD from last checkpoint."
                    )
                    self.worker_reqs["energy"][idx] = None
                    self.worker_reqs["forces"][idx] = None
                    if self.config.compute_stress:
                        self.worker_reqs["stress"][idx] = None
                    self.trajectories[idx] = atoms_full_copy(
                        self.state_manager.MD_checkpoints[idx]
                    )
                self.state_manager.trajectory_status[idx] = "running"
    
    def _training_task(self, idx: int):
        """
        Creates the dataloader of the updated dataset, updates
        the average number of neighbors, shifts and scaling factor
        and trains the ensemble members. Saves the models and checkpoints.

        Args:
            idx (int): Index of the trajectory worker.
        """
        # TODO: why is this here? why can't i use the parents method?
        self.ensemble_manager.ensemble_mace_sets = self.train_manager.prepare_training(
            mace_sets=self.ensemble_manager.ensemble_mace_sets
        )

        logging.info(f"Trajectory worker {idx} is training.")
        # we train only for some epochs before we move to the next worker which may be running MD
        # all workers train on the same models with the respective training settings for
        # each ensemble member

        self.train_manager.perform_training(idx)

        for trajectory in self.trajectories.values():
            trajectory.calc.models = [
                self.ensemble[tag] for tag in self.ensemble.keys()
            ]

        if (
            self.state_manager.trajectory_total_epochs[idx]
            >= self.config.max_epochs_worker
        ):
            self.state_manager.trajectory_status[idx] = "running"
            self.state_manager.num_workers_training -= 1
            self.state_manager.trajectory_total_epochs[idx] = 0
            if self.rank == 0:
                logging.info(f"Trajectory worker {idx} finished training.")
            # calculate true error and uncertainty on validation set

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
            idx = int(self.sys_info_buf[0])
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

    def _calculate_received(self):
        current_idx = int(self.sys_info_buf[0])
        current_num_atoms = int(self.sys_info_buf[1])
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

    def _send_result_back(self, idx, dft_result, num_atoms):
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


class ALProcedurePARSL(ALProcedure):
    """
    This class is for the PARSL implementation of the active learning procedure.
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
                "PARSL is not installed. Please install PARSL to use this feature."
            )
        super().__init__(
            mace_settings=mace_settings,
            al_settings=al_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
            use_mpi=False,
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
            handle_parsl_logger(log_dir=self.log_dir / "parsl_al.log")
            parsl.load(self.parsl_config)

        logging.info("Launching ab initio manager thread for PARSL.")
        self.ab_initio_queue = queue.Queue()
        self.ab_intio_results = {}
        self.ab_initio_counter = {
            idx: 0 for idx in range(self.config.num_trajectories)
        }
        self.results_lock = threading.Lock()
        self.kill_thread = False
        threading.Thread(target=self.ab_initio_manager, daemon=True).start()
        self.first_wait_after_restart = {
            idx: True for idx in range(self.config.num_trajectories)
        }

        if self.config.analysis:
            logging.info("Launching analysis manager thread for PARSL.")
            self.analysis_queue = queue.Queue()
            self.analysis_kill_thread = False
            self.analysis_done = False
            self.analysis_results = {}
            self.analysis_counter = {
                idx: 0 for idx in range(self.config.num_trajectories)
            }
            threading.Thread(
                target=self._analysis_manager, daemon=True
            ).start()

    def _handle_dft_call(self, idx: int):
        logging.info(f"Trajectory worker {idx} is sending point to DFT.")
        self.state_manager.trajectory_status[idx] = "waiting"
        self.state_manager.num_workers_training += 1
        self.ab_initio_queue.put((idx, self.trajectories[idx]))
        logging.info(f"Trajectory worker {idx} is waiting for job to finish.")

    def ab_initio_manager(self):
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
                futures[idx][curr_job_no] = recalc_aims_parsl(
                    positions=data.get_positions(),
                    species=data.get_chemical_symbols(),
                    cell=data.get_cell(),
                    pbc=data.pbc,
                    aims_settings=self.calc_manager.aims_settings,
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
                        # if the result is None, it means the DFT calculation did not converge
                        self.ab_intio_results[job_idx] = False
                    else:
                        # the DFT calculation converged
                        self.ab_intio_results[job_idx] = temp_result
                        logging.info(
                            f"DFT calculation number {job_no} for worker {job_idx} finished."
                        )
                    # remove the job from the futures dict to avoid double counting
                    del futures[job_idx][job_no]
                    # remove folder with results
                    if self.clean_dirs:
                        try:
                            shutil.rmtree(
                                self.calc_dir / f"worker_{job_idx}_no_{job_no}"
                            )
                        except FileNotFoundError:
                            logging.warning(
                                f"Directory {self.calc_dir / f'worker_{job_idx}_{job_no}'} not found. Skipping removal."
                            )

    def _setup_aims_calculator(self, atoms):
        pass

    def _recalc_aims(self, current_point):
        pass

    def _finalize_ab_initio(self):
        with threading.Lock():
            self.kill_thread = True
            if self.config.analysis:
                self.analysis_kill_thread = True
                while not self.analysis_done:
                    time.sleep(0.1)
        time.sleep(5)
        parsl.dfk().cleanup()

    def _waiting_task(self, idx):

        if self.config.restart and self.first_wait_after_restart[idx]:
            # if the worker is waiting and we just restarted the
            # procedure, we have to relaunch the dft job and then
            # leave the function
            logging.info(f"Worker {idx} is restarting DFT job after restart.")
            self._handle_dft_call(idx)
            self.first_wait_after_restart[idx] = False
            return None

        # with self.results_lock:
        job_result = self.ab_intio_results.get(idx, "not_done")

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
            with self.results_lock:
                # remove the job from the results dict to avoid double counting
                del self.ab_intio_results[idx]

    def _analysis_manager(self):
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
                    "Analysis manager kill switch triggered. Waiting for pending analysis jobs..."
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
                    futures[idx][current_idx] = recalc_aims_parsl(
                        positions=data.get_positions(),
                        species=data.get_chemical_symbols(),
                        cell=data.get_cell(),
                        pbc=data.pbc,
                        aims_settings=self.calc_manager.aims_settings,
                        directory=self.calc_dir
                        / f"worker_analysis{idx}_no_{current_idx}",
                        properties=self.config.properties,
                        ase_aims_command=self.launch_str,
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
                        f"SCF during analysis for worker {job_idx} no {job_no} failed. Discarding point."
                    )
                else:
                    analysis_forces = predicted_forces[job_idx][job_no]
                    true_forces = temp_result["forces"]
                    check_results = self.analysis_manager.analysis_check(
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
                        self.analysis_manager.save_analysis()
                # Remove only the completed job
                del futures[job_idx][job_no]
                del predicted_forces[job_idx][job_no]
                del current_md_steps[job_idx][job_no]
                if self.clean_dirs:
                    try:
                        shutil.rmtree(
                            self.calc_dir
                            / f"worker_analysis{job_idx}_no_{job_no}"
                        )
                    except FileNotFoundError:
                        logging.warning(
                            f"Directory {self.calc_dir / f'worker_analysis{job_idx}_no_{job_no}'} not found. Skipping removal."
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
        # Dummy to overwrite the parent method
        # processing is done in the analysis manager thread
        return
