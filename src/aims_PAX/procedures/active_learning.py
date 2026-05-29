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
    ALDFTManagerSerial,
    ALDFTReferenceManagerPARSL,
    ALTeacherModelManagerPARSL,
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
from ..settings import AimsPAXSettings, ModelSettings

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
        model_settings: ModelSettings,
        aimsPAX_settings: AimsPAXSettings,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
    ):

        super().__init__(
            model_settings=model_settings,
            aimsPAX_settings=aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
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
            mlff_manager=self.mlff_manager,
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
                logging.info("All trajectories reached maximum MD steps.")
                break

            if (
                self.ensemble_manager.train_dataset_len
                >= self.config.max_train_set_size
            ):
                logging.info("Maximum size of training set reached.")
                break

            if (
                self.state_manager.current_valid_error
                < self.config.desired_accuracy
            ):
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
            self.point = self.trajectories[idx].copy()

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
        self.ensemble_manager.ensemble_model_sets = (
            self.train_manager.prepare_training(
                model_sets=self.ensemble_manager.ensemble_model_sets
            )
        )

        logging.info(f"Trajectory worker {idx} is training.")
        self.train_manager.perform_training(idx)

        # update calculators with the new models
        self._assign_models_to_trajectories()

        if (
            self.state_manager.trajectory_total_epochs[idx]
            >= self.config.epochs_per_worker
        ):
            self.state_manager.trajectory_status[idx] = "running"
            self.state_manager.num_workers_training -= 1
            self.state_manager.trajectory_total_epochs[idx] = 0
            logging.info(f"Trajectory worker {idx} finished training.")

    def _assign_models_to_trajectories(self):
        
        if self.config.use_foundational:
            # updating only the ensemble calculator which is used for 
            # uncertainty estimation
            self.mlff_manager.mlff_calc_ensemble.models = [
                self.ensemble[tag] for tag in self.ensemble.keys()
            ]
        else:
            # updating the models that are used to propagate the MD
            for trajectory in self.trajectories.values():
                trajectory.calc.models = [
                    self.ensemble[tag] for tag in self.ensemble.keys()
                ]

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

        self.point = point
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

        logging.info("Starting active learning procedure.")

        self._al_loop()

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
            model_dir=self.config.model_settings.GENERAL.model_dir,
            current_epoch=self.state_manager.total_epoch,
            model_settings=self.config.model_settings.ARCHITECTURE,
            model_choice=self.config.model_choice
        )

        # save final results in new directory called results:
        if not self.config.converge_al:
            results_dir = self.config.output_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            save_datasets(
                ensemble=self.ensemble,
                ensemble_ase_sets=self.ensemble_manager.ensemble_ase_sets,
                path=results_dir,
            )
            save_models(
                ensemble=self.ensemble,
                training_setups=self.ensemble_manager.training_setups,
                model_dir=results_dir,
                current_epoch=self.state_manager.total_epoch,
                model_settings=self.config.model_settings.ARCHITECTURE,
                model_choice=self.config.model_choice
            )
        #  else: handled in convergence call

        if self.config.analysis:
            self.analysis_manager.save_analysis()

        if self.config.create_restart:
            self.restart_manager.update_restart_dict(
                trajectories_keys=self.trajectories.keys(),
                md_drivers=self.md_manager.md_drivers,
                save_restart=self.config.al_restart_path,
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
        model_settings: dict,
        aimsPAX_settings: dict,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
    ):

        super().__init__(
            model_settings=model_settings,
            aimsPAX_settings=aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )
        self.data_manager = ALDataManager(
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            state_manager=self.state_manager,
        )

        self.train_manager = ALTrainingManager(
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            mlff_manager=self.mlff_manager,
            state_manager=self.state_manager,
            md_manager=self.md_manager,
            restart_manager=self.restart_manager,
        )
        self.converge = self.train_manager.converge

        self.dft_manager = ALDFTManagerSerial(
            path_to_control=path_to_control,
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            state_manager=self.state_manager,
            data_manager=self.data_manager,
            path_to_geometry=path_to_geometry,
        )

        self.analysis_manager = ALAnalysisManager(
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            dft_manager=self.dft_manager,
            state_manager=self.state_manager,
            md_manager=self.md_manager,
        )

        self.run_manager = ALRunningManager(
            config=self.config,
            state_manager=self.state_manager,
            ensemble_manager=self.ensemble_manager,
            mlff_manager=self.mlff_manager,
            dft_manager=self.dft_manager,
        )
    
    def _waiting_task(self, idx):
        pass

class ALProcedurePARSL(ALProcedure):
    """
    Implementation of the active learning procedure using PARSL.
    This class handles the DFT calculations using PARSL via
    threads. Multiple DFT calculations can be run in parallel
    with ML tasks running concurrently with DFT.
    """

    def __init__(
        self,
        model_settings: ModelSettings,
        aimsPAX_settings: AimsPAXSettings,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
    ):

        if parsl is None:
            raise ImportError(
                "PARSL is not installed. Please install PARSL"
                " to use this feature."
            )
        super().__init__(
            model_settings=model_settings,
            aimsPAX_settings=aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )

        logging.info("Using following settings for the HPC environment:")
        log_yaml_block("CLUSTER:", self.config.aimsPAX_settings.CLUSTER.model_dump())
        self.data_manager = ALDataManager(
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            state_manager=self.state_manager,
        )

        self.train_manager = ALTrainingManager(
            config=self.config,
            ensemble_manager=self.ensemble_manager,
            mlff_manager=self.mlff_manager,
            state_manager=self.state_manager,
            md_manager=self.md_manager,
            restart_manager=self.restart_manager,
        )
        self.converge = self.train_manager.converge

        if self.config.use_teacher_reference:
            self.reference_manager = ALTeacherModelManagerPARSL(
                teacher_reference_settings=self.config.teacher_reference_settings,
                config=self.config,
                ensemble_manager=self.ensemble_manager,
                state_manager=self.state_manager,
            )
        else:
            self.reference_manager = ALDFTReferenceManagerPARSL(
                path_to_control=path_to_control,
                config=self.config,
                ensemble_manager=self.ensemble_manager,
                state_manager=self.state_manager,
            )
        # Backward-compatible alias
        self.dft_manager = self.reference_manager

        if self.config.analysis:
            self.analysis_manager = ALAnalysisManagerPARSL(
                config=self.config,
                ensemble_manager=self.ensemble_manager,
                dft_manager=self.reference_manager,
                state_manager=self.state_manager,
                md_manager=self.md_manager,
            )
        self.run_manager = ALRunningManager(
            config=self.config,
            state_manager=self.state_manager,
            ensemble_manager=self.ensemble_manager,
            mlff_manager=self.mlff_manager,
            dft_manager=self.reference_manager,
        )

    def _waiting_task(self, idx):

        if self.config.restart and self.first_wait_after_restart[idx]:
            # if the worker is waiting and we just restarted the
            # procedure, we have to relaunch the reference job and
            # then leave the function
            logging.info(
                f"Worker {idx} is restarting reference job after restart."
            )
            self.reference_manager.handle_reference_call(
                point=self.trajectories[idx], idx=idx
            )
            self.first_wait_after_restart[idx] = False
            return None

        job_result = self.reference_manager.reference_results.get(
            idx, "not_done"
        )

        if job_result == "not_done":
            # if the job is not done, we return None and
            # wait for the next iteration
            return None
        else:
            if not job_result:
                logging.info(
                    f"Reference calculation failed at worker {idx}. "
                    "Discarding point and restarting MD from last checkpoint."
                )
                
                self.trajectories[idx] = atoms_full_copy(
                    self.state_manager.MD_checkpoints[idx]
                )
                self.trajectories[idx].calc = (
                    self.mlff_manager.mlff_calc
                )
                self.md_manager.md_drivers[idx].atoms = (
                    self.trajectories[idx]
                )
                self.state_manager.trajectory_status[idx] = "running"

            else:
                logging.info(f"Worker {idx} received a point.")
                received_point = self.trajectories[idx].copy()
                received_point.info["REF_energy"] = job_result["energy"]
                received_point.arrays["REF_forces"] = job_result["forces"]
                if self.config.compute_stress:
                    received_point.info["REF_stress"] = job_result["stress"]
                if "hirshfeld_ratios" in job_result:
                    received_point.arrays["REF_hirshfeld_ratios"] = (
                        job_result["hirshfeld_ratios"]
                    )
                if "hirshfeld_charges" in job_result:
                    received_point.arrays["REF_charges"] = (
                        job_result["hirshfeld_charges"]
                    )

                if self.config.update_md_checkpoints:
                    self.state_manager.MD_checkpoints[idx] = atoms_full_copy(
                        received_point
                    )

                self.data_manager.handle_received_point(
                    idx=idx, received_point=received_point
                )
            with self.reference_manager.results_lock:
                # remove the job from the results dict
                del self.reference_manager.reference_results[idx]

    def _al_loop(self):
        super()._al_loop()
        # because for PARSL the analysis runs on a separate thread,
        # we have to finalize the analysis here explictly
        if self.config.analysis:
            self.analysis_manager.finalize_analysis()
