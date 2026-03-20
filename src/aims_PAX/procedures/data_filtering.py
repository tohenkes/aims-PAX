"""
Main data-filtering procedure.

DataFilteringProcedure orchestrates:
  1. Model creation / restart loading
  2. PARSL or local worker submission (DFWorkerManager)
  3. Batch evaluation loop: update threshold → add points → train
  4. Saving the filtered dataset (HDF5 always, XYZ optional)
  5. Optional final convergence pass
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch

from so3krates_torch.data.hdf5_utils import save_atoms_to_hdf5
from so3krates_torch.data.utils import KeySpecification

from aims_PAX.procedures.df_managers import (
    DFDataManager,
    DFThresholdManager,
    DFTrainingManager,
    DFWorkerManager,
)
from aims_PAX.procedures.df_preparation import (
    DFConfiguration,
    DFModelManager,
    DFRestart,
    DFStateManager,
)
from aims_PAX.tools.utilities.data_handling import save_datasets
from aims_PAX.tools.utilities.utilities import (
    log_yaml_block,
    setup_logger,
)


# ---------------------------------------------------------------------------
# DataFilteringProcedure
# ---------------------------------------------------------------------------


class DataFilteringProcedure:
    """
    Orchestrates the data-filtering workflow.

    Usage::

        df = DataFilteringProcedure(model_settings, aimsPAX_settings)
        if not df.check_df_done():
            df.run()
        if df.config.converge_best:
            df.converge()
    """

    def __init__(
        self,
        model_settings: dict,
        aimsPAX_settings: dict,
    ):
        # ---------------------------------------------------------------
        # 1. Configuration
        # ---------------------------------------------------------------
        self.config = DFConfiguration(
            model_settings=model_settings,
            aimsPAX_settings=aimsPAX_settings,
        )
        config = self.config

        # ---------------------------------------------------------------
        # 2. Logging
        # ---------------------------------------------------------------
        setup_logger(
            tag="data_filtering",
            directory=config.log_dir,
        )
        log_yaml_block("model_settings", model_settings)
        log_yaml_block("aimsPAX_settings", aimsPAX_settings)
        logging.info("Data-filtering procedure initialised.")

        # ---------------------------------------------------------------
        # 3. HDF5 dataset sizes
        # ---------------------------------------------------------------
        self._dataset_sizes: List[int] = []
        for hdf5_path in config.hdf5_paths:
            with h5py.File(hdf5_path, "r") as f:
                n = int(f.attrs.get("num_configs", 0))
            if n == 0:
                raise ValueError(
                    f"HDF5 dataset '{hdf5_path}' has zero configs "
                    f"(missing or zero 'num_configs' attribute)."
                )
            self._dataset_sizes.append(n)
            logging.info(f"Dataset {hdf5_path}: {n} configs.")

        # ---------------------------------------------------------------
        # 4. State, model, restart managers
        # ---------------------------------------------------------------
        self.state_manager = DFStateManager(config)
        self.model_manager = DFModelManager(config)
        self.restart_manager = DFRestart(config, self.state_manager)

        # ---------------------------------------------------------------
        # 5. PARSL setup (if cluster_settings provided)
        # ---------------------------------------------------------------
        self._parsl_settings = None
        if config.cluster_settings is not None:
            import parsl
            from aims_PAX.tools.utilities.parsl_utils import (
                handle_parsl_logger,
                prepare_parsl,
            )

            self._parsl_settings = prepare_parsl(
                cluster_settings=config.cluster_settings,
                output_dir=config.output_dir,
            )
            try:
                parsl.dfk()
                logging.info(
                    "PARSL already initialised; reusing existing context."
                )
            except parsl.errors.NoDataFlowKernelError:
                handle_parsl_logger(
                    log_dir=Path(config.log_dir) / "parsl_df.log"
                )
                parsl.load(self._parsl_settings["config"])
                logging.info(
                    "PARSL loaded for data-filtering workers."
                )

        # ---------------------------------------------------------------
        # 6. Model setup (scan datasets, create model)
        # ---------------------------------------------------------------
        self.model_manager.scan_datasets()

        if config.restart:
            # Restore state from checkpoint (model weights loaded later)
            self.restart_manager.load()
            logging.info("State restored from restart checkpoint.")
        else:
            # Fresh run: initialise workers
            self.state_manager.initialize_workers(self._dataset_sizes)

        self.model_manager.setup_model()

        # Load model weights from best checkpoint if restarting
        if config.restart:
            _best_ckpt = self._find_best_checkpoint()
            if _best_ckpt is not None:
                self.model_manager.load_from_checkpoint(_best_ckpt)

        # Reload collected atoms from incremental HDF5 if available.
        # Must come after setup_model() (z_table / model sets ready)
        # and after load_from_checkpoint() (model weights restored).
        if config.restart and self.state_manager.total_points_added > 0:
            self._reload_filtered_dataset()

        # ---------------------------------------------------------------
        # 7. Manager classes
        # ---------------------------------------------------------------
        self.worker_manager = DFWorkerManager(
            config=config,
            state_manager=self.state_manager,
            model_manager=self.model_manager,
        )
        self.data_manager = DFDataManager(
            config=config,
            state_manager=self.state_manager,
            model_manager=self.model_manager,
        )
        self.training_manager = DFTrainingManager(
            config=config,
            model_manager=self.model_manager,
            state_manager=self.state_manager,
            restart_manager=self.restart_manager,
        )
        self.threshold_manager = DFThresholdManager(
            config=config,
            state_manager=self.state_manager,
        )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def check_df_done(self) -> bool:
        """Return True if data-filtering is already complete (restart)."""
        return self.restart_manager.df_done

    def run(self) -> None:
        """
        Main data-filtering loop.

        Submits batch evaluation jobs, collects results, updates
        the threshold, adds exceeding points, trains the model, checks
        stopping criteria, and saves restart checkpoints.
        """
        config = self.config
        sm = self.state_manager
        wm = self.worker_manager
        dm = self.data_manager
        tm = self.training_manager
        rm = self.restart_manager

        logging.info("Starting data-filtering main loop.")

        # Save initial model for workers
        wm.save_model_for_workers()

        max_reached = False

        while not max_reached:
            # ----------------------------------------------------------
            # Inner loop: process one full pass through the dataset
            # ----------------------------------------------------------
            while not wm.all_done() and not max_reached:
                # 1. Submit jobs for idle workers
                wm.submit_batch_jobs()

                # 2. Collect completed results (poll with a short sleep)
                completed = wm.collect_completed()

                if not completed:
                    time.sleep(0.5)
                    continue

                for worker_id, dataset_idx, result in completed:
                    exceeding = result.get("exceeding_indices", [])
                    batch_errors = result.get("batch_errors", [])
                    mean_err = result.get("mean_batch_error", 0.0)

                    logging.info(
                        f"Worker {worker_id} (dataset {dataset_idx}): "
                        f"{len(exceeding)}/{len(batch_errors)} points exceeded "
                        f"threshold. Mean batch error: {mean_err:.6f}"
                    )

                    # 3. Update threshold
                    if batch_errors:
                        self.threshold_manager.update_threshold(
                            dataset_idx, batch_errors
                        )

                    # 4. Add exceeding points to dataset
                    if exceeding:
                        max_reached = dm.load_and_add_points(
                            dataset_idx, exceeding
                        )

                        # 5. Prepare dataloaders & train
                        if sm.train_points_added > 0:
                            dm.prepare_dataloaders()
                            tm.perform_training()

                            # Check desired accuracy after training
                            if (
                                sm.current_valid_error
                                <= config.desired_accuracy
                                and config.desired_accuracy > 0.0
                            ):
                                logging.info(
                                    f"Desired accuracy "
                                    f"{config.desired_accuracy} reached "
                                    f"(valid error = "
                                    f"{sm.current_valid_error:.6f}). "
                                    "Stopping."
                                )
                                max_reached = True

                            # Save updated model for workers
                            wm.save_model_for_workers()

                            # Incrementally persist collected dataset
                            if config.create_restart:
                                self._save_incremental_dataset()

                            # Analysis bookkeeping
                            if config.analysis:
                                self._record_analysis(mean_err)

                        if max_reached:
                            logging.info(
                                "Max train set size reached. "
                                "Stopping workers."
                            )
                            break

                    # 6. Save restart checkpoint after each batch
                    if config.create_restart:
                        rm.save()

                # Resubmit jobs (handles workers that just finished)
                if not max_reached:
                    wm.submit_batch_jobs()

            # ----------------------------------------------------------
            # End of pass — check whether to loop or stop
            # ----------------------------------------------------------
            if max_reached:
                break

            # Targets not met; decide whether to loop through data again
            if not config.loop_exhausted_data:
                logging.warning(
                    f"Data exhausted after pass {sm.current_pass}: all "
                    "dataset chunks evaluated but neither the desired "
                    "accuracy nor the max training set size was reached. "
                    f"Final validation error: "
                    f"{sm.current_valid_error:.6f} "
                    f"(target: {config.desired_accuracy}), "
                    f"training points collected: {sm.train_points_added} "
                    f"(target: {config.max_train_set_size})."
                )
                break

            sm.current_pass += 1
            if (
                config.max_data_passes > 0
                and sm.current_pass > config.max_data_passes
            ):
                logging.warning(
                    f"Data exhausted after {config.max_data_passes} "
                    "pass(es) without reaching targets. "
                    f"Final validation error: "
                    f"{sm.current_valid_error:.6f} "
                    f"(target: {config.desired_accuracy}), "
                    f"training points collected: {sm.train_points_added} "
                    f"(target: {config.max_train_set_size})."
                )
                break

            logging.info(
                f"Data exhausted (pass {sm.current_pass - 1} complete). "
                f"Re-shuffling dataset for pass {sm.current_pass}."
            )
            sm.initialize_workers(
                self._dataset_sizes, pass_num=sm.current_pass
            )
            if config.create_restart:
                rm.save()

        # Mark run complete and save final checkpoint
        rm.df_done = True
        if config.create_restart:
            rm.save()

        logging.info(
            f"Data-filtering loop complete. "
            f"Total points collected: {sm.total_points_added} "
            f"(train: {sm.train_points_added}, "
            f"valid: {sm.valid_points_added})."
        )

        wm.shutdown()
        self._finalize()

    def converge(self) -> None:
        """Final convergence training on the collected filtered dataset."""
        if self.state_manager.train_points_added == 0:
            logging.warning("No training data collected; skipping convergence.")
            return

        tag = "model_seed_1"
        ase_sets = self.model_manager.ensemble_ase_sets.get(tag, {})
        if not ase_sets.get("train"):
            # Atoms are not in memory (e.g. restart with df_done=True).
            # Reload them from the saved filtered HDF5 files.
            self._reload_filtered_dataset()

        # Ensure dataloaders are ready
        self.data_manager.prepare_dataloaders()
        self.training_manager.converge()

        # Save converged model
        self._save_final_model()
        logging.info("Converged model saved.")

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _reload_filtered_dataset(self) -> None:
        """
        Reload the saved filtered HDF5 dataset(s) back into memory.

        Called when converge() is invoked after a restart with df_done=True,
        where the training atoms are no longer in memory. Mirrors how AL
        reloads its dataset via load_ensemble_sets_from_folder +
        ase_to_model_ensemble_sets.
        """
        from so3krates_torch.data.hdf5_utils import load_atoms_from_hdf5
        from aims_PAX.tools.utilities.data_handling import (
            ase_to_model_ensemble_sets,
        )

        config = self.config
        mm = self.model_manager
        tag = "model_seed_1"
        dataset_dir = config.dataset_dir

        if config.use_multihead_model:
            hdf5_path = dataset_dir / "filtered_combined.h5"
        else:
            hdf5_path = dataset_dir / "filtered_dataset.h5"

        if not hdf5_path.exists():
            logging.warning(
                f"Filtered dataset not found at {hdf5_path}; "
                "cannot reload for convergence."
            )
            return

        logging.info(f"Reloading filtered dataset from {hdf5_path}.")
        all_atoms = load_atoms_from_hdf5(str(hdf5_path))

        # Split into train / valid using the same ratio as the original run
        n = len(all_atoms)
        n_valid = max(1, int(np.round(n * config.valid_ratio)))
        n_train = n - n_valid

        mm.ensemble_ase_sets[tag]["train"] = all_atoms[:n_train]
        mm.ensemble_ase_sets[tag]["valid"] = all_atoms[n_train:]

        mm.ensemble_model_sets = ase_to_model_ensemble_sets(
            ensemble_ase_sets=mm.ensemble_ase_sets,
            z_table=mm.z_table,
            r_max=config.r_max,
            r_max_lr=config.r_max_lr,
            key_specification=config.key_specification,
            all_heads=config.all_heads,
            seed=config.seed,
        )

        logging.info(
            f"Reloaded {n_train} train + {n_valid} valid structures "
            "for convergence."
        )

    def _save_incremental_dataset(self) -> None:
        """
        Write currently-collected atoms to the output HDF5 file(s).

        Called after each training step (when create_restart=True) so
        the filtered dataset is always recoverable from disk after a
        crash.  Uses the same paths as _finalize(); the final write is
        an overwrite of identical data.
        """
        config = self.config
        mm = self.model_manager
        tag = "model_seed_1"
        ase_sets = mm.ensemble_ase_sets[tag]

        all_atoms = ase_sets["train"] + ase_sets["valid"]
        if not all_atoms:
            return

        ks = config.key_specification
        save_keyspec = KeySpecification(
            info_keys={
                k: v for k, v in ks.info_keys.items() if k != "head"
            },
            arrays_keys=dict(ks.arrays_keys),
        )

        dataset_dir = config.dataset_dir
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if config.use_multihead_model:
            self._save_multihead_hdf5(all_atoms, dataset_dir, save_keyspec)
        else:
            output_path = str(dataset_dir / "filtered_dataset.h5")
            save_atoms_to_hdf5(
                atoms_iter=all_atoms,
                output_path=output_path,
                key_specification=save_keyspec,
            )
        logging.debug(
            f"Incremental dataset saved: {len(all_atoms)} structures."
        )

    def _finalize(self) -> None:
        """
        Save the filtered dataset.

        Always saves per-dataset HDF5 files (and optionally a combined
        one for multi-head runs).
        When config.save_xyz is True, also saves XYZ.
        """
        config = self.config
        mm = self.model_manager
        tag = "model_seed_1"
        ase_sets = mm.ensemble_ase_sets[tag]

        all_atoms = ase_sets["train"] + ase_sets["valid"]

        if not all_atoms:
            logging.warning("No data collected; nothing to save.")
            return

        # Write HDF5 (reuses the same logic as incremental saving)
        self._save_incremental_dataset()
        logging.info(
            f"Filtered dataset saved to {config.dataset_dir} "
            f"({len(all_atoms)} structures)."
        )

        # Optional XYZ output
        if config.save_xyz:
            xyz_dir = dataset_dir / "xyz"
            save_datasets(
                ensemble=mm.ensemble,
                ensemble_ase_sets=mm.ensemble_ase_sets,
                path=xyz_dir,
            )
            logging.info(f"XYZ datasets saved to {xyz_dir}.")

        # Save final model
        self._save_final_model()
        logging.info("Final model saved.")

    def _save_multihead_hdf5(
        self,
        all_atoms: list,
        dataset_dir: Path,
        save_keyspec: KeySpecification,
    ) -> None:
        """
        For multi-head runs: save one HDF5 per source dataset
        (based on atoms.info["head"]) and a combined HDF5.
        """
        config = self.config

        # Group atoms by head
        per_head: dict = {h: [] for h in config.all_heads}
        for atoms in all_atoms:
            head_name = atoms.info.get("head", "head_0")
            if head_name in per_head:
                per_head[head_name].append(atoms)
            else:
                per_head.setdefault(head_name, []).append(atoms)

        total = 0
        for head_name, atoms_list in per_head.items():
            if not atoms_list:
                continue
            ds_idx = int(head_name.split("_")[-1])
            src_stem = Path(config.hdf5_paths[ds_idx]).stem
            output_path = str(dataset_dir / f"filtered_{src_stem}.h5")
            save_atoms_to_hdf5(
                atoms_iter=atoms_list,
                output_path=output_path,
                key_specification=save_keyspec,
            )
            logging.info(
                f"  {head_name}: {len(atoms_list)} structures → "
                f"{output_path}"
            )
            total += len(atoms_list)

        # Combined HDF5
        combined_path = str(dataset_dir / "filtered_combined.h5")
        save_atoms_to_hdf5(
            atoms_iter=all_atoms,
            output_path=combined_path,
            key_specification=save_keyspec,
        )
        logging.info(
            f"Combined filtered dataset: {total} structures → "
            f"{combined_path}"
        )

    def _save_final_model(self) -> None:
        """Save each model in the ensemble to model_dir as <tag>.model."""
        mm = self.model_manager
        model_dir = Path(self.config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        for tag, model in mm.ensemble.items():
            out_path = model_dir / f"{tag}.model"
            torch.save(model, str(out_path))
            logging.debug(f"Model saved: {out_path}")

    def _find_best_checkpoint(self) -> Optional[str]:
        """Return path to the latest .pt checkpoint, or None."""
        ckpt_dir = Path(self.config.checkpoints_dir)
        if not ckpt_dir.exists():
            return None
        pts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        return str(pts[-1]) if pts else None

    def _record_analysis(self, mean_batch_error: float) -> None:
        """Append current metrics to analysis buffers."""
        sm = self.state_manager
        if not hasattr(sm, "collect_losses"):
            return
        sm.collect_losses["epoch"].append(sm.total_epoch)
        sm.collect_losses["avg_losses"].append(mean_batch_error)
        sm.collect_losses["ensemble_losses"].append(sm.current_valid_error)
        sm.collect_thresholds.append(
            sm.threshold
            if not self.config.use_multihead_model
            else dict(sm.threshold)
        )
