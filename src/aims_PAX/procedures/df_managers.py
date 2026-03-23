"""
Manager classes for the data-filtering procedure.

DFThresholdManager — adapts the per-(dataset/head) error threshold.
DFDataManager      — loads points from HDF5 and adds to train/valid sets.
DFTrainingManager  — wraps TrainingOrchestrator for single-model training.
DFWorkerManager    — submits and collects PARSL or ThreadPool batch jobs.
"""

import concurrent.futures
import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from so3krates_torch.data.hdf5_utils import load_atoms_from_hdf5
from so3krates_torch.data.utils import config_from_atoms

from aims_PAX.procedures.al_managers import (
    TrainingOrchestrator,
    TrainingSession,
)
from aims_PAX.procedures.df_preparation import (
    DFConfiguration,
    DFModelManager,
    DFRestart,
    DFStateManager,
)
from aims_PAX.tools.uncertainty import get_threshold
from aims_PAX.tools.utilities.data_handling import (
    create_dataloader,
    create_model_dataset,
)
from aims_PAX.tools.utilities.utilities import save_checkpoint

# ---------------------------------------------------------------------------
# DFThresholdManager
# ---------------------------------------------------------------------------


class DFThresholdManager:
    """
    Maintains adaptive thresholds based on a moving average of
    validation errors (updated after each training cycle).

    For single-dataset runs: self.state_manager.threshold is a scalar.
    For multi-head runs: self.state_manager.threshold is a dict keyed by
    head name (e.g. "head_0", "head_1", …).
    """

    def __init__(
        self,
        config: DFConfiguration,
        state_manager: DFStateManager,
    ):
        self.config = config
        self.state_manager = state_manager

    def update_from_validation(self):
        """
        Update the threshold(s) from the current validation error(s)
        obtained after training.  Appends the latest validation MAE to
        a history and recomputes the threshold as a moving average
        scaled by (1 + c_x).
        """
        sm = self.state_manager

        # Freeze check
        if sm.total_points_added >= self.config.freeze_threshold_dataset:
            logging.debug("Threshold frozen.")
            return

        _log = logging.debug if self.config.compact_logging else logging.info

        if self.config.use_multihead_model:
            for head_name in self.config.all_heads:
                mae = sm.current_valid_errors_per_head.get(head_name, None)
                if mae is None:
                    continue
                sm.validation_error_history[head_name].append(mae)
                new_threshold = get_threshold(
                    sm.validation_error_history[head_name],
                    c_x=self.config.c_x,
                )
                sm.threshold[head_name] = float(new_threshold)
                _log(
                    f"Threshold updated for {head_name}: "
                    f"{sm.threshold[head_name]:.6f} "
                    f"(from {len(sm.validation_error_history[head_name])} "
                    f"validation samples)"
                )
        else:
            mae = sm.current_valid_error
            if mae == np.inf:
                return
            sm.validation_error_history.append(mae)
            new_threshold = get_threshold(
                sm.validation_error_history,
                c_x=self.config.c_x,
            )
            sm.threshold = float(new_threshold)
            _log(
                f"Threshold updated: {sm.threshold:.6f} "
                f"(from {len(sm.validation_error_history)} "
                f"validation samples)"
            )

    def get_threshold_for_dataset(self, dataset_idx: int) -> float:
        """Return the current threshold for the given dataset."""
        if self.config.use_multihead_model:
            head_name = f"head_{dataset_idx}"
            t = self.state_manager.threshold[head_name]
            return float(t) if t != np.inf else np.inf
        else:
            t = self.state_manager.threshold
            return float(t) if t != np.inf else np.inf


# ---------------------------------------------------------------------------
# DFDataManager
# ---------------------------------------------------------------------------


class DFDataManager:
    """
    Loads atoms from HDF5 by index and adds them to the model dataset.

    Single-dataset: flat train / valid lists under "Default" head.
    Multi-head:     head_name = "head_{dataset_idx}"; combined train,
                    per-head valid.
    """

    def __init__(
        self,
        config: DFConfiguration,
        state_manager: DFStateManager,
        model_manager: DFModelManager,
    ):
        self.config = config
        self.state_manager = state_manager
        self.model_manager = model_manager

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def load_and_add_points(
        self,
        dataset_idx: int,
        indices: List[int],
    ) -> bool:
        """
        Load atoms from HDF5[dataset_idx] by the given indices, convert
        to model format, and append to train/valid sets.

        Returns:
            True if max_train_set_size has been reached, False otherwise.
        """
        hdf5_path = self.config.hdf5_paths[dataset_idx]
        head_name = (
            f"head_{dataset_idx}"
            if self.config.use_multihead_model
            else "Default"
        )

        atoms_list = load_atoms_from_hdf5(hdf5_path, index=indices)
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]

        for atoms in atoms_list:
            atoms.info["head"] = head_name
            max_reached = self._add_single_point(atoms, head_name)
            if max_reached:
                return True

        sm = self.state_manager
        logging.info(
            f"Dataset {dataset_idx}: added {len(atoms_list)} point(s). "
            f"Total: train={sm.train_points_added},"
            f" valid={sm.valid_points_added}."
        )
        return False

    def prepare_dataloaders(self) -> None:
        """
        Build PyTorch dataloaders from the current model sets and store
        them back into ensemble_model_sets under "train_loader" /
        "valid_loader" (or "train_subset" / "valid_subset").
        """
        config = self.config
        mm = self.model_manager
        tag = "model_seed_1"
        model_set = mm.ensemble_model_sets[tag]

        train_set = model_set["train"]
        valid_set = model_set["valid"]

        if not train_set:
            logging.warning("prepare_dataloaders: training set is empty.")
            return

        # Ensure every head has at least one valid point
        valid_set = self._ensure_valid_not_empty(valid_set, train_set)

        valid_batch_size = config.set_valid_batch_size

        if config.replay_strategy == "random_subset":
            assert config.train_subset_size is not None, (
                "train_subset_size must be set when "
                "replay_strategy='random_subset'."
            )
            train_n = min(config.train_subset_size, len(train_set))
            sampled_train = random.sample(model_set["train"], train_n)

            set_valid_size = (
                config.valid_subset_size
                if config.valid_subset_size is not None
                else float("inf")
            )
            sampled_valid = {}
            for head_name, head_data in valid_set.items():
                valid_n = min(int(set_valid_size), len(head_data))
                sampled_valid[head_name] = random.sample(head_data, valid_n)

            batch_size = min(config.set_batch_size, train_n)
            train_loader, valid_loaders = create_dataloader(
                sampled_train, sampled_valid, batch_size, valid_batch_size
            )
            model_set["train_subset"] = {0: train_loader}
            model_set["valid_subset"] = {0: valid_loaders}
            _log = logging.debug if config.compact_logging else logging.info
            _log(
                f"Dataloaders prepared (random_subset): "
                f"{train_n}/{len(train_set)} train,"
                f" batch_size={batch_size}."
            )
            for head_name, head_data in valid_loaders.items():
                _log(
                    f'Validation set for head "{head_name}" has '
                    f"{len(sampled_valid[head_name])} point(s) "
                    f"with {len(head_data)} batch(es)."
                )
        else:
            batch_size = min(config.set_batch_size, len(train_set))
            train_loader, valid_loaders = create_dataloader(
                train_set, valid_set, batch_size, valid_batch_size
            )
            model_set["train_loader"] = train_loader
            model_set["valid_loader"] = valid_loaders
            n_valid = sum(len(v) for v in valid_set.values())
            _log = logging.debug if config.compact_logging else logging.info
            _log(
                f"Dataloaders prepared: {len(train_set)} train,"
                f" {n_valid} valid (batch_size={batch_size})."
            )

    def prepare_convergence_dataloaders(self) -> None:
        """
        Build full-dataset dataloaders for convergence training.

        Always uses the complete train/valid sets regardless of
        replay_strategy, and stores them under "train_loader" /
        "valid_loader" so that TrainingOrchestrator can find them.
        """
        config = self.config
        mm = self.model_manager
        tag = "model_seed_1"
        model_set = mm.ensemble_model_sets[tag]

        train_set = model_set["train"]
        valid_set = model_set["valid"]

        if not train_set:
            logging.warning(
                "prepare_convergence_dataloaders: training set is empty."
            )
            return

        valid_set = self._ensure_valid_not_empty(valid_set, train_set)

        batch_size = min(config.set_batch_size, len(train_set))
        valid_batch_size = config.set_valid_batch_size
        train_loader, valid_loaders = create_dataloader(
            train_set, valid_set, batch_size, valid_batch_size
        )
        model_set["train_loader"] = train_loader
        model_set["valid_loader"] = valid_loaders

        n_valid = sum(len(v) for v in valid_set.values())
        logging.info(
            f"Convergence dataloaders: {len(train_set)} train, "
            f"{n_valid} valid (batch_size={batch_size})."
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _add_single_point(self, atoms, head_name: str) -> bool:
        """
        Add a single atoms object to train or valid set.

        Returns True if max_train_set_size reached.
        """
        sm = self.state_manager
        config = self.config
        mm = self.model_manager
        tag = "model_seed_1"

        # Convert to model format
        model_point = create_model_dataset(
            data=[atoms],
            z_table=mm.z_table,
            seed=config.seed,
            r_max=config.r_max,
            key_specification=config.key_specification,
            r_max_lr=config.r_max_lr,
            all_heads=config.all_heads,
            head_name=head_name,
        )

        # Decide train vs valid based on ratio
        validation_quota = config.valid_ratio * sm.total_points_added
        needs_validation = sm.valid_points_added < validation_quota

        if needs_validation:
            mm.ensemble_ase_sets[tag]["valid"].append(atoms)
            mm.ensemble_model_sets[tag]["valid"][head_name] += model_point
            sm.valid_points_added += 1
        else:
            mm.ensemble_ase_sets[tag]["train"].append(atoms)
            mm.ensemble_model_sets[tag]["train"] += model_point
            sm.train_points_added += 1

            # Check max size
            if sm.train_points_added >= config.max_train_set_size:
                sm.total_points_added += 1
                return True

        sm.total_points_added += 1
        return False

    def _ensure_valid_not_empty(
        self, valid_set: dict, train_set: list
    ) -> dict:
        """
        If any head's valid set is empty, sample from train_set as fallback.
        """
        for head_name, vdata in valid_set.items():
            if len(vdata) == 0 and train_set:
                valid_set[head_name] = random.sample(
                    train_set, min(1, len(train_set))
                )
        return valid_set


# ---------------------------------------------------------------------------
# DFTrainingManager
# ---------------------------------------------------------------------------


class DFTrainingManager:
    """
    Wraps TrainingOrchestrator for single-model data-filtering training.

    Uses trajectory index 0 throughout (single trajectory).
    """

    def __init__(
        self,
        config: DFConfiguration,
        model_manager: DFModelManager,
        state_manager: DFStateManager,
        restart_manager: DFRestart,
    ):
        self.config = config
        self.model_manager = model_manager
        self.state_manager = state_manager
        self.restart_manager = restart_manager

        self.orchestrator = TrainingOrchestrator(
            config=config,
            ensemble_manager=model_manager,
            state_manager=state_manager,
            restart_manager=restart_manager,
            md_manager=None,  # not needed; orchestrator stores but never calls
        )
        self.orchestrator.save_restart = False
        # Skip XYZ writing during intermediate training — atoms loaded from
        # HDF5 may have None-valued arrays that crash ASE's atoms.copy().
        # Final output is handled by DataFilteringProcedure._finalize().
        self.orchestrator._save_training_artifacts = lambda: None

    def perform_training(self) -> None:
        """
        Train the model for config.epochs_per_batch epochs using the
        current train/valid loaders.
        """
        sm = self.state_manager
        mm = self.model_manager
        config = self.config
        tag = "model_seed_1"

        session = TrainingSession(
            training_setups=mm.training_setups,
            ensemble_model_sets=mm.ensemble_model_sets,
            max_epochs=config.epochs_per_batch,
            is_convergence=False,
        )

        for _ in range(config.epochs_per_batch):
            logger = None
            for cur_tag, model in mm.ensemble.items():
                logger = self.orchestrator.train_single_epoch(
                    session, cur_tag, model, idx=0, logger=logger
                )

            self.orchestrator.validate_and_update_state(
                session, logger, trajectory_idx=0
            )

            # Update epoch counters
            sm.trajectory_intermediate_epochs[0] += 1
            sm.trajectory_total_epochs[0] += 1
            sm.total_epoch += 1

        # Save checkpoint
        for cur_tag, model in mm.ensemble.items():
            save_checkpoint(
                mm.training_setups[cur_tag]["checkpoint_handler"],
                mm.training_setups[cur_tag],
                model,
                sm.total_epoch,
                keep_last=False,
            )

        # Reset intermediate epoch counter for next batch
        sm.trajectory_intermediate_epochs[0] = 0

    def converge(self) -> None:
        """
        Final convergence training on the collected dataset with
        patience-based early stopping.
        """
        import mace.tools as tools

        mm = self.model_manager
        config = self.config
        sm = self.state_manager
        tag = "model_seed_1"

        logging.info("Starting convergence training.")

        session = TrainingSession(
            training_setups=mm.training_setups,
            ensemble_model_sets=mm.ensemble_model_sets,
            max_epochs=config.max_convergence_epochs,
            is_convergence=True,
        )
        session.ensemble_valid_losses = {tag: np.inf}

        for j in range(config.max_convergence_epochs):
            session.current_epoch = j

            logger = None
            for cur_tag, model in mm.ensemble.items():
                logger = self.orchestrator.train_single_epoch(
                    session, cur_tag, model, logger=logger
                )

            should_stop = self.orchestrator.validate_and_update_state(
                session, logger
            )

            sm.total_epoch += 1

            if should_stop:
                logging.info(
                    f"Convergence reached after {j + 1} epochs "
                    f"(best epoch: {session.best_epoch})."
                )
                break

        logging.info("Convergence training complete.")


# ---------------------------------------------------------------------------
# DFWorkerManager
# ---------------------------------------------------------------------------


class DFWorkerManager:
    """
    Submits and collects batch evaluation jobs.

    In PARSL mode: submits evaluate_batch_parsl apps via PARSL.
    In local mode: uses concurrent.futures.ThreadPoolExecutor (or
                   ProcessPoolExecutor) calling the evaluation logic
                   directly.
    """

    def __init__(
        self,
        config: DFConfiguration,
        state_manager: DFStateManager,
        model_manager: DFModelManager,
    ):
        self.config = config
        self.state_manager = state_manager
        self.model_manager = model_manager

        self.model_save_path = str(
            Path(config.output_dir) / "df_worker_model.pt"
        )

        # Pending futures: worker_id -> future
        self._futures: dict = {}

        # Determine mode
        self._use_parsl = config.cluster_settings is not None
        self._executor: Optional[concurrent.futures.Executor] = None

        if not self._use_parsl:
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=config.num_chunks
            )

        # WorkQueue resource spec — allows concurrent tasks within a block
        self.workqueue_resource_spec = None
        if self._use_parsl:
            cores = config.cluster_settings.get("cores_per_job", None)
            if cores is not None:
                memory = config.cluster_settings.get("memory_per_job", None)
                if memory is None:
                    raise ValueError(
                        "memory_per_job must be set in CLUSTER settings "
                        "when cores_per_job is set."
                    )
                disk = config.cluster_settings.get("disk_per_job", 1000)
                self.workqueue_resource_spec = {
                    "cores": cores,
                    "memory": memory,
                    "disk": disk,
                }

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def save_model_for_workers(self) -> None:
        """Save current model to shared filesystem for workers."""
        tag = "model_seed_1"
        model = self.model_manager.ensemble[tag]
        device = torch.device(self.config.device)
        model.cpu()
        try:
            torch.save(model, self.model_save_path)
        finally:
            model.to(device)
        logging.debug(f"Model saved for workers: {self.model_save_path}")

    def submit_batch_jobs(self) -> None:
        """
        Submit evaluation jobs for all idle, non-done workers.
        """
        for worker_id in range(self.config.num_chunks):
            self.submit_worker(worker_id)

    def submit_worker(self, worker_id: int) -> None:
        """
        Submit an evaluation job for a single worker if it is idle
        and not done.
        """
        sm = self.state_manager
        config = self.config

        if sm.workers_done.get(worker_id, True):
            return
        if worker_id in self._futures:
            return  # already running

        dataset_idx = sm.worker_dataset_idx[worker_id]
        start, end = sm.worker_chunks[worker_id]
        current_offset = sm.worker_offsets[worker_id]

        if current_offset >= end:
            sm.workers_done[worker_id] = True
            return

        batch_end = min(current_offset + config.eval_stride, end)
        shuffle_map = sm.shuffle_maps[dataset_idx]
        batch_indices = shuffle_map[current_offset:batch_end].tolist()

        hdf5_path = config.hdf5_paths[dataset_idx]
        threshold = self._get_threshold(dataset_idx)
        head_index = dataset_idx
        multihead = config.use_multihead_model

        _log = logging.debug if config.compact_logging else logging.info
        _log(
            f"Worker {worker_id} (dataset {dataset_idx}): submitting"
            f" indices {current_offset}-{batch_end - 1}"
            f" (threshold={threshold:.6f})"
        )

        if self._use_parsl:
            from aims_PAX.tools.utilities.parsl_utils import (
                evaluate_batch_parsl,
            )

            kwargs = {}
            if self.workqueue_resource_spec is not None:
                kwargs["parsl_resource_specification"] = (
                    self.workqueue_resource_spec
                )
            future = evaluate_batch_parsl(
                model_path=self.model_save_path,
                hdf5_path=hdf5_path,
                batch_indices=batch_indices,
                threshold=threshold,
                r_max=float(config.r_max),
                r_max_lr=config.r_max_lr,
                error_type=config.error_type,
                energy_key=config.misc["energy_key"],
                forces_key=config.misc["forces_key"],
                z_table_zs=list(self.model_manager.z_table.zs),
                head_index=head_index,
                multihead=multihead,
                eval_batch_size=config.eval_batch_size,
                device=config.worker_device,
                **kwargs,
            )
        else:
            future = self._executor.submit(
                _evaluate_batch_local,
                model_path=self.model_save_path,
                hdf5_path=hdf5_path,
                batch_indices=batch_indices,
                threshold=threshold,
                r_max=float(config.r_max),
                r_max_lr=config.r_max_lr,
                error_type=config.error_type,
                energy_key=config.misc["energy_key"],
                forces_key=config.misc["forces_key"],
                z_table_zs=list(self.model_manager.z_table.zs),
                head_index=head_index,
                multihead=multihead,
                eval_batch_size=config.eval_batch_size,
                device=config.device,
            )

        self._futures[worker_id] = (
            future,
            dataset_idx,
            batch_indices,
            batch_end,
        )

    def collect_completed(
        self,
    ) -> List[Tuple[int, int, dict]]:
        """
        Poll futures, return completed results.

        Returns:
            List of (worker_id, dataset_idx, result_dict) for completed jobs.
        """
        sm = self.state_manager
        completed = []
        done_workers = []

        for worker_id, (future, dataset_idx, batch_indices, batch_end) in list(
            self._futures.items()
        ):
            if self._use_parsl:
                is_done = future.done()
            else:
                is_done = future.done()

            if is_done:
                try:
                    result = future.result()
                except Exception as exc:
                    logging.warning(
                        f"Worker {worker_id} failed: {exc}. " "Skipping batch."
                    )
                    result = {
                        "exceeding_indices": [],
                        "batch_errors": [],
                        "mean_batch_error": 0.0,
                        "head_index": dataset_idx,
                    }

                # Advance worker offset to the next chunk position.
                # batch_end is the chunk-position end (not an HDF5 index),
                # stored at submit time to avoid confusion with the shuffled
                # HDF5 indices in batch_indices.
                sm.worker_offsets[worker_id] = batch_end

                # Check if worker is done
                _, end = sm.worker_chunks[worker_id]
                if sm.worker_offsets[worker_id] >= end:
                    sm.workers_done[worker_id] = True

                completed.append((worker_id, dataset_idx, result))
                done_workers.append(worker_id)

        for wid in done_workers:
            del self._futures[wid]

        return completed

    def all_done(self) -> bool:
        """Return True when all workers are done and no futures are pending."""
        sm = self.state_manager
        all_workers_done = all(
            sm.workers_done.get(wid, True)
            for wid in range(self.config.num_chunks)
        )
        return all_workers_done and len(self._futures) == 0

    def shutdown(self) -> None:
        """Clean up thread/process pool (PARSL cleanup handled elsewhere)."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _get_threshold(self, dataset_idx: int) -> float:
        sm = self.state_manager
        if self.config.use_multihead_model:
            head_name = f"head_{dataset_idx}"
            t = sm.threshold[head_name]
        else:
            t = sm.threshold
        return float(t) if t != np.inf else 1e12  # use large value for inf


# ---------------------------------------------------------------------------
# Local (non-PARSL) evaluation function
# ---------------------------------------------------------------------------


def _evaluate_batch_local(
    model_path: str,
    hdf5_path: str,
    batch_indices: list,
    threshold: float,
    r_max: float,
    r_max_lr,
    error_type: str,
    energy_key: str,
    forces_key: str,
    z_table_zs: list,
    head_index: int = 0,
    multihead: bool = False,
    eval_batch_size: int = 32,
    device: str = "cpu",
) -> dict:
    """
    Local version of evaluate_batch_parsl — same logic but called directly
    (in a subprocess via ProcessPoolExecutor).
    """
    import torch
    import numpy as np
    from so3krates_torch.data.hdf5_utils import load_atoms_from_hdf5
    from so3krates_torch.data.atomic_data import AtomicData
    from so3krates_torch.data.utils import KeySpecification, config_from_atoms
    from so3krates_torch.tools.utils import AtomicNumberTable
    from so3krates_torch.tools import torch_geometric as so3_torch_geometric

    _device = torch.device(device)
    z_table = AtomicNumberTable(z_table_zs)
    key_spec = KeySpecification(
        info_keys={"energy": energy_key},
        arrays_keys={"forces": forces_key},
    )

    model = torch.load(model_path, map_location=_device)
    model.eval()
    if multihead and hasattr(model, "select_heads"):
        model.select_heads = True

    atoms_list = load_atoms_from_hdf5(hdf5_path, index=batch_indices)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    all_heads = (
        [f"head_{i}" for i in range(model.num_output_heads)]
        if multihead
        else None
    )

    # When r_max_lr is None, SO3LR still needs edge_index_lr (electrostatics
    # / dispersion are enabled by default). Fall back to r_max so the graph
    # is always built with a valid cutoff.
    _cutoff_lr = r_max_lr if r_max_lr is not None else r_max

    # Build AtomicData list, tagging each entry with its global HDF5 index
    data_list = []
    for local_i, atoms in enumerate(atoms_list):
        config = config_from_atoms(atoms, key_specification=key_spec)
        data = AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=r_max,
            cutoff_lr=_cutoff_lr,
            heads=all_heads,
        )
        if multihead:
            data.head = torch.tensor([head_index], dtype=torch.long)
        data.global_idx = torch.tensor(
            [batch_indices[local_i]], dtype=torch.long
        )
        data_list.append(data)

    loader = so3_torch_geometric.dataloader.DataLoader(
        dataset=data_list,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
    )

    exceeding_indices = []
    batch_errors = []

    for mini_batch in loader:
        mini_batch = mini_batch.to(_device)
        n_structs = mini_batch.num_graphs

        # Gradients must stay enabled: SO3LR computes forces as -dE/dr
        with torch.enable_grad():
            output = model(mini_batch.to_dict())

        force_errors = None
        energy_errors = None

        if error_type in ("forces", "both"):
            ref_forces = mini_batch.forces
            pred_forces = output.get("forces")
            if ref_forces is not None and pred_forces is not None:
                per_atom_abs = (
                    torch.abs(pred_forces - ref_forces).mean(dim=-1).detach()
                )
                force_errors = torch.zeros(n_structs, device=_device)
                counts = torch.zeros(n_structs, device=_device)
                force_errors.scatter_add_(0, mini_batch.batch, per_atom_abs)
                counts.scatter_add_(
                    0,
                    mini_batch.batch,
                    torch.ones(per_atom_abs.shape[0], device=_device),
                )
                force_errors = force_errors / counts.clamp(min=1)

        if error_type in ("energy", "both"):
            ref_energy = mini_batch.energy
            pred_energy = output.get("energy")
            if ref_energy is not None and pred_energy is not None:
                n_atoms = torch.bincount(
                    mini_batch.batch, minlength=n_structs
                ).float()
                energy_errors = (
                    torch.abs(pred_energy.squeeze(-1) - ref_energy.squeeze(-1))
                    / n_atoms.clamp(min=1)
                ).detach()

        errors = torch.zeros(n_structs, device=_device)
        n_terms = 0
        if force_errors is not None:
            errors += force_errors
            n_terms += 1
        if energy_errors is not None:
            errors += energy_errors
            n_terms += 1
        if n_terms > 0:
            errors = errors / n_terms

        errors_np = errors.cpu().numpy()
        global_idxs = mini_batch.global_idx.cpu().numpy()

        for err, gidx in zip(errors_np, global_idxs):
            err_float = float(err)
            batch_errors.append(err_float)
            if err_float > threshold:
                exceeding_indices.append(int(gidx))

    mean_batch_error = float(np.mean(batch_errors)) if batch_errors else 0.0

    return {
        "exceeding_indices": exceeding_indices,
        "batch_errors": batch_errors,
        "mean_batch_error": mean_batch_error,
        "head_index": head_index,
    }
