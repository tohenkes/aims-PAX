"""
Preparation classes for the data-filtering procedure.

DFConfiguration  — reads aimsPAX.yaml DATA_FILTERING section and provides
                   all configuration parameters; duck-typed for
                   TrainingOrchestrator compatibility.
DFStateManager   — mutable runtime state (worker progress, thresholds, ...).
DFModelManager   — creates / holds the single SO3LR or MultiHeadSO3LR model.
DFRestart        — save / load restart checkpoints via numpy.
"""

import logging
import math
import os
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch

from so3krates_torch.data.hdf5_utils import (
    scan_raw_hdf5_statistics,
)
from so3krates_torch.data.utils import KeySpecification

from aims_PAX.tools.model_tools.setup_so3 import (
    setup_multihead_so3lr,
    setup_so3lr,
)
from aims_PAX.tools.model_tools.training_tools import setup_model_training
from aims_PAX.tools.utilities.utilities import (
    apply_model_settings,
    create_keyspec,
    setup_logger,
)

# ---------------------------------------------------------------------------
# DFConfiguration
# ---------------------------------------------------------------------------


class DFConfiguration:
    """
    Reads DATA_FILTERING and MISC sections from aimsPAX_settings and
    exposes all parameters used by the data-filtering procedure.

    Provides compatibility aliases so that TrainingOrchestrator can be
    reused without modification.
    """

    def __init__(
        self,
        model_settings: dict,
        aimsPAX_settings: dict,
    ):
        self.model_settings = model_settings
        self.aimsPAX_settings = aimsPAX_settings
        self.df_settings = aimsPAX_settings["DATA_FILTERING"]
        self.misc = aimsPAX_settings.get("MISC", {})
        self.cluster_settings = aimsPAX_settings.get("CLUSTER", None)

        self._setup_model_configuration()
        self._setup_df_configuration()

    # -----------------------------------------------------------------------
    # Private setup helpers
    # -----------------------------------------------------------------------

    def _setup_model_configuration(self):
        """Apply model.yaml settings (r_max, dtype, device, …)."""
        apply_model_settings(target=self, model_settings=self.model_settings)
        self.checkpoints_dir += "/df"

    def _setup_df_configuration(self):
        """Read DATA_FILTERING section and resolve paths."""
        df = self.df_settings

        # HDF5 datasets
        self.hdf5_paths: List[str] = df["hdf5_paths"]

        # Multi-head: one head per dataset when >1 path
        self.use_multihead_model = len(self.hdf5_paths) > 1
        if self.use_multihead_model:
            self.all_heads = [f"head_{i}" for i in range(len(self.hdf5_paths))]
            # Patch model settings so TrainingOrchestrator picks this up
            self.model_settings["ARCHITECTURE"]["use_multihead_model"] = True
            self.model_settings["ARCHITECTURE"]["num_multihead_heads"] = len(
                self.hdf5_paths
            )
        else:
            self.all_heads = None

        # Worker / batch settings
        self.num_chunks: int = df["num_chunks"]
        self.eval_stride: int = df["eval_stride"]
        self.error_type: str = df["error_type"]

        # Stopping criteria
        self.desired_accuracy: float = df["desired_acc"]
        self.max_train_set_size = df["max_train_set_size"]

        # Threshold
        self.c_x: float = df["c_x"]
        self.freeze_threshold_dataset = df["freeze_threshold_dataset"]

        # Training
        self.epochs_per_batch: int = df["epochs_per_batch"]
        self.valid_ratio: float = df["valid_ratio"]
        self.valid_skip: int = df["valid_skip"]
        self.replay_strategy: str = df["replay_strategy"]
        self.train_subset_size = df["train_subset_size"]
        self.valid_subset_size = df["valid_subset_size"]
        _eval_bs = df["eval_batch_size"]
        self.eval_batch_size: int = (
            _eval_bs if _eval_bs is not None else self.set_valid_batch_size
        )

        # Convergence
        self.convergence_patience: int = df["convergence_patience"]
        self.max_convergence_epochs: int = df["max_convergence_epochs"]
        self.converge_best: bool = df["converge_best"]
        self.margin: float = df["margin"]
        self.analysis: bool = df["analysis"]

        # Output
        self.save_xyz: bool = df["save_xyz"]
        self.shuffle_dataset: bool = df["shuffle_dataset"]
        self.loop_exhausted_data: bool = df.get("loop_exhausted_data", True)
        self.max_data_passes: int = df.get("max_data_passes", 0)
        self.compact_logging: bool = df.get("compact_logging", False)

        # Worker device — PARSL workers are typically CPU-only nodes even
        # when the main process trains on GPU.
        self.worker_device: str = (
            self.cluster_settings.get("worker_device", "cpu")
            if self.cluster_settings is not None
            else self.device
        )

        # Paths (all resolved relative to output_dir)
        self.output_dir = Path(self.misc.get("output_dir", ".")).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        def _r(p):
            p = Path(p)
            return p if p.is_absolute() else self.output_dir / p

        self.dataset_dir = _r(self.misc["dataset_dir"])
        self.log_dir = str(_r(self.misc["log_dir"]))
        self.checkpoints_dir = str(_r(self.checkpoints_dir))
        self.model_settings["GENERAL"][
            "checkpoints_dir"
        ] = self.checkpoints_dir
        self.model_dir = str(_r(self.model_dir))
        self.model_settings["GENERAL"]["model_dir"] = self.model_dir
        self.model_settings["GENERAL"]["loss_dir"] = str(
            _r(self.model_settings["GENERAL"]["loss_dir"])
        )

        self.df_restart_path = (
            self.output_dir / "restart" / "df" / "df_restart.npy"
        )
        self.restart = self.df_restart_path.exists()
        self.create_restart = self.misc.get("create_restart", True)

        # Keys for reading HDF5 labels
        self.key_specification = create_keyspec(
            energy_key=self.misc["energy_key"],
            forces_key=self.misc["forces_key"],
            stress_key=self.misc["stress_key"],
            dipole_key=self.misc["dipole_key"],
            polarizability_key=self.misc["polarizability_key"],
            head_key=self.misc["head_key"],
            charges_key=self.misc["charges_key"],
            total_charge_key=self.misc["total_charge_key"],
            total_spin_key=self.misc["total_spin_key"],
        )

        # TrainingOrchestrator compatibility aliases
        self.intermediate_epochs_al = self.epochs_per_batch
        self.epochs_per_worker = self.epochs_per_batch
        self.num_trajectories = 1
        self.model_choice = "so3lr"
        self.seeds_tags_dict = {"model_seed_1": self.seed}
        self.mol_idxs = None


# ---------------------------------------------------------------------------
# DFStateManager
# ---------------------------------------------------------------------------


class DFStateManager:
    """
    Holds all mutable state for a data-filtering run.

    Designed so that DFRestart can serialise / deserialise it via np.save.
    Also provides the attribute interface expected by TrainingOrchestrator.
    """

    def __init__(self, config: DFConfiguration):
        self.config = config
        num_chunks = config.num_chunks
        num_datasets = len(config.hdf5_paths)

        # Per-worker progress
        self.worker_offsets: dict = {}
        self.worker_chunks: dict = {}
        self.workers_done: dict = {}
        self.worker_dataset_idx: dict = {}

        # Threshold state — per-dataset when multi-head, scalar otherwise.
        # Initialised to 0 so the first batch always adds points and
        # immediately bootstraps training.
        if config.use_multihead_model:
            self.threshold = {h: 0.0 for h in config.all_heads}
            self.validation_error_history = {h: [] for h in config.all_heads}
        else:
            self.threshold = 0.0
            self.validation_error_history = []

        # Per-head validation errors (multi-head only, set after training)
        self.current_valid_errors_per_head: dict = {}

        # Global counters
        self.total_points_added: int = 0
        self.train_points_added: int = 0
        self.valid_points_added: int = 0
        self.new_train_count: int = 0
        self.current_valid_error: float = np.inf
        self.total_epoch: int = 0

        # TrainingOrchestrator compatibility
        self.ensemble_reset_opt = {"model_seed_1": False}
        self.ensemble_no_improvement = {"model_seed_1": 0}
        self.ensemble_best_valid = {"model_seed_1": np.inf}
        self.trajectory_intermediate_epochs = {0: 0}
        self.trajectory_total_epochs = {0: 0}

        # Shuffle maps — per-dataset index permutation (set by initialize_workers)
        self.shuffle_maps: dict = {}

        # Pass counter (incremented each time data is exhausted and re-looped)
        self.current_pass: int = 1

        # Analysis
        if config.analysis:
            self.collect_losses = {
                "epoch": [],
                "avg_losses": [],
                "ensemble_losses": [],
            }
            self.collect_thresholds = []

    def initialize_workers(self, dataset_sizes: List[int], pass_num: int = 1):
        """
        Distribute num_chunks across datasets proportionally to their
        size and partition each dataset into contiguous chunks.
        """
        num_chunks = self.config.num_chunks
        num_datasets = len(dataset_sizes)
        total = sum(dataset_sizes)

        if num_chunks > total:
            logging.warning(
                f"num_chunks ({num_chunks}) exceeds total dataset size "
                f"({total}). Effective parallelism will be limited to "
                f"{total} chunk(s)."
            )

        # Distribute chunks proportionally (at least 1 per dataset)
        if num_datasets == 1:
            workers_per_dataset = [num_chunks]
        else:
            raw = [
                max(1, round(num_chunks * s / max(total, 1)))
                for s in dataset_sizes
            ]
            # Adjust so total == num_chunks
            diff = num_chunks - sum(raw)
            for i in range(abs(diff)):
                idx = i % num_datasets
                raw[idx] += 1 if diff > 0 else -1
            workers_per_dataset = raw

        worker_id = 0
        for ds_idx, (n_workers, ds_size) in enumerate(
            zip(workers_per_dataset, dataset_sizes)
        ):
            chunk_size = math.ceil(ds_size / max(n_workers, 1))
            start = 0
            for _ in range(n_workers):
                end = min(start + chunk_size, ds_size)
                self.worker_offsets[worker_id] = start
                self.worker_chunks[worker_id] = (start, end)
                self.workers_done[worker_id] = False
                self.worker_dataset_idx[worker_id] = ds_idx
                start = end
                worker_id += 1
                if start >= ds_size:
                    break

        # Pass 1 uses seed as-is; subsequent passes offset the seed so each
        # pass produces a different permutation. On pass 2+, always shuffle
        # regardless of shuffle_dataset so looping is not identical.
        rng = np.random.default_rng(self.config.seed + pass_num - 1)
        for ds_idx, ds_size in enumerate(dataset_sizes):
            if self.config.shuffle_dataset or pass_num > 1:
                self.shuffle_maps[ds_idx] = rng.permutation(ds_size)
            else:
                self.shuffle_maps[ds_idx] = np.arange(ds_size)

        logging.info(
            f"Initialised {worker_id} workers across "
            f"{num_datasets} dataset(s). "
            f"Workers per dataset: {workers_per_dataset}"
            + (
                f" (shuffled, pass {pass_num})"
                if (self.config.shuffle_dataset or pass_num > 1)
                else ""
            )
        )


# ---------------------------------------------------------------------------
# DFModelManager
# ---------------------------------------------------------------------------


class DFModelManager:
    """
    Creates and holds the single SO3LR (or MultiHeadSO3LR) model plus
    its training setup. Mimics the interface of ALEnsemble for
    TrainingOrchestrator duck-typing.
    """

    def __init__(self, config: DFConfiguration):
        self.config = config

        # Attributes expected by TrainingOrchestrator
        self.ensemble: Optional[dict] = None
        self.training_setups: Optional[dict] = None
        self.ensemble_ase_sets: Optional[dict] = None
        self.ensemble_model_sets: Optional[dict] = None
        self.train_dataset_len: int = 0
        self.z_table = None

        # Will be populated by setup_model()
        self.e0s_dict: Optional[dict] = None
        self.avg_num_neighbors: float = 1.0

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def scan_datasets(self):
        """
        Read HDF5 metadata for E0s and avg_num_neighbors.
        Uses all datasets; union of species across datasets.
        """
        from so3krates_torch.tools.utils import AtomicNumberTable

        all_e0s: dict = {}
        all_avg_nn: list = []

        for hdf5_path in self.config.hdf5_paths:
            e0s, _, avg_nn = scan_raw_hdf5_statistics(
                hdf5_path=hdf5_path,
                r_max=float(self.config.r_max),
                r_max_lr=self.config.r_max_lr,
                keyspec=self.config.key_specification,
            )
            for z, e in e0s.items():
                if z not in all_e0s:
                    all_e0s[z] = e
                # else: use first dataset's E0 (simple heuristic)
            all_avg_nn.append(avg_nn)

        self.e0s_dict = all_e0s
        self.avg_num_neighbors = float(np.mean(all_avg_nn))
        # z_table covers all elements 1-118 (so3lr convention)
        self.z_table = AtomicNumberTable(list(range(1, 119)))

        logging.info(
            f"Dataset scan complete. "
            f"avg_num_neighbors={self.avg_num_neighbors:.2f}"
        )

    def setup_model(self):
        """Create the SO3LR or MultiHeadSO3LR model and training setup."""
        config = self.config

        if config.use_multihead_model:
            model = setup_multihead_so3lr(
                settings=config.model_settings,
                z_table=self.z_table,
                atomic_energies_dict=self.e0s_dict,
                avg_num_neighbors=self.avg_num_neighbors,
            )
        else:
            model = setup_so3lr(
                settings=config.model_settings,
                z_table=self.z_table,
                atomic_energies_dict=self.e0s_dict,
                avg_num_neighbors=self.avg_num_neighbors,
            )

        tag = "model_seed_1"
        self.ensemble = {tag: model}

        training_setup = setup_model_training(
            settings=config.model_settings,
            model=model,
            model_choice=config.model_choice,
            tag=tag,
            restart=config.restart,
            checkpoints_dir=config.checkpoints_dir,
        )
        self.training_setups = {tag: training_setup}

        # Empty dataset containers (TrainingOrchestrator expects these)
        if config.use_multihead_model:
            self.ensemble_model_sets = {
                tag: {
                    "train": [],
                    "valid": {h: [] for h in config.all_heads},
                }
            }
        else:
            self.ensemble_model_sets = {
                tag: {"train": [], "valid": {"Default": []}}
            }

        self.ensemble_ase_sets = {tag: {"train": [], "valid": []}}

        logging.info(
            "Model created: "
            + (
                f"MultiHeadSO3LR ({len(config.hdf5_paths)} heads)"
                if config.use_multihead_model
                else "SO3LR"
            )
        )

    def load_from_checkpoint(self, checkpoint_path: str):
        """Reload model weights from a saved checkpoint."""
        tag = "model_seed_1"
        device = torch.device(self.config.device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) else ckpt
        if isinstance(state_dict, dict):
            self.ensemble[tag].load_state_dict(state_dict, strict=False)
        else:
            self.ensemble[tag] = state_dict
        self.ensemble[tag].eval()
        logging.info(f"Loaded model from checkpoint: {checkpoint_path}")


# ---------------------------------------------------------------------------
# DFRestart
# ---------------------------------------------------------------------------


class DFRestart:
    """
    Saves and loads restart checkpoints for the data-filtering run via
    numpy .npy (allow_pickle=True), matching ALRestart's approach.
    """

    def __init__(
        self,
        config: DFConfiguration,
        state_manager: DFStateManager,
    ):
        self.config = config
        self.state_manager = state_manager
        self.df_done = False
        self.save_restart = False

        if config.create_restart:
            restart_dir = config.df_restart_path.parent
            restart_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # State keys to persist
    # -----------------------------------------------------------------------

    _STATE_KEYS = [
        "worker_offsets",
        "worker_chunks",
        "workers_done",
        "worker_dataset_idx",
        "shuffle_maps",
        "threshold",
        "validation_error_history",
        "current_valid_errors_per_head",
        "total_points_added",
        "train_points_added",
        "valid_points_added",
        "new_train_count",
        "current_valid_error",
        "total_epoch",
        "ensemble_reset_opt",
        "ensemble_no_improvement",
        "ensemble_best_valid",
        "trajectory_intermediate_epochs",
        "trajectory_total_epochs",
        "current_pass",
    ]

    def save(self):
        """Persist current state to disk."""
        if not self.config.create_restart:
            return

        state_dict = {
            k: getattr(self.state_manager, k) for k in self._STATE_KEYS
        }
        state_dict["df_done"] = self.df_done

        if self.config.analysis:
            state_dict["collect_losses"] = self.state_manager.collect_losses
            state_dict["collect_thresholds"] = (
                self.state_manager.collect_thresholds
            )

        np.save(self.config.df_restart_path, state_dict)
        logging.debug(
            f"Restart checkpoint saved to {self.config.df_restart_path}"
        )

    def load(self):
        """Load state from checkpoint and restore to state_manager."""
        logging.info("Restarting data-filtering procedure from checkpoint.")
        state_dict = np.load(
            self.config.df_restart_path, allow_pickle=True
        ).item()

        for k in self._STATE_KEYS:
            if k in state_dict:
                setattr(self.state_manager, k, state_dict[k])

        self.df_done = state_dict.get("df_done", False)

        if self.config.analysis:
            if "collect_losses" in state_dict:
                self.state_manager.collect_losses = state_dict[
                    "collect_losses"
                ]
            if "collect_thresholds" in state_dict:
                self.state_manager.collect_thresholds = state_dict[
                    "collect_thresholds"
                ]
