## Overview

This example shows how to use `aims-PAX-filter` to distil a large pre-labeled HDF5 dataset into a compact, representative training set.

The idea: given a dataset of millions of structures that already have reference energies and forces (e.g. from a prior DFT run or a teacher model), we train an SO3LR model on the fly by iterating through the dataset and keeping only the structures where the current model's prediction error exceeds an adaptive threshold. Redundant structures — those the model already predicts well — are discarded.

**No FHI-aims, no `control.in`, no `geometry.in` are needed.** Only `model.yaml` and `aimsPAX.yaml` are required.

---

### What to change

1. `hdf5_paths` in `aimsPAX.yaml` — point to your actual HDF5 v2.0 dataset(s).
2. `device` in `model.yaml` — set to `cpu` if no GPU is available.
3. `desired_acc` and `max_train_set_size` — tune to your convergence target and memory budget.
4. `output_dir` in `aimsPAX.yaml` — where results are written.

---

### Running

```bash
aims-PAX-filter --model-settings model.yaml --aimsPAX-settings aimsPAX.yaml
```

Default paths (`./model.yaml` and `./aimsPAX.yaml`) are used when the flags are omitted.

---

### Input dataset format

Datasets must be in **HDF5 v2.0 format** as written by `so3krates-torch`. Each structure must contain:
- atomic positions and species
- reference energies (`REF_energy` by default)
- reference forces (`REF_forces` by default)

The expected property keys are configured in `model.yaml` (`GENERAL.energy_key`, `GENERAL.forces_key`), or fall back to the so3krates defaults.

---

### Single vs. multi-dataset

**Single dataset** (`hdf5_paths` has one entry): a regular SO3LR model is trained. Output is saved as `filtered_dataset.h5` in `output_dir/datasets/`.

**Multiple datasets** (`hdf5_paths` has two or more entries): a MultiHeadSO3LR model is trained automatically, with one output head per dataset. Per-head thresholds are maintained independently (different datasets may have different error scales). Output is saved as:
- `filtered_<stem_of_source>.h5` — one file per source dataset
- `filtered_combined.h5` — all structures together

Multi-dataset example:

```yaml
DATA_FILTERING:
  hdf5_paths:
    - "/path/to/bulk_water.h5"
    - "/path/to/water_surface.h5"
  num_chunks: 8
  eval_stride: 500
  error_type: forces
  desired_acc: 0.05
  max_train_set_size: 10000
  epochs_per_batch: 50
  converge_best: True
```

---

### Key settings

| Setting | Description |
|---------|-------------|
| `num_chunks` | The dataset is split into this many sequential chunks. Each chunk is evaluated fully before moving on. |
| `eval_stride` | Number of structures passed to the model in one evaluation step. |
| `error_type` | Which error drives filtering: `forces`, `energy`, or `both`. |
| `desired_acc` | Target validation MAE (eV/Å). The run stops when this is reached. `0.0` = disabled. |
| `max_train_set_size` | Hard limit on the number of structures collected for training. |
| `c_x` | Threshold tuning factor. Negative values tighten the threshold (collect fewer points); positive values loosen it. |
| `epochs_per_batch` | Training epochs run each time a new batch of structures is added. |
| `converge_best` | If `True` (default), a final convergence pass is run on the collected dataset after filtering. |

---

### Outputs

All outputs are written under `output_dir` (default: `./filter_output`):

```
filter_output/
  logs/
    data_filtering.log        # Full run log
  datasets/
    filtered_dataset.h5       # Filtered dataset (single-dataset run)
    filtered_<name>.h5        # Per-source files  (multi-dataset run)
    filtered_combined.h5      # Combined file     (multi-dataset run)
    xyz/                      # XYZ files (only when save_xyz: True)
  model/
    model_seed_1.model        # Final trained model
  checkpoints/
    *.pt                      # Training checkpoints (for restart)
  restart/
    df_restart.npy            # Restart checkpoint
```

---

### Restart

Set `create_restart: True` in `MISC` (the default). If the run is interrupted, simply re-run the same command and it will resume from the last saved checkpoint. When the run completes, the restart file records `df_done = True`; re-running the command will skip straight to `converge()` if `converge_best` is enabled.

---

### Cluster execution

When a `CLUSTER` section is present in `aimsPAX.yaml`, evaluation workers are submitted as PARSL jobs on the cluster. Remove the section entirely to run locally with `ProcessPoolExecutor`. The `launch_str` key (used for DFT in the AL workflow) is **not** needed here.
