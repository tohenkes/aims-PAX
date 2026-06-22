## Pretrained Ensemble Active Learning

### Overview

This example shows how to run active learning starting from an already-trained ensemble and existing labeled datasets.

Use this workflow when you already have:
- A pretrained ensemble of MACE or SO3LR models
- Training and validation datasets (previously labeled with DFT or other reference methods)

This skips the initial dataset generation phase and proceeds directly to active learning: MD propagation → uncertainty checking → selective DFT labeling → retraining.


### Required Files

Place the following files in your working directory:

1. **Pretrained models** — `.model` files in the directory specified by `model.yaml` → `GENERAL.model_dir` (default: `./model/`)
   - Example: `./model/myrun-42.model`, `./model/myrun-99.model`

2. **Training dataset** — `.extxyz` or `.xyz` file(s) containing labeled structures
   - Option A (shared): single file referenced by `initial_train_dataset`
   - Option B (per-member): separate files for each ensemble member

3. **Validation dataset** — `.extxyz` or `.xyz` file(s) with additional labeled structures
   - Option A (shared): single file referenced by `initial_valid_dataset`
   - Option B (per-member): separate files for each ensemble member

4. **Standard configuration files**:
   - `geometry.in` — Initial atomic structure
   - `control.in` — FHI-aims DFT settings
   - `model.yaml` — MLFF architecture and training parameters
   - `aimsPAX.yaml` — This file


### Tag Names (Option B)

When using Option B (per-member datasets), the dictionary keys must **exactly match** the model filename stems:

| Model file | Key |
|---|---|
| `myrun-42.model` | `myrun-42` |
| `myrun-99.model` | `myrun-99` |

This mapping ensures each ensemble member trains on its designated bootstrap dataset.


### Option A vs Option B

**Option A: Single shared dataset**
```yaml
initial_train_dataset: ./train.extxyz
initial_valid_dataset: ./valid.extxyz
```
- Simplest approach

**Option B: Per-member datasets**
```yaml
initial_train_dataset:
  myrun-42: ./train_seed0.extxyz
  myrun-99: ./train_seed1.extxyz
initial_valid_dataset:
  myrun-42: ./valid_seed0.extxyz
  myrun-99: ./valid_seed1.extxyz
```
- Provides diversity by giving each member a different dataset
- Useful when you have bootstrapped or seed-dependent datasets
- Keys must match model filename stems exactly


### Multihead Models

If your `model.yaml` uses a multihead architecture, each frame in your dataset files must include a `head` annotation. Without it, frames are silently assigned to head `"Default"` and dataset construction will fail with a `KeyError`.

Add the annotation when building or converting your dataset:
```python
from ase.io import read, write

atoms_list = read("my_dataset.extxyz", index=":")
for atoms in atoms_list:
    atoms.info["head"] = "head_0"   # or "head_1", etc.
write("my_dataset_annotated.extxyz", atoms_list)
```

aims-PAX will emit a warning if it detects training frames without `head` annotations when a multihead model is configured. Single-head models are unaffected.


### Running

Start active learning with:
```bash
aims-PAX-al --model-settings model.yaml --aimsPAX-settings aimsPAX.yaml
```

The command reads your pretrained models and datasets, initializes the ensemble with the provided data, and begins active learning iterations.

