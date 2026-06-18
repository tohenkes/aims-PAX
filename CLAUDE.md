# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**aims-PAX** (ab initio molecular simulation - Parallel Active eXploration) automates the construction of machine learning force fields (MLFFs) using active learning. It integrates:
- **FHI-aims**: DFT quantum chemistry calculator (external binary, must be compiled separately)
- **MACE**: Equivariant neural network force field architecture
- **ASE**: Atomic Simulation Environment
- **PARSL**: Parallel task distribution for HPC clusters

## Installation

```bash
conda create -n my_env python=3.10
conda activate my_env
cd aims-PAX
bash setup.sh  # installs ndcctools (conda-forge) + pip dependencies + package in editable mode
```

## CLI Entry Points

```bash
aims-PAX                  # Full workflow (initial dataset → active learning → convergence)
aims-PAX-initial-ds       # Initial dataset generation only
aims-PAX-al               # Active learning only (assumes initial dataset exists)
aims-PAX-recalc           # Recalculate dataset with new DFT settings
```

All commands accept `--model-settings` and `--aimsPAX-settings` flags for custom config paths.

## Required Files (working directory)

1. `control.in` — FHI-aims DFT settings (functional, basis sets, k-points)
2. `geometry.in` — Initial atomic geometry (or path in `aimsPAX.yaml`)
3. `model.yaml` — Model architecture and training settings (MACE or SO3LR)
4. `aimsPAX.yaml` — aims-PAX workflow settings

See `example/` for a complete working configuration with `explanation.md` describing all settings.

## Code Formatting

Black formatter with 79-character line limit (enforced via pre-commit hooks):
```bash
black --line-length 79 src/
```

## Architecture

Source code is under `src/aims_PAX/` with three layers:

### CLI Layer (`cli/`)
Entry points that parse args, select serial/parallel/PARSL implementation, and invoke procedures.

### Procedures Layer (`procedures/`)
High-level workflow stages:

- **`preparation.py`** — Base classes `PrepareInitialDatasetProcedure` and `PrepareALProcedure`. Also contains `ALConfiguration` (centralized config container), `ALStateManager`, `ALEnsemble`, `ALCalculatorMLFF`, `ALMD`, and `ALRestart`. Handles YAML parsing, geometry/control file processing, directory creation, logging, model initialization, and restart checkpoint handling.
- **`initial_dataset.py`** — Three strategies:
  - `InitialDatasetAIMD`: Ab initio MD sampling with direct DFT
  - `InitialDatasetPARSL`: Parallel DFT via PARSL (foundational model sampling + PARSL DFT)
  - `InitialDatasetPARSLTeacher`: Uses a teacher ML model instead of DFT as reference
- **`active_learning.py`** — PARSL-only active learning:
  - `ALProcedurePARSL`
  - Common workflow: run MLFF MD → detect high-uncertainty structures → DFT label → retrain ensemble → adapt threshold
- **`al_managers.py`** — Manager classes decoupling concerns: `ALRunningManager`, `ALDataManager`, `ALTrainingManager` (with `TrainingSession` and `TrainingOrchestrator`), `ALReferenceManagerPARSL` and subclasses, `ALAnalysisManager*`
- **`recalculate.py`** — Recalculate existing dataset with new DFT settings (PARSL only)

### Tools Layer (`tools/`)
- **`model_tools/train_epoch.py`** — `train_epoch()` and `validate_epoch_ensemble()` for ensemble training
- **`model_tools/setup_MACE.py`** / **`model_tools/setup_so3.py`** — Ensemble initialization for MACE and SO3LR/MultiHeadSO3LR architectures
- **`model_tools/training_tools.py`** — `setup_model_training()`, optimizer/scheduler/EMA setup, loss function selection
- **`uncertainty.py`** — `MolForceUncertainty`: max atomic std-dev of forces across ensemble; `UDDCalculator`: uncertainty-driven dynamics bias potential (linear bias on SO3LR)
- **`tools/utilities/`** — Supporting utilities:
  - `data_handling.py`: Dataset I/O and splitting
  - `eval_utils.py`: Evaluation metrics
  - `input_checks.py`: Input validation (~1065 lines, very comprehensive)
  - `input_utils.py`: Unified input file reading (`read_input_files()`)
  - `utilities.py`: Ensemble ops, checkpointing, logging (~1761 lines)
  - `parsl_utils.py`: PARSL job configuration and monitoring with MPI executor support
  - `mpi_utils.py`: `CommHandler` abstraction for MPI communication; `send_points_non_blocking()` for ASE.Atoms data

## Key Design Patterns

- **Inheritance hierarchy**: Base preparation/procedure classes → specialized implementations (serial/parallel/PARSL)
- **Manager pattern**: `al_managers.py` separates data, training, DFT, and analysis concerns
- **Centralized config**: `ALConfiguration` in `preparation.py` centralizes all AL parameters
- **Ensemble-based UQ**: Multiple MACE/SO3LR models with different random seeds; uncertainty = std-dev of force predictions; configurable via `uncertainty_type` (`max_atomic_sd`, `mean_atomic_sd`, `ensemble_sd`)
- **Adaptive threshold**: Uncertainty threshold adjusts dynamically as training progresses
- **Checkpoint/restart**: State serialized via numpy; controlled via `aimsPAX.yaml` restart settings; `update_md_checkpoints` option controls whether MD checkpoints update after new DFT data

## Parallelization Modes

1. **Serial**: Baseline sequential processing
2. **Parallel**: Multi-process on a single node
3. **PARSL**: Distributed HPC with Slurm — configured via `CLUSTER` section of `aimsPAX.yaml`; supports MPI executor for distributed DFT calculations

## Notable `aimsPAX.yaml` Options (recent additions)

- `ACTIVE_LEARNING.update_md_checkpoints` (bool, default `true`): update MD checkpoints after successful DFT-labeled structures
- `ACTIVE_LEARNING.replay_strategy` (`full_dataset` or `random_subset`): training data replay strategy
- `ACTIVE_LEARNING.train_subset_size` / `valid_subset_size`: subset sizes for `random_subset` replay
- `ACTIVE_LEARNING.uncertainty_type`: choose uncertainty metric (`max_atomic_sd`, `mean_atomic_sd`, `ensemble_sd`)

## Test Suite

There is a `pytest` suite under `tests/` (run in the conda/mamba env
`aimspax_test`):

```bash
mamba run -n aimspax_test python -m pytest -q                    # full suite
mamba run -n aimspax_test python -m pytest -m "not slow" -q      # fast subset (skips integration)
mamba run -n aimspax_test python -m pytest --cov=aims_PAX --cov-report=term-missing
```

- Integration/long-running tests are marked `@pytest.mark.slow` (registered in
  `pyproject.toml` `[tool.pytest.ini_options]`); deselect with `-m "not slow"`.
- Network-dependent tests (downloading model weights) are marked
  `@pytest.mark.network`; the hermetic suite (`-m "not network"`) is the
  CI-relevant figure.
- The suite is unit-heavy (settings validation, MD setup, orchestration logic via
  stubs/mocks). A remediation plan to add numeric-kernel and integration coverage
  lives in `docs/redo_tests/`.

### Coverage figures (measured 2026-06-18)

| Suite | Command | Passed | Total coverage |
|---|---|---|---|
| Fast (unit only) | `-m "not slow"` | 361 | **51%** |
| Hermetic (CI) | `-m "not network"` | 384 | **70%** |

The gap between fast and hermetic reflects the teacher-student e2e tests
(`test_teacher_student_e2e.py`) and other slow integration tests that exercise
the orchestration core. Key per-module figures from the hermetic run:

| Module | Hermetic coverage |
|---|---|
| `procedures/active_learning.py` | 84% (was 47% fast-only) |
| `procedures/al_managers.py` | 61% (was 32% fast-only) |
| `procedures/initial_dataset.py` | 82% |
| `procedures/preparation.py` | 78% |
| `tools/model_tools/train_epoch.py` | 61% (was 15% fast-only) |
| `tools/model_tools/setup_so3.py` | 81% (was 34% fast-only) |
| `tools/utilities/eval_utils.py` | 61% (was 31% fast-only) |
| `tools/utilities/utilities.py` | 67% (was 49% fast-only) |

End-to-end validation can still be done by running the example in `example/` and
inspecting logs in `./logs/` (`initial_dataset.log`, `active_learning.log`).
