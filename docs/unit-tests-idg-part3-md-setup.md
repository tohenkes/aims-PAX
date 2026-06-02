# Plan: Unit tests — Part 3: MDynamics setup

## Context

`setup_md` is the method in `PrepareInitialDatasetProcedure` that constructs ASE molecular dynamics integrators from configuration dicts. It dispatches on `stat_ensemble` (NVT/NPT) and then on thermostat/barostat type (Langevin, Berendsen, MTK, IsotropicMTK). Velocities are optionally initialized unless restarting. Optional Berendsen parameters (`taup`, `taut`, `compressibility_au`, `fixcm`) are forwarded if present. Unrecognised values raise `ValueError` at all dispatch levels after an implicit-UnboundLocalError bug was fixed.

**File to test**: `src/aims_PAX/procedures/preparation.py`
**Test file**: `tests/test_idg_md_setup.py`
**Framework**: pytest (no mocking needed — ASE MD classes construct with just atoms + numeric params)

---

### 3.0 Setup / imports

```python
import numpy as np
import pytest
from types import SimpleNamespace

import ase
import ase.build
from ase import units
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nose_hoover_chain import MTKNPT, IsotropicMTKNPT

from aims_PAX.procedures.preparation import PrepareInitialDatasetProcedure
```

Test harness:
```python
def call_setup_md(atoms, md_settings, *, restart=False):
    stub = SimpleNamespace(restart=restart)
    return PrepareInitialDatasetProcedure.setup_md(stub, atoms, md_settings)
```

Key design decision: `setup_md` only reads `self.restart` from the instance, so `SimpleNamespace(restart=...)` is used as `self` to bypass full class init. This avoids needing to construct the heavy `PrepareInitialDatasetProcedure` class.

Baseline dicts used in all tests:

```python
NVT_LANGEVIN = dict(
    stat_ensemble="nvt",
    thermostat="langevin",
    timestep=0.5,
    friction=0.001,
    temperature=300.0,
    MD_seed=42,
)

NPT_BERENDSEN = dict(
    stat_ensemble="npt",
    barostat="berendsen",
    timestep=0.5,
    temperature=300.0,
    pressure=101325.0,
)

NPT_MTK = dict(
    stat_ensemble="npt",
    barostat="mtk",
    timestep=0.5,
    temperature=300.0,
    pressure=101325.0,
    tdamp=50.0,
    pdamp=500.0,
    tchain=3,
    pchain=3,
    tloop=1,
    ploop=1,
)

NPT_ISOMTK = NPT_MTK | dict(barostat="isomtk")
```

Fixture:
```python
@pytest.fixture
def cu_atoms():
    return ase.build.bulk("Cu", "fcc", a=3.6)
```

---

### 3.1 NVT/Langevin dispatch

Single test for the NVT thermostat dispatch:

| Test | Input | Expected |
|------|-------|----------|
| `test_nvt_langevin_returns_langevin` | `stat_ensemble="nvt"`, `thermostat="langevin"`, standard NVT_LANGEVIN dict | ASE `Langevin` instance |

---

### 3.2 NPT ensemble dispatches

Three tests for NPT barostat dispatch, with special attention to inheritance:

| Test | Input barostat | Expected type | Notes |
|-------|-----------------|-----------------|-------|
| `test_npt_berendsen_returns_nptberendsen` | `"berendsen"` | `NPTBerendsen` | Standard barostat dispatch |
| `test_npt_mtk_returns_mtknpt` | `"mtk"` | `MTKNPT` AND `not IsotropicMTKNPT` | `IsotropicMTKNPT` is a subclass of `MTKNPT`, so must check both conditions to disambiguate |
| `test_npt_isomtk_returns_isotropicmtknpt` | `"isomtk"` | `IsotropicMTKNPT` | Only `isinstance(dyn, IsotropicMTKNPT)` needed; parent is implicitly checked |

Implementation note: `NPT_ISOMTK = NPT_MTK | dict(barostat="isomtk")` — Python 3.9+ dict merge used to avoid keyword argument collision (`barostat` appears in both dicts).

---

### 3.3 Restart gate for velocity initialization

`MaxwellBoltzmannDistribution` is guarded by `if not self.restart`. Velocities are initialized only on fresh runs; restart preserves momenta.

| Test | `restart` flag | Initial momenta state | Expected behavior |
|------|----------------|-----------------------|-------------------|
| `test_restart_false_sets_velocities` | `False` | `cu_atoms.get_momenta()` is all-zero | `setup_md` should set non-zero momenta |
| `test_restart_true_preserves_velocities` | `True` | `cu_atoms.get_momenta()` is all-zero | `setup_md` should NOT touch momenta; remain all-zero |

Both tests check momentum state **after** `setup_md` returns.

---

### 3.4 Berendsen optional kwargs forwarding

`taup`, `taut`, `compressibility_au`, `fixcm` are conditionally forwarded to `NPTBerendsen` with `if md_settings.get(k, False)`. ASE defaults are `taup=1e3*units.fs` and `taut=0.5e3*units.fs`.

Three tests covering edge cases:

| Test | MD settings kwargs | Expected state |
|------|-------------------|-----------------|
| `test_berendsen_without_optional_kwargs` | No extra keys | `dyn.taup == approx(1e3 * units.fs)`, `dyn.taut == approx(0.5e3 * units.fs)` |
| `test_berendsen_with_taup` | `"taup": 2000.0` | `dyn.taup == approx(2000.0 * units.fs)` |
| `test_berendsen_with_taut` | `"taut": 100.0` | `dyn.taut == approx(100.0 * units.fs)` |

Implementation notes:
- `.get(k, False)` is used instead of `.get(k, None)` to default optional keys to falsy values. This allows the code to skip kwarg forwarding when a key is absent.
- Latent edge case: `taup=0.0` would silently use the ASE default instead of setting zero (not tested, as this scenario is unrealistic for MD timesteps).

---

### 3.5 Unrecognised values raise ValueError

Three dispatch levels must explicitly raise `ValueError` when given bad values (not silently fall through with an implicit `UnboundLocalError`). This tests that the bug fix is in place.

| Test | Field | Bad value | Expected |
|-------|-------|-----------|----------|
| `test_unknown_stat_ensemble_raises` | `stat_ensemble` | `"mcmc"` | `ValueError` with substring `"stat_ensemble"` |
| `test_nvt_unknown_thermostat_raises` | `thermostat` (within NVT branch) | `"velocity_rescaling"` | `ValueError` with substring `"thermostat"` |
| `test_npt_unknown_barostat_raises` | `barostat` (within NPT branch) | `"andersen"` | `ValueError` with substring `"barostat"` |

---

### 3.6 Dead code and latent edge cases

These are documented but not tested:

1. **Dead code path**: `barostat == "npt"` inside the NPT branch. This can never be reached via Pydantic because `"npt"` is a valid `stat_ensemble` value, not a `barostat` value. Kept in place for now; not tested.

2. **Latent edge case**: `md_settings.get("taup", False)` defaults to `False`, so `taup=0.0` would fail the truthiness check and use ASE defaults instead of setting zero. This is unrealistic for MD timesteps (always positive), so not tested.

---

## File layout

```
tests/
├── conftest.py          # Minimal — pytest standard fixtures only
└── test_idg_md_setup.py # 12 test functions, all self-contained
```

---

## Verification

```bash
pytest tests/test_idg_md_setup.py -v
# Expected: 12 tests pass; no ASE import errors; no mocking required
```

---

## Notes

- Parts 1, 2, 4, 5 are sketched in `docs/unit-tests-idg.md`
- Part 3 depends only on ASE being installed (no other internal dependencies)
- Parts 4 and 5 depend on actual class instantiation; tackle after ASI4py removal is complete (see `docs/remove-asi4py.md`) since the class hierarchy will change
