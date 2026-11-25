import numpy as np
import pytest
from types import SimpleNamespace

import ase
import ase.build
from ase import units

from aims_PAX.procedures.preparation import PrepareInitialDatasetProcedure
from tests.helpers import MD_CASES, NVT_LANGEVIN, NPT_BERENDSEN, NPT_MTK


def call_setup_md(atoms, md_settings, *, restart=False):
    stub = SimpleNamespace(restart=restart)
    return PrepareInitialDatasetProcedure.setup_md(stub, atoms, md_settings)


@pytest.fixture
def cu_atoms():
    return ase.build.bulk("Cu", "fcc", a=3.6)


# ===========================================================================
# §3.1 — Shared type checks (parametrized)
# ===========================================================================


@pytest.mark.parametrize("name,expected_type,settings", MD_CASES)
def test_dynamics_type(cu_atoms, name, expected_type, settings):
    dyn = call_setup_md(cu_atoms, settings)
    assert type(dyn) is expected_type


# ===========================================================================
# §3.2 — Shared physics assertions
# ===========================================================================


def test_nvt_langevin_friction(cu_atoms):
    dyn = call_setup_md(cu_atoms, NVT_LANGEVIN)
    assert dyn.fr == pytest.approx(NVT_LANGEVIN["friction"] / units.fs)


def test_nvt_langevin_rng(cu_atoms):
    dyn = call_setup_md(cu_atoms, NVT_LANGEVIN)
    assert isinstance(dyn.rng, np.random.RandomState)


def test_npt_berendsen_pressure(cu_atoms):
    dyn = call_setup_md(cu_atoms, NPT_BERENDSEN)
    assert dyn.pressure == pytest.approx(
        NPT_BERENDSEN["pressure"] * units.Pascal
    )


def test_npt_mtk_chain_params(cu_atoms):
    dyn = call_setup_md(cu_atoms, NPT_MTK)
    assert dyn._thermostat._tdamp == pytest.approx(NPT_MTK["tdamp"] * units.fs)
    assert dyn._barostat._pdamp == pytest.approx(NPT_MTK["pdamp"] * units.fs)
    assert dyn._thermostat._tchain == NPT_MTK["tchain"]
    assert dyn._barostat._pchain == NPT_MTK["pchain"]
    assert dyn._thermostat._tloop == NPT_MTK["tloop"]
    assert dyn._barostat._ploop == NPT_MTK["ploop"]


# ===========================================================================
# §3.3 — Restart gate
# ===========================================================================


def test_restart_false_sets_velocities(cu_atoms):
    assert not np.any(cu_atoms.get_momenta())
    call_setup_md(cu_atoms, NVT_LANGEVIN, restart=False)
    assert np.any(cu_atoms.get_momenta())


def test_restart_true_preserves_velocities(cu_atoms):
    assert not np.any(cu_atoms.get_momenta())
    call_setup_md(cu_atoms, NVT_LANGEVIN, restart=True)
    assert not np.any(cu_atoms.get_momenta())


# ===========================================================================
# §3.4 — Berendsen optional kwargs (IDG-specific)
# ===========================================================================


def test_berendsen_without_optional_kwargs(cu_atoms):
    dyn = call_setup_md(cu_atoms, NPT_BERENDSEN)
    assert dyn.taup == pytest.approx(1e3 * units.fs)
    assert dyn.taut == pytest.approx(0.5e3 * units.fs)


def test_berendsen_with_taup(cu_atoms):
    settings = {**NPT_BERENDSEN, "taup": 2000.0}
    dyn = call_setup_md(cu_atoms, settings)
    assert dyn.taup == pytest.approx(2000.0 * units.fs)


def test_berendsen_with_taut(cu_atoms):
    settings = {**NPT_BERENDSEN, "taut": 100.0}
    dyn = call_setup_md(cu_atoms, settings)
    assert dyn.taut == pytest.approx(100.0 * units.fs)


def test_berendsen_with_taup_zero(cu_atoms):
    settings = {**NPT_BERENDSEN, "taup": 0.0}
    dyn = call_setup_md(cu_atoms, settings)
    assert dyn.taup == pytest.approx(0.0)


# ===========================================================================
# §3.5 — Unrecognised values raise ValueError
# ===========================================================================


def test_unknown_stat_ensemble_raises(cu_atoms):
    settings = {**NVT_LANGEVIN, "stat_ensemble": "mcmc"}
    with pytest.raises(ValueError, match="stat_ensemble"):
        call_setup_md(cu_atoms, settings)


def test_nvt_unknown_thermostat_raises(cu_atoms):
    settings = {**NVT_LANGEVIN, "thermostat": "velocity_rescaling"}
    with pytest.raises(ValueError, match="thermostat"):
        call_setup_md(cu_atoms, settings)


def test_npt_unknown_barostat_raises(cu_atoms):
    settings = {**NPT_BERENDSEN, "barostat": "andersen"}
    with pytest.raises(ValueError, match="barostat"):
        call_setup_md(cu_atoms, settings)


# ===========================================================================
# SO3LR-tauri as foundational model, teacher and AL-MD driver
# ===========================================================================

import numpy as np
import torch
from ase.build import molecule

from aims_PAX.procedures.initial_dataset import InitialDatasetPARSL
from aims_PAX.procedures.preparation import ALCalculatorMLFF
from aims_PAX.settings.project import So3lrFMSettings
from aims_PAX.tools.utilities.parsl_utils import _build_teacher_calculator


@pytest.fixture
def restore_default_dtype():
    previous = torch.get_default_dtype()
    yield
    torch.set_default_dtype(previous)


def _idg_stub():
    return SimpleNamespace(
        device="cpu",
        dtype="float32",
        compute_stress=False,
        key_specification=None,
        enable_cueq=False,
    )


def _assert_finite_water(calc):
    atoms = molecule("H2O")
    atoms.calc = calc
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()


def test_fm_tauri_calculator_runs(restore_default_dtype):
    settings = So3lrFMSettings(model="tauri-s-2606", theory_level=5)
    calc = InitialDatasetPARSL._setup_foundational(
        _idg_stub(), "so3lr", settings
    )
    _assert_finite_water(calc)


def test_fm_v1_default_matches_so3lr_calculator(restore_default_dtype):
    from so3krates_torch.calculator.so3 import SO3LRCalculator

    calc = InitialDatasetPARSL._setup_foundational(
        _idg_stub(), "so3lr", So3lrFMSettings()
    )
    reference = SO3LRCalculator(
        model="v1", device="cpu", default_dtype="float32"
    )
    atoms = molecule("H2O")
    atoms.calc = calc
    ref_atoms = molecule("H2O")
    ref_atoms.calc = reference
    np.testing.assert_allclose(
        atoms.get_potential_energy(), ref_atoms.get_potential_energy()
    )
    np.testing.assert_allclose(atoms.get_forces(), ref_atoms.get_forces())


def test_fm_passes_lr_settings(restore_default_dtype):
    settings = So3lrFMSettings(r_max_lr=10.0, dispersion_lr_damping=1.5)
    calc = InitialDatasetPARSL._setup_foundational(
        _idg_stub(), "so3lr", settings
    )
    assert calc.models[0].dispersion_energy_cutoff_lr_damping == 1.5
    assert calc.models[0].r_max_lr == 10.0


def test_teacher_accepts_keyword(restore_default_dtype):
    calc = _build_teacher_calculator(
        "so3lr",
        "tauri-s-2606",
        {"theory_level": 5, "device": "cpu", "default_dtype": "float32"},
    )
    _assert_finite_water(calc)


def test_teacher_passes_lr_damping(restore_default_dtype):
    calc = _build_teacher_calculator(
        "so3lr",
        "v1",
        {
            "dispersion_lr_damping": 1.5,
            "r_max_lr": 12.0,
            "device": "cpu",
            "default_dtype": "float32",
        },
    )
    assert calc.models[0].dispersion_energy_cutoff_lr_damping == 1.5


def test_teacher_so3krates_still_uses_model_paths(monkeypatch):
    import so3krates_torch.calculator.so3 as so3

    captured = {}

    class FakeCalc:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(so3, "TorchkratesCalculator", FakeCalc)
    _build_teacher_calculator("so3krates", "some.model", {"device": "cpu"})
    assert captured["model_paths"] == "some.model"


def _al_stub(tmp_path, foundational_model_settings):
    # the foundational branch also loads the ensemble from model_dir
    from aims_PAX.tools.model_tools.so3lr_loading import load_so3lr

    torch.save(
        load_so3lr("v1", "cpu", "float32"), tmp_path / "ensemble_0.model"
    )
    config = SimpleNamespace(
        use_foundational=True,
        foundational_model_settings=foundational_model_settings,
        device="cpu",
        dtype="float32",
        compute_stress=False,
        key_specification=None,
        model_dir=str(tmp_path),
        model_choice="so3lr",
        use_multihead_model=False,
        r_max_lr=12.0,
        dispersion_energy_cutoff_lr_damping=2.0,
        enable_cueq=False,
    )
    return SimpleNamespace(config=config, models=None, mlff_calc=None)


def test_al_foundational_tauri_keyword(tmp_path, restore_default_dtype):
    stub = _al_stub(
        tmp_path,
        {
            "model_type": "so3lr",
            "model_path": "tauri-s-2606",
            "theory_level": 5,
        },
    )
    ALCalculatorMLFF._setup_mlff_calculator(stub)
    _assert_finite_water(stub.mlff_calc)


def test_al_foundational_lr_damping(tmp_path, restore_default_dtype):
    stub = _al_stub(
        tmp_path,
        {
            "model_type": "so3lr",
            "model_path": "v1",
            "r_max_lr": 12.0,
            "dispersion_lr_damping": 1.5,
        },
    )
    ALCalculatorMLFF._setup_mlff_calculator(stub)
    assert stub.mlff_calc.models[0].dispersion_energy_cutoff_lr_damping == 1.5
