import numpy as np
import pytest
from types import SimpleNamespace

import ase
import ase.build
from ase import units

from aims_PAX.procedures.preparation import ALMD
from tests.helpers import MD_CASES, NVT_LANGEVIN, NPT_BERENDSEN, NPT_MTK


def call_setup_md(atoms, md_settings, *, restart=False):
    stub = SimpleNamespace(config=SimpleNamespace(restart=restart))
    stub._initialize_velocities = lambda a, s: ALMD._initialize_velocities(
        stub, a, s
    )
    stub._create_dynamics_engine = (
        lambda a, s, e, i: ALMD._create_dynamics_engine(stub, a, s, e, i)
    )
    stub._create_nvt_dynamics = lambda a, s, i: ALMD._create_nvt_dynamics(
        stub, a, s, i
    )
    stub._create_npt_dynamics = lambda a, s, i: ALMD._create_npt_dynamics(
        stub, a, s, i
    )
    stub._create_berendsen_npt = lambda a, s, i: ALMD._create_berendsen_npt(
        stub, a, s, i
    )
    stub._create_mkt_npt = lambda a, s, i, iso=False: ALMD._create_mkt_npt(
        stub, a, s, i, iso
    )
    return ALMD._setup_md_dynamics(stub, atoms, md_settings, idx=0)


@pytest.fixture
def cu_atoms():
    return ase.build.bulk("Cu", "fcc", a=3.6)


# ===========================================================================
# §1 — Shared type checks (parametrized)
# ===========================================================================


@pytest.mark.parametrize("name,expected_type,settings", MD_CASES)
def test_dynamics_type(cu_atoms, name, expected_type, settings):
    dyn = call_setup_md(cu_atoms, settings)
    assert type(dyn) is expected_type


# ===========================================================================
# §2 — Shared physics assertions
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
# §3 — Restart gate
# ===========================================================================


def test_velocities_initialized_when_not_restart(cu_atoms):
    assert not np.any(cu_atoms.get_momenta())
    call_setup_md(cu_atoms, NVT_LANGEVIN, restart=False)
    assert np.any(cu_atoms.get_momenta())


def test_velocities_not_initialized_when_restart(cu_atoms):
    assert not np.any(cu_atoms.get_momenta())
    call_setup_md(cu_atoms, NVT_LANGEVIN, restart=True)
    assert not np.any(cu_atoms.get_momenta())


# ===========================================================================
# §4 — Invalid values raise ValueError
# ===========================================================================


def test_invalid_stat_ensemble_raises(cu_atoms):
    settings = {**NVT_LANGEVIN, "stat_ensemble": "invalid"}
    with pytest.raises(ValueError):
        call_setup_md(cu_atoms, settings)


def test_invalid_thermostat_raises(cu_atoms):
    settings = {**NVT_LANGEVIN, "thermostat": "invalid"}
    with pytest.raises(ValueError):
        call_setup_md(cu_atoms, settings)


def test_invalid_barostat_raises(cu_atoms):
    settings = {**NPT_BERENDSEN, "barostat": "invalid"}
    with pytest.raises(ValueError):
        call_setup_md(cu_atoms, settings)


# ---------------------------------------------------------------------------
# MLFF calculators read charge/spin under the configured MISC keys
# ---------------------------------------------------------------------------


def _calc_setup_stub(monkeypatch, *, foundational, model_choice, multihead):
    import aims_PAX.procedures.preparation as prep
    from aims_PAX.tools.utilities.utilities import create_keyspec

    recorded = {}

    def recorder(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(prep, "MultiHeadSO3LRCalculator", recorder)
    monkeypatch.setattr(prep, "TorchkratesCalculator", recorder)
    monkeypatch.setattr(
        prep,
        "make_so3lr_calculator",
        lambda *a, **k: (recorded.update(k) or SimpleNamespace()),
    )
    monkeypatch.setattr(prep, "list_files_in_directory", lambda d: [])

    config = SimpleNamespace(
        use_foundational=foundational,
        foundational_model_settings={
            "model_type": model_choice,
            "model_path": "dummy",
        },
        model_choice=model_choice,
        use_multihead_model=multihead,
        model_dir="unused",
        device="cpu",
        dtype="float32",
        compute_stress=False,
        r_max_lr=None,
        dispersion_energy_cutoff_lr_damping=None,
        enable_cueq=False,
        key_specification=create_keyspec(
            total_charge_key="q", total_spin_key="s"
        ),
    )
    stub = SimpleNamespace(config=config)
    return prep.ALCalculatorMLFF, stub, recorded


@pytest.mark.parametrize(
    "foundational,model_choice,multihead",
    [
        (True, "so3lr", False),
        (True, "so3krates", False),
        (False, "so3lr", True),
        (False, "so3lr", False),
        (False, "so3krates", False),
    ],
)
def test_so3lr_md_calculator_uses_misc_charge_spin_keys(
    monkeypatch, foundational, model_choice, multihead
):
    cls, stub, recorded = _calc_setup_stub(
        monkeypatch,
        foundational=foundational,
        model_choice=model_choice,
        multihead=multihead,
    )
    cls._setup_mlff_calculator(stub)
    info_keys = recorded["key_specification"].info_keys
    assert info_keys["total_charge"] == "q"
    assert info_keys["total_spin"] == "s"


def test_so3lr_foundational_ensemble_calculator_uses_misc_keys(monkeypatch):
    """Foundational MD + the ensemble (uncertainty) calculator both get it."""
    import aims_PAX.procedures.preparation as prep

    cls, stub, recorded = _calc_setup_stub(
        monkeypatch, foundational=True, model_choice="so3lr", multihead=True
    )
    calls = []
    monkeypatch.setattr(
        prep,
        "MultiHeadSO3LRCalculator",
        lambda **kw: calls.append(kw) or SimpleNamespace(),
    )
    cls._setup_mlff_calculator(stub)
    assert len(calls) == 1
    assert calls[0]["key_specification"].info_keys["total_charge"] == "q"
