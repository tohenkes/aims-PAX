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
