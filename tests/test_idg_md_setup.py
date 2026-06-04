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


def call_setup_md(atoms, md_settings, *, restart=False):
    stub = SimpleNamespace(restart=restart)
    return PrepareInitialDatasetProcedure.setup_md(stub, atoms, md_settings)


NVT_LANGEVIN = dict(
    stat_ensemble="nvt",
    thermostat="langevin",
    timestep=0.5,
    friction=0.001,
    temperature=300.0,
    seed=42,
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


@pytest.fixture
def cu_atoms():
    return ase.build.bulk("Cu", "fcc", a=3.6)


# ===========================================================================
# §3.1 — NVT/Langevin
# ===========================================================================


def test_nvt_langevin_returns_langevin(cu_atoms):
    dyn = call_setup_md(cu_atoms, NVT_LANGEVIN)
    assert isinstance(dyn, Langevin)


def test_nvt_langevin_friction(cu_atoms):
    dyn = call_setup_md(cu_atoms, NVT_LANGEVIN)
    # Source passes friction=md_settings["friction"] / units.fs; ASE
    # stores as dyn.fr
    assert dyn.fr == pytest.approx(NVT_LANGEVIN["friction"] / units.fs)


def test_nvt_langevin_rng(cu_atoms):
    dyn = call_setup_md(cu_atoms, NVT_LANGEVIN)
    assert isinstance(dyn.rng, np.random.RandomState)


# ===========================================================================
# §3.2 — NPT ensembles
# ===========================================================================


def test_npt_berendsen_returns_nptberendsen(cu_atoms):
    dyn = call_setup_md(cu_atoms, NPT_BERENDSEN)
    assert isinstance(dyn, NPTBerendsen)


def test_npt_berendsen_pressure(cu_atoms):
    dyn = call_setup_md(cu_atoms, NPT_BERENDSEN)
    assert dyn.pressure == pytest.approx(
        NPT_BERENDSEN["pressure"] * units.Pascal
    )


def test_npt_mtk_returns_mtknpt(cu_atoms):
    dyn = call_setup_md(cu_atoms, NPT_MTK)
    assert isinstance(dyn, MTKNPT)
    assert not isinstance(dyn, IsotropicMTKNPT)
    assert dyn._thermostat._tdamp == pytest.approx(NPT_MTK["tdamp"] * units.fs)
    assert dyn._barostat._pdamp == pytest.approx(NPT_MTK["pdamp"] * units.fs)
    assert dyn._thermostat._tchain == NPT_MTK["tchain"]
    assert dyn._barostat._pchain == NPT_MTK["pchain"]
    assert dyn._thermostat._tloop == NPT_MTK["tloop"]
    assert dyn._barostat._ploop == NPT_MTK["ploop"]


def test_npt_isomtk_returns_isotropicmtknpt(cu_atoms):
    dyn = call_setup_md(cu_atoms, NPT_ISOMTK)
    assert isinstance(dyn, IsotropicMTKNPT)
    assert dyn._thermostat._tdamp == pytest.approx(
        NPT_ISOMTK["tdamp"] * units.fs
    )
    assert dyn._barostat._pdamp == pytest.approx(
        NPT_ISOMTK["pdamp"] * units.fs
    )
    assert dyn._thermostat._tchain == NPT_ISOMTK["tchain"]
    assert dyn._barostat._pchain == NPT_ISOMTK["pchain"]
    assert dyn._thermostat._tloop == NPT_ISOMTK["tloop"]
    assert dyn._barostat._ploop == NPT_ISOMTK["ploop"]


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
# §3.4 — Berendsen optional kwargs
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
