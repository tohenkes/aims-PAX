import numpy as np
import pytest
from types import SimpleNamespace

import ase
import ase.build
from ase import units
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nose_hoover_chain import MTKNPT, IsotropicMTKNPT

from aims_PAX.procedures.preparation import ALMD


def call_setup_md(atoms, md_settings, *, restart=False):
    stub = SimpleNamespace(
        config=SimpleNamespace(restart=restart)
    )
    stub._initialize_velocities = (
        lambda a, s: ALMD._initialize_velocities(stub, a, s)
    )
    stub._create_dynamics_engine = (
        lambda a, s, e, i: ALMD._create_dynamics_engine(stub, a, s, e, i)
    )
    stub._create_nvt_dynamics = (
        lambda a, s, i: ALMD._create_nvt_dynamics(stub, a, s, i)
    )
    stub._create_npt_dynamics = (
        lambda a, s, i: ALMD._create_npt_dynamics(stub, a, s, i)
    )
    stub._create_berendsen_npt = (
        lambda a, s, i: ALMD._create_berendsen_npt(stub, a, s, i)
    )
    stub._create_mkt_npt = (
        lambda a, s, i, iso=False: ALMD._create_mkt_npt(stub, a, s, i, iso)
    )
    return ALMD._setup_md_dynamics(stub, atoms, md_settings, idx=0)


NVT_LANGEVIN = dict(
    stat_ensemble="nvt",
    thermostat="langevin",
    timestep=1.0,
    friction=0.01,
    temperature=300,
    seed=42,
)

NPT_BERENDSEN = dict(
    stat_ensemble="npt",
    barostat="berendsen",
    timestep=1.0,
    temperature=300,
    pressure=101325.0,
)

NPT_MTK = dict(
    stat_ensemble="npt",
    barostat="mtk",
    timestep=1.0,
    temperature=300,
    pressure=101325.0,
    tdamp=100.0,
    pdamp=100.0,
    tchain=3,
    pchain=3,
    tloop=2,
    ploop=2,
)

NPT_ISOMTK = NPT_MTK | dict(barostat="isomtk")


@pytest.fixture
def cu_atoms():
    return ase.build.bulk("Cu", "fcc", a=3.6)


# ===========================================================================
# §1 — NVT/Langevin type
# ===========================================================================


def test_nvt_langevin_type(cu_atoms):
    dyn = call_setup_md(cu_atoms, NVT_LANGEVIN)
    assert isinstance(dyn, Langevin)


# ===========================================================================
# §2 — NVT/Langevin friction
# ===========================================================================


def test_nvt_langevin_friction(cu_atoms):
    dyn = call_setup_md(cu_atoms, NVT_LANGEVIN)
    assert dyn.fr == pytest.approx(NVT_LANGEVIN["friction"] / units.fs)


# ===========================================================================
# §3 — NVT/Langevin RNG
# ===========================================================================


def test_nvt_langevin_rng(cu_atoms):
    dyn = call_setup_md(cu_atoms, NVT_LANGEVIN)
    assert isinstance(dyn.rng, np.random.RandomState)


# ===========================================================================
# §4 — NPT Berendsen type
# ===========================================================================


def test_npt_berendsen_type(cu_atoms):
    dyn = call_setup_md(cu_atoms, NPT_BERENDSEN)
    assert isinstance(dyn, NPTBerendsen)


# ===========================================================================
# §5 — NPT Berendsen pressure
# ===========================================================================


def test_npt_berendsen_pressure(cu_atoms):
    dyn = call_setup_md(cu_atoms, NPT_BERENDSEN)
    assert dyn.pressure == pytest.approx(
        NPT_BERENDSEN["pressure"] * units.Pascal
    )


# ===========================================================================
# §6 — NPT MTK type
# ===========================================================================


def test_npt_mtk_type(cu_atoms):
    dyn = call_setup_md(cu_atoms, NPT_MTK)
    assert isinstance(dyn, MTKNPT)
    assert not isinstance(dyn, IsotropicMTKNPT)


# ===========================================================================
# §7 — NPT IsotropicMTK type
# ===========================================================================


def test_npt_isomtk_type(cu_atoms):
    dyn = call_setup_md(cu_atoms, NPT_ISOMTK)
    assert isinstance(dyn, IsotropicMTKNPT)


# ===========================================================================
# §8 — NPT MTK chain parameters
# ===========================================================================


def test_npt_mtk_chain_params(cu_atoms):
    dyn = call_setup_md(cu_atoms, NPT_MTK)
    assert dyn._thermostat._tdamp == pytest.approx(
        NPT_MTK["tdamp"] * units.fs
    )
    assert dyn._barostat._pdamp == pytest.approx(
        NPT_MTK["pdamp"] * units.fs
    )
    assert dyn._thermostat._tchain == NPT_MTK["tchain"]
    assert dyn._barostat._pchain == NPT_MTK["pchain"]
    assert dyn._thermostat._tloop == NPT_MTK["tloop"]
    assert dyn._barostat._ploop == NPT_MTK["ploop"]


# ===========================================================================
# §9 — Velocities initialized when not restart
# ===========================================================================


def test_velocities_initialized_when_not_restart(cu_atoms):
    assert not np.any(cu_atoms.get_momenta())
    call_setup_md(cu_atoms, NVT_LANGEVIN, restart=False)
    assert np.any(cu_atoms.get_momenta())


# ===========================================================================
# §10 — Velocities not initialized when restart
# ===========================================================================


def test_velocities_not_initialized_when_restart(cu_atoms):
    assert not np.any(cu_atoms.get_momenta())
    call_setup_md(cu_atoms, NVT_LANGEVIN, restart=True)
    assert not np.any(cu_atoms.get_momenta())


# ===========================================================================
# §11 — Invalid stat_ensemble raises ValueError
# ===========================================================================


def test_invalid_stat_ensemble_raises(cu_atoms):
    settings = {**NVT_LANGEVIN, "stat_ensemble": "invalid"}
    with pytest.raises(ValueError):
        call_setup_md(cu_atoms, settings)


# ===========================================================================
# §12 — Invalid thermostat raises ValueError
# ===========================================================================


def test_invalid_thermostat_raises(cu_atoms):
    settings = {**NVT_LANGEVIN, "thermostat": "invalid"}
    with pytest.raises(ValueError):
        call_setup_md(cu_atoms, settings)


# ===========================================================================
# §13 — Invalid barostat raises ValueError
# ===========================================================================


def test_invalid_barostat_raises(cu_atoms):
    settings = {**NPT_BERENDSEN, "barostat": "invalid"}
    with pytest.raises(ValueError):
        call_setup_md(cu_atoms, settings)
