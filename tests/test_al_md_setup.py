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
