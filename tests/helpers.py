from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nose_hoover_chain import IsotropicMTKNPT, MTKNPT

NVT_LANGEVIN = dict(
    stat_ensemble="nvt",
    thermostat="langevin",
    timestep=1.0,
    friction=0.001,
    temperature=300.0,
    seed=42,
)

NPT_BERENDSEN = dict(
    stat_ensemble="npt",
    barostat="berendsen",
    timestep=1.0,
    temperature=300.0,
    pressure=101325.0,
)

NPT_MTK = dict(
    stat_ensemble="npt",
    barostat="mtk",
    timestep=1.0,
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

MD_CASES = [
    ("nvt_langevin", Langevin, NVT_LANGEVIN),
    ("npt_berendsen", NPTBerendsen, NPT_BERENDSEN),
    ("npt_mtk", MTKNPT, NPT_MTK),
    ("npt_isomtk", IsotropicMTKNPT, NPT_ISOMTK),
]
