from pathlib import Path

import ase.io
from mace.tools import AtomicNumberTable
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nose_hoover_chain import IsotropicMTKNPT, MTKNPT

import so3krates_torch.tools.torch_geometric as so3_torch_geometric

from aims_PAX.settings import ModelSettings
from aims_PAX.tools.model_tools.setup_MACE import setup_mace
from aims_PAX.tools.utilities.data_handling import create_model_dataset
from aims_PAX.tools.utilities.utilities import (
    compute_average_E0s,
    create_keyspec,
)

_TEST_DATA = Path(__file__).parent / "test_data"
_TRAIN_XYZ = (
    _TEST_DATA
    / "datasets/initial/training/combined_initial_train_set.xyz"
)


def build_si_model(
    z_table=None,
    r_max=5.0,
    num_interactions=1,
    hidden_irreps="8x0e",
    MLP_irreps="8x0e",
    correlation=2,
):
    """Build a small MACE model suitable for Si testing.

    Args:
        z_table: AtomicNumberTable; defaults to [14] (Si).
        r_max: Cutoff radius. Defaults to 5.0.
        num_interactions: Number of interaction layers. Defaults to 1.
        hidden_irreps: Hidden irreps string. Defaults to "8x0e".
        MLP_irreps: MLP irreps string. Defaults to "8x0e".
        correlation: Correlation order. Defaults to 2.

    Returns:
        torch.nn.Module: Initialised MACE model.
    """
    if z_table is None:
        z_table = AtomicNumberTable([14])
    atoms_list = ase.io.read(_TRAIN_XYZ, index=":")
    energies = [a.info["REF_energy"] for a in atoms_list]
    zs = [a.get_atomic_numbers() for a in atoms_list]
    atomic_energies_dict = compute_average_E0s(energies, zs, z_table)
    settings = ModelSettings(
        **{
            "GENERAL": {
                "name_exp": "test",
                "seed": 42,
                "default_dtype": "float64",
            },
            "ARCHITECTURE": {
                "model_choice": "mace",
                "num_channels": 8,
                "num_interactions": num_interactions,
                "max_L": 0,
            },
            "MISC": {"device": "cpu"},
        }
    )
    return setup_mace(settings, z_table, atomic_energies_dict)


def make_loader(
    atoms_list,
    keyspec,
    z_table,
    r_max,
    batch_size=2,
    shuffle=False,
):
    """Create a DataLoader from a list of ase.Atoms.

    Args:
        atoms_list: List of ASE Atoms objects with REF_energy/REF_forces.
        keyspec: KeySpecification for the dataset.
        z_table: AtomicNumberTable matching the model.
        r_max: Cutoff radius matching the model.
        batch_size: Batch size. Defaults to 2.
        shuffle: Whether to shuffle. Defaults to False.

    Returns:
        DataLoader: so3krates DataLoader over the dataset.
    """
    dataset = create_model_dataset(
        data=atoms_list,
        seed=42,
        z_table=z_table,
        r_max=r_max,
        key_specification=keyspec,
    )
    loader = so3_torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
    return loader

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
