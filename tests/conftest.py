"""
Fixtures used for aims-PAX tests
"""

import os
import tempfile

import pytest
from pathlib import Path

import yaml
from pyfhiaims.geometry import AimsGeometry
from pymatgen.core import Molecule

from aims_PAX.settings import AimsPAXSettings


@pytest.fixture(autouse=True)
def _restore_torch_default_dtype():
    """Code under test sets the global torch default dtype (e.g.
    ``apply_model_settings``); restore it so it cannot leak into the
    next test."""
    import torch

    previous_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(previous_dtype)


@pytest.fixture(scope="session")
def data_dir():
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def species_dir(data_dir) -> Path:
    return data_dir / "species_defaults" / "light"


@pytest.fixture(scope="session")
def control_molecule(data_dir) -> Path:
    return data_dir / "control_files" / "molecule.in"


@pytest.fixture(scope="session")
def control_periodic(data_dir) -> Path:
    return data_dir / "control_files" / "periodic.in"


@pytest.fixture(scope="session")
def aspirin(data_dir) -> Path:
    return data_dir / "structures" / "aspirin.in"


@pytest.fixture(scope="session")
def si(data_dir) -> Path:
    return data_dir / "structures" / "Si.in"


@pytest.fixture
def project_settings(data_dir, species_dir):
    def get_settings(
        path_to_control: Path, path_to_geometry: Path, output_dir: Path
    ):
        settings_file = data_dir / "project_settings" / "aimsPAX.yaml"
        with open(settings_file) as f:
            data = yaml.safe_load(f)
        # Patch relative species_dir before pydantic validates paths
        data["INITIAL_DATASET_GENERATION"]["species_dir"] = str(species_dir)
        data["ACTIVE_LEARNING"]["species_dir"] = str(species_dir)
        settings = AimsPAXSettings.model_validate(data)
        settings.MISC.path_to_control = path_to_control
        settings.MISC.path_to_geometry = path_to_geometry
        settings.MISC.output_dir = output_dir
        return settings

    return get_settings


@pytest.fixture(scope="session")
def clean_dir(debug_mode):
    old_cwd = Path.cwd()
    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    os.chdir(tmp_dir)
    yield Path(tmp_dir)
    os.chdir(old_cwd)
    if not debug_mode:
        tmp_dir_obj.cleanup()
    else:
        print(f"Tests ran in {tmp_dir}")


@pytest.fixture(scope="session")
def debug_mode():
    return True


@pytest.fixture
def z_table():
    from mace.tools import AtomicNumberTable

    return AtomicNumberTable([1, 8])


@pytest.fixture
def so3lr_model_settings():
    """Factory: ``ModelSettings`` for a small SO3LR model.

    Long-range terms are off by default; keyword arguments override the
    ``ARCHITECTURE`` entries.
    """
    from aims_PAX.settings import ModelSettings
    from tests.helpers import so3lr_cfg

    def build(ARCHITECTURE=None):
        arch = {
            "zbl_repulsion_bool": True,
            "electrostatic_energy_bool": False,
            "dispersion_energy_bool": False,
            **(ARCHITECTURE or {}),
        }
        return ModelSettings.model_validate(so3lr_cfg(**arch))

    return build


@pytest.fixture
def pretrained_settings():
    """Factory: ``ModelSettings`` for a multi-head SO3LR started from a
    pretrained model.

    Each keyword argument goes to the settings section that has a field
    of that name (``TRAINING``, ``ARCHITECTURE`` or ``GENERAL``), so only
    the passed keys count as set by the user. The global default dtype
    is restored afterwards.
    """
    import torch

    from aims_PAX.settings import ModelSettings
    from aims_PAX.settings.model import (
        GeneralSettings,
        SO3LRArchitectureSettings,
        TrainingSettings,
    )

    previous_dtype = torch.get_default_dtype()

    def build(**kwargs):
        sections = {
            "GENERAL": {"name_exp": "test"},
            "ARCHITECTURE": {
                "model_choice": "so3lr",
                "use_multihead_model": True,
            },
            "TRAINING": {},
            "MISC": {"device": "cpu"},
        }
        for key, value in kwargs.items():
            if key in TrainingSettings.model_fields:
                sections["TRAINING"][key] = value
            elif key in SO3LRArchitectureSettings.model_fields:
                sections["ARCHITECTURE"][key] = value
            elif key in GeneralSettings.model_fields:
                sections["GENERAL"][key] = value
            else:
                raise KeyError(f"unknown settings key {key!r}")
        return ModelSettings.model_validate(sections)

    yield build
    torch.set_default_dtype(previous_dtype)


@pytest.fixture
def target():
    """Empty object for ``apply_model_settings``; restores the global
    default dtype it sets."""
    from types import SimpleNamespace

    import torch

    previous_dtype = torch.get_default_dtype()
    yield SimpleNamespace()
    torch.set_default_dtype(previous_dtype)
