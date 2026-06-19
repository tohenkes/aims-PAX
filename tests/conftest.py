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
