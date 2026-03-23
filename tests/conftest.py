"""
Fixtures used for aims-PAX tests
"""
import os
import tempfile

import pytest
from pathlib import Path

from pyfhiaims.geometry import AimsGeometry
from pymatgen.core import Molecule

from aims_PAX.settings import AimsPAXSettings


@pytest.fixture(scope="session")
def data_dir():
    return Path(__file__).parent / "test_data"

@pytest.fixture(scope="session")
def species_dir(data_dir):
    return data_dir / "species_defaults" / "light"

@pytest.fixture(scope="session")
def control_molecule(data_dir):
    return data_dir / "control_files" / "molecule.in"

@pytest.fixture(scope="session")
def aspirin(data_dir) -> Molecule:
    structure_file = data_dir / "structures" / "aspirin.in"
    return AimsGeometry.from_file(structure_file).to_structure()


@pytest.fixture
def project_settings(data_dir, species_dir):
    def get_settings(path_to_control: Path, path_to_geometry: Path, output_dir: Path):
        settings_file = data_dir / "project_settings" / "aimsPAX.yaml"
        settings = AimsPAXSettings.from_file(settings_file)
        settings.INITIAL_DATASET_GENERATION.species_dir = species_dir
        settings.ACTIVE_LEARNING.species_dir = species_dir
        settings.MISC.path_to_control = path_to_control
        settings.MISC.path_to_geometry = path_to_geometry
        settings.MISC.output_dir = output_dir
        return settings
    return get_settings


@pytest.fixture(scope="session")
def clean_dir(debug_mode):
    old_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(delete=not debug_mode) as tmp_dir:
        os.chdir(tmp_dir)
        yield tmp_dir
        os.chdir(old_cwd)
    if debug_mode:
        print(f"Tests ran in {tmp_dir}")


@pytest.fixture(scope="session")
def debug_mode():
    return True
