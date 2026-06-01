"""
Fixtures used for aims-PAX tests
"""

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def data_dir():
    return Path(__file__).parent / "test_data"
