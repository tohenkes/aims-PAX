"""
Fixtures specific to Atomate2 workflows
"""
import pytest

from atomate2.aims.jobs.core import StaticMaker as AimsStaticMaker
from atomate2.forcefields.md import ForceFieldMDMaker


@pytest.fixture
def reference_maker(data_dir):
    return AimsStaticMaker()


@pytest.fixture
def md_maker():
    return ForceFieldMDMaker()