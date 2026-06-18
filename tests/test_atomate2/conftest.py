"""
Fixtures specific to Atomate2 workflows
"""

import pytest

try:
    from atomate2.aims.jobs.core import StaticMaker as AimsStaticMaker
    from atomate2.forcefields.md import ForceFieldMDMaker

    _ATOMATE2_AVAILABLE = True
except ImportError:
    AimsStaticMaker = None
    ForceFieldMDMaker = None
    _ATOMATE2_AVAILABLE = False


@pytest.fixture
def reference_maker(data_dir):
    if not _ATOMATE2_AVAILABLE:
        pytest.skip("atomate2 not installed")
    return AimsStaticMaker()


@pytest.fixture
def md_maker():
    if not _ATOMATE2_AVAILABLE:
        pytest.skip("atomate2 not installed")
    return ForceFieldMDMaker()
