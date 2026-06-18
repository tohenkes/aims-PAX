from typing import Union

try:
    from atomate2.aims.jobs.core import StaticMaker as AimsStaticMaker
    from atomate2.forcefields.md import ForceFieldMDMaker

    AllowedReferenceMakers = Union[AimsStaticMaker]
    AllowedMDMakers = Union[ForceFieldMDMaker]
except ImportError:
    AimsStaticMaker = None
    ForceFieldMDMaker = None
    AllowedReferenceMakers = None
    AllowedMDMakers = None