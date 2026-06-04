from typing import Union

from atomate2.aims.jobs.core import StaticMaker as AimsStaticMaker
from atomate2.forcefields.md import ForceFieldMDMaker

AllowedReferenceMakers = Union[AimsStaticMaker]
AllowedMDMakers = Union[ForceFieldMDMaker]