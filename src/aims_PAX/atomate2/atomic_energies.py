"""
This is a representation of `ensemble_atomic_energies` and `ensemble_atomic_energies_dict`
"""
from dataclasses import dataclass

import numpy as np
from monty.json import MSONable


@dataclass
class AtomicEnergies(MSONable):
    _e: dict[int, float]
    need_updating: bool = False

    @classmethod
    def from_z(cls, zs: list[int], need_updating: bool = False) -> "AtomicEnergies":
        """Create an AtomicEnergies object from a list of tags and a list of atomic numbers."""
        return cls(_e={z: 0 for z in np.sort(np.unique(zs))},
                   need_updating=need_updating)

    @classmethod
    def from_e(cls, e: dict[int, float], need_updating: bool = False):
        return cls(_e=e, need_updating=need_updating)

    def as_list(self):
        return list(self._e.values())

    def as_dict(self):
        return self._e
