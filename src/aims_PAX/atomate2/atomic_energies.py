"""
This is a representation of `ensemble_atomic_energies` and `ensemble_atomic_energies_dict`
"""
from dataclasses import dataclass

import numpy as np
from monty.json import MSONable


@dataclass
class AtomicEnergies(MSONable):
    _e: dict[str, dict[int, float]]

    @classmethod
    def from_z(cls, tags: list[str], zs: list[int]) -> "AtomicEnergies":
        """Create an AtomicEnergies object from a list of tags and a list of atomic numbers."""
        return cls(_e={
            tag: {z: 0 for z in np.sort(np.unique(zs))}
            for tag in tags
        })

    @classmethod
    def from_e(cls, tags: list[str], e: dict[int, float]):
        return cls(_e={tag: e for tag in tags})

    def get(self, tag: str) -> dict[int, float]:
        return self._e[tag]

    def get_array(self, tag: str) -> np.ndarray:
        return np.array(self._e[tag].values())
