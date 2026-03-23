"""
This module contains a serializable version of np.randon.RandomState
"""

import numpy as np
from monty.json import MSONable


class RandomState(MSONable):
    def __init__(self, seed=None):
        self._rng = np.random.RandomState(seed)

    def as_dict(self):
        d = super().as_dict()
        state = list(self._rng.get_state())
        state[1] = state[1].tolist()
        d["state"] = state
        return d

    @classmethod
    def from_dict(cls, d):
        state = list(d["state"])
        state[1] = np.array(state[1], dtype=np.uint32)
        obj = cls(None)
        obj._rng.set_state(tuple(state))
        return obj

    def __getattr__(self, name):
        if name == "_rng":
            raise AttributeError("_rng")
        return getattr(self._rng, name)
