"""
This module contains an MSONable version of the MACE model.
"""
from pathlib import Path
import torch
from mace.modules.models import MACE, ScaleShiftMACE
from monty.json import MSONable


class MSONMixin(MSONable):

    @classmethod
    def from_parent(cls, instance):
        instance.__class__ = cls
        return instance

    def as_dict(self) -> dict:
        path = Path.cwd() / "model.pt"
        torch.save(self, path)
        return {"checkpoint": str(path)}

    @classmethod
    def from_dict(cls, d):
        return cls.from_parent(torch.load(d["checkpoint"], weights_only=False))


class MSONMACE(MACE, MSONMixin):
    pass


class MSONScaleShiftMACE(ScaleShiftMACE, MSONMixin):
    pass