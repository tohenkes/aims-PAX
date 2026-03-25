"""
This module contains an MSONable version of the MACE model.
"""
from pathlib import Path
import torch
from mace.modules.models import MACE as MACE_
from monty.json import MSONable


class MACE(MACE_, MSONable):

    def as_dict(self) -> dict:
        torch.save(self, Path.cwd() / "model.pt")
        return {"checkpoint": Path.cwd() / "model.pt"}

    @classmethod
    def from_dict(cls, d):
        return torch.load(d["checkpoint"], weights_only=False)
