"""
This is a module with an MSONable version of the Model classes.
"""
from dataclasses import dataclass, Field

from mace.tools import AtomicNumberTable
from monty.json import MSONable

from .serialization import MSONableMACE
from ...settings import ModelSettings


@dataclass
class MaceModel(MSONable):
    model: MSONableMACE
    settings: ModelSettings
    zs: list[int]
    atomic_energies: dict[int, float]
    train_set_ase: list = None
    valid_set_ase: list = None

    def __post_init__(self):
        self.train_set_ase = []
        self.valid_set_ase = []

    def __getattr__(self, name):
        # avoid infinite recursion if 'model' itself isn't set yet
        if name == "model":
            raise AttributeError(name)
        return getattr(self.model, name)

    @classmethod
    def from_parent(cls, instance, **kwargs):
        return cls(model=MSONableMACE.from_parent(instance), **kwargs)

    @property
    def z_table(self):
        return AtomicNumberTable(self.zs)

    @property
    def train_set(self):
        return self.train_set_ase

    @property
    def valid_set(self):
        return self.valid_set_ase

    def update_datasets(self, train_set: list, valid_set: list):
        pass