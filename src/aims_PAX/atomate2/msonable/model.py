"""
This is a module with an MSONable version of the Model classes.
"""
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms
from mace.data import KeySpecification
from mace.tools import AtomicNumberTable
from monty.json import MSONable
from pymatgen.io.ase import MSONAtoms

from .serialization import MSONableMACE, MSONableModel
from ..atomic_energies import AtomicEnergies
from ...settings import ModelSettings
from ...tools.utilities.data_handling import create_model_dataset, create_dataloader
from ...tools.utilities.utilities import update_model_auxiliaries


@dataclass
class ModelTrainer(MSONable):
    model: MSONableModel
    settings: ModelSettings
    tag: str
    zs: list[int]
    atomic_energies: AtomicEnergies
    key_specification: KeySpecification
    train_set: list = None
    valid_set: list = None
    valid_losses: float = np.inf

    def __post_init__(self):
        """Sets the train and valid sets to empty lists."""
        self.train_set = []
        self.valid_set = []

    @property
    def z_table(self):
        return AtomicNumberTable(self.zs)

    def update_train_set(self, train_set: list[MSONAtoms | Atoms]):
        train_set = [a if isinstance(a, MSONAtoms) else MSONAtoms(a) for a in train_set]
        self.train_set += train_set

    def update_valid_set(self, valid_set: list[MSONAtoms | Atoms]):
        valid_set = [a if isinstance(a, MSONAtoms) else MSONAtoms(a) for a in valid_set]
        self.valid_set += valid_set

    def train(self, training_setup: dict[str, Any]):
        """Train a model on given model sets."""
        raise NotImplementedError


@dataclass
class MaceModelTrainer(ModelTrainer):
    model: MSONableMACE
    head: str = "Default"

    @classmethod
    def from_model(cls, instance, **kwargs):
        return cls(model=MSONableMACE.from_parent(instance), **kwargs)

    @property
    def train_model_set(self):
        return self._create_model_set(self.train_set)

    @property
    def valid_model_set(self):
        return {self.head: self._create_model_set(self.valid_set)}

    @property
    def batch_size(self):
        """Returns the batch size for training."""
        return (
            1
            if len(self.train_set) < self.settings.TRAINING.batch_size
            else self.settings.TRAINING.batch_size
            )

    @property
    def valid_batch_size(self):
        """Returns the batch size for validation."""
        smallest_valid_set = 0
        # This should be subclassed for a multihead model
        # MACE models are always single head
        for valid_set in self.valid_model_set.values():
            if (
                    smallest_valid_set == 0
                    or len(valid_set) < smallest_valid_set
            ):
                smallest_valid_set = len(valid_set)

        return (
            1
            if smallest_valid_set
               < self.settings.TRAINING.valid_batch_size
            else self.settings.TRAINING.valid_batch_size
            )

    def train(self, training_setup: dict[str, Any]):
        """Trains a model on given model sets"""
        train_model_set = self.train_model_set
        valid_model_set = self.valid_model_set
        # create dataloaders
        train_loader, valid_loader = create_dataloader(
            train_model_set,
            valid_model_set,
            self.batch_size,
            self.valid_batch_size,
        )
        model_sets = dict(
            train=train_model_set,
            valid=valid_model_set,
            train_loader=train_loader,
            valid_loader=valid_loader
        )
        logging.info(
            f"Training set size for '{self.tag}': {len(train_model_set)}; "
            f"Validation set size: "
        )
        for head, valid_set in valid_model_set.items():
            logging.info(f" - Head '{head}': {len(valid_set)}")

    def _update_auxiliaries(self, model_sets: dict[str, Any]):
        update_model_auxiliaries(
            model=self.model,
            model_choice="mace",
            model_sets=model_sets,
            # scaling is set for MACE models only
            scaling=self.settings.ARCHITECTURE.scaling,
            z_table=self.z_table,
            atomic_energies_list=self.ensemble_atomic_energies[self.tag],
            atomic_energies_dict=self.ensemble_atomic_energies_dict[self.tag],
            update_atomic_energies=self.update_atomic_energies,
            update_avg_num_neighbors=self.settings.TRAINING.update_avg_num_neighbors,
            dtype=self.settings.GENERAL.default_dtype,
            device=self.settings.MISC.device,
        )


    def _create_model_set(self, data: list[MSONAtoms]):
        return create_model_dataset(
            data=data,
            z_table=self.z_table,
            seed=self.settings.GENERAL.seed,
            r_max=self.settings.ARCHITECTURE.r_max,
            # r_max_lr is used only in SO3LR / So3krates
            r_max_lr=None,
            key_specification=self.key_specification
        )

