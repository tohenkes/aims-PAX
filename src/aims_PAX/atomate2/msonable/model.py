"""
This is a module with an MSONable version of the Model classes.
"""
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ase import Atoms
from mace import tools
from mace.data import KeySpecification
from mace.tools import AtomicNumberTable
from monty.json import MSONable
from pymatgen.io.ase import MSONAtoms
from so3krates_torch.tools import torch_geometric as so3_torch_geometric

from .cache import Cache
from .serialization import MSONableMACE, MSONableModel
from ..atomic_energies import AtomicEnergies
from ..utils import to_msonatoms
from ...settings import ModelSettings
from ...tools.model_tools.train_epoch import train_epoch, evaluate
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
    valid_loss: float = np.inf
    eval_metric: dict[str, Any] = None
    _cache: Cache = field(default_factory=Cache, init=False, repr=False, compare=False)

    def __post_init__(self):
        """Sets the train and valid sets to empty lists. Initialize the caching mechanism"""
        self.train_set = []
        self.valid_set = []
        self._cache = Cache()

    @property
    def z_table(self):
        return AtomicNumberTable(self.zs)

    @property
    def train_model_set(self):
        return self._cache.cached(
            "train_model_set",
            lambda: self._create_model_set(self.train_set))

    @property
    def valid_model_set(self):
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    def train_loader(self):
        return self._cache.cached(
            "train_loader",
            lambda: so3_torch_geometric.dataloader.DataLoader(
            dataset=self.train_model_set,
            batch_size=self.batch_size,
            # shuffle=True,
            # drop_last=True,
            )
        )

    @property
    def valid_loader(self):
        raise NotImplementedError

    @property
    def ase_sets(self):
        return {"train": self.train_set, "valid": self.valid_set}

    def update_train_set(self, train_set: list[MSONAtoms | Atoms]):
        train_set = [a if isinstance(a, MSONAtoms) else to_msonatoms(a) for a in train_set]
        self.train_set += train_set
        self._cache.invalidate("train_model_set")
        self._cache.invalidate("train_loader")

    def update_valid_set(self, valid_set: list[MSONAtoms | Atoms]):
        valid_set = [a if isinstance(a, MSONAtoms) else to_msonatoms(a) for a in valid_set]
        self.valid_set += valid_set
        self._cache.invalidate("valid_model_set")
        self._cache.invalidate("valid_loader")

    def train(self, training_setup: dict[str, Any], n_epochs: int):
        """
        Trains a model based on the provided training setup and the number of epochs.

        Args:
            training_setup (dict[str, Any]): A dictionary containing the configuration
                for training.
            n_epochs (int): The number of epochs for which the model should be trained.

        Raises:
            NotImplementedError: Indicates that the method is not yet implemented.
        """
        raise NotImplementedError

    def validate(self, training_setup: dict[str, Any]):
        raise NotImplementedError

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


@dataclass
class MaceModelTrainer(ModelTrainer):
    model: MSONableMACE
    head: str = "Default"

    @classmethod
    def from_model(cls, instance, **kwargs):
        """Get a ModelTrainer from a MACE model"""
        return cls(model=MSONableMACE.from_parent(instance), **kwargs)

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

    @property
    def valid_model_set(self):
        return self._cache.cached(
            "valid_model_set",
            lambda: {self.head: self._create_model_set(self.valid_set)})

    @property
    def valid_loader(self):
        return self._cache.cached(
            "valid_loader",
            lambda: {k: so3_torch_geometric.dataloader.DataLoader(
            dataset=v,
            batch_size=self.valid_batch_size,
            # shuffle=True,
            # drop_last=True,
            ) for k, v in self.valid_model_set.items()}
        )

    def train(self, training_setup: dict[str, Any], n_epochs: int):
        model_sets = dict(
            train=self.train_model_set,
            valid=self.valid_model_set,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader
        )
        logging.info(
            f"Training set size for '{self.tag}': {len(self.train_model_set)}; "
            f"Validation set size: "
        )
        for head, valid_set in self.valid_model_set.items():
            logging.info(f" - Head '{head}': {len(valid_set)}")
        self._update_auxiliaries(model_sets)
        logger = tools.MetricsLogger(
            directory=self.settings.GENERAL.loss_dir.as_posix(),
            tag=f"{self.tag}_train",
        )
        for i in range(n_epochs):
            train_epoch(
                model=self.model,
                train_loader=self.train_loader,
                loss_fn=training_setup["loss_fn"],
                optimizer=training_setup["optimizer"],
                lr_scheduler=training_setup["lr_scheduler"],
                epoch=i,
                start_epoch=0,
                valid_loss=self.valid_loss,
                logger=logger,
                device=training_setup["device"],
                max_grad_norm=training_setup["max_grad_norm"],
                output_args=training_setup["output_args"],
                ema=training_setup["ema"],
            )

    def validate(self, training_setup: dict[str, Any]):
        """Validates the model on the validation set, returns losses"""
        ema = training_setup["ema"]
        loss_fn = training_setup["loss_fn"]
        device = training_setup["device"]
        output_args = training_setup["output_args"]

        # get context based on ema presence
        ctx = ema.average_parameters() if ema is not None else contextlib.nullcontext()
        with ctx:
            self.valid_loss, self.eval_metric = evaluate(
                model=self.model,
                loss_fn=loss_fn,
                data_loader=self.valid_loader["Default"],
                output_args=output_args,
                device=device,
            )
        return self.valid_loss, self.eval_metric

    def _update_auxiliaries(self, model_sets: dict[str, Any]):
        _, atomic_energies = update_model_auxiliaries(
            model=self.model,
            model_choice="mace",
            model_sets=model_sets,
            # scaling is set for MACE models only
            scaling=self.settings.ARCHITECTURE.scaling,
            z_table=self.z_table,
            atomic_energies_list=self.atomic_energies.as_list(),
            atomic_energies_dict=self.atomic_energies.as_dict(),
            update_atomic_energies=self.atomic_energies.need_updating,
            update_avg_num_neighbors=self.settings.TRAINING.update_avg_num_neighbors,
            dtype=self.settings.GENERAL.default_dtype,
            device=self.settings.MISC.device,
        )
        self.atomic_energies = AtomicEnergies.from_e(atomic_energies, need_updating=False)
