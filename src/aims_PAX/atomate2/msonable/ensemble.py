"""
This module contains an MSONable version of the Ensemble model and Ensemble.
 Ensemble is basically a dict connecting Tag and Ensemble model.
"""
from dataclasses import dataclass
from typing import Any

from ase import Atoms
from mace import tools
from monty.json import MSONable
from pymatgen.io.ase import MSONAtoms

from aims_PAX.atomate2.atomic_energies import AtomicEnergies
from aims_PAX.atomate2.msonable.checkpoint_handler import MSONableCheckpointHandler
from aims_PAX.atomate2.msonable.ema import MSONableEMA
from aims_PAX.atomate2.msonable.model import MaceModelTrainer, ModelTrainer
from aims_PAX.atomate2.msonable.serialization import wrap
from aims_PAX.settings import ModelSettings
from aims_PAX.tools.model_tools.setup_MACE import setup_mace
from aims_PAX.tools.model_tools.train_epoch import validate_epoch_ensemble
from aims_PAX.tools.model_tools.training_tools import setup_model_training
from aims_PAX.tools.utilities.data_handling import KeySpecification


@dataclass
class Ensemble(MSONable):
    models: dict[str, ModelTrainer]
    training_setups: dict[str, dict[str, Any]]
    log_settings: dict[str, Any]
    epoch = 0

    @classmethod
    def from_scratch(cls,
                     tags: list[str],
                     model_settings: ModelSettings,
                     atomic_energies: dict[str, AtomicEnergies],
                     model_inputs: dict[str, Any],
                     key_specification: KeySpecification) -> "Ensemble":
        """
        Generate an ensemble from model settings and tags list.

        Args:
            tags: a list of tags.
            model_settings: a ModelSettings object.
            atomic_energies: atomic energies.
            model_inputs: model-dependent inputs (z_table)
            key_specification: a KeySpecification object.

        Returns:
            an Ensemble object.
        """
        # can we put calculate model inputs here?
        ensemble = {}
        training_setups = {}

        model_choice = model_settings.GENERAL.model_choice
        log_settings = dict(
            loss_dir=model_settings.GENERAL.loss_dir.as_posix(),
            log_errors=model_settings.MISC.error_table
        )
        for tag in tags:
            if model_choice == "mace":
                model = setup_mace(
                    model_settings,
                    model_inputs["z_table"],
                    atomic_energies[tag].as_dict()
                )
                ensemble[tag] = MaceModelTrainer.from_model(
                    model,
                    settings=model_settings,
                    tag=tag,
                    head="Default",
                    zs=model_inputs["z_table"].zs,
                    atomic_energies=atomic_energies[tag],
                    key_specification=key_specification
                )
            else:
                raise NotImplementedError(f"{model_choice} is not supported yet. "
                                          f"Supported models are: ['mace']")
            # leave training setups separate for now
            # should it live on a Model?
            training_setups[tag] = setup_model_training(
                settings=model_settings,
                model=model,
                model_choice=model_choice,
                tag=tag,
                restart=False,
                checkpoints_dir=model_settings.GENERAL.checkpoints_dir.as_posix()
            )
            # make some of the training setup parameters serializable
            if "ema" in training_setups[tag]:
                training_setups[tag]["ema"] = MSONableEMA.from_parent(training_setups[tag]["ema"])
            if "checkpoint_handler" in training_setups[tag]:
                training_setups[tag]["checkpoint_handler"] = (
                    MSONableCheckpointHandler.from_parent(training_setups[tag]["checkpoint_handler"]))

            for key in ["lr_scheduler", "loss_fn", "optimizer"]:
                if key in training_setups[tag]:
                    training_setups[tag][key] = wrap(training_setups[tag][key])

        return cls(models=ensemble,
                   training_setups=training_setups,
                   log_settings=log_settings
                  )

    @classmethod
    def from_restart_data(cls):
        raise NotImplementedError

    def get_model(self, tag: str) -> ModelTrainer:
        return self.models[tag]

    def get_training_setup(self, tag: str) -> dict[str, Any]:
        return self.training_setups[tag]

    def update_datasets(self,
                        training_sets: dict[str, list[MSONAtoms | Atoms]]  = None,
                        valid_sets: dict[str, list[MSONAtoms | Atoms]] = None
                        ):
        """
        Updates the datasets of the models in the ensemble.
        Args:
            training_sets: a dict of training sets, keyed by tag.
            valid_sets: a dict of validation sets, keyed by tag.

        """
        training_sets = training_sets or {}
        valid_sets = valid_sets or {}
        # update models' datasets
        for tag in self.models:
            if tag in training_sets:
                self.models[tag].update_train_set(training_sets[tag])
            if tag in valid_sets:
                self.models[tag].update_valid_set(valid_sets[tag])

    @property
    def ensemble(self):
        """A dictionary of models in the ensemble."""
        return {tag: m.model for tag, m in self.models.items()}

    def train(self,
              n_epochs: int,
              valid_skip: int = 1):
        """Train all models in the ensemble."""

        logger = tools.MetricsLogger(
            directory=self.log_settings["loss_dir"],
            tag="ensemble_train",
        )

        for i_epoch in range(n_epochs):
            for tag in self.models:
                self.models[tag].train(self.training_setups[tag], 1)

            if self.epoch % valid_skip == 0:
                valid_loaders = {}
                (
                    ensemble_valid_losses,
                    valid_loss,
                    metrics,
                    _
                ) = validate_epoch_ensemble(
                    ensemble=self.ensemble,
                    training_setups=self.training_setups,
                    valid_loaders=valid_loaders,
                    logger=logger,
                    log_errors=self.log_settings["log_errors"],
                    epoch=self.epoch
                )
            self.epoch += 1
