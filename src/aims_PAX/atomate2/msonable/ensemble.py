"""
This module contains an MSONable version of the Ensemble model and Ensemble.
 Ensemble is basically a dict connecting Tag and Ensemble model.
"""
from dataclasses import dataclass
from typing import Any

from mace.tools import AtomicNumberTable
from monty.json import MSONable
from pymatgen.io.ase import MSONAtoms

from aims_PAX.atomate2.atomic_energies import AtomicEnergies
from aims_PAX.atomate2.msonable.checkpoint_handler import MSONableCheckpointHandler
from aims_PAX.atomate2.msonable.ema import MSONableEMA
from aims_PAX.atomate2.msonable.model import MaceModel
from aims_PAX.atomate2.msonable.serialization import MSONableModel, wrap
from aims_PAX.settings import ModelSettings
from aims_PAX.tools.model_tools.setup_MACE import setup_mace
from aims_PAX.tools.model_tools.training_tools import setup_model_training
from aims_PAX.tools.utilities.data_handling import KeySpecification


@dataclass
class Ensemble(MSONable):
    models: dict[str, MSONableModel]
    training_setups: dict[str, dict[str, Any]]
    ase_sets: dict[str, dict[str, list[Any]]]
    model_sets: dict[str, dict[str, Any]]
    key_specification: KeySpecification

    @classmethod
    def from_scratch(cls,
                     tags: list[str],
                     model_settings: ModelSettings,
                     atomic_energies: AtomicEnergies,
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
        ase_sets = {}
        model_sets = {}

        model_choice = model_settings.GENERAL.model_choice.lower()
        for tag in tags:
            if model_choice == "mace":
                model = setup_mace(model_settings, model_inputs["z_table"], atomic_energies.get(tag))
                ensemble[tag] = MaceModel.from_parent(
                    model,
                    settings=model_settings,
                    zs=model_inputs["z_table"].zs,
                    atomic_energies=atomic_energies.get(tag)
                )
            else:
                raise NotImplementedError(f"{model_choice} is not supported yet. "
                                          f"Supported models are: ['mace']")
            # TODO: checkpoints_dir is unneeded as model settings is already there
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

            ase_sets[tag] = {"train": [], "valid": []}
            # TODO: add multihead model support
            model_sets[tag] = {"train": [], "valid": {"Default": []}}
        return cls(models=ensemble,
                   training_setups=training_setups,
                   ase_sets=ase_sets,
                   model_sets=model_sets,
                   key_specification=key_specification,
                   zs=model_inputs["z_table"].zs)

    @classmethod
    def from_restart_data(cls):
        raise NotImplementedError

    def get_model(self, tag: str) -> MSONableModel:
        return self.models[tag]

    def get_training_setup(self, tag: str) -> dict[str, Any]:
        return self.training_setups[tag]

    @property
    def z_table(self):
        return AtomicNumberTable(self.zs)

    def update_datasets(self, tag, train_set: list[MSONAtoms], test_set: list[MSONAtoms]):
        """Updates datasets for the training and testing of the model.
        """
