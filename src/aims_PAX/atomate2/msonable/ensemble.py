"""
This module contains an MSONable version of the Ensemble model and Ensemble.
 Ensemble is basically a dict connecting Tag and Ensemble model.
"""
from dataclasses import dataclass
from typing import Any

from monty.json import MSONable

from aims_PAX.atomate2.atomic_energies import AtomicEnergies
from aims_PAX.atomate2.msonable.serialization import MSONableModel, wrap
from aims_PAX.settings import ModelSettings
from aims_PAX.tools.model_tools.setup_MACE import setup_mace
from aims_PAX.tools.model_tools.training_tools import setup_model_training


@dataclass
class Ensemble(MSONable):
    models: dict[str, MSONableModel]
    training_setups: dict[str, dict[str, Any]]
    ase_sets: dict[str, dict[str, list[Any]]]
    model_sets: dict[str, dict[str, Any]]

    @classmethod
    def from_scratch(cls,
                     tags: list[str],
                     model_settings: ModelSettings,
                     atomic_energies: AtomicEnergies,
                     model_inputs: dict[str, Any]) -> "Ensemble":
        """
        Generate an ensemble from model settings and tags list.

        Args:
            tags: a list of tags.
            model_settings: a ModelSettings object.
            atomic_energies: atomic energies.
            model_inputs: model-dependent inputs (z_table)

        Returns:
            an Ensemble object.
        """
        ensemble = {}
        training_setups = {}
        ase_sets = {}
        model_sets = {}

        model_choice = model_settings.GENERAL.model_choice.lower()
        for tag in tags:
            if model_choice == "mace":
                model = setup_mace(model_settings, model_inputs["z_table"], atomic_energies.get(tag))
                ensemble[tag] = wrap(model)
            else:
                raise NotImplementedError(f"{model_choice} is not supported yet. "
                                          f"Supported models are: {['mace']}")
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
            for key in ["lr_scheduler", "ema", "loss_fn", "checkpoint_handler", "optimizer"]:
                if key in training_setups[tag]:
                    training_setups[tag][key] = wrap(training_setups[tag][key])

            ase_sets[tag] = {"train": [], "valid": []}
            # TODO: add multihead model support
            # Why there are ase_sets and model_sets separately?
            model_sets[tag] = {"train": [], "valid": {"Default": []}}
        return cls(models=ensemble,
                   training_setups=training_setups,
                   ase_sets=ase_sets,
                   model_sets=model_sets)

    @classmethod
    def from_restart_data(cls):
        raise NotImplementedError

    def get_model(self, tag: str) -> MSONableModel:
        return self.models[tag]

    def get_training_setup(self, tag: str) -> dict[str, Any]:
        return self.training_setups[tag]

