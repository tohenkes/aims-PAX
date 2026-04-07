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


@dataclass
class Ensemble(MSONable):
    _ensemble: dict[str, MSONableModel]

    @classmethod
    def from_model_settings(cls,
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
        model_choice = model_settings.GENERAL.model_choice.lower()
        for tag in tags:
            if model_choice == "mace":
                model = setup_mace(model_settings, model_inputs["z_table"], atomic_energies.get(tag))
                ensemble[tag] = wrap(model)
            else:
                raise NotImplementedError(f"{model_choice} is not supported yet. "
                                          f"Supported models are: {['mace']}")

        return cls(_ensemble=ensemble)
