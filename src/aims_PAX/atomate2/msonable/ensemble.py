"""
This module contains an MSONable version of the Ensemble model and Ensemble.
 Ensemble is basically a dict connecting Tag and Ensemble model.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from mace import tools
from monty.json import MSONable
from pymatgen.io.ase import MSONAtoms
from so3krates_torch.tools.train import valid_err_log

from aims_PAX.atomate2.atomic_energies import AtomicEnergies
from aims_PAX.atomate2.msonable.checkpoint_handler import MSONableCheckpointHandler
from aims_PAX.atomate2.msonable.ema import MSONableEMA
from aims_PAX.atomate2.msonable.losses import LossesCollection
from aims_PAX.atomate2.msonable.model import MaceModelTrainer, ModelTrainer
from aims_PAX.atomate2.msonable.serialization import wrap
from aims_PAX.settings import ModelSettings
from aims_PAX.settings.project import MiscSettings
from aims_PAX.tools.model_tools.setup_MACE import setup_mace
from aims_PAX.tools.model_tools.training_tools import setup_model_training
from aims_PAX.tools.utilities.data_handling import save_datasets
from aims_PAX.tools.utilities.utilities import save_checkpoint, create_keyspec


class Stage(Enum):
    # the values are chosen to be consistent with the restart file names
    IDG = "initial_ds"
    AL = "al"


@dataclass
class Ensemble(MSONable):
    """
    Represents an ensemble of machine learning models and their training setups.

    The Ensemble class is used to manage multiple models that are part of an ensemble,
    as well as their respective training setups. It provides methods for model creation,
    training, dataset updating, and managing evaluation metrics. It is specifically
    designed to work with different learning stages such as initial dataset generation
    or active learning and supports operations like on-the-fly dataset updates and multimodel training.

    Attributes:
        stage (Stage): The learning stage associated with the ensemble (e.g., "idg" or "al");
        models (dict[str, ModelTrainer]): A dictionary containing the models in the ensemble,
            keyed by unique tags;
        training_setups (dict[str, dict[str, Any]]): A dictionary storing the training setups
            for each model, keyed by unique tags;
        log_settings (dict[str, Any]): Logging configurations for the ensemble, such as loss
            directory and error table settings;
        epoch (int): The current epoch number for the ensemble training process;
        step (int): The current training step for the ensemble, which can consist of multiple epochs;
        valid_loss (float): The validation loss of the ensemble. Defaults to infinity;
        eval_metrics (dict[str, Any]): A dictionary containing evaluation metrics for the
            ensemble.
    """
    stage: Stage
    models: dict[str, ModelTrainer]
    training_setups: dict[str, dict[str, Any]]
    log_settings: dict[str, Any]
    losses: LossesCollection
    epoch: int = 0
    step: int = 0
    valid_loss: float = np.inf
    eval_metrics: dict[str, Any] = None
    done: bool = False

    @classmethod
    def from_scratch(cls,
                     stage: Stage,
                     tags: list[str],
                     misc_settings: MiscSettings,
                     model_settings: ModelSettings,
                     atomic_energies: dict[str, AtomicEnergies],
                     model_inputs: dict[str, Any]) -> "Ensemble":
        """
        Generate an ensemble from model settings and tags list.

        Args:
            stage: a learning stage (idg or al).
            tags: a list of tags.
            misc_settings: aims-PAX miscellaneous project settings
            model_settings: model settings in a ModelSettings object.
            atomic_energies: atomic energies.
            model_inputs: model-dependent inputs (z_table)

        Returns:
            an Ensemble object.
        """
        # can we put calculate model inputs here?
        ensemble = {}
        training_setups = {}

        model_choice = model_settings.GENERAL.model_choice
        # this goes to the instance attribute
        log_settings = dict(
            output_dir=misc_settings.output_dir.as_posix(),
            dataset_dir=misc_settings.dataset_dir.as_posix(),
            loss_dir=model_settings.GENERAL.loss_dir.as_posix(),
            log_errors=model_settings.MISC.error_table
        )
        # this is used just here
        key_specification = create_keyspec(
            energy_key=misc_settings.energy_key,
            forces_key=misc_settings.forces_key,
            stress_key=misc_settings.stress_key,
            dipole_key=misc_settings.dipole_key,
            polarizability_key=misc_settings.polarizability_key,
            head_key=misc_settings.head_key,
            charges_key=misc_settings.charges_key,
            total_charge_key=misc_settings.total_charge_key,
            total_spin_key=misc_settings.total_spin_key,
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
            # should they live on a Model?
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

        return cls(
            stage=stage,
            models=ensemble,
            training_setups=training_setups,
            log_settings=log_settings,
            losses=LossesCollection.from_scratch()
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
              valid_skip: int = 1,
              analysis: bool = False,
              desired_accuracy: float = 0.01) -> bool:
        """Train all models in the ensemble for one step."""

        self.step += 1
        logger = tools.MetricsLogger(
            directory=self.log_settings["loss_dir"],
            tag="ensemble_train",
        )

        for i_epoch in range(n_epochs):
            for tag in self.models:
                self.models[tag].train(self.training_setups[tag], 1)

            valid_losses = []
            eval_metrics = []
            if self.epoch % valid_skip == 0:
                for tag in self.models:
                    valid_loss, eval_metric = self.models[tag].validate(self.training_setups[tag])
                    valid_losses.append(valid_loss)
                    eval_metrics.append(eval_metric)

                self.valid_loss = np.mean(valid_losses, dtype=float)
                self.eval_metrics = {
                    key: np.mean([m[key] for m in eval_metrics])
                    for key in eval_metrics[0]
                    if key not in {"mode", "epoch", "head"}
                }
                self.eval_metrics["mode"] = "eval"
                self.eval_metrics["epoch"] = self.epoch

                valid_err_log(
                    valid_loss=self.valid_loss,
                    eval_metrics=self.eval_metrics,
                    logger=logger,
                    log_errors=self.log_settings["log_errors"],
                    epoch=self.epoch,
                )

                current_valid = self.eval_metrics["mae_f"]
                if analysis:
                    self.losses.update_and_save(
                        epoch=self.epoch,
                        valid_loss=self.valid_loss,
                        ensemble_valid_losses={tag: m.valid_loss for tag, m in self.models.items()},
                        save_path=Path(self.log_settings["output_dir"]) / "analysis" / f"{self.stage.value}_losses.npz",
                    )

                for tag in self.models:
                    save_checkpoint(
                        checkpoint_handler=self.training_setups[tag][
                            "checkpoint_handler"
                        ],
                        training_setup=self.training_setups[tag],
                        model=self.models[tag].model,
                        epoch=self.epoch,
                        keep_last=False,
                    )
                ensemble_ase_sets = {tag: m.ase_sets for tag, m in self.models.items()}
                save_datasets(
                    self.ensemble,
                    ensemble_ase_sets,
                    path=Path(self.log_settings["dataset_dir"]) / self.stage.value,
                    initial=(self.stage is Stage.IDG),
                )

                if desired_accuracy >= current_valid:
                    logging.info(
                        f"Accuracy criterion reached at epoch {self.epoch}. Breaking."
                    )
                    logging.info(
                        f"Criterion: {desired_accuracy}; Current accuracy: {current_valid}."
                    )
                    self.done = True
                    break
            self.epoch += 1
        return self.done
