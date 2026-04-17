"""
An MSONable Losses collection for Ensemble
"""
from pathlib import Path

import numpy as np
from monty.json import MSONable

from dataclasses import dataclass, field

@dataclass
class LossesCollection(MSONable):
    epochs: list = field(default_factory=list)
    avg_losses: list = field(default_factory=list)
    ensemble_losses: list = field(default_factory=list)

    @classmethod
    def from_scratch(cls):
        """Create a LossesCollection object from scratch."""
        return cls(
            epochs=[],
            avg_losses=[],
            ensemble_losses=[],
        )

    @classmethod
    def from_file(cls, path: Path):
        """Create a LossesCollection object from a npz file."""
        data = np.load(path)
        return cls(
            epochs=data["epoch"].tolist(),
            avg_losses=data["avg_losses"].tolist(),
            ensemble_losses=data["ensemble_losses"].tolist(),
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, epoch: int = None):
        """Return the losses for a given epoch. If epoch is None, return all losses."""
        if epoch is None:
            return {
                "epoch": self.epochs,
                "avg_loss": self.avg_losses,
                "ensemble_losses": self.ensemble_losses
            }
        if epoch not in self.epochs:
            raise ValueError(f"Epoch {epoch} not found in losses collection.")
        idx = self.epochs.index(epoch)
        return {
            "epoch": epoch,
            "avg_loss": self.avg_losses[idx],
            "ensemble_losses": self.ensemble_losses[idx]
        }

    def update_and_save(
        self,
        epoch: int,
        valid_loss: float,
        ensemble_valid_losses: dict,
        save_path: Path,
    ):
        """
        Collects number of epochs, average validation loss, and
        per ensemble member validation losses and saves in a
        npz file.

        Args:
            epoch (int): Current epoch;
            valid_loss (float): Averaged validation loss over the ensemble;
            ensemble_valid_losses (dict): Per ensemble member
                                                    validation losses;
            save_path (Path): Path to save the analysis data.
        """
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        self.epochs.append(epoch)
        self.avg_losses.append(valid_loss)
        self.ensemble_losses.append(ensemble_valid_losses)
        np.savez(save_path,
                 epoch=self.epochs,
                 avg_losses=self.avg_losses,
                 ensemble_losses=self.ensemble_losses)