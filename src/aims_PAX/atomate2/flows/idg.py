"""
Atomate2 flows for initial dataset generation in aims-PAX.
Return an aims-PAX checkpoint from which the AL can be restarted.
"""

from dataclasses import dataclass

from jobflow import Maker, job
from pymatgen.core import Structure, Molecule

from aims_PAX.atomate2.utils import get_idg_makers_from_settings
from aims_PAX.settings import ModelSettings
from aims_PAX.settings.project import IDGSettings, MiscSettings, MDSettings


@dataclass
class InitialDatasetGenerator(Maker):
    """
    A Root Maker to create an ensemble of models for aims-PAX.

    This class is responsible for generating the initial dataset that serves as
    the starting point for active learning workflows in aims-PAX. It creates an
    ensemble of models and returns a checkpoint from which the active learning
    procedure can be initiated or restarted.

    Attributes
    ----------
    settings: IDGSettings
        The settings for initial dataset generation as given by aims-PAX
    misc_settings: MiscSettings
        Project miscellaneous settings as given by aims-PAX
    model_settings: ModelSettings
        the settings for the models to train as given by aims-PAX
    name : str
        The name of the Maker.

    Notes
    -----
    This maker integrates with the atomate2 workflow framework and is designed
    to work within the aims-PAX active learning infrastructure.
    """
    structure: Structure | Molecule
    settings: IDGSettings
    md_settings: MDSettings
    misc_settings: MiscSettings
    model_settings: ModelSettings
    name: str = "initial dataset generator"

    def make(self):
        """Create a job to make an initial dataset. This is a sequence of steps (`self.step` jobs)
        made until one of the stopping criteria is reached."""
        md_makers, reference_maker = get_idg_makers_from_settings(idg_settings=self.settings,
                                                                  md_settings=self.md_settings,
                                                                  misc_settings=self.misc_settings)

        return md_makers[1].make(self.structure)


    @job
    def step(self):
        """Do one step of the initial dataset creation: MD -> DFT -> training ensemble of models"""

    def create_reference_jobs(self, structures: list[Structure | Molecule]):
        """Create a set of reference (DFT/reference model) jobs for the initial dataset generation."""
