"""
Atomate2 flows for initial dataset generation in aims-PAX.
Return an aims-PAX checkpoint from which the AL can be restarted.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from pickle import UnpicklingError

import numpy as np
from jobflow import Maker, job, Flow, Response
from pymatgen.core import Structure, Molecule
from pymatgen.io.ase import MSONAtoms, AseAtomsAdaptor

from aims_PAX.atomate2 import AllowedMDMakers
from aims_PAX.atomate2.atomic_energies import AtomicEnergies
from aims_PAX.atomate2.msonable.ensemble import Ensemble
from aims_PAX.atomate2.utils import get_model_dependent_inputs, get_idg_makers_from_settings
from aims_PAX.settings import ModelSettings
from aims_PAX.settings.project import IDGSettings, MiscSettings, MDSettings
from aims_PAX.tools.utilities.input_utils import read_geometry
from aims_PAX.tools.utilities.utilities import setup_logger, log_yaml_block, get_seeds, create_seeds_tags_dict, \
    get_atomic_energies_from_pt, create_keyspec


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
    settings: IDGSettings
    md_settings: MDSettings
    misc_settings: MiscSettings
    model_settings: ModelSettings
    name: str = "initial dataset generator"

    def make(self):
        """Create a job to make an initial dataset. This is a sequence of steps (`self.step` jobs)
        made until one of the stopping criteria is reached."""
        # use dtype from model settings for the foundational model as well
        logger = self.get_logger()
        logger.info("Initializing initial dataset procedure.")
        logger.info(f"Procedure runs with atomate2.")
        logger.info("Using following settings for the initial dataset procedure:")
        log_yaml_block(
            "INITIAL_DATASET_GENERATION",
            self.settings.model_dump(),
        )
        # check for the restart
        restart_path = (
            self.misc_settings.output_dir
            / "restart"
            / "initial_ds"
            / "initial_ds_restart.npy"
        )
        restart = restart_path.exists()

        # set seeds for reproducibility
        ensemble_seeds = get_seeds(self.model_settings.GENERAL.seed,
                          self.settings.ensemble_size)
        seeds_tags_dict = create_seeds_tags_dict(
            seeds=ensemble_seeds,
            model_settings=self.model_settings,
            dataset_dir=self.misc_settings.dataset_dir,
        )
        tags = list(seeds_tags_dict.keys())
        logger.debug(f"Using seeds: {seeds_tags_dict}")

        # create MD and reference Makers (to create jobs later on)

        # No CuEQ training during initial dataset generation
        # (because avg_num_neighbors, mean, std, atomic energies etc
        # are changing all the time; not possible to modify with CuEQ)
        model_settings = {
            "device": self.model_settings.MISC.device,
            "default_dtype": self.model_settings.GENERAL.default_dtype,
            "enable_cueq_train": False
        }

        # there can be several MD settings (hence, several MD makers)
        # and several control.in files. How to deal with that here?
        md_makers, reference_maker = get_idg_makers_from_settings(
            idg_settings=self.settings,
            md_settings=self.md_settings,
            misc_settings=self.misc_settings,
            model_settings=model_settings)

        # create the initial job (create / read in trajectories / energies / etc)
        if restart:
            prepare_job = self.restart(
                restart_path=restart_path,
                seeds_tags_dict=seeds_tags_dict,)
        else:
            prepare_job = self.run_from_scratch(tags=tags)

        md_makers[1] = md_makers[1].update_kwargs(
            {"n_steps": 100}
        )

        md_job = self.create_md_job(md_makers, prepare_job.output.trajectories)
        dummy_job = self.dummy_job(md_job.output)
        return Flow([prepare_job, md_job, dummy_job])


    def get_logger(self):
        """Get the logger for the initial dataset generation."""
        logger_level = getattr(logging, self.model_settings.MISC.log_level, logging.INFO)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        return setup_logger(
            level=logger_level,
            tag="initial_dataset",
            directory=self.misc_settings.log_dir.as_posix()
        )


    @job(name="prepare")
    def restart(self, restart_path: Path, seeds_tags_dict: dict[str, int]):
        """Restart the initial dataset generation from a checkpoint."""
        logger = self.get_logger()
        logger.info("Restarting initial dataset acquisition from checkpoint.")
        try:
            restart_dict = np.load(
                restart_path,
                allow_pickle=True,
            ).item()
        except (UnpicklingError, EOFError) as e:
            logging.error(
                f"Could not read restart data at '{restart_path}'")
            raise e
        trajectories = restart_dict["trajectories"]
        step = restart_dict["step"]
        if not self.misc_settings.create_restart:
            restart_dict = None

        logging.info(f"Running initial dataset acquisition with {len(trajectories)} geometries.")
        # get model-dependent inputs (zs, etc) to the trained model
        model_inputs = get_model_dependent_inputs(self.model_settings.GENERAL.model_choice,
                                                  trajectories=trajectories)
        logger.debug("Loading atomic energies from checkpoint.")

        # get atomic energies
        # TODO: this needs just the tags list, no need for a full dict
        _, ensemble_atomic_energies_dict = get_atomic_energies_from_pt(
            path_to_checkpoints=self.model_settings.GENERAL.checkpoints_dir,
            z=model_inputs["z"],
            seeds_tags_dict=seeds_tags_dict,
            dtype=self.model_settings.GENERAL.default_dtype,
            model_choice=self.model_settings.GENERAL.model_choice,
        )
        ensemble_atomic_energies = AtomicEnergies.from_e(list(seeds_tags_dict.keys()),
                                                         ensemble_atomic_energies_dict)

        return {
            "trajectories": {k: MSONAtoms(v) for k, v in trajectories.items()},
            "step": step,
            "restart_dict": restart_dict,
            "atomic_energies": ensemble_atomic_energies,
        }

    @job(name="prepare")
    def run_from_scratch(self, tags: list[str]):
        """
        Run the job from scratch based on the provided tags.
        """
        logger = self.get_logger()
        # create restart dict
        restart_dict = None
        if self.misc_settings.create_restart:
            logger.debug("Creating restart dictionary.")
            restart_dict = {
                "trajectories": None,
                "last_initial_losses": None,
                "initial_ds_done": False,
            }
        trajectories = read_geometry(self.misc_settings.path_to_geometry, log=True)
        step = 0

        logging.info(f"Running initial dataset acquisition with {len(trajectories)} geometries.")
        # get model-dependent inputs (zs, etc) to the trained model
        model_inputs = get_model_dependent_inputs(self.model_settings.GENERAL.model_choice,
                                                  trajectories=trajectories)

        # get atomic energies
        default_atomic_energies = self.model_settings.ARCHITECTURE.atomic_energies
        if default_atomic_energies is None:
            logger.info("No atomic energies specified. Fitting to training data.")
            ensemble_atomic_energies = {
                tag: AtomicEnergies.from_z(model_inputs["z"], need_updating=True)
                for tag in tags
            }
        else:
            logger.info("Using specified atomic energies.")
            ensemble_atomic_energies = {
                tag: AtomicEnergies.from_e(default_atomic_energies)
                for tag in tags
            }
        key_specification = create_keyspec(
            energy_key=self.misc_settings.energy_key,
            forces_key=self.misc_settings.forces_key,
            stress_key=self.misc_settings.stress_key,
            dipole_key=self.misc_settings.dipole_key,
            polarizability_key=self.misc_settings.polarizability_key,
            head_key=self.misc_settings.head_key,
            charges_key=self.misc_settings.charges_key,
            total_charge_key=self.misc_settings.total_charge_key,
            total_spin_key=self.misc_settings.total_spin_key,
        )
        # get ensemble dicts
        ensemble = Ensemble.from_scratch(tags,
                                         self.model_settings,
                                         ensemble_atomic_energies,
                                         model_inputs,
                                         key_specification)

        collect_losses = {
            "epoch": [],
            "avg_losses": [],
            "ensemble_losses": [],
        }

        return {
            "trajectories": {k: MSONAtoms(v) for k, v in trajectories.items()},
            "step": step,
            "restart_dict": restart_dict,
            "ensemble": ensemble,
            "epoch": 0,
            "losses": collect_losses
        }

    @job
    def step(self):
        """Do one step of the initial dataset creation: MD -> DFT -> training ensemble of models"""

    @job(name="md")
    def create_md_job(self, makers: dict[str, AllowedMDMakers], trajectories: dict[str, MSONAtoms]):
        """Create an MD job for the initial dataset generation."""
        # We will use the first structure in the trajectories for IDG.
        # If I have several structures, what should happen then?
        atoms = trajectories['0']
        struct = AseAtomsAdaptor.get_structure(atoms) if all(atoms.get_pbc()) else AseAtomsAdaptor.get_molecule(atoms)
        md_job = makers['1'].make(struct)
        return Response(replace=md_job)

    @job
    def dummy_job(self, output):
        print(len(output.output.ionic_steps))
        return {}



    def create_reference_jobs(self, structures: list[Structure | Molecule]):
        """Create a set of reference (DFT/reference model) jobs for the initial dataset generation."""
