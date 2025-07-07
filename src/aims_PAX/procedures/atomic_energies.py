import ase.build
import numpy as np
from mace import tools
from aims_PAX.tools.utilities import (
    Z_from_geometry_in,
    AIMSControlParser,
)
import ase
import logging
from mpi4py import MPI
from asi4py.asecalc import ASI_ASE_calculator
from typing import List

WORLD_COMM = MPI.COMM_WORLD
WORLD_SIZE = WORLD_COMM.Get_size()
RANK = WORLD_COMM.Get_rank()
from yaml import safe_load


# TODO: make option to pass a list of geometry.in files


class E0Calculator:
    def __init__(
        self,
        basis_dir: str = None,
        path_to_control: str = "./control.in",
        path_to_geometry: str = "./geometry.in",
        Zs: List[int] = None,
        aims_path: str = None,
    ):

        if path_to_control is None:
            raise FileNotFoundError("No control.in provided.")

        logging.basicConfig(
            filename="compute_E0s.log",
            encoding="utf-8",
            level=logging.DEBUG,
            force=True,
        )
        tools.setup_logger(
            level="INFO",
            #    tag=tag,
            directory=".",
        )
        self.control_parser = AIMSControlParser()

        if basis_dir is not None:
            self.basis_dir = basis_dir
        else:
            try:
                with open("./active_learning_settings.yaml", "r") as file:
                    al_settings = safe_load(file)
                    self.basis_dir = al_settings["ACTIVE_LEARNING"][
                        "species_dir"
                    ]
            except FileNotFoundError:
                if RANK == 0:
                    logging.error(
                        "No basis_dir provided or found in active_learning_settings.yaml."
                    )

        if aims_path is not None:
            self.ASI_path = aims_path
        else:
            try:
                with open("./active_learning_settings.yaml", "r") as file:
                    al_settings = safe_load(file)
                    self.ASI_path = al_settings["ACTIVE_LEARNING"][
                        "aims_lib_path"
                    ]
            except FileNotFoundError:
                if RANK == 0:
                    logging.error(
                        "No ASI_path provided or found in active_learning_settings.yaml."
                    )

        if Zs is None:
            try:
                self.Zs = Z_from_geometry_in(path_to_geometry)
            except FileNotFoundError:
                if RANK == 0:
                    logging.error(
                        "No elements provided or found in geometry.in."
                    )
        else:
            self.Zs = Zs

        self.handle_aims_settings(path_to_control)

    def __call__(self):
        MPI.COMM_WORLD.Barrier()
        self.get_atomic_energies()
        if RANK == 0:
            logging.info(f"Saving atomic energies to ./atomic_energies.npz")
            np.savez("./atomic_energies.npz", self.atomic_energies_dict)

    def handle_aims_settings(self, path_to_control: str):
        """
        Parses the AIMS control file to get the settings for the AIMS calculator.

        Args:
            path_to_control (str): Path to the AIMS control file.
            species_dir (str): Path to the species directory of AIMS.
        """

        self.aims_settings = self.control_parser(path_to_control)
        self.aims_settings["species_dir"] = self.basis_dir

    def get_atomic_energies(self):
        """
        Calculates the isolated atomic energies for the elements in the geometry using AIMS.
        """
        if RANK == 0:
            logging.info("Calculating the isolated atomic energies.")
        self.atomic_energies_dict = {}
        unique_atoms = np.unique(self.Zs)
        for element in unique_atoms:
            atom = ase.Atoms([int(element)], positions=[[0, 0, 0]])
            if RANK == 0:
                logging.info(
                    f"Calculating the atomic energy energy of {atom.symbols[0]}."
                )
            self.setup_calculator(atom)
            self.atomic_energies_dict[element] = atom.get_potential_energy()
            atom.calc.close()  # kills AIMS process so we can start a new one later

        if RANK == 0:
            logging.info(f"{self.atomic_energies_dict}")

    def setup_calculator(
        self,
        atoms: ase.Atoms,
    ) -> ase.Atoms:
        """
        Attaches the AIMS calculator to the atoms object. Uses the AIMS settings
        from the control.in to set up the calculator.

        Args:
            atoms (ase.Atoms): Atoms object to attach the calculator to.
            pbc (bool, optional): If periodic boundry conditions are required or not.
            Defaults to False.

        Returns:
            ase.Atoms: Atoms object with the calculator attached.
        """

        aims_settings = self.aims_settings.copy()

        if aims_settings.get("k_grid") is not None:
            aims_settings.pop("k_grid")

        # 1-2 atoms are too small to use scalapack KS solver
        # so we fall back to the serial one for calculating single atoms
        aims_settings["KS_method"] = "serial"

        def init_via_ase(asi):

            from ase.calculators.aims import Aims

            calc = Aims(**aims_settings)
            calc.write_input(asi.atoms)

        atoms.calc = ASI_ASE_calculator(
            self.ASI_path, init_via_ase, MPI.COMM_WORLD, atoms
        )
        return atoms
