import ase.build
from mace import tools
from FHI_AL.tools.utilities import (
    AIMSControlParser,
)
import ase
from ase.io import read, write
import logging
from mpi4py import MPI
from asi4py.asecalc import ASI_ASE_calculator
from yaml import safe_load


WORLD_COMM = MPI.COMM_WORLD
WORLD_SIZE = WORLD_COMM.Get_size()
RANK = WORLD_COMM.Get_rank()

class ReCalculator:
    def __init__(
        self,
        path_to_data: str, 
        basis_dir: str = None,
        path_to_control: str = "./control.in",
        aims_path: str = None,
        start_idx: int = 0,
        save_interval: int = 10
        ):
        
        if path_to_control is None:
            raise FileNotFoundError("No control.in provided.")
        
        logging.basicConfig(
            filename="recalculate.log",
            encoding="utf-8",
            level=logging.DEBUG,
            force=True,
        )
        tools.setup_logger(
            level='INFO',
            #    tag=tag,
            directory="."
        )
        self.save_interval = save_interval
        self.control_parser = AIMSControlParser()
        
        if basis_dir is not None:
            self.basis_dir = basis_dir
        else:
            try:
                with open("./active_learning_settings.yaml", "r") as file:
                    al_settings = safe_load(file)
                    self.basis_dir = al_settings["ACTIVE_LEARNING"]['species_dir']
            except FileNotFoundError:
                if RANK == 0:
                    logging.error("No basis_dir provided or found in active_learning_settings.yaml.")
        
        if aims_path is not None:
            self.ASI_path = aims_path
        else:
            try:
                with open("./active_learning_settings.yaml", "r") as file:
                    al_settings = safe_load(file)
                    self.ASI_path = al_settings['ACTIVE_LEARNING']['aims_lib_path']
            except FileNotFoundError:
                if RANK == 0:
                    logging.error("No ASI_path provided or found in active_learning_settings.yaml.")
        
        self.handle_aims_settings(path_to_control)
        self.data = read(path_to_data, index=f'{start_idx}:')
        self.calc = self.setup_calculator(self.data[0])


    def __call__(self):
        MPI.COMM_WORLD.Barrier()
        self.recalculate()
        if RANK == 0:
            write('recalculated_data.xyz', self.recalc_data)
        
    def handle_aims_settings(
        self,
        path_to_control: str
        ):
        """
        Parses the AIMS control file to get the settings for the AIMS calculator.

        Args:
            path_to_control (str): Path to the AIMS control file.
            species_dir (str): Path to the species directory of AIMS.
        """
        
        self.aims_settings = self.control_parser(path_to_control)
        self.aims_settings['species_dir'] = self.basis_dir
        self.aims_settings['compute_forces'] = True

    def recalculate(self):
        """
        Calculates the isolated atomic energies for the elements in the geometry using AIMS.
        """     
        if RANK == 0:
            logging.info(f'Recalculating dataset.')

        self.recalc_data = []
        for i, mol in enumerate(self.data):
            MPI.COMM_WORLD.Barrier()
            point = mol.copy()
            self.calc.calculate(mol, properties=["energy","forces"])
            if RANK == 0:
                energies = self.calc.results['energy']
                forces = self.calc.results['forces']
                point.info['energy'] = energies
                point.arrays['forces'] = forces
                self.recalc_data.append(point)
                if i % self.save_interval == 0:
                    logging.info(f'Calculated {i}/{len(self.data)-1}')
                    write('recalculated_data.xyz', self.recalc_data)
        
    def setup_calculator(
            self,
            atoms: ase.Atoms,
            ) -> ase.Atoms:
        """
        Attaches the AIMS calculator to the atoms object. Uses the AIMS settings
        from the control.in to set up the calculator.

        Args:
            atoms (ase.Atoms): Atoms object to attach the calculator to.

        Returns:
            ase.Atoms: Atoms object with the calculator attached.
        """

        aims_settings = self.aims_settings.copy()
        def init_via_ase(asi):
            
            from ase.calculators.aims import Aims
            calc = Aims(**aims_settings)
            calc.write_input(asi.atoms)

        calc = ASI_ASE_calculator(
            self.ASI_path,
            init_via_ase,
            MPI.COMM_WORLD,
            atoms
            )
        return calc
    