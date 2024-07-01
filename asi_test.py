import ase.build
import numpy as np
from scipy.linalg import eigh
import sys
import os
from ase.build import molecule, bulk
from ase.io import read, write
import asi4py
from asi4py.asecalc import ASI_ASE_calculator
from ctypes import POINTER, c_int32, py_object, cast
from mpi4py import MPI
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
import ase
from mace.calculators import MACECalculator

write('geometry.in',
ase.build.molecule("H2O"))

exit()
ASI_LIB_PATH = "/home/thenkes/FHI_aims/libaims.240410.mpi.so"


dictionary_settings = {
  'xc': 'pbe',
  'relativistic': 'atomic_zora scalar',
  'species_dir': '/home/thenkes/FHI_aims/FHIaims/species_defaults/defaults_2020/light/',
  'compute_forces': True,
  'many_body_dispersion': '',
}

def setup_calculator(atoms):
    def init_via_ase(asi):
        from ase.calculators.aims import Aims
        calc = Aims(xc=dictionary_settings['xc'],
            relativistic=dictionary_settings['relativistic'],
            species_dir=dictionary_settings['species_dir'],
            compute_forces=dictionary_settings['compute_forces'],
            #many_body_dispersion=dictionary_settings['many_body_dispersion'],
            )
        calc.write_input(asi.atoms)

    atoms.calc = ASI_ASE_calculator(
        ASI_LIB_PATH,
        init_via_ase,
        MPI.COMM_WORLD,
        atoms
        )
    return atoms
  
  
  
def get_calculator(atoms):
    def init_via_ase(asi):
        from ase.calculators.aims import Aims
        calc = Aims(xc=dictionary_settings['xc'],
            relativistic=dictionary_settings['relativistic'],
            species_dir=dictionary_settings['species_dir'],
            compute_forces=dictionary_settings['compute_forces'],
            #many_body_dispersion=dictionary_settings['many_body_dispersion'],
            )
        calc.write_input(asi.atoms)

    el_calculo = ASI_ASE_calculator(
        ASI_LIB_PATH,
        init_via_ase,
        MPI.COMM_WORLD,
        atoms
        )
    return el_calculo
  
  
def get_atomic_energies(z):

    if MPI.COMM_WORLD.Get_rank() == 0:
        print('Calculating isolated atomic energies.')
    atomic_energies_dict = {}        
    unique_atoms = np.unique(z)
    for element in unique_atoms:
        if MPI.COMM_WORLD.Get_rank()== 0:
            print(f'Calculating energy for element {element}.')
        atom = ase.Atoms([int(element)],positions=[[0,0,0]])
        setup_calculator(atom)
        atomic_energies_dict[element] = atom.get_potential_energy()
        atom.calc.close()
  
sampled_points = [] 
def run_MD(dyn):
  for i in range(5):
    dyn.step(1)
    if MPI.COMM_WORLD.Get_rank() == 0:
      print(f'Step {i} done.')  
      current_point = atoms
      current_point.info['energy'] = np.array(current_point.get_potential_energy())
      current_point.arrays['forces'] = np.array(current_point.get_forces())
      sampled_points.append(
          current_point
      )
def setup_md(atoms):
  dyn = Langevin(atoms, timestep=1 * units.fs, friction=0.002, temperature_K=300,
               rng=np.random.RandomState(42))
  MaxwellBoltzmannDistribution(atoms, temperature_K=300)
  return dyn


atoms = ase.build.molecule("H2O")
atoms2 = ase.build.molecule("H2")

mr_calc_alot = get_calculator(atoms)
mr_calc_alot2 = get_calculator(atoms2)

print(mr_calc_alot.calculate(atoms))
print(mr_calc_alot2.calculate(atoms2))

mr_calc_alot.close()
mr_calc_alot2.close()
exit()


def init_via_ase(asi):
  from ase.calculators.aims import Aims
  calc = Aims(xc='pbe',
    relativistic="atomic_zora scalar",
    species_dir="/home/thenkes/FHI_aims/FHIaims/species_defaults/defaults_2020/light/",
    compute_forces=True,
    many_body_dispersion="",
    )
  calc.write_input(asi.atoms)
  
  
atoms = molecule("H2O")
atoms.calc = ASI_ASE_calculator(ASI_LIB_PATH, init_via_ase, MPI.COMM_WORLD, atoms)
dyn = Langevin(atoms, timestep=1 * units.fs, friction=0.002, temperature_K=300,
               rng=np.random.RandomState(42))
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

for i in range(10):
    dyn.step(1)
    #print(atoms.get_potential_energy())
    #print(atoms.get_forces())    