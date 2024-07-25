import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from FHI_AL.custom_MACECalculator import MACECalculator
from utilities import list_files_in_directory
import torch
from ase.build import molecule
from ase.md.langevin import Langevin
from ase import units

model_paths = list_files_in_directory(
    "/home/users/u101418/al_aims/asi/FHI_AL/model"
    )

models = [
    torch.load(f=model_path, map_location="cpu") for model_path in model_paths
]
# the calculator needs to be updated consistently see below
mace_calc = MACECalculator(
    models=[models[0]],
    device="cpu",
    default_dtype="float32")

mace_calc2 = MACECalculator(
    models=[models[1]],
    device="cpu",
    default_dtype="float32")

mol = molecule("H2O")

dyn = Langevin(
    mol,
    timestep=0.5 * units.fs,
    friction=1.0 / units.fs,
    temperature_K=200
    )

mol.calc = mace_calc

dyn.step()

print(mol.calc.results['forces'])


mol.calc = mace_calc2

dyn.step()
print(mol.calc.results['forces'])

