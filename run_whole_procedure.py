import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from FHI_AL.procedures import InitalDatasetProcedure, ALProcedure, StandardMACEEnsembleProcedure
from yaml import safe_load
import argparse
from ase.io import read
import random
import numpy as np
from mpi4py import MPI


with open("./mace_settings.yaml", "r") as file:
    mace_settings = safe_load(file)
with open("./active_learning_settings.yaml", "r") as file:
    al_settings = safe_load(file)


initial_ds = InitalDatasetProcedure(
    mace_settings=mace_settings,
    al_settings=al_settings,
    #path_to_aims_lib="/home/thenkes/FHI_aims/libaims.240410.mpi.so",
    #species_dir="/home/thenkes/FHI_aims/FHIaims/species_defaults/defaults_2020/light/",
    path_to_aims_lib="/home/tobias/FHI-aims/libaims.240410.mpi.so",
    species_dir="/home/tobias/FHI-aims/FHIaims/species_defaults/defaults_2020/light"
)

if not al_settings['ACTIVE_LEARNING']["scheduler_initial"]:
    mace_settings["lr_scheduler"] = None
        


initial_ds.run()
if al_settings['ACTIVE_LEARNING']["converge_initial"]:
    initial_ds.converge() 

MPI.COMM_WORLD.Barrier()
MPI.Finalize()
