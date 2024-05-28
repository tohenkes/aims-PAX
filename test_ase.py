import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from FHI_AL.procedures import InitalDatasetProcedure, ALProcedure, StandardMACEEnsembleProcedure
from yaml import safe_load
import numpy as np
from ase.io import read
import random
# Set the stack size limit to unlimited
import resource
resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))




with open("./mace_settings.yaml", "r") as file:
    mace_settings = safe_load(file)
model_dir = mace_settings["GENERAL"]["model_dir"]
with open("./active_learning_settings.yaml", "r") as file:
    al_settings = safe_load(file)["ACTIVE_LEARNING"]

random.seed(42)
np.random.seed(42)
ensemble_seeds = np.random.randint(0, 1000, al_settings["ensemble_size"])


initial_ds = InitalDatasetProcedure(
    mace_settings=mace_settings,
    al_settings=al_settings,
    initial_geometry="initial_geo.xyz",
    ensemble_seeds=ensemble_seeds
    )
initial_ds.run()