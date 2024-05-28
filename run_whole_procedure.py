import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from FHI_AL.procedures import InitalDatasetProcedure, ALProcedure
from yaml import safe_load

with open("./mace_settings.yaml", "r") as file:
    mace_settings = safe_load(file)
with open("./active_learning_settings.yaml", "r") as file:
    al_settings = safe_load(file)["ACTIVE_LEARNING"]


initial_ds = InitalDatasetProcedure(
    mace_settings=mace_settings,
    al_settings=al_settings,
    path_to_trajectory="/home/thenkes/Documents/Uni/Promotion/Research/aims_MLFF/data/naphtalene/nve_80K_extxyz/out-1/naph_80K_1.xyz",
    #path_to_trajectory="/home/tobias/Uni/Promotion/Research/aims_MLFF/data/naphtalene/nve_80K_extxyz/out-1/naph_80K_1.xyz"
)

if not al_settings["scheduler_initial"]:
    mace_settings["lr_scheduler"] = None

initial_ds.run()
if al_settings["converge_initial"]:
    initial_ds.converge()   

al = ALProcedure(
    mace_settings=mace_settings,
    al_settings=al_settings,
    #path_to_trajectories= '/home/tobias/Uni/Promotion/Research/aims_MLFF/data/naphtalene/nve_80K_extxyz/trajectories_for_sampling',
    path_to_trajectories='/home/thenkes/Documents/Uni/Promotion/Research/aims_MLFF/data/naphtalene/nve_80K_extxyz/trajectories_for_sampling'
)
al.run()
al.converge()
