import sys
import os
import logging
#sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#sys.path.append("/home/tobias/Uni/Promotion/Research/aims_MLFF/active_learning")
#sys.path.append("/home/tobias/Uni/Promotion/Research/aims_MLFF/active_learning")
sys.path.append("/home/thenkes/Documents/Uni/Promotion/Research/aims_MLFF/active_learning")
from FHI_AL.procedures import InitalDatasetProcedure
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import logging
from yaml import safe_load


with open("../../mace_settings.yaml", "r") as file:
    mace_settings = safe_load(file)
with open("../../active_learning_settings.yaml", "r") as file:
    al_settings = safe_load(file)["ACTIVE_LEARNING"]

initial_ds = InitalDatasetProcedure(
    mace_settings=mace_settings,
    al_settings=al_settings,
    #ensemble_seeds=[20, 10],
)
initial_ds.z = [1,8,1]
sampled_points = []

def my_function(energy, coords, forces, step_number):
    global initial_ds, sampled_points, initial_ds, epoch
    status = 0
    if (step_number )% initial_ds.skip_step == 0 and step_number !=0 :
        atoms = initial_ds.data_to_ase(energy, coords, forces)
        sampled_points.append(atoms)
        if len(sampled_points) % (initial_ds.n_samples * initial_ds.ensemble_size) == 0:
            logging.info(f"Sampled points at step {step_number}: {len(sampled_points)}")
            status = initial_ds.run(sampled_points)
            if status == 1 and al_settings["converge_initial"]:
                initial_ds.converge()
                
    return status
