import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from FHI_AL.procedures import InitalDatasetProcedure, ALProcedure, StandardMACEEnsembleProcedure
from yaml import safe_load
import argparse
from ase.io import read
import random
import numpy as np

with open("./mace_settings.yaml", "r") as file:
    mace_settings = safe_load(file)
with open("./active_learning_settings.yaml", "r") as file:
    al_settings = safe_load(file)["ACTIVE_LEARNING"]

parser = argparse.ArgumentParser()
parser.add_argument("--initial_trajectory", type=str, help="Path to initial trajectory")
parser.add_argument("--sampling_trajectories", type=str, help="Path to trajectories for sampling")
parser.add_argument("--n_test", type=str, help="Number of test points.", default=1000)
parser.add_argument("--test_data", type=str, help="Path to test data", default=None)
args = parser.parse_args()
initial_trajectory = args.initial_trajectory
sampling_trajectories = args.sampling_trajectories
n_test = args.n_test



initial_ds = InitalDatasetProcedure(
    mace_settings=mace_settings,
    al_settings=al_settings,
    path_to_trajectory=initial_trajectory,
)

if not al_settings["scheduler_initial"]:
    mace_settings["lr_scheduler"] = None

initial_ds.run()
if al_settings["converge_initial"]:
    initial_ds.converge()   

al = ALProcedure(
    mace_settings=mace_settings,
    al_settings=al_settings,
    path_to_trajectories=sampling_trajectories
)
al.run()
al.converge()

test_data = args.test_data
if test_data is not None:
    test_data = read(test_data, index=':')
    random.shuffle(test_data)
    test_data = test_data[:n_test]

    al_metrics = al.evaluate_ensemble(
        ase_atoms_list=test_data
    )
    results = {}
    results['al_metrics'] = al_metrics
    results['total_aims_calls'] = al.point_added + initial_ds.point_added
    results['t_intervals'] = al.t_intervals
    results['sanity_checks'] = al.sanity_checks
    results['sanity_checks_valid'] = al.sanity_checks_valid
    np.savez(f"./results/results_{mace_settings['GENERAL']['test']}.npz", **results)