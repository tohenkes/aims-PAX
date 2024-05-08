import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from FHI_AL.procedures import InitalDatasetProcedure, ALProcedure, StandardMACEEnsembleProcedure
from yaml import safe_load
import numpy as np
from ase.io import read
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, help="Start index")
parser.add_argument("--end", type=int, help="End index")
parser.add_argument("--test_data", type=int, help="Path to test data")
parser.add_argument("--initial_trajectory", type=str, help="Path to initial trajectory")
parser.add_argument("--sampling_trajectories", type=str, help="Path to trajectories for sampling")
args = parser.parse_args()

start = args.start
end = args.end
test_data = args.test_data
initial_trajectory = args.initial_trajectory
sampling_trajectories = args.sampling_trajectories

lamb = np.array([2.,3.])
intermediate_epochs = np.array([5,10])
converge_or_not = np.array([False,True])
scheduler_or_not_initial = np.array([False,True])
max_epochs_worker = np.array([1,2]) # times the intermediate epochs
scheduler_or_not_al = np.array([False,True])
uncert_cx = np.array([0.1,0.3])

# grid of all possible values
A,B,C,D,E,F,G = np.meshgrid(
    lamb,
    intermediate_epochs,
    converge_or_not,
    scheduler_or_not_initial,
    max_epochs_worker,
    scheduler_or_not_al,
    uncert_cx
    )
# one array with all the combinations
parameter_set = np.stack([A.ravel(),B.ravel(),C.ravel(),D.ravel(),E.ravel(),F.ravel(),G.ravel()],axis=1)
parameter_set = parameter_set[start:end]



with open("./mace_settings.yaml", "r") as file:
    mace_settings = safe_load(file)
model_dir = mace_settings["GENERAL"]["model_dir"]
with open("./active_learning_settings.yaml", "r") as file:
    al_settings = safe_load(file)["ACTIVE_LEARNING"]

random.seed(42)
np.random.seed(42)
ensemble_seeds = np.random.randint(0, 1000, al_settings["ensemble_size"])

test_data = read(test_data, index=':')
random.shuffle(test_data)
test_data = test_data[:1000]


for parameters in parameter_set:
    print(f"########################## Next parameter set: {parameters} ##########################")
    mace_settings["GENERAL"]["model_dir"] = model_dir
    results = {}
    parameter_tag = "-".join([str(x) for x in parameters])
    test_lamb = parameters[0]
    test_intermediate_epochs = parameters[1]
    test_converge_or_not = parameters[2]
    test_scheduler_or_not_initial = parameters[3]
    test_max_epochs_worker = parameters[4]
    test_scheduler_or_not_al = parameters[5]
    test_uncert_cx = parameters[6]

    al_settings['lambda'] = test_lamb
    al_settings['intermediate_epochs'] = int(test_intermediate_epochs)
    if not test_scheduler_or_not_initial:
        mace_settings["lr_scheduler"] = None
    al_settings["max_epochs_worker"] = int(test_max_epochs_worker * test_intermediate_epochs)
    al_settings["c_x"] = test_uncert_cx
    

    initial_ds = InitalDatasetProcedure(
        mace_settings=mace_settings,
        al_settings=al_settings,
        path_to_trajectory=initial_trajectory,
        ensemble_seeds=ensemble_seeds
    )

    initial_ds.run()
    if test_converge_or_not:
        initial_ds.converge()

    al = ALProcedure(
        mace_settings=mace_settings,
        al_settings=al_settings,
        path_to_trajectories= sampling_trajectories,
    )
    if test_scheduler_or_not_al:
        al.use_scheduler = True

    al.run()
    al.converge()
    al_metrics = al.evaluate_ensemble(
        ase_atoms_list=test_data
    )
    print("########################## Training and evaluation of the ensemble from scratch. ##########################")
    # train from scratch:
    scratch_procedure = StandardMACEEnsembleProcedure(
        mace_settings=mace_settings,
        ensemble_ase_sets=al.ensemble_ase_sets,
        seeds=np.array(list(al.seeds.values())),
    )
    mace_settings['GENERAL']['model_dir'] = './scratch_model'
    scratch_procedure.train()
    scratch_metrics = scratch_procedure.evaluate_ensemble(
        ase_atoms_list=test_data
    )

    results['al_metrics'] = al_metrics
    results['scratch_metrics'] = scratch_metrics
    results['total_aims_calls'] = al.point_added
    results['parameter_tag'] = parameter_tag
    results['t_intervals'] = al.t_intervals
    results['sanity_checks'] = al.sanity_checks
    results['sanity_checks_valid'] = al.sanity_checks_valid
    np.savez(f"./results/results_{parameter_tag}.npz", **results)








