from FHI_AL.procedures.initial_dataset import (
    InitialDatasetAIMD,
    InitialDatasetFoundational,
    InitialDatasetFoundationalParallel,
    InitialDatasetPARSL
    )
from yaml import safe_load
from time import perf_counter
import time
from FHI_AL.tools.utilities import GPUMonitor
from FHI_AL.tools.utilities_parsl import create_parsl_config, parsl_test_app


def main():
    with open("./mace_settings.yaml", "r") as file:
        mace_settings = safe_load(file)
    with open("./active_learning_settings.yaml", "r") as file:
        al_settings = safe_load(file)

    if al_settings['ACTIVE_LEARNING']["initial_sampling"].lower() == "aimd":
        initial_ds = InitialDatasetAIMD(
            mace_settings=mace_settings,
            al_settings=al_settings
        )
    elif al_settings['ACTIVE_LEARNING']["initial_sampling"].lower() == "mace-mp0":    
        if al_settings['ACTIVE_LEARNING'].get('parallel', False):
            initial_ds = InitialDatasetFoundationalParallel(
                mace_settings=mace_settings,
                al_settings=al_settings
            )
        elif al_settings.get('CLUSTER', False):
            initial_ds = InitialDatasetPARSL(
                mace_settings=mace_settings,
                al_settings=al_settings
            )
        else:
            initial_ds = InitialDatasetFoundational(
                mace_settings=mace_settings,
                al_settings=al_settings
            )
    #MPI.COMM_WORLD.Barrier()
    if not initial_ds.check_initial_ds_done():
        #monitor = GPUMonitor(1, 'gpu_utilization.csv')
        #start_time = perf_counter()
        initial_ds.run()
        #end_time = perf_counter()
        #monitor.stop()
        #execution_time = end_time - start_time
        #if MPI.COMM_WORLD.Get_rank() == 0:
        #    with open("initial_ds_execution_time.txt", "a") as file:
        #        file.write(f".run() Execution Time: {execution_time:.6f} seconds\n")
    if al_settings['ACTIVE_LEARNING'].get("converge_initial", False):
        initial_ds.converge() 

    #MPI.COMM_WORLD.Barrier()
    #MPI.Finalize()


if __name__ == "__main__":
    main()
