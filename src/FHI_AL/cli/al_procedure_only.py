from FHI_AL.procedures.active_learning import ALProcedure, ALProcedureParallel
from yaml import safe_load
from mpi4py import MPI
from FHI_AL.tools.utilities import GPUMonitor
from time import perf_counter

def main():
    with open("./mace_settings.yaml", "r") as file:
        mace_settings = safe_load(file)
    with open("./active_learning_settings.yaml", "r") as file:
        al_settings = safe_load(file)

    if al_settings['ACTIVE_LEARNING'].get('parallel', False):
        al = ALProcedureParallel(
            mace_settings=mace_settings,
            al_settings=al_settings
        )

    else:
        al = ALProcedure(
            mace_settings=mace_settings,
            al_settings=al_settings
        )

    if not al.check_al_done():
        start_time = perf_counter()
        al.run()
        end_time = perf_counter()

    
    if al_settings['ACTIVE_LEARNING'].get("converge_al", False):
        al.converge()

if __name__ == "__main__":
    main()