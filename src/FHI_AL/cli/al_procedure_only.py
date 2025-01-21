from FHI_AL.procedures.active_learning import ALProcedure, ALProcedureParallel
from yaml import safe_load
from mpi4py import MPI


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
        
    MPI.COMM_WORLD.Barrier()

    if not al.check_al_done():
        al.run()
    
    if al_settings['ACTIVE_LEARNING'].get("converge_al", False):
        al.converge()

if __name__ == "__main__":
    main()