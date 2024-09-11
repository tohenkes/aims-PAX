from FHI_AL.procedures.initial_dataset import InitalDatasetProcedure, InitialDSFoundational
from yaml import safe_load
from mpi4py import MPI


def main():
    with open("./mace_settings.yaml", "r") as file:
        mace_settings = safe_load(file)
    with open("./active_learning_settings.yaml", "r") as file:
        al_settings = safe_load(file)

    if al_settings['ACTIVE_LEARNING']["initial_foundational_size"] is None:
        initial_ds = InitalDatasetProcedure(
            mace_settings=mace_settings,
            al_settings=al_settings
        )
    else:
        initial_ds = InitialDSFoundational(
            mace_settings=mace_settings,
            al_settings=al_settings
        )
    MPI.COMM_WORLD.Barrier()
    
    #check if initial_ds_done.txt exists
    if not initial_ds.check_initial_ds_done():
        initial_ds.run()
        
    if al_settings['ACTIVE_LEARNING']["converge_initial"]:
        initial_ds.converge() 

    MPI.COMM_WORLD.Barrier()
    MPI.Finalize()

if __name__ == "__main__":
    main()