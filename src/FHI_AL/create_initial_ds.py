from FHI_AL.procedures import InitalDatasetProcedure
from yaml import safe_load
from mpi4py import MPI


def main():
    with open("./mace_settings.yaml", "r") as file:
        mace_settings = safe_load(file)
    with open("./active_learning_settings.yaml", "r") as file:
        al_settings = safe_load(file)

    initial_ds = InitalDatasetProcedure(
        mace_settings=mace_settings,
        al_settings=al_settings
    )

    MPI.COMM_WORLD.Barrier()

    initial_ds.run()
    if al_settings['ACTIVE_LEARNING']["converge_initial"]:
        initial_ds.converge() 

    MPI.COMM_WORLD.Barrier()
    MPI.Finalize()

if __name__ == "__main__":
    main()