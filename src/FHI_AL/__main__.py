from FHI_AL.procedures import InitalDatasetProcedure, ALProcedure
from yaml import safe_load
from mpi4py import MPI


def main():
    with open("./mace_settings.yaml", "r") as file:
        mace_settings = safe_load(file)
    with open("./active_learning_settings.yaml", "r") as file:
        al_settings = safe_load(file)

    species_dir = al_settings['ACTIVE_LEARNING']["species_dir"]
    aims_lib_path = al_settings['ACTIVE_LEARNING']["aims_lib_path"]

    initial_ds = InitalDatasetProcedure(
        mace_settings=mace_settings,
        al_settings=al_settings,
        path_to_aims_lib=aims_lib_path,
        species_dir=species_dir,
        atomic_energies_dict=mace_settings['ARCHITECTURE'].get("atomic_energies", None)
    )

    if not al_settings['ACTIVE_LEARNING']["scheduler_initial"]:
        mace_settings["lr_scheduler"] = None

    MPI.COMM_WORLD.Barrier()
    initial_ds.run()
    if al_settings['ACTIVE_LEARNING']["converge_initial"]:
        initial_ds.converge() 

    MPI.COMM_WORLD.Barrier()
    al = ALProcedure(
        mace_settings=mace_settings,
        al_settings=al_settings,
        path_to_aims_lib=aims_lib_path,
        species_dir=species_dir
    )
    MPI.COMM_WORLD.Barrier()

    al.run()
    al.converge()

    MPI.COMM_WORLD.Barrier()
    MPI.Finalize()

if __name__ == "__main__":
    main()