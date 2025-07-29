from aims_PAX.procedures.active_learning import (
    ALProcedureSerial,
    ALProcedureParallel,
    ALProcedurePARSL,
)
from yaml import safe_load

# from time import perf_counter


def main():
    with open("./mace_settings.yaml", "r") as file:
        mace_settings = safe_load(file)
    with open("./active_learning_settings.yaml", "r") as file:
        al_settings = safe_load(file)

    path_to_control = al_settings["MISC"].get(
        "path_to_control", "./control.in"
    )
    path_to_geometry = al_settings["MISC"].get(
        "path_to_geometry", "./geometry.in"
    )

    if al_settings["ACTIVE_LEARNING"].get("parallel", False):
        al = ALProcedureParallel(
            mace_settings=mace_settings,
            al_settings=al_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )
    elif al_settings.get("CLUSTER", False):
        al = ALProcedurePARSL(
            mace_settings=mace_settings,
            al_settings=al_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )
    else:
        al = ALProcedureSerial(
            mace_settings=mace_settings,
            al_settings=al_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )

    if not al.check_al_done():
        # start_time = perf_counter()
        al.run()
        # end_time = perf_counter()
        # with open("./al_procedure_time.txt", "w") as f:
        #    f.write(
        #        f"Active Learning Procedure Time: {end_time - start_time} seconds\n"
        #    )

    if al_settings["ACTIVE_LEARNING"].get("converge_al", False):
        al.converge()


if __name__ == "__main__":
    main()
