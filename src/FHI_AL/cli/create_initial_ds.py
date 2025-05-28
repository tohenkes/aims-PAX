from FHI_AL.procedures.initial_dataset import (
    InitialDatasetAIMD,
    InitialDatasetFoundational,
    InitialDatasetFoundationalParallel,
    InitialDatasetPARSL
    )
from yaml import safe_load
from time import perf_counter
import time


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

    if not initial_ds.check_initial_ds_done():
        initial_ds.run()
    if al_settings['ACTIVE_LEARNING'].get("converge_initial", False):
        initial_ds.converge() 


if __name__ == "__main__":
    main()
