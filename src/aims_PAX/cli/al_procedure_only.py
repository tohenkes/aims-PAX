from aims_PAX.procedures.active_learning import (
    ALProcedureSerial,
    ALProcedureParallel,
    ALProcedurePARSL,
)
import argparse
from aims_PAX.tools.utilities.utilities import read_input_files
import os
import time
import threading
from time import perf_counter

def main():
    parser = argparse.ArgumentParser(
        description="Create initial dataset for AIMLFF."
    )
    parser.add_argument(
        "--mace-settings",
        type=str,
        default="./mace.yaml",
        help="Path to mace.yaml file",
    )
    parser.add_argument(
        "--aimsPAX-settings",
        type=str,
        default="./aimsPAX.yaml",
        help="Path to aimsPAX settings file",
    )
    args = parser.parse_args()

    (mace_settings, aimsPAX_settings, path_to_control, path_to_geometry) = (
        read_input_files(
            path_to_mace_settings=args.mace_settings,
            path_to_aimsPAX_settings=args.aimsPAX_settings,
            procedure="al",
        )
    )

    if aimsPAX_settings["ACTIVE_LEARNING"].get("parallel", False):
        al = ALProcedureParallel(
            mace_settings=mace_settings,
            aimsPAX_settings=aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )
    elif aimsPAX_settings.get("CLUSTER", False):
        al = ALProcedurePARSL(
            mace_settings=mace_settings,
            aimsPAX_settings=aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )
    else:
        al = ALProcedureSerial(
            mace_settings=mace_settings,
            aimsPAX_settings=aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )

    if not al.check_al_done():
        accumulator_file = "./al_procedure_time.txt"

        # Load previously accumulated time
        if os.path.exists(accumulator_file):
            with open(accumulator_file, "r") as f:
                accumulated_time = float(f.read().strip())
        else:
            accumulated_time = 0.0

        start_time = perf_counter()
        # Periodically sync to file (every 1 minute)
        def sync_time_periodically(interval=60):
            def sync():
                while True:
                    time.sleep(interval)
                    current = perf_counter() - start_time
                    try:
                        with open(accumulator_file, "w") as f:
                            f.write(str(accumulated_time + current))
                    except Exception as e:
                        print("Failed to sync time:", e)
            t = threading.Thread(target=sync, daemon=True)
            t.start()
        sync_time_periodically()
        al.run()

    if aimsPAX_settings["ACTIVE_LEARNING"].get("converge_al", False):
        al.converge()


if __name__ == "__main__":
    main()
