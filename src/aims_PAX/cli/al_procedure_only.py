from aims_PAX.procedures.active_learning import (
    ALProcedureSerial,
    ALProcedureParallel,
    ALProcedurePARSL,
)
import argparse
from aims_PAX.tools.utilities.input_utils import read_input_files
import os
import time
import threading
from time import perf_counter

def main():
    parser = argparse.ArgumentParser(
        description="Create initial dataset for AIMLFF."
    )
    parser.add_argument(
        "--model-settings",
        type=str,
        default="./model.yaml",
        help="Path to model.yaml file",
    )
    parser.add_argument(
        "--aimsPAX-settings",
        type=str,
        default="./aimsPAX.yaml",
        help="Path to aimsPAX settings file",
    )
    args = parser.parse_args()

    (model_settings, aimsPAX_settings, path_to_control, path_to_geometry) = (
        read_input_files(
            path_to_model_settings=args.model_settings,
            path_to_aimsPAX_settings=args.aimsPAX_settings,
            procedure="al",
        )
    )

    if aimsPAX_settings["ACTIVE_LEARNING"].get("parallel", False):
        al = ALProcedureParallel(
            model_settings=model_settings,
            aimsPAX_settings=aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )
    elif aimsPAX_settings.get("CLUSTER", False):
        al = ALProcedurePARSL(
            model_settings=model_settings,
            aimsPAX_settings=aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )
    else:
        al = ALProcedureSerial(
            model_settings=model_settings,
            aimsPAX_settings=aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )

    if not al.check_al_done():
        al.run()

    if aimsPAX_settings["ACTIVE_LEARNING"].get("converge_al", False):
        al.converge()


if __name__ == "__main__":
    main()
