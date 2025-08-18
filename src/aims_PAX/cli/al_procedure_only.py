from aims_PAX.procedures.active_learning import (
    ALProcedureSerial,
    ALProcedureParallel,
    ALProcedurePARSL,
)
import argparse
from aims_PAX.tools.utilities.utilities import read_input_files


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
        al.run()

    if aimsPAX_settings["ACTIVE_LEARNING"].get("converge_al", False):
        al.converge()


if __name__ == "__main__":
    main()
