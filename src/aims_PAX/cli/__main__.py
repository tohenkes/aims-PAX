from aims_PAX.procedures.initial_dataset import (
    InitialDatasetAIMD,
    InitialDatasetFoundational,
    InitialDatasetPARSL,
)
from aims_PAX.procedures.active_learning import (
    ALProcedureSerial,
    ALProcedureParallel,
    ALProcedurePARSL,
)
from aims_PAX.tools.utilities.input_utils import read_input_files
import argparse

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None


def main():

    parser = argparse.ArgumentParser(
        description="Create initial dataset for aims-PAX."
    )
    parser.add_argument(
        "--model-settings",
        type=str,
        default="./model.yaml",
        help="Path to model settings file",
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
            procedure="full",
        )
    )

    if (
        aimsPAX_settings["INITIAL_DATASET_GENERATION"][
            "initial_sampling"
        ].lower()
        == "aimd"
    ):
        initial_ds = InitialDatasetAIMD(
            model_settings=model_settings,
            al_settings=aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )
    elif (
        aimsPAX_settings["INITIAL_DATASET_GENERATION"][
            "initial_sampling"
        ].lower()
        == "foundational"
    ):
        if aimsPAX_settings.get("CLUSTER", False):
            initial_ds = InitialDatasetPARSL(
                model_settings=model_settings,
                aimsPAX_settings=aimsPAX_settings,
                path_to_control=path_to_control,
                path_to_geometry=path_to_geometry,
                close_parsl=False,
            )
        else:
            initial_ds = InitialDatasetFoundational(
                model_settings=model_settings,
                aimsPAX_settings=aimsPAX_settings,
                path_to_control=path_to_control,
                path_to_geometry=path_to_geometry,
            )

    if not initial_ds.check_initial_ds_done():
        initial_ds.run()

    if aimsPAX_settings["ACTIVE_LEARNING"].get("converge_initial", False):
        initial_ds.converge()

    if aimsPAX_settings["ACTIVE_LEARNING"].get("parallel", False):
        al = ALProcedureParallel(
            model_settings=model_settings,
            al_settings=aimsPAX_settings,
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

    if MPI is not None:
        MPI.Finalize()


if __name__ == "__main__":
    main()
