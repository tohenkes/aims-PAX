import argparse

from aims_PAX.procedures.initial_dataset import (
    InitialDatasetAIMD,
    InitialDatasetPARSL,
    InitialDatasetPARSLTeacher,
)
from aims_PAX.tools.utilities.input_utils import read_input_files


def main():
    parser = argparse.ArgumentParser(
        description="Create initial dataset for aims-PAX."
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
            procedure="initial-ds",
        )
    )

    idg = aimsPAX_settings.INITIAL_DATASET_GENERATION
    if idg.initial_sampling.lower() == "aimd":
        initial_ds = InitialDatasetAIMD(
            model_settings=model_settings,
            aimsPAX_settings=aimsPAX_settings,
            path_to_control=path_to_control,
            path_to_geometry=path_to_geometry,
        )
    elif idg.initial_sampling.lower() == "foundational":
        if aimsPAX_settings.CLUSTER is None:
            raise ValueError(
                "Foundational initial dataset generation requires CLUSTER "
                "settings. Please add a CLUSTER section to aimsPAX.yaml."
            )
        if idg.use_teacher_reference:
            initial_ds = InitialDatasetPARSLTeacher(
                model_settings=model_settings,
                aimsPAX_settings=aimsPAX_settings,
                path_to_control=path_to_control,
                path_to_geometry=path_to_geometry,
            )
        else:
            initial_ds = InitialDatasetPARSL(
                model_settings=model_settings,
                aimsPAX_settings=aimsPAX_settings,
                path_to_control=path_to_control,
                path_to_geometry=path_to_geometry,
            )
    else:
        raise ValueError(
            f"Unknown initial_sampling: {idg.initial_sampling!r}. "
            "Expected 'aimd' or 'foundational'."
        )

    if not initial_ds.check_initial_ds_done():
        initial_ds.run()
    if idg.converge_initial:
        initial_ds.converge()


if __name__ == "__main__":
    main()
