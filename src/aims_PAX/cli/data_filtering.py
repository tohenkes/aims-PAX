import argparse

from aims_PAX.procedures.data_filtering import DataFilteringProcedure
from aims_PAX.tools.utilities.input_utils import read_input_files


def main():
    parser = argparse.ArgumentParser(
        description="Filter large labeled HDF5 datasets into representative "
        "subsets using adaptive error thresholding."
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

    model_settings, aimsPAX_settings, _, _ = read_input_files(
        path_to_model_settings=args.model_settings,
        path_to_aimsPAX_settings=args.aimsPAX_settings,
        procedure="data-filtering",
    )

    df = DataFilteringProcedure(
        model_settings=model_settings,
        aimsPAX_settings=aimsPAX_settings,
    )

    if not df.check_df_done():
        df.run()

    if aimsPAX_settings["DATA_FILTERING"].get("converge_best", True):
        df.converge()


if __name__ == "__main__":
    main()
