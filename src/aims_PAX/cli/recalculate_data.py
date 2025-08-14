from aims_PAX.procedures.recalculate import ReCalculatorPARSL
from mpi4py import MPI
import argparse


def main():

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Calculator arguments")
        parser.add_argument(
            "--data", type=str, help="Path to data file", default=None
        )
        parser.add_argument(
            "--basis_dir",
            type=str,
            help="Path to species directory",
            default=None,
        )
        parser.add_argument(
            "--control",
            type=str,
            help="Path to control input file",
            default="./control.in",
        )
        parser.add_argument(
            "--properties",
            type=str,
            nargs="+",
            help="Properties to calculate",
            default=["energy", "forces"],
        )
        parser.add_argument(
            "--settings",
            type=str,
            help="Path to active learning settings file",
            default="./aimsPAX.yaml",
        )
        parser.add_argument(
            "--start_idx",
            type=int,
            help="Index to start recalculation",
            default=0,
        )
        parser.add_argument(
            "--end_idx",
            type=int,
            help="Index to end recalculation",
            default=None,
        )
        parser.add_argument(
            "--save_interval",
            type=int,
            help="Interval to save data",
            default=10,
        )
        return parser.parse_args()

    args = parse_arguments()

    recalc = ReCalculatorPARSL(
        path_to_data=args.data,
        basis_dir=args.basis_dir,
        path_to_control=args.control,
        path_to_settings=args.settings,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        save_interval=args.save_interval,
        properties=args.properties,
    )
    recalc()

if __name__ == "__main__":
    main()
