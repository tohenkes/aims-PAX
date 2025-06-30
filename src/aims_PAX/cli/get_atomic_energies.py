from aims_PAX.procedures.atomic_energies import E0Calculator
from mpi4py import MPI
import argparse


def main():
    
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Calculator arguments')
        parser.add_argument('--control', type=str, help='Path to control input file', default='./control.in')
        parser.add_argument('--basis_dir', type=str, help='Path to species directory',default=None)
        parser.add_argument('--geometry', type=str, help='Path to geometry input file',default='geometry.in')
        parser.add_argument('--aims-lib-path', type=str, help='Path to aims library',default=None)
        parser.add_argument('--Zs', type=int, nargs='+', help='Atomic numbers of the atoms in the system', default=None)
        return parser.parse_args()

    args = parse_arguments()
    
    E0s_calc = E0Calculator(
        basis_dir=args.basis_dir,
        path_to_control=args.control,
        path_to_geometry=args.geometry,
        aims_path=args.aims_lib_path,
        Zs=args.Zs
    )

    MPI.COMM_WORLD.Barrier()
    
    E0s_calc()

    MPI.COMM_WORLD.Barrier()
    MPI.Finalize()

if __name__ == "__main__":
    main()