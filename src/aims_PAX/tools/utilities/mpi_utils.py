import numpy as np
import ase


class CommHandler:
    def __init__(self, use_mpi=True):
        self.use_mpi = use_mpi
        if self.use_mpi:
            try:
                from mpi4py import MPI
            except ImportError:
                raise ImportError(
                    "mpi4py is not installed."
                    "Please install it to use MPI features."
                )
            self.comm = MPI.COMM_WORLD
            self.mpi = MPI
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.rank = 0
            self.size = 1

    def barrier(self):
        if self.use_mpi:
            self.comm.Barrier()

    def bcast(self, data, root=0):
        if self.use_mpi:
            return self.comm.bcast(data, root=root)
        return data

    def get_rank(self):
        return self.rank

    def get_size(self):
        return self.size


def send_points_non_blocking(
    idx: int, point_data: ase.Atoms, tag: int, world_comm
):
    # send idx, pbc, cell, positions, species in a non-blocking way

    positions = np.asarray(point_data.get_positions(), dtype=np.float64)
    species = point_data.get_atomic_numbers()
    pbc = point_data.pbc
    cell = point_data.get_cell()
    num_atoms = len(positions)

    idx_num_atoms_pbc_cell = np.array(
        [idx, num_atoms, *pbc.flatten(), *cell.flatten()], dtype=np.float64
    )
    species_array = np.array(
        [[element, element, element] for element in species],
        dtype=np.float64,
    )

    positions_species = np.empty(
        shape=(2, positions.shape[0], 3),
    )
    positions_species[0] = species_array
    positions_species[1] = positions

    world_comm.Isend(buf=idx_num_atoms_pbc_cell, dest=1, tag=tag)
    world_comm.Isend(buf=positions_species, dest=1, tag=tag + 1)
