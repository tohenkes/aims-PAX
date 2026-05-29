class CommHandler:
    """Stub communicator for serial/PARSL execution (no MPI)."""

    def __init__(self, use_mpi: bool = False):
        self.use_mpi = False
        self.comm = None
        self.rank = 0
        self.size = 1

    def barrier(self):
        pass

    def bcast(self, data, root=0):
        return data

    def get_rank(self) -> int:
        return self.rank

    def get_size(self) -> int:
        return self.size
