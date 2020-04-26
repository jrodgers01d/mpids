from mpi4py import MPI
import numpy as np

import mpids.MPInumpy as mpi_np

if __name__ == "__main__":

    #Capture default communicator, MPI process rank, and number of MPI processes
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    note = "Note: creation routines are using their default MPI related kwargs."
    note += "\nDefault kwargs:"
    note += " routine(..., comm=MPI.COMM_WORLD, root=0, dist='b')\n"
    print(note) if rank == 0 else None

    #Empty, shape based non-intialized distributed array
    print('From empty(array_shape) Routine:') if rank == 0 else None
    array_shape = (size, size)
    mpi_empty_array = mpi_np.empty(array_shape)
    print('Local Array Result Rank {}:\n{}'.format(rank, mpi_empty_array))
    print() if rank == 0 else None
