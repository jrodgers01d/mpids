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

    #Array, distributed array-like data
    print('From array(array_like_data) Routine') if rank == 0 else None
    array_like_data = list(range(size * 5))
    mpi_array = mpi_np.array(array_like_data)
    print('Local Array Result Rank {}: {}'.format(rank, mpi_array))
    print() if rank == 0 else None
