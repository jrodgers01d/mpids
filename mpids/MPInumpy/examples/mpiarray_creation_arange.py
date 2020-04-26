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

    #Arange, evenly spaced values within specified interval
    print('From arange(start, stop, step) Routine') if rank == 0 else None
    mpi_arange = mpi_np.arange(size * 5)
    print('Local Arange Result Rank {}: {}'.format(rank, mpi_arange))
    print() if rank == 0 else None
