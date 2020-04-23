from mpi4py import MPI
import numpy as np

import mpids.MPInumpy as mpi_np

if __name__ == "__main__":

    #Capture default communicator and MPI process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    note = "Note: creation routines are using their default MPI related kwargs."
    note += "\nDefault kwargs:"
    note += " routine(..., comm=MPI.COMM_WORLD, root=0, dist='b')\n"
    comm.Barrier()
    print(note) if rank == 0 else None

    #Arange, evenly spaced values within specified interval
    print('From arange() Routine') if rank == 0 else None
    print('Local Arange Result Rank {}: {}'\
        .format(rank, mpi_np.arange(10)))
    comm.Barrier()
    print() if rank == 0 else None
    comm.Barrier()

    #Array, distributed array-like data
    array_like_data = list(range(10, 20, 1))
    print('From array() Routine') if rank == 0 else None
    print('Local Array Result Rank {}: {}'\
        .format(rank, mpi_np.array(array_like_data)))
    comm.Barrier()
    print() if rank == 0 else None
    comm.Barrier()

    #Empty, shape based non-intialized distributed array
    array_shape = (4, 4)
    print('From empty() Routine:') if rank == 0 else None
    print('Local Array Result Rank {}:\n{}'\
        .format(rank, mpi_np.empty(array_shape)))
    comm.Barrier()
    print() if rank == 0 else None
    comm.Barrier()

    #Ones, shape based distributed array initialized with ones
    print('From ones() Routine:') if rank == 0 else None
    print('Local Array Result Rank {}:\n{}'\
        .format(rank, mpi_np.ones(array_shape)))
    comm.Barrier()
    print() if rank == 0 else None
    comm.Barrier()

    #Zeros, shape based distributed array initialized with zeros
    print('From zeros() Routine:') if rank == 0 else None
    print('Local Array Result Rank {}:\n{}'\
        .format(rank, mpi_np.zeros(array_shape)))
    comm.Barrier()
    print() if rank == 0 else None
