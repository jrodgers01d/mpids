from mpi4py import MPI
import numpy as np

import mpids.MPInumpy as mpi_np

if __name__ == "__main__":

    #Capture default communicator and MPI process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #Arrays  elements (values 0-24)
    mpi_array = mpi_np.arange(25, dist='b').reshape(5,5)

    print('Rank {} Local Array Contents: \n{}\n'.format(rank, mpi_array))
    comm.Barrier()

    #Capture distributed array data
    #Method 1: Use indexing notation to select all elements
    replicated_mpi_array_method_1 = mpi_array[:]
    #Method 2: Use method used by getter (__getitem__()) to produce result
    replicated_mpi_array_method_2 = mpi_array.collect_data()

    #Confirm both methods return same result
    assert np.allclose(replicated_mpi_array_method_1,
                       replicated_mpi_array_method_2)

    if rank == 0:
        print('Collected Distributed Array Contents: \n{}\n'\
            .format(replicated_mpi_array_method_1))
    comm.Barrier()
