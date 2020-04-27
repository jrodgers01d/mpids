from mpi4py import MPI

import mpids.MPInumpy as mpi_np


if __name__ == "__main__":

    #Capture default communicator and MPI process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #Arrays  elements (values 0-24)
    mpi_array = mpi_np.arange(25, dist='b').reshape(5,5)

    #Capture distributed array data
    replicated_mpi_array = mpi_array[:]

    if rank == 0:
        print('Distributed Array Contents: \n{}\n'.format(replicated_mpi_array))
    comm.Barrier()
    print('Rank {} Local Array Contents: \n{}\n'.format(rank, mpi_array))
    comm.Barrier()

    #Local iteration of array
    for row_index, row in enumerate(mpi_array):
        #Method 1: Use Python generated result from a 'for' loop (__iter__())
        print('Rank {} Local Iteration Method 1: {}'.format(rank, row))
        #Method 2: Use index directly (__getitem__())
        print('Rank {} Local Iteration Method 2: {}'\
            .format(rank, mpi_array.local[row_index]))

    print()
    comm.Barrier()

    #Global iteration of array
    for row_index in range(mpi_array.globalshape[0]):
        #Method similar to local iter. method 2, but here we use global index
        print('Rank {} Global Iteration Method: {}'\
            .format(rank, mpi_array[row_index]))

    print() if rank == 0 else None
    comm.Barrier()
