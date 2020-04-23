from mpi4py import MPI
import numpy as np

import mpids.MPInumpy as mpi_np

if __name__ == "__main__":

    #Capture default communicator and MPI process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #Arrays  elements (values 0-15)
    data_1D = np.arange(16)

    #Block data distribution
    block_mpi_array_1D = mpi_np.array(data_1D, comm=comm, dist='b')

    print('1D Global data:\n{}\n\n'.format(data_1D)) if rank == 0 else None
    comm.Barrier()
    print('1D Blocked Data Rank {}:\n{}'.format(rank, block_mpi_array_1D))

    #Replicated data distribution
    replicated_mpi_array_1D = mpi_np.array(data_1D, comm=comm, dist='r')

    comm.Barrier()
    print('1D Replicated Data Rank {}:\n{}'\
        .format(rank, replicated_mpi_array_1D))
