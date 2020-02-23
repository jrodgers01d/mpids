from mpi4py import MPI
import numpy as np

import mpids.MPInumpy as mpi_np

if __name__ == "__main__":

    #Capture default communicator and MPI process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #Arrays  elements (values 0-15)
    data_1D = np.arange(16)
    data_2D = data_1D.reshape(4, 4)
    data_3D = data_2D.reshape(4, 2, 2)

    #Block data distribution
    block_mpi_array_1D = mpi_np.array(data_1D, comm=comm, dist='b')
    block_mpi_array_2D = mpi_np.array(data_2D, comm=comm, dist='b')
    block_mpi_array_3D = mpi_np.array(data_3D, comm=comm, dist='b')

    print('1D Global data:\n{}'.format(data_1D)) if rank == 0 else None
    print('2D Global data:\n{}'.format(data_2D)) if rank == 0 else None
    print('3D Global data:\n{}\n\n'.format(data_3D)) if rank == 0 else None
    comm.Barrier()
    print('1D Blocked Data Rank {}:\n{}'.format(rank, block_mpi_array_1D.local))
    comm.Barrier()
    print('2D Blocked Data Rank {}:\n{}'.format(rank, block_mpi_array_2D.local))
    comm.Barrier()
    print('3D Blocked Data Rank {}:\n{}'.format(rank, block_mpi_array_3D.local))

    #Undistributed data distribution
    undistributed_mpi_array_1D = mpi_np.array(data_1D, comm=comm, dist='u')
    undistributed_mpi_array_2D = mpi_np.array(data_2D, comm=comm, dist='u')
    undistributed_mpi_array_3D = mpi_np.array(data_3D, comm=comm, dist='u')

    print('1D Undistributed Data Rank {}:\n{}'\
        .format(rank, undistributed_mpi_array_1D.local))
    comm.Barrier()
    print('2D Undistributed Data Rank {}:\n{}'\
        .format(rank, undistributed_mpi_array_2D.local))
    comm.Barrier()
    print('3D Undistributed Data Rank {}:\n{}'\
        .format(rank, undistributed_mpi_array_3D.local))
