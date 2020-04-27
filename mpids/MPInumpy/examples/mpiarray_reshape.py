from mpi4py import MPI

import mpids.MPInumpy as mpi_np

if __name__ == "__main__":

    #Capture default communicator and MPI process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #Arrays  elements (values 0-20)
    mpi_array = mpi_np.arange(21, dist='b').reshape(7, 3)
    #Reshape (Redistribute) Array Data
    reshaped_mpi_array = mpi_array.reshape(3, 7)

    #Capture distributed array data
    replicated_mpi_array = mpi_array[:]
    #Capture reshaped distributed array data
    replicated_reshaped_mpi_array = reshaped_mpi_array[:]

    if rank == 0:
        print('Original Distributed Array Contents: \n{}\n'\
            .format(replicated_mpi_array))
    if rank == 0:
        print('Reshaped Distributed Array Contents: \n{}\n'\
            .format(replicated_reshaped_mpi_array))
    comm.Barrier()

    print('Rank {} Original Local Array Contents: \n{}\n'\
        .format(rank, mpi_array))
    comm.Barrier()

    print('Rank {} Reshaped Local Array Contents: \n{}\n'\
        .format(rank, reshaped_mpi_array))
    comm.Barrier()
