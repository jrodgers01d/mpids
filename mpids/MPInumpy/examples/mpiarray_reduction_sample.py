from mpi4py import MPI

import mpids.MPInumpy as mpi_np

#Sample normalization of all columns by their mean value

if __name__ == "__main__":

    #Capture default communicator and MPI process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #Arrays  elements (values 0-24)
    block_mpi_array = mpi_np.arange(25, dist='b').reshape(5,5)

    #Capture distributed array data
    replicated_mpi_array = block_mpi_array[:]

    if rank == 0:
        print('Distributed Array Contents:\n{}\n'.format(replicated_mpi_array))
    comm.Barrier()

    block_mpi_array_col_mean = block_mpi_array.mean(axis=0)
    block_mpi_array_col_normalized = block_mpi_array / block_mpi_array_col_mean

    #Capture distributed array data after setter routine update
    updated_replicated_mpi_array = block_mpi_array_col_normalized[:]
    if rank == 0:
        print('Distributed Array Contents After Column Normalization: \n{}\n'\
            .format(updated_replicated_mpi_array))
