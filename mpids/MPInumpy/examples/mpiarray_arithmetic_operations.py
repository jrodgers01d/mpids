from mpi4py import MPI

import mpids.MPInumpy as mpi_np

if __name__ == "__main__":

    #Capture default communicator and MPI process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #Arrays  elements (values 0-9)
    block_mpi_array = mpi_np.arange(10)

    #Capture distributed array data
    replicated_mpi_array = block_mpi_array[:]

    if rank == 0:
        print('Distributed Array Contents: {}\n'.format(replicated_mpi_array))
    comm.Barrier()

    output = 'Rank {} Local Array Contents {}: \n'.format(rank, block_mpi_array)
    output += '\t local_array * 2 = {} \n'.format(block_mpi_array * 2)
    output += '\t local_array - 3 = {} \n'.format(block_mpi_array - 3)
    output += '\t local_array + 7 = {} \n'.format(block_mpi_array + 7)
    output += '\t local_array / 0.5 = {} \n'.format(block_mpi_array / 0.5)
    output += '\t local_array // 3 = {} \n'.format(block_mpi_array // 3)
    output += '\t local_array % 2 = {} \n'.format(block_mpi_array % 2)
    output += '\t local_array ** 2 = {} \n'.format(block_mpi_array ** 2)
    print(output)
