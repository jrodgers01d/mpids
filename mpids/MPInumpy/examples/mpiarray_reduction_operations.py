from mpi4py import MPI

import mpids.MPInumpy as mpi_np

if __name__ == "__main__":

    #Capture default communicator and MPI process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #Arrays  elements (values 0-15)
    block_mpi_array = mpi_np.arange(16).reshape(4, 4)

    #Capture distributed array data
    replicated_mpi_array = block_mpi_array[:]

    if rank == 0:
        print('Distributed Array Contents:\n{}\n'.format(replicated_mpi_array))
    comm.Barrier()

    output = 'Rank {} Local Array Contents:\n{}\n\n'.format(rank, block_mpi_array)
    output += 'Rank {} Reduction Results:\n'.format(rank)
    output += '\tarray.max() = {}\n'.format(block_mpi_array.max())
    output += '\tarray.max(axis=0) = {}\n'.format(block_mpi_array.max(axis=0))
    output += '\tarray.max(axis=1) = {}\n'.format(block_mpi_array.max(axis=1))
    output += '\tarray.mean() = {}\n'.format(block_mpi_array.mean())
    output += '\tarray.mean(axis=0) = {}\n'.format(block_mpi_array.mean(axis=0))
    output += '\tarray.mean(axis=1) = {}\n'.format(block_mpi_array.mean(axis=1))
    output += '\tarray.min() = {}\n'.format(block_mpi_array.min())
    output += '\tarray.min(axis=0) = {}\n'.format(block_mpi_array.min(axis=0))
    output += '\tarray.min(axis=1) = {}\n'.format(block_mpi_array.min(axis=1))
    output += '\tarray.std() = {}\n'.format(block_mpi_array.std())
    output += '\tarray.std(axis=0) = {}\n'.format(block_mpi_array.std(axis=0))
    output += '\tarray.std(axis=1) = {}\n'.format(block_mpi_array.std(axis=1))
    output += '\tarray.sum() = {}\n'.format(block_mpi_array.sum())
    output += '\tarray.sum(axis=0) = {}\n'.format(block_mpi_array.sum(axis=0))
    output += '\tarray.sum(axis=1) = {}\n'.format(block_mpi_array.sum(axis=1))
    print(output)
