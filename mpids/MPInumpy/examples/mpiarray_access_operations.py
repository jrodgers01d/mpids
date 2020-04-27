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

    #Getter routines return globally resolved result to all processes
    mpi_array_index_00 = mpi_array[0,0]
    mpi_array_slice = mpi_array[::2]

    print('Rank {} Globally Indexed Result: {}\n'.format(rank,
                                                         mpi_array_index_00))
    comm.Barrier()
    print('Rank {} Globally Sliced Result: \n{}\n'.format(rank,
                                                          mpi_array_slice))
    comm.Barrier()


    #Setter routines only modify local result based on global index
    mpi_array[4,4] = 9999

    #Capture distributed array data after setter routine update
    updated_replicated_mpi_array = mpi_array[:]
    if rank == 0:
        print('Distributed Array Contents After Setter Update: \n{}\n'\
            .format(updated_replicated_mpi_array))
