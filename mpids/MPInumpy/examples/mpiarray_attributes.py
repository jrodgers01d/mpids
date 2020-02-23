from mpi4py import MPI
import platform

import mpids.MPInumpy as mpi_np

if __name__ == "__main__":

    #Capture default communicator and MPI process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #Create array-like data with 8 elements, values 0-7
    data = list(range(8))
    #Create MPInumpy array, passing data and default communicator
    mpi_array = mpi_np.array(data, comm=comm)

    #Print mpi_array attributes for each MPI process
    output = '{} Rank {} MPIArray Attributes: \n'.format(platform.node(), rank)
    output += '\t mpi_array.base = {} \n'.format(mpi_array.base)
    output += '\t mpi_array.dtype = {} \n'.format(mpi_array.dtype)
    #Common distributed local and global properties
    output += '\t mpi_array.shape = {} \n'.format(mpi_array.shape)
    output += '\t mpi_array.globalshape = {} \n'.format(mpi_array.globalshape)
    output += '\t mpi_array.size = {} \n'.format(mpi_array.size)
    output += '\t mpi_array.globalsize = {} \n'.format(mpi_array.globalsize)
    output += '\t mpi_array.nbytes = {} \n'.format(mpi_array.nbytes)
    output += '\t mpi_array.globalnbytes = {} \n'.format(mpi_array.globalnbytes)
    output += '\t mpi_array.ndim = {} \n'.format(mpi_array.ndim)
    output += '\t mpi_array.globalndim = {} \n'.format(mpi_array.globalndim)
    #Unique properties to MPIArray
    output += '\t mpi_array.dist = {} \n'.format(mpi_array.dist)
    output += '\t mpi_array.comm = {} \n'.format(mpi_array.comm)
    output += '\t mpi_array.comm_dims = {} \n'.format(mpi_array.comm_dims)
    output += '\t mpi_array.comm_coord = {} \n'.format(mpi_array.comm_coord)
    output += \
        '\t mpi_array.local_to_global = {} \n'.format(mpi_array.local_to_global)
    print(output)
