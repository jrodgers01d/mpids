from mpi4py import MPI

import mpids

if __name__ == "__main__":

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        data = list(range(100))
        array = mpids.MPInumpy.array(data, dtype = 'i', comm=comm)

        array_size = array.size()
        array_data = array.data()
        array_dtype = array.dtype()
        array_shape = array.shape()
        array_strides = array.strides()

        print('Rank{} Array Attributes: {} size, {} data, {} dtype, {} shape, {} strides'.format(rank, array_size, array_data, array_dtype, array_shape, array_strides))
