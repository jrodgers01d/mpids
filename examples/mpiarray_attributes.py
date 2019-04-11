from mpi4py import MPI

import mpids

if __name__ == "__main__":

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        data = list(range(100))
        array = mpids.MPInumpy.array(data, dtype = 'i', comm)

        array_size = array.size()
        array_data = array.data()
        array_dtype = array.dtype()
        array_shape = array.shape()
        array_strides = array.strides()

        print(f'Rank{rank} Array Attributes: {array_size} size, {array_data} data, {array_dtype} dtype, {array_shape} shape, {array_strides} strides')
