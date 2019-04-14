from mpi4py import MPI

import mpids

if __name__ == "__main__":

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        data = list(range(100))
        array = mpids.MPInumpy.array(data, comm=comm)

        # array_size = array.size()
        # array_data = array.data()
        # array_dtype = array.dtype()
        # array_shape = array.shape()
        # array_strides = array.strides()

        #Print array attributes for each proc in comm
        output = 'Rank {} Array Attributes: '.format(rank)
        output = output + '{} size, '.format(array.size)
        output = output + '{} data, '.format(array.data)
        output = output + '{} dtype, '.format(array.dtype)
        output = output + '{} shape, '.format(array.shape)
        output = output + '{} strides, '.format(array.strides)
        output = output + '{} strides, '.format(array.comm_size)
        print(output)
