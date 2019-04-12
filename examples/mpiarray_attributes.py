from mpi4py import MPI

import mpids

if __name__ == "__main__":

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        data = list(range(100))
        array = mpids.MPInumpy.array(data, comm=comm)

        array_size = array.size()
        array_data = array.data()
        array_dtype = array.dtype()
        array_shape = array.shape()
        array_strides = array.strides()

        #Print array attributes for each proc in comm
        output = 'Rank {} Array Attributes: '.format(rank)
        output = output + '{} size, '.format(array_size)
        output = output + '{} data, '.format(array_data)
        output = output + '{} dtype, '.format(array_dtype)
        output = output + '{} shape, '.format(array_shape)
        output = output + '{} strides, '.format(array_strides)
        print(output)
