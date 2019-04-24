from mpi4py import MPI
import platform

import mpids

if __name__ == "__main__":

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        data = list(range(103))
        array = mpids.MPInumpy.array(data, comm=comm)

        #Print array attributes for each proc in comm
        output = '{} Rank {} Array Attributes: '.format(platform.node(), rank)
        output += '{} size, '.format(array.size)
        output += '{} data, '.format(array.data)
        output += '{} dtype, '.format(array.dtype)
        output += '{} shape, '.format(array.shape)
        output += '{} strides, '.format(array.strides)
        output += '{} global_size'.format(array.global_size)
        print(output)
