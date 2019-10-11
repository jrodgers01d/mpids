from mpi4py import MPI
from mpids.MPIArray import MPIArray
import numpy as np

def array(array_object, dtype=None, copy=True, order=None, subok=False, ndmin=0,
          comm=MPI.COMM_WORLD, scatter=True):
        """ Create MPINpArray Object on all procs in comm. """
        np_array = np.array(array_object, dtype=dtype, copy=copy, order=order,
                            subok=subok, ndmin=ndmin)

        if not scatter: return MPIArray(np_array, comm)

        size = comm.Get_size()
        rank = comm.Get_rank()

        scatter_size = __get_scatterv_size(len(np_array), size, rank)
        np_array_recv = np.empty(scatter_size, dtype=np_array.dtype)

        #Currently limited to just basic C-types
        ##Leveraging MPI datatype auto-discovery for Numpy arrays
        comm.Scatterv(np_array, np_array_recv, root=0)

        return MPIArray(np_array_recv, comm)

def __get_scatterv_size(length, procs, rank):
        """ Return length required for scatter"""
        num = int(length / procs)
        rem = int(length % procs)

        if rank < rem:
                scatter_size = num + 1
        else:
                scatter_size = num

        return scatter_size
