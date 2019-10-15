from mpi4py import MPI
from mpids.MPInumpy.MPIArray import MPIArray
from mpids.MPInumpy.utils import get_local_data

def array(array_data, dtype=None, copy=True, order=None, subok=False, ndmin=0,
          comm=MPI.COMM_WORLD, distribution='Equal'):
        """ Create MPINpArray Object on all procs in comm. """

        if distribution != 'Equal':
                print('Only equal blocked distributions currently supported.')
                return None

        size = comm.Get_size()
        rank = comm.Get_rank()

        local_data = get_local_data(array_data, distribution, size, rank)

        return MPIArray(local_data, comm)
