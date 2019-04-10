from mpi4py import MPI
from mpids.MPIArray import MPIArray
import numpy as np

def array(object, dtype=None, comm=MPI.COMM_WORLD):
        """
        Create MPINpArray Object on all procs in comm.
        Currently only supports duplication of object.
        """
        np_array = np.array(object, dtype)

        return MPIArray(np_array, comm)
