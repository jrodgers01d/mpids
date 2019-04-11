from mpi4py import MPI
from mpids.MPIArray import MPIArray
import numpy as np

def array(object, dtype=None, comm=MPI.COMM_WORLD):
        """
        Create MPINpArray Object on all procs in comm.
        Currently only supports duplication of object.
        """

        return MPIArray(np.array(object, dtype), comm)
