from mpi4py import MPI
from numpy import matmul as np_matmul
from mpids.MPInumpy.MPIArray import MPIArray
from mpids.MPInumpy.errors import NotSupportedError

def matmul(a, b, out=None, comm=MPI.COMM_WORLD, dist='b'):
        if out is not None:
                raise NotSupportedError("'out' field not supported")

        if not isinstance(a, MPIArray) and not isinstance(b, MPIArray):
                return MPIArray(np_matmul(a, b), comm=comm, dist=dist)

        return MPIArray(np_matmul(a, b), comm=comm, dist=dist)
