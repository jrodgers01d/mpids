from mpi4py import MPI
from numpy import matmul as np_matmul
from petsc4py import PETSc

from mpids.MPInumpy.MPIArray import MPIArray
from mpids.MPInumpy.distributions import Distribution_Dict
from mpids.MPInumpy.errors import NotSupportedError
from mpids.MPInumpy.utils import get_comm_dims, get_cart_coords

__all__ = ['matmul']

def matmul(a, b, out=None, comm=MPI.COMM_WORLD, dist='b'):
    if out is not None:
        raise NotSupportedError("'out' field not supported")

    #Numpy only arrays
    if not isinstance(a, MPIArray) and not isinstance(b, MPIArray):
        return Distribution_Dict[dist](np_matmul(a, b), comm=comm)
    #Numpy and MPIArray
    if not isinstance(a, MPIArray) or not isinstance(b, MPIArray):
        return Distribution_Dict[dist](np_matmul(a, b), comm=comm)
    #Replicated MPIArrays
    if a.dist == b.dist == 'r':
        return Distribution_Dict[dist](np_matmul(a, b), comm=comm)

    return _block_mat_mult(a, b, comm=comm)


def _block_mat_mult(a, b, comm=MPI.COMM_WORLD):
    a_global_rows, a_global_cols = a.globalshape
    a_local_rows, a_local_cols = a.shape
    b_global_rows, b_global_cols = b.globalshape
    b_local_rows, b_local_cols = b.shape

    A = PETSc.Mat().create(comm=comm)
    A.setSizes((a_global_rows, a_global_cols))
    A.setFromOptions()
    A.setPreallocationNNZ((a_global_rows, a_global_cols))
    A_row_start, A_row_end = A.getOwnershipRange()

    A.setValues(range(A_row_start, A_row_end), range(a_global_cols), a)
    A.assemblyBegin()
    A.assemblyEnd()

    B = PETSc.Mat().create(comm=comm)
    B.setSizes((b_global_rows, b_global_cols))
    B.setFromOptions()
    B.setPreallocationNNZ((b_global_rows, b_global_cols))
    B_row_start, B_row_end = B.getOwnershipRange()

    B.setValues(range(B_row_start, B_row_end), range(b_global_cols), b)
    B.assemblyBegin()
    B.assemblyEnd()

    C = A.matMult(B)

    size = comm.Get_size()
    rank = comm.Get_rank()

    comm_dims = get_comm_dims(size, 'b')
    comm_coord = get_cart_coords(comm_dims, size, rank)

    return Distribution_Dict['b'](C.getValues(range(A_row_start, A_row_end),
                                              range(b_global_cols)),
                                              comm=comm,
                                              comm_dims=comm_dims,
                                              comm_coord=comm_coord)
