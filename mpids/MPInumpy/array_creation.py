from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.distributions import Distribution_Dict
from mpids.MPInumpy.utils import determine_local_data,            \
                                 determine_local_data_from_shape, \
                                 get_comm_dims,                   \
                                 get_cart_coords
from mpids.MPInumpy.mpi_utils import broadcast_shape

__all__ = ['array', 'empty', 'ones', 'zeros']

def array(array_data, dtype=None, copy=True, order=None, subok=False, ndmin=0,
          comm=MPI.COMM_WORLD, root=0, dist='b'):
    """ Create MPInumpyArray Object on all procs in comm.
        See docstring for mpids.MPInumpy.MPIArray

    Parameters
    ----------
    array_data : array_like
        Array like data to be distributed among processes.
    dtype : data-type, optional
        Desired data-type for the array.
    copy : bool, optional
        Default 'True' results in copied object, if 'False' copy
        only made when base class '__array__' returns a copy.
    order: {'K','A','C','F'}, optional
        Specified memory layout of the array.
    subok : bool, optional
        Default 'False' returned array will be forced to be
        base-class array, if 'True' then sub-classes will be
        passed-through.
    ndmin : int, optional
        Specifies the minimum number of dimensions that the
        resulting array should have.
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD
    dist : str, list, tuple
        Specified distribution of data among processes.
        Default value 'b' : Block, *
        Supported types:
            'b' : Block, *
            'u' : Undistributed

    Returns
    -------
    MPIArray : numpy.ndarray sub class
        Distributed among processes.
    """
    size = comm.Get_size()
    rank = comm.Get_rank()

    comm_dims = get_comm_dims(size, dist)
    comm_coord = get_cart_coords(comm_dims, size, rank)

    local_data, local_to_global = \
        determine_local_data(array_data, dist, comm_dims, comm_coord)

    np_local_data = np.array(local_data,
                             dtype=dtype,
                             copy=copy,
                             order=order,
                             subok=subok,
                             ndmin=ndmin)

    return Distribution_Dict[dist](np_local_data,
                                   comm=comm,
                                   comm_dims=comm_dims,
                                   comm_coord=comm_coord,
                                   local_to_global=local_to_global)


def empty(shape, dtype=np.float64, order='C',
          comm=MPI.COMM_WORLD, root=0, dist='b'):
    """ Create an empty MPInumpyArray Object, without initializing entries,
        on all procs in comm. See docstring for mpids.MPInumpy.MPIArray

    Parameters
    ----------
    shape : int, tuple of int
        Shape of empty array
    dtype : data-type, optional
        Desired data-type for the array. Default is np.float64
    order: {'C','F'}, optional
        Specified memory layout of the array.
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD
    dist : str, list, tuple
        Specified distribution of data among processes.
        Default value 'b' : Block, *
        Supported types:
            'b' : Block, *
            'u' : Undistributed

    Returns
    -------
    MPIArray : numpy.ndarray sub class
        Distributed among processes with unintialized values.
    """
    size = comm.Get_size()
    rank = comm.Get_rank()

    comm_dims = get_comm_dims(size, dist)
    comm_coord = get_cart_coords(comm_dims, size, rank)

    array_shape = shape if rank == root else None
    array_shape = broadcast_shape(array_shape, comm=comm, root=root)

    local_shape, local_to_global  = \
        determine_local_data_from_shape(array_shape, dist, comm_dims, comm_coord)

    np_local_data = np.empty(local_shape, dtype=dtype, order=order)

    return Distribution_Dict[dist](np_local_data,
                                   comm=comm,
                                   comm_dims=comm_dims,
                                   comm_coord=comm_coord,
                                   local_to_global=local_to_global)


def ones(shape, dtype=np.float64, order='C',
         comm=MPI.COMM_WORLD, root=0, dist='b'):
    """ Create an MPInumpyArray Object with entries filled with ones
        on all procs in comm. See docstring for mpids.MPInumpy.MPIArray

    Parameters
    ----------
    shape : int, tuple of int
        Shape of array
    dtype : data-type, optional
        Desired data-type for the array. Default is np.float64
    order: {'C','F'}, optional
        Specified memory layout of the array.
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD
    dist : str, list, tuple
        Specified distribution of data among processes.
        Default value 'b' : Block, *
        Supported types:
            'b' : Block, *
            'u' : Undistributed

    Returns
    -------
    MPIArray : numpy.ndarray sub class
        Distributed among processes with values all equal to one.
    """
    size = comm.Get_size()
    rank = comm.Get_rank()

    comm_dims = get_comm_dims(size, dist)
    comm_coord = get_cart_coords(comm_dims, size, rank)

    array_shape = shape if rank == root else None
    array_shape = broadcast_shape(array_shape, comm=comm, root=root)

    local_shape, local_to_global  = \
        determine_local_data_from_shape(array_shape, dist, comm_dims, comm_coord)

    np_local_data = np.ones(local_shape, dtype=dtype, order=order)

    return Distribution_Dict[dist](np_local_data,
                                   comm=comm,
                                   comm_dims=comm_dims,
                                   comm_coord=comm_coord,
                                   local_to_global=local_to_global)


def zeros(shape, dtype=np.float64, order='C',
          comm=MPI.COMM_WORLD, root=0, dist='b'):
    """ Create an MPInumpyArray Object with entries filled with zeros
        on all procs in comm. See docstring for mpids.MPInumpy.MPIArray

    Parameters
    ----------
    shape : int, tuple of int
        Shape of array
    dtype : data-type, optional
        Desired data-type for the array. Default is np.float64
    order: {'C','F'}, optional
        Specified memory layout of the array.
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD
    dist : str, list, tuple
        Specified distribution of data among processes.
        Default value 'b' : Block, *
        Supported types:
            'b' : Block, *
            'u' : Undistributed

    Returns
    -------
    MPIArray : numpy.ndarray sub class
        Distributed among processes with values all equal to zero.
    """
    size = comm.Get_size()
    rank = comm.Get_rank()

    comm_dims = get_comm_dims(size, dist)
    comm_coord = get_cart_coords(comm_dims, size, rank)

    array_shape = shape if rank == root else None
    array_shape = broadcast_shape(array_shape, comm=comm, root=root)

    local_shape, local_to_global  = \
        determine_local_data_from_shape(array_shape, dist, comm_dims, comm_coord)

    np_local_data = np.zeros(local_shape, dtype=dtype, order=order)

    return Distribution_Dict[dist](np_local_data,
                                   comm=comm,
                                   comm_dims=comm_dims,
                                   comm_coord=comm_coord,
                                   local_to_global=local_to_global)
