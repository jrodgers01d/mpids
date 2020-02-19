from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.distributions import Distribution_Dict
from mpids.MPInumpy.errors import TypeError, ValueError
from mpids.MPInumpy.utils import distribute_array, distribute_shape

__all__ = ['arange', 'array', 'empty', 'ones', 'zeros']

def arange(*args, dtype=None, comm=MPI.COMM_WORLD, root=0, dist='b'):
    """ Create an empty MPInumpyArray Object, without initializing entries,
        on all procs in comm. See docstring for mpids.MPInumpy.MPIArray

    Parameters
    ----------
    start : int, optional
        Start of interval
    stop : int
        End of interval
    step : int, optional
        Spacing between values. Default step size is 1.
    dtype : data-type, optional
        Desired data-type for the array. Default is None
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD
    root : int, optional
        Rank of root process that has the global start, stop, step information.
        If none specified defaults to 0.
    dist : str
        Specified distribution of data among processes.
        Default value 'b' : Block
        Supported types:
            'b' : Block
            'u' : Undistributed

    Returns
    -------
    MPIArray : numpy.ndarray sub class
        Distributed array of evenly spaced arguments among processes.
    """
#TODO: There's got to be a more elegant way to do this
    array_data = np.arange(*args, dtype=dtype) if comm.rank == root else None

    local_data, comm_dims, comm_coord, local_to_global = \
        distribute_array(array_data, dist, comm=comm, root=root)

    np_local_data = np.array(local_data, dtype=dtype)

    return Distribution_Dict[dist](np_local_data,
                                   comm=comm,
                                   comm_dims=comm_dims,
                                   comm_coord=comm_coord,
                                   local_to_global=local_to_global)


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
    root : int, optional
        Rank of root process that has the global array data. If none specified
        defaults to 0.
    dist : str
        Specified distribution of data among processes.
        Default value 'b' : Block
        Supported types:
            'b' : Block
            'u' : Undistributed

    Returns
    -------
    MPIArray : numpy.ndarray sub class
        Distributed among processes.
    """
    local_data, comm_dims, comm_coord, local_to_global = \
        distribute_array(array_data, dist, comm=comm, root=root)

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


def empty(*args, dtype=np.float64, order='C',
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
    root : int, optional
        Rank of root process that has the global shape data. If none specified
        defaults to 0.
    dist : str
        Specified distribution of data among processes.
        Default value 'b' : Block
        Supported types:
            'b' : Block
            'u' : Undistributed

    Returns
    -------
    MPIArray : numpy.ndarray sub class
        Distributed among processes with unintialized values.
    """
    shape = _validate_shape(*args)
    local_shape, comm_dims, comm_coord, local_to_global = \
        distribute_shape(shape, dist, comm=comm, root=root)

    np_local_data = np.empty(local_shape, dtype=dtype, order=order)

    return Distribution_Dict[dist](np_local_data,
                                   comm=comm,
                                   comm_dims=comm_dims,
                                   comm_coord=comm_coord,
                                   local_to_global=local_to_global)


def ones(*args, dtype=np.float64, order='C',
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
    root : int, optional
        Rank of root process that has the global shape data. If none specified
        defaults to 0.
    dist : str
        Specified distribution of data among processes.
        Default value 'b' : Block
        Supported types:
            'b' : Block
            'u' : Undistributed

    Returns
    -------
    MPIArray : numpy.ndarray sub class
        Distributed among processes with values all equal to one.
    """
    shape = _validate_shape(*args)
    local_shape, comm_dims, comm_coord, local_to_global = \
        distribute_shape(shape, dist, comm=comm, root=root)

    np_local_data = np.ones(local_shape, dtype=dtype, order=order)

    return Distribution_Dict[dist](np_local_data,
                                   comm=comm,
                                   comm_dims=comm_dims,
                                   comm_coord=comm_coord,
                                   local_to_global=local_to_global)


def zeros(*args, dtype=np.float64, order='C',
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
    root : int, optional
        Rank of root process that has the global shape data. If none specified
        defaults to 0.
    dist : str
        Specified distribution of data among processes.
        Default value 'b' : Block
        Supported types:
            'b' : Block
            'u' : Undistributed

    Returns
    -------
    MPIArray : numpy.ndarray sub class
        Distributed among processes with values all equal to zero.
    """
    shape = _validate_shape(*args)
    local_shape, comm_dims, comm_coord, local_to_global = \
        distribute_shape(shape, dist, comm=comm, root=root)

    np_local_data = np.zeros(local_shape, dtype=dtype, order=order)

    return Distribution_Dict[dist](np_local_data,
                                   comm=comm,
                                   comm_dims=comm_dims,
                                   comm_coord=comm_coord,
                                   local_to_global=local_to_global)


def _validate_shape(*args):
    """ Helper method for shape based array creation routines.
        Verifies user specified shape is either int or tuple of ints.
        Additional work in case of in is to format it for upcoming broadcast.
    """
    if len(args) != 1:
        raise TypeError('only positional argument should be shape.')
    shape = args[0]
    #Special case for non-root ranks/processes
    if shape is None:
        return shape
    if isinstance(shape, int):
        return (shape,)
    if isinstance(shape, tuple):
        if all([isinstance(dim, int) for dim in shape]):
            return shape

    raise ValueError('shape must be int or tuple of ints.')
