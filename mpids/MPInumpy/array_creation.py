from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.distributions import Distribution_Dict
from mpids.MPInumpy.utils import determine_local_shape_and_mapping, \
                                 get_comm_dims,                   \
                                 get_cart_coords,                 \
                                 is_undistributed
from mpids.MPInumpy.mpi_utils import all_gather_v,                \
                                     broadcast_array,             \
                                     broadcast_shape,             \
                                     scatter_v                    \

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

    if is_undistributed(dist):
        local_data = broadcast_array(np.asarray(array_data),
                                     comm=comm,
                                     root=root)
        comm_dims = None
        comm_coord = None
        local_to_global = None
    else:
        comm_dims = get_comm_dims(size, dist)
        comm_coord = get_cart_coords(comm_dims, size, rank)
        array_shape = np.shape(array_data) if rank == root else None
        array_shape = broadcast_shape(array_shape, comm=comm, root=root)

        local_shape, local_to_global = \
            determine_local_shape_and_mapping(array_shape,
                                              dist,
                                              comm_dims,
                                              comm_coord)
        shapes = all_gather_v(np.asarray(local_shape),
                              shape=(size, len(local_shape)),
                              comm=comm)
        # Creation and conditioning of displacements list
        displacements = np.roll(np.cumsum(np.prod(shapes, axis=1)),1)
        displacements[0] = 0

        local_data = scatter_v(np.asarray(array_data),
                               displacements,
                               shapes,
                               comm=comm,
                               root=root)

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
        determine_local_shape_and_mapping(array_shape,
                                          dist,
                                          comm_dims,
                                          comm_coord)

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
        determine_local_shape_and_mapping(array_shape,
                                          dist,
                                          comm_dims,
                                          comm_coord)

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
        determine_local_shape_and_mapping(array_shape,
                                          dist,
                                          comm_dims,
                                          comm_coord)

    np_local_data = np.zeros(local_shape, dtype=dtype, order=order)

    return Distribution_Dict[dist](np_local_data,
                                   comm=comm,
                                   comm_dims=comm_dims,
                                   comm_coord=comm_coord,
                                   local_to_global=local_to_global)
