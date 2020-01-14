from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.errors import IndexError, InvalidDistributionError, \
                                  NotSupportedError

from mpids.MPInumpy.mpi_utils import all_gather_v,                \
                                     all_to_all,                  \
                                     broadcast_array,             \
                                     broadcast_shape,             \
                                     get_comm_size, get_rank,     \
                                     scatter_v

__all__ = ['determine_local_shape_and_mapping',
           'determine_redistribution_counts_from_shape',
           'determine_global_offset', 'distribute_array', 'distribute_shape',
           'get_block_index', 'get_cart_coords', 'get_comm_dims',
           'global_to_local_key', 'distribution_to_dimensions',
           'is_undistributed', 'is_row_block_distributed',
           'slice_local_data_and_determine_mapping']


def determine_local_shape_and_mapping(array_shape, dist, comm_dims, comm_coord):
    """ Determine expected distributed local shape and global mapping based on
        passed distribution and global array shape.

    Parameters
    ----------
    array_shape : int, tuple of int
        Shape of data to distribute.
    dist : str, list, tuple
        Specified distribution of data among processes.
        Default value 'b' : Block, *
        Supported types:
            'b' : Block, *
            'u' : Undistributed
    comm_dims : list, None
        Dimensions of cartesian grid
    coordinates : list, None
        Coordinates of rank in grid

    Returns
    -------
    local_shape : tuple
        Local shape determined for process(rank)
    local_to_global : dictionary
        Dictionary specifying global index start/end of data by axis.
        Format:
            key, value = axis, [inclusive start, exclusive end)
            {0: (start_index, end_index),
             1: (start_index, end_index),
             ...}
    """
    if is_undistributed(dist):
        local_to_global = None
        return array_shape, local_to_global

    local_to_global = {}
    local_shape = []
    for axis, axis_length in enumerate(array_shape):
        if axis == 0:
            row_start, row_end = get_block_index(axis_length,
                                                 comm_dims[0],
                                                 comm_coord[0])
            local_to_global[axis] = (int(row_start), int(row_end))
            local_shape.append(row_end - row_start)
        else:
            local_to_global[axis] = (0, int(axis_length))
            local_shape.append(axis_length)

    return tuple(local_shape), local_to_global


def determine_global_offset(index, global_shape):
    """ Determine global offset of specified index based on shape of global
    array.  The result is conceptually equivalent to the offset(in data items)
    of the element at the specified index if it was one contiguous vector of
    elements.

    Parameters
    ----------
    index : list of ints
        Global indices.
    global_shape: list, tuple
        Shape of distributed array.

    Returns
    -------
    global_offset: int
        Offset in number of elements from start of global array.
    """
    if not isinstance(index, list): raise TypeError("index must be a list")
    if len(index) != len(global_shape):
        raise ValueError(
            "number of elements in index and global shape must be equal")

    if len(global_shape) == 1:
        global_offset = index[0]
    else:
        strides = [1]
        for i, stride in enumerate(global_shape[1:]):
            strides.append(strides[i] * stride)

        index.reverse()
        global_offset = 0
        for dim, num in enumerate(index):
            global_offset += num * strides[dim]

    return int(global_offset)


def determine_redistribution_counts_from_shape(current_shape, desired_shape,
                                               dist, comm=MPI.COMM_WORLD):
    """ Determine information required to redistribute distributed array.

    Parameters
    ----------
    current_shape : int, tuple of int
        Current global shape of distributed array.
    desired_shape : int, tuple of int
        Global shape array data should be mapped to.
#TODO: Rething the need for dist, it's practically irrevelant here
    dist : str, list, tuple
        Specified distribution of data among processes.
        Default value 'b' : Block, *
        Supported types:
            'b' : Block, *
            'u' : Undistributed
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD

    Returns
    -------
    send_counts : numpy.ndarray
        Constructed local numpy array of length communicator size that lists how
        much data it will be sending to a given process.  The position in the
        array is the equivalent rank of the receiving process.
    recv_counts : numpy.ndarray
        Constructed local numpy array of length communicator size that lists how
        much data it will be receiving for a given process.  The position in the
        array is the equivalent rank of the sending process.
    """
    size = comm.Get_size()
    rank = comm.Get_rank()
    current_ndim = len(current_shape)
    current_leading_dim = current_shape[0]
    #Zeroing rest of indices to determine global offset of current partition
    current_index_pad = [0] * (current_ndim - 1)

    current_partition_start, current_partition_end = \
        get_block_index(current_leading_dim, size, rank)
    current_index = [current_partition_start] + current_index_pad
    current_offset = determine_global_offset(current_index, current_shape)

    desired_leading_dim = desired_shape[0]
    #Distribute work of determining desired partitioning
    desired_partioning = np.zeros(2, dtype=np.int32)
    desired_partioning[0], desired_partioning[1] = \
        get_block_index(desired_leading_dim, size, rank)
    #Collect global start/stops of rank based partitions
    global_desired_partioning = all_gather_v(desired_partioning,
                                             shape=(size, 2),
                                             comm=comm)

    current_over_paritioning = current_partition_end - current_partition_start
    current_remaining_dim = np.prod(current_shape[1:]) * current_over_paritioning
    desired_remaining_dim = np.prod(desired_shape[1:])
    send_counts = np.zeros(size, dtype=np.int32)
    for offset in range(current_offset, current_offset + current_remaining_dim):
        for global_rank in range(size):
            partition_start = global_desired_partioning[global_rank][0]
            partition_stop  = global_desired_partioning[global_rank][1]

            if (offset >= partition_start * desired_remaining_dim and
                offset <  partition_stop * desired_remaining_dim):
                send_counts[global_rank] += 1

    #Use all to all to distribute what's being sent
    recv_counts = all_to_all(send_counts, comm=comm)

    return send_counts, recv_counts


def distribute_array(array_data, dist, comm=MPI.COMM_WORLD, root=0):
    """ Distribute global array like object among MPI processes base on
    specified distribution.

    Parameters
    ----------
    array_data : array_like
        Array like data to be distributed among processes.
    dist : str, list, tuple
        Specified distribution of data among processes.
        Default value 'b' : Block, *
        Supported types:
            'b' : Block, *
            'u' : Undistributed
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD
    root : int, optional
        Rank of root process that has the local array data. If none specified
        defaults to 0.

    Returns
    -------
    local_shape : tuple
        Local shape determined for process(rank)
    comm_dims : list, None
        Dimensions of cartesian grid
    coordinates : list, None
        Coordinates of rank in grid
    local_to_global : dictionary
        Dictionary specifying global index start/end of data by axis.
        Format:
            key, value = axis, [inclusive start, exclusive end)
            {0: (start_index, end_index),
             1: (start_index, end_index),
             ...}
    """
    if is_undistributed(dist):
        local_data = broadcast_array(np.asarray(array_data),
                                     comm=comm,
                                     root=root)
        comm_dims = None
        comm_coord = None
        local_to_global = None
    else:
        size = comm.Get_size()
        rank = comm.Get_rank()

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

    return local_data, comm_dims, comm_coord, local_to_global


def distribute_shape(shape, dist, comm=MPI.COMM_WORLD, root=0):
    """ Distribute global array shape among MPI processes base on specified
    distribution.

    Parameters
    ----------
    array_shape : int, tuple of int
        Shape of data to distribute.
    dist : str, list, tuple
        Specified distribution of data among processes.
        Default value 'b' : Block, *
        Supported types:
            'b' : Block, *
            'u' : Undistributed
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD
    root : int, optional
        Rank of root process that has the local shape data. If none specified
        defaults to 0.

    Returns
    -------
    local_shape : tuple
        Local shape determined for process(rank)
    comm_dims : list, None
        Dimensions of cartesian grid
    coordinates : list, None
        Coordinates of rank in grid
    local_to_global : dictionary
        Dictionary specifying global index start/end of data by axis.
        Format:
            key, value = axis, [inclusive start, exclusive end)
            {0: (start_index, end_index),
             1: (start_index, end_index),
             ...}
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

    return local_shape, comm_dims, comm_coord, local_to_global


def distribution_to_dimensions(distribution, procs):
    """ Convert specified distribution to cartesian dimensions

    Parameters
    ----------
    distribution : str, list, tuple
        Specified distribution of data among processes.
        Default value 'b' : Block, *
        Supported types:
            'b' : Block, *
            'u' : Undistributed
    procs: int
        Size/number of processes in communicator

    Returns
    -------
    dimensions : int, list
        Seed for determinging processes per cartesian coordinate
        direction.
    """
    if is_row_block_distributed(distribution):
        return 1
    raise InvalidDistributionError(
        'Invalid distribution encountered: {}'.format(distribution))


def _format_indexed_result(global_key, indexed_result):
    """ Helper method to format __getitem__ index based result
        distribution as a MPIArray.

        Ensures results have:
        - shapes:
            In the case of scalar values
        - correctly populated shapes(value and length):
            In the case of empty slices

        Parameters
        ----------
        global_key : int, slice, tuple
            Selection indices, i.e. keys to object access dunder methods
            __getitem__, __setitem__, ...
        indexed_result : numpy.ndarray
            Result of calling __getitem__ with global_key

        Returns
        -------
        formatted_indexed_result : numpy.ndarray
            Original array with properties necessary for distribution
    """
    #Avoid empty tuples for shape
    if indexed_result.ndim == 0:
        indexed_result = np.array([indexed_result])

    #Adjust shape for processes with nothing sliced
    if indexed_result.size == 0:
        if isinstance(global_key, int):
            indexed_result = indexed_result.reshape(0)
        if isinstance(global_key, slice):
            indexed_result = \
                indexed_result.reshape([0] * len(indexed_result.shape))
        if isinstance(global_key, tuple):
            if all(isinstance(dim_key, int) for dim_key in global_key):
                indexed_result = indexed_result.reshape(0)
            else:
                indexed_result = \
                    indexed_result.reshape([0] * len(indexed_result.shape))
    return indexed_result


def get_block_index(axis_len, axis_size, axis_coord):
    """ Get start/end array index range along axis for data block.

    Parameters
    ----------
    axis_len : int
        Length of array data along axis.
    axis_size : int
        Number of processes along axis.
    axis_coord : int
        Cartesian coorindate along axis for local process.

    Returns
    -------
    [start_index, end_index) : tuple
        Index range along axis for data block.
    """
    axis_num = axis_len // axis_size
    axis_rem = axis_len % axis_size

    if axis_coord < axis_rem:
        local_len = axis_num + 1
        start_index = axis_coord * local_len
    else:
        local_len = axis_num
        start_index = \
            axis_rem * (axis_num + 1) + (axis_coord - axis_rem) * axis_num
    end_index = start_index + local_len

    return (start_index, end_index)


def get_cart_coords(comm_dims, procs, rank):
    """ Get coordinates of process placed on cartesian grid.
        Implementation based on OpenMPI.mca.topo.topo_base_cart_coords

    Parameters
    ----------
    comm_dims : list
        Division of processes in cartesian grid
    procs: int
        Size/number of processes in communicator
    rank : int
        Process rank in communicator

    Returns
    -------
    coordinates : list, None
        Coordinates of rank in grid
    """
    if comm_dims == None:
        return None

    coordinates = []
    rem_procs = procs

    for dim in comm_dims:
        rem_procs = rem_procs // dim
        coordinates.append(rank // rem_procs)
        rank = rank % rem_procs

    return coordinates


def get_comm_dims(procs, dist):
    """ Get dimensions of cartesian grid as determined by specified
        distribution.

    Parameters
    ----------
    procs: int
        Size/number of processes in communicator
    dist : str, list, tuple
        Specified distribution of data among processes.
        Default value 'b' : Block, *
        Supported types:
            'b' : Block, *
            'u' : Undistributed

    Returns
    -------
    comm_dims : list, None
        Dimensions of cartesian grid
    """
    if is_undistributed(dist):
        return None
    return MPI.Compute_dims(procs, distribution_to_dimensions(dist, procs))


def global_to_local_key(global_key, globalshape, local_to_global_dict):
    """ Determine array like data to be distributed among processes
        Convert global slice/index key to process local key

    Parameters
    ----------
    global_key : int, slice, tuple
        Selection indices, i.e. keys to object access dunder methods
        __getitem__, __setitem__, ...
    globalshape : list, tuple
        Combined shape of distributed array.
    local_to_global_dict : dictionary
        Dictionary specifying global index start/end of data by axis.
        Format:
            key, value = axis, (inclusive start, exclusive end)
            {0: [start_index, end_index),
             1: [start_index, end_index),
             ...}

    Returns
    -------
    local_key : int, slice, tuple
        Selection indices present in locally distributed array.
    """
    __global_to_local_key_map = {int   : _global_to_local_key_int,
                                 slice : _global_to_local_key_slice,
                                 tuple : _global_to_local_key_tuple}

    def __unsupported_key(*args):
        raise NotSupportedError('index/slice key ' +
                                '{} '.format(global_key) +
                                'is not supported')

    local_key = __global_to_local_key_map\
        .get(type(global_key), __unsupported_key)(global_key,
                                                  globalshape,
                                                  local_to_global_dict)
    return local_key


def _global_to_local_key_int(global_key, globalshape,
                             local_to_global_dict, axis=0):
    """ Helper method to process int keys """
    # Handle negative/reverse access case
    if global_key < 0:
        global_key += globalshape[axis]
    if global_key < 0 or global_key >= globalshape[axis]:
        raise IndexError('index {}'.format(global_key) +
                         ' is out of bounds for axis 0 with' +
                         ' global shape {}'.format(globalshape[axis]))
    if local_to_global_dict is None:
        return global_key

    global_min, global_max = local_to_global_dict[axis]
    if global_key >= global_min and global_key < global_max:
        local_key = global_key - global_min
    else: #Don't slice/access
        local_key = slice(0, 0)

    return local_key


def _global_to_local_key_slice(global_key, globalshape,
                               local_to_global_dict, axis=0):
    """ Helper method to process slice keys """
    if global_key == slice(None):
        return global_key
    if local_to_global_dict is None:
        return global_key

    global_start, global_stop, global_step = \
        global_key.indices(globalshape[axis])
    global_min, global_max = local_to_global_dict[axis]

    #Condition slice start/stop
    global_start = global_start if global_start is not None else 0
    global_stop = \
        global_stop if global_stop is not None else globalshape[axis]
    #Bias start/stop by local min/max
    local_start = global_start - global_min
    local_stop = global_stop - global_min

    local_key = slice(local_start, local_stop, global_step)

    return local_key


def _global_to_local_key_tuple(global_key, globalshape, local_to_global_dict):
    """ Helper method to process tuple of int or slice keys """
    if len(global_key) > len(globalshape):
        raise IndexError('too many indices for array with'  +
                         ' global shape {}'.format(globalshape))

    local_key = []
    for axis, dim_key in enumerate(global_key):
        if isinstance(dim_key, int):
            local_key.append(_global_to_local_key_int(dim_key,
                                                      globalshape,
                                                      local_to_global_dict,
                                                      axis))
        if isinstance(dim_key, slice):
            local_key.append(_global_to_local_key_slice(dim_key,
                                                        globalshape,
                                                        local_to_global_dict,
                                                        axis))

    return tuple(local_key)


def is_undistributed(distribution):
    """ Check if distribution is of type undistributed

    Parameters
    ----------
    distribution : str, list, tuple
        Specified distribution of data among processes.
        Default value 'b' : Block, *
        Supported types:
            'b' : Block, *
            'u' : Undistributed

    Returns
    -------
    result : boolean
    """
    return distribution == 'u'


def is_row_block_distributed(distribution):
    """ Check if distribution is of type row block

    Parameters
    ----------
    distribution : str, list, tuple
        Specified distribution of data among processes.
        Default value 'b' : Block, *
        Supported types:
            'b' : Block, *
            'u' : Undistributed

    Returns
    -------
    result : boolean
    """
    return distribution[0] == 'b' and len(distribution) == 1

#NOTE: Legacy method, good candidate for removal
def slice_local_data_and_determine_mapping(array_data, dist, comm_dims, comm_coord):
    """ Slice array like data to be distributed among processes and determine
        its local to global mapping

    Parameters
    ----------
    array_data : array_like
        Array like data to be distributed among processes.
    dist : str, list, tuple
        Specified distribution of data among processes.
        Default value 'b' : Block, *
        Supported types:
            'b' : Block, *
            'u' : Undistributed
    comm_dims : list
        Division of processes in cartesian grid
    comm_coord : list
        Coordinates of rank in grid

    Returns
    -------
    local_data : array_like
        Array data which is responsibility of process(rank).
    local_to_global : dictionary
        Dictionary specifying global index start/end of data by axis.
        Format:
            key, value = axis, [inclusive start, exclusive end)
            {0: (start_index, end_index),
             1: (start_index, end_index),
             ...}
    """
    if is_undistributed(dist):
        local_to_global = None
        return array_data, local_to_global

    local_to_global = {}
    for axis, axis_length in enumerate(np.shape(array_data)):
        if axis == 0:
            row_start, row_end = get_block_index(axis_length,
                                                 comm_dims[0],
                                                 comm_coord[0])
            local_to_global[axis] = (int(row_start), int(row_end))
        else:
            local_to_global[axis] = (0, int(axis_length))

    return array_data[slice(row_start, row_end)], local_to_global
