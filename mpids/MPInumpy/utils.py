from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.errors import IndexError, InvalidDistributionError

__all__ = ['determine_local_data', 'get_block_index', 'get_cart_coords',
           'get_comm_dims', 'global_to_local_key', 'distribution_to_dimensions',
           'is_undistributed', 'is_row_block_distributed',
           'is_column_block_distributed', 'is_block_block_distributed']

def determine_local_data(array_data, dist, comm_dims, comm_coord):
        """ Determine array like data to be distributed among processes

        Parameters
        ----------
        array_data : array_like
                Array like data to be distributed among processes.
        dist : str, list, tuple
                Specified distribution of data among processes.
                Default value 'b' : Block, *
                Supported types:
                    'b' : Block, *
                    ('*', 'b') : *, Block
                    ('b','b') : Block-Block
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
                        key, value = axis, (inclusive start, exclusive end)
                        {0: [start_index, end_index),
                         1: [start_index, end_index),
                         ...}
        """
        if is_undistributed(dist):
                return array_data, None

        local_to_global = {}
        if len(comm_dims) == 1:
                for axis in range(len(np.shape(array_data))):
                        if axis == 0:
                                row_start, row_end = \
                                            get_block_index(len(array_data),
                                                            comm_dims[0],
                                                            comm_coord[0])
                                local_to_global[axis] = (row_start, row_end)
                        else:
                                local_to_global[axis] = (0, len(array_data[0]))

                return array_data[slice(row_start, row_end)], local_to_global

        try:
                for axis in range(len(comm_dims)):
                        if axis == 0:
                                row_start, row_end = \
                                        get_block_index(len(array_data),
                                                        comm_dims[axis],
                                                        comm_coord[axis])
                                local_to_global[axis] = (row_start, row_end)
                        else:
                                col_start, col_end =  \
                                        get_block_index(len(array_data[0]),
                                                        comm_dims[axis],
                                                        comm_coord[axis])
                                local_to_global[axis] = (col_start, col_end)
#TODO: Find more elegant solution than try catch
        except TypeError: # Case when dim of specified dist != dim input array
                raise InvalidDistributionError(
                        'Invalid distribution encountered: {}'.format(dist))

        return [array_data[row][slice(col_start, col_end)] \
                        for row in range(row_start, row_end)], local_to_global


def distribution_to_dimensions(distribution, procs):
        """ Convert specified distribution to cartesian dimensions

        Parameters
        ----------
        distribution : str, list, tuple
                Specified distribution of data among processes.
                Default value 'b' : Block, *
                Supported types:
                    'b' : Block, *
                    ('*', 'b') : *, Block
                    ('b','b') : Block-Block
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
        if is_column_block_distributed(distribution):
                return [1, procs]
        if is_block_block_distributed(distribution):
                return len(distribution)

        raise InvalidDistributionError(
                'Invalid distribution encountered: {}'.format(distribution))


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
                start_index = axis_rem * (axis_num + 1) + \
                              (axis_coord - axis_rem) * axis_num
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
        coordinates : list
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
                    ('*', 'b') : *, Block
                    ('b','b') : Block-Block
                    'u' : Undistributed

        Returns
        -------
        comm_dims : list
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
        if local_to_global_dict is None: #Undistributed Case
                return global_key

        if isinstance(global_key, int):
                local_key = _global_to_local_key_int(global_key,
                                                     globalshape,
                                                     local_to_global_dict)
        if isinstance(global_key, slice):
                local_key = _global_to_local_key_slice(global_key,
                                                       globalshape,
                                                       local_to_global_dict)
        if isinstance(global_key, tuple):
                local_key = _global_to_local_key_tuple(global_key,
                                                       globalshape,
                                                       local_to_global_dict)
        return local_key


def _global_to_local_key_int(global_key, globalshape,
                             local_to_global_dict, axis=0):
        """ Helper method to process int keys """
        global_min, global_max = local_to_global_dict[axis]
        # Handle negative/reverse access case
        if global_key < 0:
                global_key += globalshape[axis]
        if global_key < 0 or global_key >= globalshape[axis]:
                raise IndexError(' index {}'.format(global_key) +
                                 ' is out of bounds for axis 0 with ' +
                                 ' global shape {}'.format(globalshape[axis]))
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

        global_start, global_stop, global_step = \
                global_key.indices(globalshape[axis])
        global_min, global_max = local_to_global_dict[axis]

        #Condition slice start/stop
        global_start = \
                global_start if global_start is not None else 0
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
                raise IndexError(' too many indices for array with'  +
                                 ' global shape {}'.format(globalshape))

        local_key = []
        axis = 0
        for dim_key in global_key:
                if isinstance(dim_key, int):
                        local_key.append(
                                _global_to_local_key_int(dim_key,
                                                         globalshape,
                                                         local_to_global_dict,
                                                         axis))
                if isinstance(dim_key, slice):
                        local_key.append(
                                _global_to_local_key_slice(dim_key,
                                                           globalshape,
                                                           local_to_global_dict,
                                                           axis))
                axis += 1

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
                    ('*', 'b') : *, Block
                    ('b','b') : Block-Block
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
                    ('*', 'b') : *, Block
                    ('b','b') : Block-Block
                    'u' : Undistributed

        Returns
        -------
        result : boolean
        """
        return distribution[0] == 'b' and len(distribution) == 1


def is_column_block_distributed(distribution):
        """ Check if distribution is of type column block

        Parameters
        ----------
        distribution : str, list, tuple
                Specified distribution of data among processes.
                Default value 'b' : Block, *
                Supported types:
                    'b' : Block, *
                    ('*', 'b') : *, Block
                    ('b','b') : Block-Block
                    'u' : Undistributed

        Returns
        -------
        result : boolean
        """
        if len(distribution) != 2:
                return False
        return distribution[0] == '*' and distribution[1] == 'b'


def is_block_block_distributed(distribution):
        """ Check if distribution is of type block-block

        Parameters
        ----------
        distribution : str, list, tuple
                Specified distribution of data among processes.
                Default value 'b' : Block, *
                Supported types:
                    'b' : Block, *
                    ('*', 'b') : *, Block
                    ('b','b') : Block-Block
                    'u' : Undistributed

        Returns
        -------
        result : boolean
        """
        if len(distribution) != 2:
                return False
        return distribution[0] == distribution[1] == 'b'
