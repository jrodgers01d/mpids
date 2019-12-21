from mpi4py import MPI
from mpids.MPInumpy.errors import InvalidDistributionError

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
        """
        if is_undistributed(dist):
                return array_data

        if len(comm_dims) == 1:
                start, end = get_block_index(len(array_data), comm_dims[0], comm_coord[0])
                return array_data[slice(start, end)]

        try:
                for axis in range(len(comm_dims)):
                        if axis == 0:
                                row_start, row_end = \
                                        get_block_index(len(array_data),
                                                        comm_dims[axis],
                                                        comm_coord[axis])
                        else:
                                col_start, col_end =  \
                                        get_block_index(len(array_data[0]),
                                                        comm_dims[axis],
                                                        comm_coord[axis])
#TODO: Find more elegant solution than try catch
        except TypeError: # Case when dim of specified dist != dim input array
                raise InvalidDistributionError(
                        'Invalid distribution encountered: {}'.format(dist))

        return [array_data[row][slice(col_start, col_end)] \
                        for row in range(row_start, row_end)]


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
        if distribution[0] != 'b':
                return False
        return len(distribution) == 1 or distribution[1] == '*'


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
