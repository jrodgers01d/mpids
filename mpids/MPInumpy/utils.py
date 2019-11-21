from mpi4py import MPI
from mpids.MPInumpy.errors import InvalidDistributionError

#TODO: Potentially move MPI.Compute_dims to parameter, treating it as service
def determine_local_data(array_data, dist, procs, rank):
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
        procs : int
                Number of processes in communicator.
        rank : int
                Process rank(index) in communicator.

        Returns
        -------
        local_data : array_like
                Array data which is responsibility of process(rank).
        """
        if dist[0] == 'u':
                return array_data

        dims = MPI.Compute_dims(procs, distribution_to_dimensions(dist, procs))
        coord = get_cart_coords(dims, procs, rank)

        if len(dims) == 1:
                start, end = get_block_index(len(array_data), dims[0], coord[0])
                return array_data[slice(start, end)]

        try:
                for axis in range(len(dims)):
                        if axis == 0:
                                row_start, row_end = \
                                        get_block_index(len(array_data),
                                                        dims[axis],
                                                        coord[axis])
                        else:
                                col_start, col_end =  \
                                        get_block_index(len(array_data[0]),
                                                        dims[axis],
                                                        coord[axis])
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


def get_cart_coords(dims, procs, rank):
        """ Get coordinates of process placed on cartesian grid.
            Implementation based on OpenMPI.mca.topo.topo_base_cart_coords

        Parameters
        ----------
        dims : list
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
        coordinates = []
        rem_procs = procs

        for dim in dims:
                rem_procs = rem_procs // dim
                coordinates.append(rank // rem_procs)
                rank = rank % rem_procs

        return coordinates


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
        # Row-block
        if len(distribution) == 1 or distribution[1] == '*':
                return 1

        # Two Dim
        if len(distribution) == 2:
                # block-block
                if distribution[0] == distribution[1] == 'b':
                        return len(distribution)

                # column-block
                if distribution[0] == '*' and distribution[1] == 'b':
                        return [1, procs]

        raise InvalidDistributionError(
                'Invalid distribution encountered: {}'.format(distribution))
