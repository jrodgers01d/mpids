from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.errors import TypeError

__all__ = ['all_gather_v', 'all_to_all', 'all_to_all_v', 'broadcast_array',
           'broadcast_shape', 'get_comm', 'get_comm_size',
           'get_rank', 'scatter_v']

def all_gather_v(array_data, shape=None, comm=MPI.COMM_WORLD):
    """ Gather distributed array data to all processes

    Parameters
    ----------
    array_data : numpy.ndarray
        Numpy array data distributed among processes.
    shape : int, tuple of int, None
        Final desired shape of gathered array data
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD

    Returns
    -------
    gathered_array : numpy.ndarray
        Collected numpy array from all process in MPI Comm.
    """
    if not isinstance(array_data, np.ndarray):
        raise TypeError('invalid data type for all_gather_v.')

    comm_size = comm.Get_size()
    local_displacement = np.empty(1, dtype=np.int32)
    displacements = np.empty(comm_size, dtype=np.int32)
    local_count = np.asarray(array_data.size, dtype=np.int32)
    counts = np.empty(comm_size, dtype=np.int32)
    total_count = np.empty(1, dtype=np.int32)

    #Exclusive scan to determine displacements
    comm.Exscan(local_count, local_displacement, op=MPI.SUM)
    comm.Allreduce(local_count, total_count, op=MPI.SUM)
    comm.Allgather(local_displacement, displacements)
    comm.Allgather(local_count, counts)

    gathered_array = np.empty(total_count, dtype=array_data.dtype)
    #Reshape if necessary
    if shape is not None:
        gathered_array = gathered_array.reshape(shape)
    # Final conditioning of displacements list
    displacements[0] = 0

    mpi_dtype = MPI._typedict[np.sctype2char(array_data.dtype)]
    comm.Allgatherv(array_data,
                    [gathered_array, (counts, displacements), mpi_dtype])

    return gathered_array


def all_to_all(array_data, comm=MPI.COMM_WORLD):
    """ All to all exchange of distributed array data among processes in
        communicator.

    Parameters
    ----------
    array_data : numpy.ndarray
        Numpy array data distributed among processes.
        Requirements:
            Array length must be equal to the number of ranks in specified comm.
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD

    Returns
    -------
    all_to_all_exchanged : numpy.ndarray
        Collected numpy array from all process in MPI Comm.
    """
    size = comm.Get_size()

    if not isinstance(array_data, np.ndarray):
        raise TypeError('invalid data type for all_to_all.')
    if array_data.ndim > 1:
        raise ValueError('array must be a single dimension.')
    if len(array_data) != size or array_data.ndim != 1:
        raise ValueError('length of array must be equal to size of communicator.')

    all_to_all_exchanged = np.empty(array_data.size, dtype=array_data.dtype)
    comm.Alltoall(array_data, all_to_all_exchanged)

    return all_to_all_exchanged


def all_to_all_v(array_data, send_counts, recv_counts, send_displacements=None,
                 recv_displacements=None, recv_shape=None, comm=MPI.COMM_WORLD):
    """ All to all exchange of distributed array data among processes in
        communicator where each exchange is unique.

    Parameters
    ----------
    array_data : numpy.ndarray
        Numpy array data distributed among processes.
    send_counts : numpy.ndarray
        Constructed local array of length communicator size that lists how
        much data it will be sending to a given process.  The position in the
        array is the equivalent rank of the receiving process.
        Requirements:
            Array length must be equal to the number of ranks in specified comm.
        Format:
            send_counts[rank] = number of elements destined for rank
            ex:
                send_counts[0] = 0
                send_counts[1] = 3
                send_counts[2] = 4
                ...
    recv_counts : numpy.ndarray
        Constructed local numpy array of length communicator size that lists
        how much data it will be receiving for a given process.  The position in
        the array is the equivalent rank of the sending process.
        Requirements:
            Array length must be equal to the number of ranks in specified comm.
        Format:
            recv_counts[rank] = number of elements being transmitted from rank
            ex:
                recv_counts[0] = 0
                recv_counts[1] = 3
                recv_counts[2] = 4
                ...
    send_displacements : numpy.ndarray, None
        Optional numpy array of integers that specifies the element start local
        in the original array_data array that should be transmitted to a given
        process.
        Notes:
            If not supplied it is assumed that all data is being exchanged and
            the displacements will be determined by the send_counts.
        Requirements:
            Array length must be equal to the number of ranks in specified comm.
            The send_displacements will be unique to each rank/process in the
            communicator.
        Format:
            send_disp[rank] = start index in array_data buffer for given rank
            send_disp[0] = 0
            send_disp[1] = length of array data assigned to rank 0
            send_disp[2] = length of array data assigned to rank 0 + 1
            send_disp[3] = length of array data assigned to rank 0 + 1 + 2
            ...
    recv_displacements : numpy.ndarray, None
        Optional numpy array of integers that specifies the element start local
        in the local distributed array that should be received from a given
        process.
        Notes:
            If not supplied it is assumed that all data is being exchanged
            and the displacements will be determined by the recv_counts.
        Requirements:
            Array length must be equal to the number of ranks in specified comm.
            The recv_displacements will be unique to each rank/process in the
            communicator.
        Format:
            recv_disp[rank] = start index in local array buffer for given rank
            recv_disp[0] = 0
            recv_disp[1] = length of array data assigned to rank 0
            recv_disp[2] = length of array data assigned to rank 0 + 1
            recv_disp[3] = length of array data assigned to rank 0 + 1 + 2
            ...
    recv_shape : tuple, None
        Optional tuple describing the desired local shape for the receiving
        process/rank.
        Notes:
            If nothing is supplied the result will default to a 1-D array.
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD

    Returns
    -------
    recv_local_array : numpy.ndarray
        Constructed local numpy array from all process in MPI Comm.
    """
#TODO: Think about type checking here
    #Perform calculation on non-user provided information
    if send_displacements is None:
        send_displacements = _displacments_from_counts(send_counts)

    if recv_displacements is None:
        recv_displacements = _displacments_from_counts(recv_counts)

    recv_local_array = np.empty(recv_counts.sum(), dtype=array_data.dtype)
    mpi_dtype = MPI._typedict[np.sctype2char(array_data.dtype)]
    comm.Alltoallv(
        [array_data, (send_counts, send_displacements), mpi_dtype],
        [recv_local_array, (recv_counts, recv_displacements), mpi_dtype])

    if recv_shape is None:
        recv_shape = (recv_counts.sum(),)
    #Reshape
    recv_local_array = recv_local_array.reshape(recv_shape)

    return recv_local_array


def _displacments_from_counts(counts):
    """ Helper method to compute displacements from send/recv_counts.
        Note: Assumes entire local array contents is being replaced.

        Parameters
        ----------
        counts : np.ndarray
            See all_to_all_v docstring

        Returns
        -------
        displacments : np.ndarray
            Computed displacements
    """
    displacements = np.copy(counts)
    #Zero last non-zero element
    displacements[(displacements != 0).cumsum().argmax()] = 0
    #Add up all elements
    displacements = displacements.cumsum()
    #Roll contents by 1 index
    displacements = np.roll(displacements, 1)
    #Zero first element
    displacements[0] = 0
    return displacements

#TODO find elegant way to handle type checking in this
def broadcast_array(array_data, comm=MPI.COMM_WORLD, root=0):
    """ Broadcast array to all processes

    Parameters
    ----------
    array_data : numpy.ndarray
        Numpy array data local to root process.
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD
    root : int, optional
        Rank of root process that has the local data. If none specified
        defaults to 0.

    Returns
    -------
    array_data : numpy.ndarray
        Broadcasted(Distributed) array to all processes in MPI Comm.
    """
    rank = comm.Get_rank()
    #Transmit information needed to reconstruct array
    array_shape = array_data.shape if rank == root else None
    array_shape = broadcast_shape(array_shape, comm=comm, root=root)

#TODO: Look into str/char buffer send for this operation
    array_dtype = np.sctype2char(array_data.dtype) if rank == root else None
    array_dtype = comm.bcast(array_dtype, root=root)

    #Create empty buffer on non-root ranks
    if rank != root:
        array_data = np.empty(array_shape, dtype=np.dtype(array_dtype))

    #Broadcast the array
    mpi_dtype = MPI._typedict[array_dtype]
    comm.Bcast([array_data, array_data.size, mpi_dtype], root=root)

    return array_data

#TODO find elegant way to handle type checking in this
def broadcast_shape(shape, comm=MPI.COMM_WORLD, root=0):
    """ Broadcast shape to all processes

    Parameters
    ----------
    shape : int, tuple of int
        Shape representation of numpy.ndarray object
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD
    root : int, optional
        Rank of root process that has the local shape data. If none specified
        defaults to 0.

    Returns
    -------
    array_shape : numpy.ndarray
        Broadcasted(Distributed) shape to all processes in MPI Comm.
    """
    rank = comm.Get_rank()
    #Transmit number of dimensions
    if rank == root:
        shape_ndim = np.asarray(len(shape), dtype=np.int32)
    else:
        shape_ndim = np.empty(1, dtype=np.int32)
    comm.Bcast(shape_ndim, root=root)

    #Transmit shape values
    if rank == root:
        array_shape = np.asarray(shape, dtype=np.int32)
    else:
        array_shape = np.empty(shape_ndim, dtype=np.int32)
    comm.Bcast(array_shape, root=root)

    return array_shape


def get_comm():
    """ Get default world communicator

    Parameters
    ----------
    None

    Returns
    -------
    World_Comm : MPI.COMM_WORLD
        Default communicator
    """
    return MPI.COMM_WORLD


def get_comm_size(comm=MPI.COMM_WORLD):
    """ Get number of MPI processes(size) in communicator

    Parameters
    ----------
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD

    Returns
    -------
    size : int
        Number of MPI Processes in communicator
    """
    return comm.Get_size()


def get_rank(comm=MPI.COMM_WORLD):
    """ Get rank of MPI process in communicator

    Parameters
    ----------
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD

    Returns
    -------
    rank : int
        Rank of process
    """
    return comm.Get_rank()

#TODO find elegant way to handle type checking in this
def scatter_v(array_data, displacements, shapes, comm=MPI.COMM_WORLD, root=0):
    """ Scatter local array data to all processes

    Parameters
    ----------
    array_data : numpy.ndarray
        Numpy array data scattered(distributed) among processes.
    displacements : numpy.ndarray
        Numpy array of integers that specifies the element start local
        in the original array_data array that should be scattered to a given
        process.
        Requirements:
            Array length must be equal to the number of ranks in specified comm.
        Format:
            disp[rank] = start index in array_data buffer for given rank
            disp[0] = 0
            disp[1] = length of array data assigned to rank 0
            disp[2] = length of array data assigned to rank 0 + 1
            disp[3] = length of array data assigned to rank 0 + 1 + 2
            ...
    shapes : numpy.ndarray
        Numpy array of numpy.ndarray shape representations that specifies the
        final desired shape of the scattered array data to a given process.
        Notes:
            Acting as counts in typical scatter_v operation, with the
            benefit of allowing you to reconstruct a given numpy.ndarray shape.
        Requirements:
            Array length must be equal to the number of ranks in specified comm.
        Format:
            shapes[rank] = (length_axis0, length_axis1, ...)
            ex:
                shapes[0] = (2, 3)
                shapes[1] = (1, 3)
                shapes[2] = (1, 3)
                ...
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD
    root : int, optional
        Rank of root process that has the local shape data. If none specified
        defaults to 0.

    Returns
    -------
    local_data : numpy.ndarray
        Scattered numpy array as determined by the displacements and shapes
        arrays to processes in MPI Comm.
    """
    rank = comm.Get_rank()
    #Transmit information needed to reconstruct array
    displacements = broadcast_array(displacements, root=root)
    shapes = broadcast_array(shapes, root=root)

#TODO: Look into str/char buffer send for this operation
    array_dtype = np.sctype2char(array_data.dtype) if rank == root else None
    array_dtype = comm.bcast(array_dtype, root=root)

    counts = [np.prod(shape) for shape in shapes]
    local_data = np.empty(shapes[rank], dtype=np.dtype(array_dtype))

    #Scatter the array
    mpi_dtype = MPI._typedict[array_dtype]
    comm.Scatterv([array_data, counts, displacements, mpi_dtype],
                  local_data, root=root)

    return local_data
