from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.errors import TypeError

__all__ = ['all_gather_v', 'broadcast_array', 'scatter_v']

def all_gather_v(array_data, shape=None, comm=MPI.COMM_WORLD):
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

#TODO find elegant way to handle type checking in this
def broadcast_array(array_data, comm=MPI.COMM_WORLD, root=0):
    rank = comm.Get_rank()
    #Transmit information needed to reconstruct array
    if rank == root:
        array_ndim = np.asarray(array_data.ndim, dtype=np.int32)
    else:
        array_ndim = np.empty(1, dtype=np.int32)
    comm.Bcast(array_ndim, root=root)

    if rank == root:
        array_shape = np.asarray(array_data.shape, dtype=np.int32)
    else:
        array_shape = np.empty(array_ndim, dtype=np.int32)
    comm.Bcast(array_shape, root=root)

#TODO: Look into str/char buffer send for this operation
    if rank == root:
        array_dtype = np.sctype2char(array_data.dtype)
    else:
        array_dtype = None
    array_dtype = comm.bcast(array_dtype, root=root)

    #Create empty buffer on non-root ranks
    if rank != root:
        array_data = np.empty(array_shape, dtype=np.dtype(array_dtype))

    #Broadcast the array
    mpi_dtype = MPI._typedict[array_dtype]
    comm.Bcast([array_data, array_data.size, mpi_dtype], root=root)

    return array_data


#TODO find elegant way to handle type checking in this
def scatter_v(array_data, displacements, shapes, comm=MPI.COMM_WORLD, root=0):
    rank = comm.Get_rank()
    #Transmit information needed to reconstruct array
    displacements = broadcast_array(displacements, root=root)
    shapes = broadcast_array(shapes, root=root)
#TODO: Look into str/char buffer send for this operation
    if rank == root:
        array_dtype = np.sctype2char(array_data.dtype)
    else:
        array_dtype = None
    array_dtype = comm.bcast(array_dtype, root=root)

    counts = [np.prod(shape) for shape in shapes]
    local_data = np.empty(shapes[rank], dtype=np.dtype(array_dtype))

    #Scatter the array
    mpi_dtype = MPI._typedict[array_dtype]
    comm.Scatterv([array_data, counts, displacements, mpi_dtype],
                  local_data, root=root)

    return local_data
