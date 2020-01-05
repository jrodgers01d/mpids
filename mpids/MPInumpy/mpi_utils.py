from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.errors import TypeError

__all__ = ['all_gather_v']

def all_gather_v(array_data, shape=None, comm=MPI.COMM_WORLD):
    if not isinstance(array_data, np.ndarray):
        raise TypeError('invalid data type for all_gather_v.')

    comm_size = comm.Get_size()
    local_displacement = np.zeros(1, dtype='int')
    displacements = np.zeros(comm_size, dtype='int')
    local_count = np.asarray(array_data.size, dtype='int')
    counts = np.zeros(comm_size, dtype='int')
    total_count = np.zeros(1, dtype='int')

    #Exclusive scan to determine displacements
    comm.Exscan(local_count, local_displacement, op=MPI.SUM)
    comm.Allreduce(local_count, total_count, op=MPI.SUM)
    comm.Allgather(local_displacement, displacements)
    comm.Allgather(local_count, counts)

    gathered_array = np.zeros(total_count, dtype=array_data.dtype)
    #Reshape if necessary
    if shape is not None:
        gathered_array = gathered_array.reshape(shape)
    # Final conditioning of displacements list
    displacements[0] = 0

    mpi_dtype = MPI._typedict[np.sctype2char(array_data.dtype)]
    comm.Allgatherv(array_data,
                    [gathered_array, (counts, displacements), mpi_dtype])

    return gathered_array
