from mpi4py import MPI
import numpy as np

class MPIArray:
        def __init__(self, np_array, comm=MPI.COMM_WORLD):
                self.array = np_array
                self.comm = comm
                self.size = np_array.size
                self.data = np_array.data
                self.dtype = np_array.dtype
                self.shape = np_array.shape
                self.strides = np_array.strides
                self.global_size = self.__find_global_size()

        def __find_global_size(self):
                comm_size = np.array(0)
                self.comm.Allreduce(np.array(self.size), comm_size, op=MPI.SUM)
                return comm_size
