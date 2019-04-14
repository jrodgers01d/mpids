from mpi4py import MPI

class MPIArray:
        def __init__(self, np_array, comm=MPI.COMM_WORLD):
                self.array = np_array
                self.comm = comm
                self.size = np_array.size
                self.data = np_array.data
                self.dtype = np_array.dtype
                self.shape = np_array.shape
                self.strides = np_array.strides
                self.comm_size = __find_global_size()
                self.comm_shape = None

        def __find_global_size(self):
                rank = self.comm.Get_rank()
                self.comm.Allreduce(self.size, self.comm_size, op=MPI.SUM)

        # def size(self):
        #         return self.array.size
        #
        # def data(self):
        #         return self.array.data
        #
        # def dtype(self):
        #         return self.array.dtype
        #
        # def shape(self):
        #         return self.array.shape
        #
        # def strides(self):
        #         return self.array.strides
