class MPIArray:
        def __init__(self, np_array, comm=MPI.COMM_WORLD):
                self.data = np_array.data
                self.size = np_array.size
                # self.size_global = None
                self.dtype = np_array.dtype
                self.shape = np_array.shape
                # self.shape_global = None
                self.strides = np_array.strides
                self.comm = comm

        # def size(self):
        #         rank = comm.Get_rank()
        #         if rank == 0:
        #                 return self.shape_global
