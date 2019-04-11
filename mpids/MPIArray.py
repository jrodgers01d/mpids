from mpi4py import MPI

class MPIArray:
        def __init__(self, np_array, comm=MPI.COMM_WORLD):
                self.array = np_array
                self.size_global = None
                self.shape_global = None
                self.comm = comm

        def size(self):
                return self.array.size

        def data(self):
                return self.array.data

        def dtype(self):
                return self.array.dtype

        def shape(self):
                return self.array.shape

        def strides(self):
                return self.array.strides
