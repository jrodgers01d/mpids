from mpi4py import MPI
import numpy as np

class MPIArray(np.ndarray):
        def __new__(cls, array_data, comm=MPI.COMM_WORLD):
                obj = np.asarray(array_data).view(cls)
                obj._comm = comm

                return obj

        #Unique properties to MPIArray
        @property
        def comm(self):
                return self._comm

        @property
        def globalsize(self):
                comm_size = np.zeros(1, dtype='int')
                self.comm.Allreduce(np.array(self.size), comm_size, op=MPI.SUM)
                return comm_size

        @property
        def globalnbytes(self):
                comm_nbytes = np.zeros(1, dtype='int')
                self.comm.Allreduce(np.array(self.nbytes), comm_nbytes, op=MPI.SUM)
                return comm_nbytes

# TODO:  global_shape, global_strides

        def __repr__(self):
                return '{}'.format(self.__class__.__name__)
                # return '{}(global_size={}, global_shape={}, dtype={})' \
                #         .format(self.__class__.__name__,
                #                 self.global_size,
                #                 self.global_shape,
                #                 self.dtype)
