from mpi4py import MPI
import numpy as np

class MPIArray(object):
        def __init__(self, array_data, comm=MPI.COMM_WORLD):
                self._data = array_data
                self._comm = comm
                # self.size = np_array.size
                # self.data = np_array.data
                # self.dtype = np_array.dtype
                # self.shape = np_array.shape
                # self.strides = np_array.strides
                # self.global_size = self.__find_global_size()
                # self.global_shape = self.__find_global_shape()

        def __repr__(self):
                return '{}'.format(self.__class__.__name__)
                # return '{}(global_size={}, global_shape={}, dtype={})' \
                #         .format(self.__class__.__name__,
                #                 self.global_size,
                #                 self.global_shape,
                #                 self.dtype)

        def __array__(self):
                return np.array(self._data)

        @property
        def data(self):
                return self._data

        @property
        def comm(self):
                return self._comm

        # def __find_global_size(self):
        #         comm_size = np.zeros(1)
        #         self.comm.Allreduce(np.array(self.size), comm_size, op=MPI.SUM)
        #         return comm_size
        #
        # def __find_global_shape(self):
        #         return -9999
