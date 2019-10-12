from mpi4py import MPI
import numpy as np

class MPIArray(object):
        def __init__(self, array_data, comm=MPI.COMM_WORLD):
                self._data = array_data
                self._comm = comm

        def __repr__(self):
                return '{}'.format(self.__class__.__name__)
                # return '{}(global_size={}, global_shape={}, dtype={})' \
                #         .format(self.__class__.__name__,
                #                 self.global_size,
                #                 self.global_shape,
                #                 self.dtype)


        def __array__(self):
                return np.array(self._data)

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

        #Expected properties to numpy.ndarray
        @property
        def T(self):
                return np.array(self._data).T

        @property
        def data(self):
                return np.array(self._data).data

        @property
        def dtype(self):
                return np.array(self._data).dtype

# TODO: Flags?
        # @property
        # def flags(self)
        #         return None

# TODO: Flat?
        # @property
        # def flat(self)
        #         return None

        @property
        def imag(self):
                return np.array(self._data).imag

        @property
        def real(self):
                return np.array(self._data).real

        @property
        def size(self):
                return np.array(self._data).size

        @property
        def itemsize(self):
                return np.array(self._data).itemsize

        @property
        def nbytes(self):
                return np.array(self._data).nbytes

        @property
        def ndim(self):
                return np.array(self._data).ndim

        @property
        def shape(self):
                return np.array(self._data).shape

        @property
        def strides(self):
                return np.array(self._data).strides

# TODO: ctypes?
        # @property
        # def ctypes(self)
        #         return None

# TODO: base?
        # @property
        # def base(self)
        #         return None
