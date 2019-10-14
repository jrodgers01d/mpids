from mpi4py import MPI
import numpy as np

class MPIArray(object):
        def __init__(self, array_data, comm=MPI.COMM_WORLD):
                self._data = array_data
                self._array = np.array(array_data)
                self._comm = comm

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
                return self._array.T

        @property
        def data(self):
                return self._array.data

        @property
        def dtype(self):
                return self._array.dtype

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
                return self._array.imag

        @property
        def real(self):
                return self._array.real

        @property
        def size(self):
                return self._array.size

        @property
        def itemsize(self):
                return self._array.itemsize

        @property
        def nbytes(self):
                return self._array.nbytes

        @property
        def ndim(self):
                return self._array.ndim

        @property
        def shape(self):
                return self._array.shape

        @property
        def strides(self):
                return self._array.strides

# TODO: ctypes?
        # @property
        # def ctypes(self)
        #         return None

# TODO: base?
        # @property
        # def base(self)
        #         return None

        def __repr__(self):
                return '{}'.format(self.__class__.__name__)
                # return '{}(global_size={}, global_shape={}, dtype={})' \
                #         .format(self.__class__.__name__,
                #                 self.global_size,
                #                 self.global_shape,
                #                 self.dtype)

        def __str__(self):
                return self._array.__str__()

        def __array__(self):
                return self._array

        #Binary Operations
        def __add__(self, other):
                return getattr(self._array, '__add__')(other)

        def __radd__(self, other):
                return getattr(self._array, '__radd__')(other)

        def __sub__(self, other):
                return getattr(self._array, '__sub__')(other)

        def __rsub__(self, other):
                return getattr(self._array, '__rsub__')(other)

        def __mul__(self, other):
                return getattr(self._array, '__mul__')(other)

        def __rmul__(self, other):
                return getattr(self._array, '__rmul__')(other)

        def __floordiv__(self, other):
                return getattr(self._array, '__floordiv__')(other)

        def __rfloordiv__(self, other):
                return getattr(self._array, '__rfloordiv__')(other)

        def __truediv__(self, other):
                return getattr(self._array, '__truediv__')(other)

        def __rtruediv__(self, other):
                return getattr(self._array, '__rtruediv__')(other)

        def __mod__(self, other):
                return getattr(self._array, '__mod__')(other)

        def __rmod__(self, other):
                return getattr(self._array, '__rmod__')(other)

        def __pow__(self, other):
                return getattr(self._array, '__pow__')(other)

        def __rpow__(self, other):
                return getattr(self._array, '__rpow__')(other)

        def __lshift__(self, other):
                return getattr(self._array, '__lshift__')(other)

        def __rlshift__(self, other):
                return getattr(self._array, '__rlshift__')(other)

        def __rshift__(self, other):
                return getattr(self._array, '__rshift__')(other)

        def __rrshift__(self, other):
                return getattr(self._array, '__rrshift__')(other)

        def __and__(self, other):
                return getattr(self._array, '__and__')(other)

        def __rand__(self, other):
                return getattr(self._array, '__rand__')(other)

        def __or__(self, other):
                return getattr(self._array, '__or__')(other)

        def __ror__(self, other):
                return getattr(self._array, '__ror__')(other)

        def __xor__(self, other):
                return getattr(self._array, '__xor__')(other)

        def __rxor__(self, other):
                return getattr(self._array, '__rxor__')(other)

# TODO: unary and comparison operators
