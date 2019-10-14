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

## NOTE: MAY BE POSSIBLE TO REMOVE THE BELOW IF WE TRUST '__array_ufunc__'
        #Binary Operations
        def __add__(self, value):
                return getattr(self._array, '__add__')(value)

        def __radd__(self, value):
                return getattr(self._array, '__radd__')(value)

        def __sub__(self, value):
                return getattr(self._array, '__sub__')(value)

        def __rsub__(self, value):
                return getattr(self._array, '__rsub__')(value)

        def __mul__(self, value):
                return getattr(self._array, '__mul__')(value)

        def __rmul__(self, value):
                return getattr(self._array, '__rmul__')(value)

        def __floordiv__(self, value):
                return getattr(self._array, '__floordiv__')(value)

        def __rfloordiv__(self, value):
                return getattr(self._array, '__rfloordiv__')(value)

        def __truediv__(self, value):
                return getattr(self._array, '__truediv__')(value)

        def __rtruediv__(self, value):
                return getattr(self._array, '__rtruediv__')(value)

        def __mod__(self, value):
                return getattr(self._array, '__mod__')(value)

        def __rmod__(self, value):
                return getattr(self._array, '__rmod__')(value)

        def __pow__(self, value):
                return getattr(self._array, '__pow__')(value)

        def __rpow__(self, value):
                return getattr(self._array, '__rpow__')(value)

        def __lshift__(self, value):
                return getattr(self._array, '__lshift__')(value)

        def __rlshift__(self, value):
                return getattr(self._array, '__rlshift__')(value)

        def __rshift__(self, value):
                return getattr(self._array, '__rshift__')(value)

        def __rrshift__(self, value):
                return getattr(self._array, '__rrshift__')(value)

        def __and__(self, value):
                return getattr(self._array, '__and__')(value)

        def __rand__(self, value):
                return getattr(self._array, '__rand__')(value)

        def __or__(self, value):
                return getattr(self._array, '__or__')(value)

        def __ror__(self, value):
                return getattr(self._array, '__ror__')(value)

        def __xor__(self, value):
                return getattr(self._array, '__xor__')(value)

        def __rxor__(self, value):
                return getattr(self._array, '__rxor__')(value)

        #Unary Operations
        def __neg__(self):
                return getattr(self._array, '__neg__')()

        def __pos__(self):
                return getattr(self._array, '__pos__')()

        def __abs__(self):
                return getattr(self._array, '__abs__')()

        def __invert__(self):
                return getattr(self._array, '__invert__')()

# TODO: remaining unary operators
        # def __complex__(self):
        #         return getattr(self._array, '__complex__')()
        #
        # def __int__(self):
        #         return getattr(self._array, '__int__')()
        #
        # def __long__(self):
        #         return getattr(self._array, '__long__')()
        #
        # def __float__(self):
        #         return getattr(self._array, '__float__')()
        #
        # def __oct__(self):
        #         return getattr(self._array, '__oct__')()
        #
        # def __hex__(self):
        #         return getattr(self._array, '__hex__')()

        #Comparison Operations
        def __lt__(self, value):
                return getattr(self._array, '__lt__')(value)

        def __le__(self, value):
                return getattr(self._array, '__le__')(value)

        def __eq__(self, value):
                return getattr(self._array, '__eq__')(value)

        def __ne__(self, value):
                return getattr(self._array, '__ne__')(value)

        def __gt__(self, value):
                return getattr(self._array, '__gt__')(value)

        def __ge__(self, value):
                return getattr(self._array, '__ge__')(value)
