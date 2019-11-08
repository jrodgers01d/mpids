from mpi4py import MPI
import numpy as np
from mpids.MPInumpy.errors import ValueError, NotSupportedError

class MPIArray(np.ndarray):
        """ MPIArray subclass of numpy.ndarray """

        def __new__(cls, array_data, dtype=None, copy=True, order=None,
                    subok=False, ndmin=0, comm=MPI.COMM_WORLD, dist='b'):
                """ Create MPIArray from process local array data.

                Parameters
                ----------
                array_data : array_like
                        Array like data local to each process.
                dtype : data-type, optional
                        Desired data-type for the array.
                copy : bool, optional
                        Default 'True' results in copied object, if 'False' copy
                        only made when base class '__array__' returns a copy.
                order: {'K','A','C','F'}, optional
                        Specified memory layout of the array.
                subok : bool, optional
                        Default 'False' returned array will be forced to be
                        base-class array, if 'True' then sub-classes will be
                        passed-through.
                ndmin : int, optional
                        Specifies the minimum number of dimensions that the
                        resulting array should have.
                comm : MPI Communicator, optional
                        MPI process communication object.  If none specified
                        defaults to MPI.COMM_WORLD
                dist : str, list, tuple
                        Specified distribution of data among processes.
                        Default value 'b' : Block, *
                        Supported types:
                            'b' : Block, *
                            ('*', 'b') : *, Block
                            ('b','b') : Block-Block
                            'u' : Undistributed

                Returns
                -------
                MPIArray : numpy.ndarray sub class
                """
                obj = np.array(array_data,
                               dtype=dtype,
                               copy=copy,
                               order=order,
                               subok=subok,
                               ndmin=ndmin).view(cls)
                obj.comm = comm
                obj.dist = dist
                return obj


        def __array_finalize__(self, obj):
                if obj is None: return
                self.comm = getattr(obj, 'comm', None)
                self.dist = getattr(obj, 'dist', None)


        def __repr__(self):
                return '{}(globalsize={}, dist={}, dtype={})' \
                       .format(self.__class__.__name__,
                               getattr(self, 'globalsize', None),
                               getattr(self, 'dist', None),
                               getattr(self, 'dtype', None))
                # return '{}(global_size={}, global_shape={}, distribution={}, dtype={})' \
                #         .format(self.__class__.__name__,
                #                 self.global_size,
                #                 self.dist,
                #                 self.global_shape,
                #                 self.dtype)


        #Unique properties to MPIArray
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
        # Custom method implementations
        def sum(self, axis=None, dtype=None, out=None):
                if dtype is None: dtype = self.dtype
                if axis is not None and axis > self.ndim - 1:
                        raise ValueError("'axis' entry is out of bounds")
                if out is not None:
                        raise NotSupportedError("'out' field not supported")

                # Compute local sum with specified parms on local base array
                local_sum = np.asarray(self.base.sum(axis=axis, dtype=dtype))
                global_sum = np.zeros(local_sum.size, dtype=dtype)
                self.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)

                return MPIArray(global_sum, global_sum.dtype, dist='u')

        def min(self, axis=None, out=None):
                if axis is not None and axis > self.ndim - 1:
                        raise ValueError("'axis' entry is out of bounds")
                if out is not None:
                        raise NotSupportedError("'out' field not supported")

                # Compute local sum with specified parms on local base array
                local_min = np.asarray(self.base.min(axis=axis))
                global_min = np.zeros(local_min.size, dtype=self.dtype)
                self.comm.Allreduce(local_min, global_min, op=MPI.MIN)

                return MPIArray(global_min, global_min.dtype, dist='u')

        def max(self, axis=None, out=None):
                if axis is not None and axis > self.ndim - 1:
                        raise ValueError("'axis' entry is out of bounds")
                if out is not None:
                        raise NotSupportedError("'out' field not supported")

                # Compute local sum with specified parms on local base array
                local_max = np.asarray(self.base.max(axis=axis))
                global_max = np.zeros(local_max.size, dtype=self.dtype)
                self.comm.Allreduce(local_max, global_max, op=MPI.MAX)

                return MPIArray(global_max, global_max.dtype, dist='u')
