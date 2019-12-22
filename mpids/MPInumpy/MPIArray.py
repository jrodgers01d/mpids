from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.utils import is_undistributed, is_row_block_distributed, \
                                 is_column_block_distributed, is_block_block_distributed
from mpids.MPInumpy.errors import ValueError, NotSupportedError

class MPIArray(np.ndarray):
        """ MPIArray subclass of numpy.ndarray """

        def __new__(cls, array_data, dtype=None, copy=True, order=None,
                    subok=False, ndmin=0, comm=MPI.COMM_WORLD, dist='b',
                    comm_dims=None, comm_coord=None):
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
                comm_dims: list
                        Specified dimensions of processes in cartesian grid
                        for communicator.
                comm_coord : list
                        Rank/Procses cartesian coordinate in communicator
                        process grid.

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
                obj.comm_dims = comm_dims
                obj.comm_coord = comm_coord
                return obj


        def __array_finalize__(self, obj):
                if obj is None: return
                self.comm = getattr(obj, 'comm', None)
                self.dist = getattr(obj, 'dist', None)
                self.comm_dims = getattr(obj, 'comm_dims', None)
                self.comm_coord = getattr(obj, 'comm_coord', None)


        def __repr__(self):
                return '{}(globalsize={}, globalshape={}, dist={}, dtype={})' \
                       .format(self.__class__.__name__,
                               getattr(self, 'globalsize', None),
                               list(getattr(self, 'globalshape', None)),
                               getattr(self, 'dist', None),
                               getattr(self, 'dtype', None))

        #Unique properties to MPIArray
        @property
        def globalsize(self):
                if is_undistributed(self.dist):
                        return self.size

                comm_size = np.zeros(1, dtype='int')
                self.comm.Allreduce(np.array(self.size), comm_size, op=MPI.SUM)
                return comm_size


        @property
        def globalnbytes(self):
                if is_undistributed(self.dist):
                        return self.nbytes

                comm_nbytes = np.zeros(1, dtype='int')
                self.comm.Allreduce(np.array(self.nbytes), comm_nbytes, op=MPI.SUM)
                return comm_nbytes


        @property
        def globalshape(self):
                if is_undistributed(self.dist):
                        return self.shape

                comm_shape = np.zeros(self.ndim, dtype='int')
                self.comm.Allreduce(np.array(self.shape), comm_shape, op=MPI.SUM)

                if is_row_block_distributed(self.dist):
                        comm_shape[1] = self.shape[1]
                if is_column_block_distributed(self.dist):
                        comm_shape[0] = self.shape[0]
# TODO below behavior questionable for non-mulitiple of comm_dims shapes
                if is_block_block_distributed(self.dist):
                        comm_dims = MPI.Compute_dims(self.comm.size, 2)
                        comm_shape = comm_shape / comm_dims

                return comm_shape

        #Custom reduction method implementations
        def sum(self, axis=None, dtype=None, out=None):
                if dtype is None: dtype = self.dtype
                if axis is not None and axis > self.ndim - 1:
                        raise ValueError("'axis' entry is out of bounds")
                if out is not None:
                        raise NotSupportedError("'out' field not supported")

                # Compute local sum with specified parms on local base array
                local_sum = np.asarray(self.base.sum(axis=axis, dtype=dtype))

                if is_undistributed(self.dist):
                        global_sum = local_sum

                if axis is None and not is_undistributed(self.dist):
                        global_sum = np.zeros(local_sum.size, dtype=dtype)
                        self.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)

                if is_row_block_distributed(self.dist):
                        if axis == 0:
                                global_sum = np.zeros(local_sum.size, dtype=dtype)
                                self.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
                        if axis == 1:
                                global_sum = np.zeros(local_sum.size * self.comm.size,
                                                      dtype=dtype)
                                self.comm.Allgather(local_sum, global_sum)

                if is_column_block_distributed(self.dist):
                        if axis == 0:
                                global_sum = np.zeros(local_sum.size * self.comm.size,
                                                      dtype=dtype)
                                self.comm.Allgather(local_sum, global_sum)
                        if axis == 1:
                                global_sum = np.zeros(local_sum.size, dtype=dtype)
                                self.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)

                if is_block_block_distributed(self.dist):
                        row_comm = self.comm.Split(color = self.comm_coord[0],
                                                   key = self.comm.Get_rank())
                        col_comm = self.comm.Split(color = self.comm_coord[1],
                                                   key = self.comm.Get_rank())

                        if axis == 0:
                                col_sum = np.zeros(local_sum.size, dtype=dtype)
                                col_comm.Allreduce(local_sum, col_sum, op=MPI.SUM)
                                global_sum = np.zeros(local_sum.size * self.comm_dims[1],
                                                      dtype=dtype)
                                row_comm.Allgather(col_sum, global_sum)

                        if axis == 1:
                                row_sum = np.zeros(local_sum.size, dtype=dtype)
                                row_comm.Allreduce(local_sum, row_sum, op=MPI.SUM)
                                global_sum = np.zeros(local_sum.size * self.comm_dims[0],
                                                      dtype=dtype)
                                col_comm.Allgather(row_sum, global_sum)

                return self.__class__(global_sum,
                                      dtype=global_sum.dtype,
                                      comm=self.comm,
                                      dist='u')


        def min(self, axis=None, out=None):
                if axis is not None and axis > self.ndim - 1:
                        raise ValueError("'axis' entry is out of bounds")
                if out is not None:
                        raise NotSupportedError("'out' field not supported")

                # Compute local min with specified parms on local base array
                local_min = np.asarray(self.base.min(axis=axis))

                if is_undistributed(self.dist):
                        global_min = local_min

                if axis is None and not is_undistributed(self.dist):
                        global_min = np.zeros(local_min.size, dtype=self.dtype)
                        self.comm.Allreduce(local_min, global_min, op=MPI.MIN)

                if is_row_block_distributed(self.dist):
                        if axis == 0:
                                global_min = np.zeros(local_min.size, dtype=self.dtype)
                                self.comm.Allreduce(local_min, global_min, op=MPI.MIN)
                        if axis == 1:
                                global_min = np.zeros(local_min.size * self.comm.size,
                                                      dtype=self.dtype)
                                self.comm.Allgather(local_min, global_min)

                if is_column_block_distributed(self.dist):
                        if axis == 0:
                                global_min = np.zeros(local_min.size * self.comm.size,
                                                      dtype=self.dtype)
                                self.comm.Allgather(local_min, global_min)
                        if axis == 1:
                                global_min = np.zeros(local_min.size, dtype=self.dtype)
                                self.comm.Allreduce(local_min, global_min, op=MPI.MIN)

                if is_block_block_distributed(self.dist):
                        row_comm = self.comm.Split(color = self.comm_coord[0],
                                                   key = self.comm.Get_rank())
                        col_comm = self.comm.Split(color = self.comm_coord[1],
                                                   key = self.comm.Get_rank())
                        if axis == 0:
                                col_min = np.zeros(local_min.size, dtype=self.dtype)
                                col_comm.Allreduce(local_min, col_min, op=MPI.MIN)
                                global_min = np.zeros(local_min.size * self.comm_dims[1],
                                                      dtype=self.dtype)
                                row_comm.Allgather(col_min, global_min)

                        if axis == 1:
                                row_min = np.zeros(local_min.size, dtype=self.dtype)
                                row_comm.Allreduce(local_min, row_min, op=MPI.MIN)
                                global_min = np.zeros(local_min.size * self.comm_dims[0],
                                                      dtype=self.dtype)
                                col_comm.Allgather(row_min, global_min)

                return self.__class__(global_min,
                                      dtype=global_min.dtype,
                                      comm=self.comm,
                                      dist='u')


        def max(self, axis=None, out=None):
                if axis is not None and axis > self.ndim - 1:
                        raise ValueError("'axis' entry is out of bounds")
                if out is not None:
                        raise NotSupportedError("'out' field not supported")

                # Compute local max with specified parms on local base array
                local_max = np.asarray(self.base.max(axis=axis))

                if is_undistributed(self.dist):
                        global_max = local_max

                if axis is None and not is_undistributed(self.dist):
                        global_max = np.zeros(local_max.size, dtype=self.dtype)
                        self.comm.Allreduce(local_max, global_max, op=MPI.MAX)

                if is_row_block_distributed(self.dist):
                        if axis == 0:
                                global_max = np.zeros(local_max.size, dtype=self.dtype)
                                self.comm.Allreduce(local_max, global_max, op=MPI.MAX)
                        if axis == 1:
                                global_max = np.zeros(local_max.size * self.comm.size,
                                                      dtype=self.dtype)
                                self.comm.Allgather(local_max, global_max)

                if is_column_block_distributed(self.dist):
                        if axis == 0:
                                global_max = np.zeros(local_max.size * self.comm.size,
                                                      dtype=self.dtype)
                                self.comm.Allgather(local_max, global_max)
                        if axis == 1:
                                global_max = np.zeros(local_max.size, dtype=self.dtype)
                                self.comm.Allreduce(local_max, global_max, op=MPI.MAX)

                if is_block_block_distributed(self.dist):
                        row_comm = self.comm.Split(color = self.comm_coord[0],
                                                   key = self.comm.Get_rank())
                        col_comm = self.comm.Split(color = self.comm_coord[1],
                                                   key = self.comm.Get_rank())
                        if axis == 0:
                                col_max = np.zeros(local_max.size, dtype=self.dtype)
                                col_comm.Allreduce(local_max, col_max, op=MPI.MAX)
                                global_max = np.zeros(local_max.size * self.comm_dims[1],
                                                      dtype=self.dtype)
                                row_comm.Allgather(col_max, global_max)

                        if axis == 1:
                                row_max = np.zeros(local_max.size, dtype=self.dtype)
                                row_comm.Allreduce(local_max, row_max, op=MPI.MAX)
                                global_max = np.zeros(local_max.size * self.comm_dims[0],
                                                      dtype=self.dtype)
                                col_comm.Allgather(row_max, global_max)

                return self.__class__(global_max,
                                      dtype=global_max.dtype,
                                      comm=self.comm,
                                      dist='u')
