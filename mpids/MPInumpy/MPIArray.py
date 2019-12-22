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
                local_shape = self.shape
                if is_undistributed(self.dist):
                        return local_shape

                comm_shape = []
                axis = 0
                for axis_dim in local_shape:
                    axis_length = self._custom_reduction(MPI.SUM,
                                                         np.asarray(local_shape[axis]),
                                                         axis = axis)
                    comm_shape.append(axis_length[0])
                    axis += 1

                return comm_shape


        #Custom reduction method implementations
        def sum(self, **kwargs):
                """ Sum of array elements in distributed matrix over a
                given axis.

                Parameters
                ----------
                axis : None or int
                        Axis or axes along which the sum is performed.
                dtype : dtype, optional
                        Specified data type of returned array and of the
                        accumulator in which the elements are summed.

                Returns
                -------
                MPIArray : numpy.ndarray sub class
                        MPIArray with sum values along specified axis with
                        undistributed(copies on all procs) distribution.
                """
                self._check_reduction_parms(**kwargs)
                local_sum = np.asarray(self.base.sum(**kwargs))
                global_sum = self._custom_reduction(MPI.SUM, local_sum, **kwargs)
                return self.__class__(global_sum,
                                      dtype=global_sum.dtype,
                                      comm=self.comm,
                                      dist='u')


        def min(self, **kwargs):
                """ Min of array elements in distributed matrix over a
                given axis.

                Parameters
                ----------
                axis : None or int
                        Axis or axes along which the sum is performed.
                dtype : dtype, optional
                        Specified data type of returned array and of the
                        accumulator in which the elements are summed.

                Returns
                -------
                MPIArray : numpy.ndarray sub class
                        MPIArray with min values along specified axis with
                        undistributed(copies on all procs) distribution.
                """
                self._check_reduction_parms(**kwargs)
                local_min = np.asarray(self.base.min(**kwargs))
                global_min = self._custom_reduction(MPI.MIN, local_min, **kwargs)
                return self.__class__(global_min,
                                      dtype=global_min.dtype,
                                      comm=self.comm,
                                      dist='u')


        def max(self, **kwargs):
                """ Max of array elements in distributed matrix over a
                given axis.

                Parameters
                ----------
                axis : None or int
                        Axis or axes along which the sum is performed.
                dtype : dtype, optional
                        Specified data type of returned array and of the
                        accumulator in which the elements are summed.

                Returns
                -------
                MPIArray : numpy.ndarray sub class
                        MPIArray with max values along specified axis with
                        undistributed(copies on all procs) distribution.
                """
                self._check_reduction_parms(**kwargs)
                local_max = np.asarray(self.base.max(**kwargs))
                global_max = self._custom_reduction(MPI.MAX, local_max, **kwargs)
                return self.__class__(global_max,
                                      dtype=global_max.dtype,
                                      comm=self.comm,
                                      dist='u')


        def _check_reduction_parms(self, axis=None, dtype=None, out=None):
                if axis is not None and axis > self.ndim - 1:
                        raise ValueError("'axis' entry is out of bounds")
                if out is not None:
                        raise NotSupportedError("'out' field not supported")
                return


        def _custom_reduction(self, operation, local_red, axis=None,
                              dtype=None, out=None):
                if dtype is None: dtype = self.dtype

                if is_undistributed(self.dist):
                        global_red = local_red

                if axis is None and not is_undistributed(self.dist):
                        global_red = np.zeros(local_red.size, dtype=dtype)
                        self.comm.Allreduce(local_red, global_red, op=operation)

                if is_row_block_distributed(self.dist):
                        if axis == 0:
                                global_red = np.zeros(local_red.size, dtype=dtype)
                                self.comm.Allreduce(local_red, global_red, op=operation)
                        if axis == 1:
                                global_red = np.zeros(local_red.size * self.comm.size,
                                                      dtype=dtype)
                                self.comm.Allgather(local_red, global_red)

                if is_column_block_distributed(self.dist):
                        if axis == 0:
                                global_red = np.zeros(local_red.size * self.comm.size,
                                                      dtype=dtype)
                                self.comm.Allgather(local_red, global_red)
                        if axis == 1:
                                global_red = np.zeros(local_red.size, dtype=dtype)
                                self.comm.Allreduce(local_red, global_red, op=operation)

                if is_block_block_distributed(self.dist):
                        row_comm = self.comm.Split(color = self.comm_coord[0],
                                                   key = self.comm.Get_rank())
                        col_comm = self.comm.Split(color = self.comm_coord[1],
                                                   key = self.comm.Get_rank())

                        if axis == 0:
                                col_red = np.zeros(local_red.size, dtype=dtype)
                                col_comm.Allreduce(local_red, col_red, op=operation)
                                global_red = np.zeros(local_red.size * self.comm_dims[1],
                                                      dtype=dtype)
                                row_comm.Allgather(col_red, global_red)

                        if axis == 1:
                                row_red = np.zeros(local_red.size, dtype=dtype)
                                row_comm.Allreduce(local_red, row_red, op=operation)
                                global_red = np.zeros(local_red.size * self.comm_dims[0],
                                                      dtype=dtype)
                                col_comm.Allgather(row_red, global_red)

                return global_red
