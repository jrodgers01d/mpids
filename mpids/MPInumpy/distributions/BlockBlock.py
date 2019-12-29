from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.MPIArray import MPIArray
from mpids.MPInumpy.distributions.Undistributed import Undistributed


"""
    Block-Block implementation of MPIArray abstract base class.
"""
class BlockBlock(MPIArray):

        #Unique properties to MPIArray
        @property
        def dist(self):
                return ('b', 'b')


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


        @property
        def globalshape(self):
                local_shape = self.shape
                comm_shape = []
                axis = 0
                for axis_dim in local_shape:
                    axis_length = self.custom_reduction(MPI.SUM,
                                                          np.asarray(local_shape[axis]),
                                                          axis = axis)
                    comm_shape.append(axis_length[0])
                    axis += 1

                return comm_shape


        #Custom reduction method implementations
        def max(self, **kwargs):
                self.check_reduction_parms(**kwargs)
                local_max = np.asarray(self.base.max(**kwargs))
                global_max = self.custom_reduction(MPI.MAX, local_max, **kwargs)
                return Undistributed(global_max,
                                      dtype=global_max.dtype,
                                      comm=self.comm)

        def mean(self, **kwargs):
                global_sum = self.sum(**kwargs)
                axis = kwargs.get('axis')
                if axis is not None:
                        global_mean = global_sum * 1. / self.globalshape[axis]
                else:
                        global_mean = global_sum * 1. / self.globalsize

                return Undistributed(global_mean,
                                      dtype=global_mean.dtype,
                                      comm=self.comm)


        def min(self, **kwargs):
                self.check_reduction_parms(**kwargs)
                local_min = np.asarray(self.base.min(**kwargs))
                global_min = self.custom_reduction(MPI.MIN, local_min, **kwargs)
                return Undistributed(global_min,
                                      dtype=global_min.dtype,
                                      comm=self.comm)


        def std(self, **kwargs):
                axis = kwargs.get('axis')
                local_mean = self.mean(**kwargs)

                if axis == 0:
                        col_min, col_max = self.local_to_global[1]
                        local_mean = local_mean[col_min: col_max]
                if axis == 1:
                        row_min, row_max = self.local_to_global[0]
                        local_mean = local_mean[row_min: row_max]
#TODO: Explore np kwarg 'keepdims' to avoid force transpose
                        #Force a transpose
                        local_mean = local_mean.reshape(self.shape[0], 1)

                local_square_diff = (self - local_mean)**2
                local_sum_square_diff = \
                        np.asarray(local_square_diff.base.sum(**kwargs))
                global_sum_square_diff = \
                        self.custom_reduction(MPI.SUM,
                                                local_sum_square_diff,
                                                dtype = local_sum_square_diff.dtype,
                                                **kwargs)
                if axis is not None:
                        global_std = np.sqrt(
                                global_sum_square_diff * 1. / self.globalshape[axis])
                else:
                        global_std = np.sqrt(
                                global_sum_square_diff * 1. / self.globalsize)

                return Undistributed(global_std,
                                      dtype=global_std.dtype,
                                      comm=self.comm)


        def sum(self, **kwargs):
                self.check_reduction_parms(**kwargs)
                local_sum = np.asarray(self.base.sum(**kwargs))
                global_sum = self.custom_reduction(MPI.SUM, local_sum, **kwargs)
                return Undistributed(global_sum,
                                      dtype=global_sum.dtype,
                                      comm=self.comm)


        def custom_reduction(self, operation, local_red, axis=None, dtype=None,
                             out=None):
                if dtype is None: dtype = self.dtype

                if axis is None:
                        global_red = np.zeros(local_red.size, dtype=dtype)
                        self.comm.Allreduce(local_red, global_red, op=operation)

                else:
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
