from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.MPIArray import MPIArray

class ColumnBlock(MPIArray):

        #Unique properties to MPIArray
        @property
        def dist(self):
                return ('*', 'b')


        def std(self, **kwargs):
                axis = kwargs.get('axis')
                local_mean = self.mean(**kwargs)

                if axis == 0:
                        col_min, col_max = self.local_to_global[1]
                        local_mean = local_mean[col_min: col_max]
#TODO: Explore np kwarg 'keepdims' to avoid force transpose
                if axis == 1: #Force a transpose
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

                return self.__class__(global_std,
                                      dtype=global_std.dtype,
                                      comm=self.comm)


        def custom_reduction(self, operation, local_red, axis=None, dtype=None,
                             out=None):
                if dtype is None: dtype = self.dtype

                if axis == 0:
                        global_red = np.zeros(local_red.size * self.comm.size,
                                              dtype=dtype)
                        self.comm.Allgather(local_red, global_red)
                if axis is None or axis == 1:
                        global_red = np.zeros(local_red.size, dtype=dtype)
                        self.comm.Allreduce(local_red, global_red, op=operation)

                return global_red
