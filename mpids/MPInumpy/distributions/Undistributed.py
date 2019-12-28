from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.MPIArray import MPIArray

class Undistributed(MPIArray):

        #Unique properties to MPIArray
        @property
        def dist(self):
                return 'u'


        @property
        def globalsize(self):
                return self.size


        @property
        def globalnbytes(self):
                return self.nbytes


        @property
        def globalshape(self):
                return self.shape


        def std(self, **kwargs):
                axis = kwargs.get('axis')
                local_mean = self.mean(**kwargs)

#TODO: Explore np kwarg 'keepdims' to avoid force transpose
                if axis == 1:
                        #Force a tranpose
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
                return local_red
