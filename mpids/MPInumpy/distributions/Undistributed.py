from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.MPIArray import MPIArray
from mpids.MPInumpy.errors import ValueError
from mpids.MPInumpy.utils import global_to_local_key

"""
    Undistributed implementation of MPIArray abstract base class.
"""
class Undistributed(MPIArray):

#TODO: Resolve this namespace requirement
    def __getitem__(self, key):
        local_key = global_to_local_key(key,
                                        self.globalshape,
                                        self.local_to_global)
        indexed_result = self.base.__getitem__(key)
        #Return undistributed copy of data
        return self.__class__(indexed_result, comm=self.comm)


    #Unique properties to MPIArray
    @property
    def dist(self):
        return 'u'


    @property
    def globalshape(self):
        if self._globalshape is None:
            self._globalshape = self.shape
        return self._globalshape


    @property
    def globalsize(self):
        if self._globalsize is None:
            self._globalsize = self.size
        return self._globalsize


    @property
    def globalnbytes(self):
        if self._globalnbytes is None:
            self._globalnbytes = self.nbytes
        return self._globalnbytes


    @property
    def globalndim(self):
        if self._globalndim is None:
            self.__globalndim()
        return self._globalndim

    def __globalndim(self):
        self._globalndim = int(len(self.globalshape))


    #Custom reduction method implementations
    def max(self, **kwargs):
        self.check_reduction_parms(**kwargs)
        local_max = np.asarray(self.base.max(**kwargs))
        global_max = self.custom_reduction(MPI.MAX, local_max, **kwargs)
        return Undistributed(global_max, comm=self.comm)


    def mean(self, **kwargs):
        global_sum = self.sum(**kwargs)
        axis = kwargs.get('axis')
        if axis is not None:
            global_mean = global_sum * 1. / self.globalshape[axis]
        else:
            global_mean = global_sum * 1. / self.globalsize

        return Undistributed(global_mean, comm=self.comm)


    def min(self, **kwargs):
        self.check_reduction_parms(**kwargs)
        local_min = np.asarray(self.base.min(**kwargs))
        global_min = self.custom_reduction(MPI.MIN, local_min, **kwargs)
        return Undistributed(global_min, comm=self.comm)


    def std(self, **kwargs):
        axis = kwargs.get('axis')
        local_mean = self.mean(**kwargs)

#TODO: Explore np kwarg 'keepdims' to avoid force transpose
        if axis == 1:
            #Force a tranpose
            local_mean = local_mean.reshape(self.shape[0], 1)

        local_square_diff = (self - local_mean)**2
        local_sum_square_diff = np.asarray(local_square_diff.base.sum(**kwargs))
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

        return Undistributed(global_std, comm=self.comm)


    def sum(self, **kwargs):
        self.check_reduction_parms(**kwargs)
        local_sum = np.asarray(self.base.sum(**kwargs))
        global_sum = self.custom_reduction(MPI.SUM, local_sum, **kwargs)
        return Undistributed(global_sum, comm=self.comm)


    def custom_reduction(self, operation, local_red, axis=None, dtype=None,
                         out=None):
        return local_red


    def collect_data(self):
        return self.__class__(self, comm=self.comm)


    def reshape(self, *args):
        if np.prod(args) != self.globalsize:
            raise ValueError("cannot reshape global array of size",
                             self.globalsize,"into shape", tuple(args))
        return self.__class__(self.base.reshape(*args), comm=self.comm)
