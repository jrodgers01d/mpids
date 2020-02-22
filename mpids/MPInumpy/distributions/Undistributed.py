from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.MPIArray import MPIArray
from mpids.MPInumpy.errors import ValueError
from mpids.MPInumpy.utils import global_to_local_key

"""
    Undistributed implementation of MPIArray abstract base class.
"""
class Undistributed(MPIArray):

    def __getitem__(self, key):
        local_key = global_to_local_key(key,
                                        self.globalshape,
                                        self.local_to_global)
        indexed_result = self.base.__getitem__(key)
        #Return undistributed copy of data
        return self.__class__(indexed_result, comm=self.comm)


    def __setitem__(self, key, value):
        #Check input, will throw np.ValueError if data type of passed
        ## value can't be converted to objects type
        np_value = np.asarray(value, dtype=self.dtype)
        local_key = global_to_local_key(key,
                                        self.globalshape,
                                        self.local_to_global)
        self.base.__setitem__(local_key, np_value)


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
            self._globalndim = self.ndim
        return self._globalndim


    #Custom reduction method implementations
    def max(self, **kwargs):
        self.check_reduction_parms(**kwargs)
        return Undistributed(np.asarray(self.base.max(**kwargs)),
                             comm=self.comm)


    def mean(self, **kwargs):
        self.check_reduction_parms(**kwargs)
        return Undistributed(np.asarray(self.base.mean(**kwargs)),
                             comm=self.comm)


    def min(self, **kwargs):
        self.check_reduction_parms(**kwargs)
        return Undistributed(np.asarray(self.base.min(**kwargs)),
                             comm=self.comm)


    def std(self, **kwargs):
        self.check_reduction_parms(**kwargs)
        return Undistributed(np.asarray(self.base.std(**kwargs)),
                             comm=self.comm)


    def sum(self, **kwargs):
        self.check_reduction_parms(**kwargs)
        return Undistributed(np.asarray(self.base.sum(**kwargs)),
                             comm=self.comm)


    def collect_data(self):
        return self.__class__(self, comm=self.comm)


    def reshape(self, *args):
        if np.prod(args) != self.globalsize and np.prod(args) > 0:
            raise ValueError("cannot reshape global array of size",
                             self.globalsize,"into shape", tuple(args))
        return self.__class__(self.base.reshape(*args), comm=self.comm)
