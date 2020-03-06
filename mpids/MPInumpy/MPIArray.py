from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.errors import ValueError, NotSupportedError
from mpids.MPInumpy.utils import global_to_local_key

__all__ = ['MPIArray']

"""
    Abstract base numpy array subclass for the individual distributions.
    See mpids.MPInumpy.distributions for implementations.
"""
class MPIArray(np.ndarray):
    """ MPIArray subclass of numpy.ndarray """

    def __new__(cls, local_array, comm=MPI.COMM_WORLD, comm_dims=None,
                comm_coord=None, local_to_global=None):
        """ Create MPIArray from process local array data.

        Parameters
        ----------
        local_array : array_like, numpy array
            Array like data or Numpy array local to each process.
        comm : MPI Communicator, optional
            MPI process communication object.  If none specified
            defaults to MPI.COMM_WORLD
        comm_dims: list
            Specified dimensions of processes in cartesian grid
            for communicator.
        comm_coord : list
            Rank/Procses cartesian coordinate in communicator
            process grid.
        local_to_global: dict, None
            Dictionary specifying global index start/end of data by axis.
            Format:
                key, value = axis, [inclusive start, exclusive end)
                {0: (start_index, end_index),
                 1: (start_index, end_index),
                 ...}

        Returns
        -------
        MPIArray : numpy.ndarray sub class
        """
        if not isinstance(local_array, np.ndarray):
            local_array = np.asarray(local_array)

        obj = local_array.view(cls)
        obj.comm = comm
        obj.comm_dims = comm_dims
        obj.comm_coord = comm_coord
        obj.local_to_global = local_to_global
        return obj


    def __init__(self, *args, **kwargs):
        #Initialize unique properties
        self._globalshape = None
        self._globalsize = None
        self._globalnbytes = None
        self._globalndim = None


    def __array_finalize__(self, obj):
        if obj is None: return
        self.comm = getattr(obj, 'comm', None)
        self.comm_dims = getattr(obj, 'comm_dims', None)
        self.comm_coord = getattr(obj, 'comm_coord', None)
        self.local_to_global = getattr(obj, 'local_to_global', None)
        self._globalshape = getattr(obj, '_globalshape', None)
        self._globalsize = getattr(obj, '_globalsize', None)
        self._globalnbytes = getattr(obj, '_globalnbytes', None)
        self._globalndim = getattr(obj, '_globalndim', None)


    def __iter__(self):
        return self.base.__iter__()


    def __getitem__(self, key):
        raise NotImplementedError(
            "Implement a custom __getitem__ method")


    def __repr__(self):
        return '{}(globalsize={}, globalshape={}, dist={}, dtype={})' \
                   .format('MPIArray',
                           getattr(self, 'globalsize', None),
                           getattr(self, 'globalshape', None),
                           getattr(self, 'dist', None),
                           getattr(self, 'dtype', None))


    def __setitem__(self, key, value):
        raise NotImplementedError(
            "Implement a custom __setitem__ method")


    def __str__(self):
        return self.base.__str__()


    #Unique properties to MPIArray
    @property
    def dist(self):
        """ Specified distribution of data among processes.

        Returns
        -------
        dist : str
            Default value 'b' : Block
            Supported types:
                'b' : Block
                'r' : Replicated
        """
        raise NotImplementedError("Define a distribution")


    @property
    def globalshape(self):
        """ Combined shape of distributed array.

        Returns
        -------
        globalshape : tuple
        """
        raise NotImplementedError("Define a globalshape implmentation")


    @property
    def globalsize(self):
        """ Combined size of distributed array.

        Returns
        -------
        globalsize : int
        """
        raise NotImplementedError("Define a globalsize implmentation")


    @property
    def globalnbytes(self):
        """ Combined number of bytes of distributed array.

        Returns
        -------
        globalnbytes : int
        """
        raise NotImplementedError("Define a globalnbytes implmentation")


    @property
    def globalndim(self):
        """ Combined number of dimensions of distributed array.

        Returns
        -------
        globalndim : int
        """
        raise NotImplementedError("Define a globalndim implmentation")


    @property
    def local(self):
        """ Base ndarray object local to each process.

        Returns
        -------
        MPIArray.base : numpy.ndarray
        """
        return self.base


    #Custom reduction method implementations
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
            replicated(copies on all procs) distribution.
        """
        raise NotImplementedError("Implement a custom max method")


    def mean(self, **kwargs):
        """ Mean of array elements in distributed matrix over a
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
            MPIArray with mean values along specified axis with
            replicated(copies on all procs) distribution.
        """
        raise NotImplementedError("Implement a custom mean method")


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
            replicated(copies on all procs) distribution.
        """
        raise NotImplementedError("Implement a custom min method")


    def std(self, **kwargs):
        """ Standard deviation of array elements in distributed matrix
        over a given axis.

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
            MPIArray with std values along specified axis with
            replicated(copies on all procs) distribution.
        """
        raise NotImplementedError("Implement a custom std method")


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
            replicated(copies on all procs) distribution.
        """
        raise NotImplementedError("Implement a custom sum method")


    def check_reduction_parms(self, axis=None, dtype=None, out=None):
        if axis is not None and axis > self.ndim - 1:
            raise ValueError("'axis' entry is out of bounds")
        if out is not None:
            raise NotSupportedError("'out' field not supported")
        return


    #General methods
    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        """ Cast to specified data type.
        Parameters
        ----------
        dtype : data-type
            Desired casted array data-type.
        order: {'K','A','C','F'}, optional
            Specified memory layout of the array.
        casting : {‘no’, ‘equiv’, ‘safe’, ‘same_kind’, ‘unsafe’}, optional
            Controls what kind of data casting may occur.
            See docstring for np.ndarray.astype
        subok : bool, optional
            Default 'True' returned array will be forced to be
            base-class array, if 'True' then sub-classes will be
            passed-through.
        copy : bool, optional
            Default 'True' results in copied object, if 'False' copy
            only made when base class '__array__' returns a copy.
        Returns
        -------
        MPIArray : numpy.ndarray sub class
            Distributed among processes.
        """
        local_casted_result = self.base.astype(dtype,
                                               order=order,
                                               casting=casting,
                                               subok=subok,
                                               copy=copy)
        return self.__class__(local_casted_result,
                              comm=self.comm,
                              comm_dims=self.comm_dims,
                              comm_coord=self.comm_coord,
                              local_to_global=self.local_to_global)


    def collect_data(self):
        """ Collect/Reconstruct distributed array.

        Parameters
        ----------
        None

        Returns
        -------
        MPIArray : numpy.ndarray sub class
            Replicated(resconstructed) MPIArray.
        """
        raise NotImplementedError(
            "Implement a method to collect distributed array")


    def reshape(self, *args):
        """ Reshape distributed array.

        Parameters
        ----------
        new_shape : int, tuple of ints
            Desired shape that's compatible with the previous one.  To be
            compatible the total number of elements(product of shape) must
            be equal to the existing shape.

        Returns
        -------
        MPIArray : numpy.ndarray sub class
            Distributed MPIArray with new shape.
        """
        raise NotImplementedError(
            "Implement a method to reshape distributed array")
