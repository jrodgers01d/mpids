from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.errors import ValueError, NotSupportedError

"""
    Abstract base numpy array subclass for the individual distributions.
"""
class MPIArray(np.ndarray):
        """ MPIArray subclass of numpy.ndarray """

        def __new__(cls, array_data, dtype=None, copy=True, order=None,
                    subok=False, ndmin=0, comm=MPI.COMM_WORLD, comm_dims=None,
                    comm_coord=None, local_to_global=None):
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
                comm_dims: list
                        Specified dimensions of processes in cartesian grid
                        for communicator.
                comm_coord : list
                        Rank/Procses cartesian coordinate in communicator
                        process grid.
                local_to_global: dict, None
                        Dictionary specifying global index start/end of data by axis.
                        Format:
                                key, value = axis, (inclusive start, exclusive end)
                                {0: [start_index, end_index),
                                 1: [start_index, end_index),
                                 ...}

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
                obj.comm_dims = comm_dims
                obj.comm_coord = comm_coord
                obj.local_to_global = local_to_global
                return obj


        def __array_finalize__(self, obj):
                if obj is None: return
                self.comm = getattr(obj, 'comm', None)
                self.comm_dims = getattr(obj, 'comm_dims', None)
                self.comm_coord = getattr(obj, 'comm_coord', None)
                self.local_to_global = getattr(obj, 'local_to_global', None)


        def __repr__(self):
                return '{}(globalsize={}, globalshape={}, dist={}, dtype={})' \
                       .format('MPIArray',
                               getattr(self, 'globalsize', None),
                               list(getattr(self, 'globalshape', None)),
                               getattr(self, 'dist', None),
                               getattr(self, 'dtype', None))

        #Unique properties to MPIArray
        @property
        def dist(self):
                raise NotImplementedError("Define a distribution")


        @property
        def globalsize(self):
                raise NotImplementedError("Define a globalsize implmentation")


        @property
        def globalnbytes(self):
                raise NotImplementedError("Define a globalnbytes implmentation")


        @property
        def globalshape(self):
                raise NotImplementedError("Define a globalshape implmentation")


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
                        undistributed(copies on all procs) distribution.
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
                        undistributed(copies on all procs) distribution.
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
                        undistributed(copies on all procs) distribution.
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
                        undistributed(copies on all procs) distribution.
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
                        undistributed(copies on all procs) distribution.
                """
                raise NotImplementedError("Implement a custom sum method")


        def check_reduction_parms(self, axis=None, dtype=None, out=None):
                if axis is not None and axis > self.ndim - 1:
                        raise ValueError("'axis' entry is out of bounds")
                if out is not None:
                        raise NotSupportedError("'out' field not supported")
                return


        def custom_reduction(self, operation, local_red, axis=None,
                               dtype=None, out=None):
                raise NotImplementedError("Implement a custom reduction")
