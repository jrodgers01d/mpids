from mpi4py import MPI
import numpy as np

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


        # #Unique properties to MPIArray
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
