from mpi4py import MPI
from mpids.MPInumpy.MPIArray import MPIArray
from mpids.MPInumpy.utils import determine_local_data

def array(array_data, dtype=None, copy=True, order=None, subok=False, ndmin=0,
          comm=MPI.COMM_WORLD, dist='b'):
        """ Create MPInumpyArray Object on all procs in comm.
            See docstring for mpids.MPInumpy.MPIArray

        Parameters
        ----------
        array_data : array_like
                Array like data to be distributed among processes.
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

        Returns
        -------
        MPIArray : numpy.ndarray sub class
                Distributed among processes.
        """
        size = comm.Get_size()
        rank = comm.Get_rank()

        local_data = determine_local_data(array_data, dist, size, rank)

        return MPIArray(local_data,
                        dtype=dtype,
                        copy=copy,
                        order=order,
                        subok=subok,
                        ndmin=ndmin,
                        comm=comm)
