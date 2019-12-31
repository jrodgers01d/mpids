import unittest
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np
from mpids.MPInumpy.errors import IndexError

class MPIArraySlicingDefaultTest(unittest.TestCase):

        def create_setUp_parms(self):
                parms = {}
                parms['comm'] = MPI.COMM_WORLD
                # Default distribution
                parms['dist'] = 'b'
                np_data = (np.array(list(range(25))).reshape(5,5))
                parms['data'] = np_data.tolist()
                return parms


        def setUp(self):
                parms = self.create_setUp_parms()
                self.comm = parms.get('comm')
                self.dist = parms.get('dist')
                self.data = parms.get('data')

                self.mpi_array = mpi_np.array(self.data, comm=self.comm, dist=self.dist)


        def test_custom_slicing_exceptions_behavior(self):
                #Check Index Error for index outside global range
                with self.assertRaises(IndexError):
                        shape_axis0, _ = self.mpi_array.globalshape
                        self.mpi_array[shape_axis0 + 1]
                with self.assertRaises(IndexError):
                        _, shape_axis1 = self.mpi_array.globalshape
                        self.mpi_array[shape_axis1 + 1]

                #Check Index Error for index with more dims than distributed array
                with self.assertRaises(IndexError):
                        ndims = len(self.mpi_array.globalshape)
                        overslice = tuple([0] * (ndims + 1))
                        self.mpi_array[overslice]


class MPIArraySlicingUndistributedTest(MPIArraySlicingDefaultTest):

        def create_setUp_parms(self):
                parms = {}
                parms['comm'] = MPI.COMM_WORLD
                # Undistributed distribution
                parms['dist'] = 'u'
                np_data = (np.array(list(range(25))).reshape(5,5))
                parms['data'] = np_data.tolist()
                return parms


class MPIArraySlicingColBlockTest(MPIArraySlicingDefaultTest):

        def create_setUp_parms(self):
                parms = {}
                parms['comm'] = MPI.COMM_WORLD
                # Column block distribution
                parms['dist'] = ('*', 'b')
                np_data = (np.array(list(range(25))).reshape(5,5))
                parms['data'] = np_data.tolist()
                return parms


class MPIArraySlicingBlockBlockTest(MPIArraySlicingDefaultTest):

        def create_setUp_parms(self):
                parms = {}
                parms['comm'] = MPI.COMM_WORLD
                # Block block distribution
                parms['dist'] = ('b', 'b')
                np_data = (np.array(list(range(25))).reshape(5,5))
                parms['data'] = np_data.tolist()
                return parms


if __name__ == '__main__':
        unittest.main()
