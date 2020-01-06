import unittest
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np
from mpids.MPInumpy.distributions import *
from mpids.MPInumpy.errors import InvalidDistributionError

class ArrayCreationErrorsPropegatedTest(unittest.TestCase):
    def test_unsupported_distribution(self):
        data = np.array(list(range(10)))
        comm = MPI.COMM_WORLD
        with self.assertRaises(InvalidDistributionError):
            mpi_np.array(data, comm=comm, dist='bananas')
        # Test cases where dim input data != dim distribution
        with self.assertRaises(InvalidDistributionError):
            mpi_np.array(data, comm=comm, dist=('*', 'b'))
        with self.assertRaises(InvalidDistributionError):
            mpi_np.array(data, comm=comm, dist=('b','b'))


class ArrayDefaultTest(unittest.TestCase):
    """ MPIArray creation routine.
        See mpids.MPInumpy.tests.MPIArray_test for more exhaustive evaluation
    """

    def create_setUp_parms(self):
        parms = {}
        data = np.array(list(range(20))).reshape(5,4)
        parms['comm'] = MPI.COMM_WORLD
        # Default distribution
        parms['dist'] = 'b'
        parms['dist_class'] = RowBlock
        parms['mpi_np_array'] = mpi_np.array(data, comm=parms['comm'])
        return parms


    def setUp(self):
        parms = self.create_setUp_parms()
        self.comm = parms['comm']
        self.dist = parms['dist']
        self.dist_class = parms['dist_class']
        self.mpi_np_array = parms['mpi_np_array']


    def test_return_behavior(self):
        self.assertTrue(isinstance(self.mpi_np_array, mpi_np.MPIArray))
        self.assertTrue(isinstance(self.mpi_np_array, self.dist_class))
        self.assertEqual(self.mpi_np_array.comm, self.comm)
        self.assertEqual(self.mpi_np_array.dist, self.dist)


class ArrayUndistributedTest(ArrayDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        data = np.array(list(range(20))).reshape(5,4)
        parms['comm'] = MPI.COMM_WORLD
        # Undistributed distribution
        parms['dist'] = 'u'
        parms['dist_class'] = Undistributed
        parms['mpi_np_array'] = mpi_np.array(data,
                                             comm=parms['comm'],
                                             dist=parms['dist'])
        return parms


class EmptyDefaultTest(unittest.TestCase):

    def create_setUp_parms(self):
        parms = {}
        shape = (5, 4)
        parms['comm'] = MPI.COMM_WORLD
        # Default distribution
        parms['dist'] = 'b'
        parms['dist_class'] = RowBlock
        parms['mpi_np_empty'] = mpi_np.empty(shape,
                                             comm=parms['comm'],
                                             dist=parms['dist'])
        return parms


    def setUp(self):
        parms = self.create_setUp_parms()
        self.comm = parms['comm']
        self.dist = parms['dist']
        self.dist_class = parms['dist_class']
        self.mpi_np_empty = parms['mpi_np_empty']


    def test_return_behavior(self):
        self.assertTrue(isinstance(self.mpi_np_empty, mpi_np.MPIArray))
        self.assertTrue(isinstance(self.mpi_np_empty, self.dist_class))
        self.assertEqual(self.mpi_np_empty.comm, self.comm)
        self.assertEqual(self.mpi_np_empty.dist, self.dist)


class EmptyUndistributedTest(EmptyDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        shape = (5, 4)
        parms['comm'] = MPI.COMM_WORLD
        # Undistributed distribution
        parms['dist'] = 'u'
        parms['dist_class'] = Undistributed
        parms['mpi_np_empty'] = mpi_np.empty(shape,
                                             comm=parms['comm'],
                                             dist=parms['dist'])
        return parms


class OnesDefaultTest(unittest.TestCase):

    def create_setUp_parms(self):
        parms = {}
        shape = (5, 4)
        parms['comm'] = MPI.COMM_WORLD
        # Default distribution
        parms['dist'] = 'b'
        parms['dist_class'] = RowBlock
        parms['mpi_np_ones'] = mpi_np.ones(shape,
                                            comm=parms['comm'],
                                            dist=parms['dist'])
        return parms


    def setUp(self):
        parms = self.create_setUp_parms()
        self.comm = parms['comm']
        self.dist = parms['dist']
        self.dist_class = parms['dist_class']
        self.mpi_np_ones = parms['mpi_np_ones']


    def test_return_behavior(self):
        self.assertTrue(isinstance(self.mpi_np_ones, mpi_np.MPIArray))
        self.assertTrue(isinstance(self.mpi_np_ones, self.dist_class))
        self.assertEqual(self.mpi_np_ones.comm, self.comm)
        self.assertEqual(self.mpi_np_ones.dist, self.dist)
        self.assertTrue(np.alltrue((self.mpi_np_ones) == (1)))


class OnesUndistributedTest(OnesDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        shape = (5, 4)
        parms['comm'] = MPI.COMM_WORLD
        # Undistributed distribution
        parms['dist'] = 'u'
        parms['dist_class'] = Undistributed
        parms['mpi_np_ones'] = mpi_np.ones(shape,
                                           comm=parms['comm'],
                                           dist=parms['dist'])
        return parms


class ZerosDefaultTest(unittest.TestCase):

    def create_setUp_parms(self):
        parms = {}
        shape = (5, 4)
        parms['comm'] = MPI.COMM_WORLD
        # Default distribution
        parms['dist'] = 'b'
        parms['dist_class'] = RowBlock
        parms['mpi_np_zeros'] = mpi_np.zeros(shape,
                                             comm=parms['comm'],
                                             dist=parms['dist'])
        return parms


    def setUp(self):
        parms = self.create_setUp_parms()
        self.comm = parms['comm']
        self.dist = parms['dist']
        self.dist_class = parms['dist_class']
        self.mpi_np_zeros = parms['mpi_np_zeros']


    def test_return_behavior(self):
        self.assertTrue(isinstance(self.mpi_np_zeros, mpi_np.MPIArray))
        self.assertTrue(isinstance(self.mpi_np_zeros, self.dist_class))
        self.assertEqual(self.mpi_np_zeros.comm, self.comm)
        self.assertEqual(self.mpi_np_zeros.dist, self.dist)
        self.assertTrue(np.alltrue((self.mpi_np_zeros) == (0)))


class ZerosUndistributedTest(ZerosDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        shape = (5, 4)
        parms['comm'] = MPI.COMM_WORLD
        # Undistributed distribution
        parms['dist'] = 'u'
        parms['dist_class'] = Undistributed
        parms['mpi_np_zeros'] = mpi_np.zeros(shape,
                                             comm=parms['comm'],
                                             dist=parms['dist'])
        return parms


if __name__ == '__main__':
    unittest.main()
