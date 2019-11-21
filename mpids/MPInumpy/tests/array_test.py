import unittest
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np
from mpids.MPInumpy.errors import InvalidDistributionError

class ArrayTest(unittest.TestCase):

        def setUp(self):
                self.comm = MPI.COMM_WORLD
                self.data = list(range(10))
                self.data_2d = np.array(list(range(20))).reshape(5,4)
                self.mpi_np_array = mpi_np.array(self.data, comm=self.comm)


        def test_unsupported_distribution(self):
                with self.assertRaises(InvalidDistributionError):
                        mpi_np.array(self.data, comm=self.comm, dist='bananas')
                # Test cases where dim input data != dim distribution
                with self.assertRaises(InvalidDistributionError):
                        mpi_np.array(self.data, comm=self.comm, dist=('*', 'b'))
                with self.assertRaises(InvalidDistributionError):
                        mpi_np.array(self.data, comm=self.comm, dist=('b','b'))


        def test_supported_distributions(self):
                self.assertEqual(mpi_np.array(self.data, dist='u').dist, 'u')
                self.assertEqual(mpi_np.array(self.data_2d, dist=('*', 'b')).dist, ('*', 'b'))
                self.assertEqual(mpi_np.array(self.data_2d, dist=('b','b')).dist, ('b','b'))


        def test_default_behavior(self):
                self.assertTrue(isinstance(self.mpi_np_array, mpi_np.MPIArray))
                self.assertEqual(self.mpi_np_array.comm, self.comm)
                self.assertEqual(self.mpi_np_array.dist, 'b')


        def test_array(self):
                rank = self.comm.Get_rank()
                rank_data_map = {0: [0, 1, 2],
                                 1: [3, 4, 5],
                                 2: [6, 7],
                                 3: [8, 9]}

                self.assertEqual(rank_data_map[rank], self.mpi_np_array.data.tolist())


if __name__ == '__main__':
        unittest.main()
