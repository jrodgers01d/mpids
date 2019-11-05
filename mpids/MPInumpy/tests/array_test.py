import unittest
from mpi4py import MPI
import mpids.MPInumpy as mpi_np
from mpids.MPInumpy.errors import InvalidDistributionError

class ArrayTest(unittest.TestCase):

        def setUp(self):
                self.comm = MPI.COMM_WORLD
                self.data = list(range(10))
                self.mpi_np_array = mpi_np.array(self.data, comm=self.comm)


        def test_unsupported_distribution(self):
            with self.assertRaises(InvalidDistributionError):
                    mpi_np.array(self.data, comm=self.comm, dist='bananas')


        def test_array(self):
                rank = self.comm.Get_rank()
                rank_data_map = {0: [0, 1, 2],
                                 1: [3, 4, 5],
                                 2: [6, 7],
                                 3: [8, 9]}

                self.assertEqual(rank_data_map[rank], self.mpi_np_array.data.tolist())


if __name__ == '__main__':
        unittest.main()
