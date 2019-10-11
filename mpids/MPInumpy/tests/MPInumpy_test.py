import unittest
from mpi4py import MPI
from mpids.MPInumpy.MPInumpy import array

class MPIArrayTest(unittest.TestCase):

        def setUp(self):
                self.comm = MPI.COMM_WORLD
                self.data = list(range(10))

        def test_array(self):
                mpi_np_array = array(self.data, comm=self.comm, distribution='bananas')
                self.assertEqual(None, mpi_np_array)


                rank = self.comm.Get_rank()
                rank_data_map = {0: [0, 1],
                                 1: [2, 3, 4],
                                 2: [5, 6],
                                 3: [7, 8, 9]}

                mpi_np_array = array(self.data, comm=self.comm)
                self.assertEqual(rank_data_map[rank], mpi_np_array.data)


if __name__ == '__main__':
        unittest.main()
