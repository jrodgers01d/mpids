import unittest
import numpy as np
from mpi4py import MPI
from mpids.MPInumpy.utils import get_cart_coords, get_block_index, get_local_data

class UtilsTest(unittest.TestCase):

        def setUp(self):
                self.procs = 3
                self.ranks = [0, 1, 2]
                self.data = list(range(10))
                self.data_length = len(self.data)


        def test_get_cart_coords(self):
                procs = 4
                dist = 'b'
                dims = MPI.Compute_dims(procs, len(dist))
                self.assertEqual([4], dims)
                self.assertEqual([0], get_cart_coords(dims, procs, 0))
                self.assertEqual([1], get_cart_coords(dims, procs, 1))
                self.assertEqual([2], get_cart_coords(dims, procs, 2))
                self.assertEqual([3], get_cart_coords(dims, procs, 3))

                procs = 4
                dist = ('b', 'b')
                dims = MPI.Compute_dims(procs, len(dist))
                self.assertEqual([2, 2], dims)
                self.assertEqual([0, 0], get_cart_coords(dims, procs, 0))
                self.assertEqual([0, 1], get_cart_coords(dims, procs, 1))
                self.assertEqual([1, 0], get_cart_coords(dims, procs, 2))
                self.assertEqual([1, 1], get_cart_coords(dims, procs, 3))

                procs = 3
                dist = ('b', 'b')
                dims = MPI.Compute_dims(procs, len(dist))
                self.assertEqual([3, 1], dims)
                self.assertEqual([0, 0], get_cart_coords(dims, procs, 0))
                self.assertEqual([1, 0], get_cart_coords(dims, procs, 1))
                self.assertEqual([2, 0], get_cart_coords(dims, procs, 2))


        def test_get_block_index(self):
                self.assertEqual((0, 4), get_block_index(self.data_length, 3, 0))
                self.assertEqual((4, 7), get_block_index(self.data_length, 3, 1))
                self.assertEqual((7, 10), get_block_index(self.data_length, 3, 2))


        def test_local_data_default_distribution(self):
                local_data_rank0 = self.data[0:4]
                local_data_rank1 = self.data[4:7]
                local_data_rank2 = self.data[7:10]

                self.assertEqual(local_data_rank0, get_local_data(self.data,
                                                                  'b',
                                                                  self.procs,
                                                                  self.ranks[0]))

                self.assertEqual(local_data_rank1, get_local_data(self.data,
                                                                  'b',
                                                                  self.procs,
                                                                  self.ranks[1]))

                self.assertEqual(local_data_rank2, get_local_data(self.data,
                                                                  'b',
                                                                  self.procs,
                                                                  self.ranks[2]))

                procs = 4
                dist = ('b', 'b')
                data_2d = np.array(list(range(20))).reshape(5,4)
                local_data_rank0 = data_2d[[slice(0, 3), slice(0, 2)]]
                local_data_rank1 = data_2d[[slice(0, 3), slice(2, 4)]]
                local_data_rank2 = data_2d[[slice(3, 5), slice(0, 2)]]
                local_data_rank3 = data_2d[[slice(3, 5), slice(2, 4)]]
                self.assertTrue(np.alltrue(
                    local_data_rank0 == get_local_data(data_2d, dist, procs, 0)))
                self.assertTrue(np.alltrue(
                    local_data_rank1 == get_local_data(data_2d, dist, procs, 1)))
                self.assertTrue(np.alltrue(
                    local_data_rank2 == get_local_data(data_2d, dist, procs, 2)))
                self.assertTrue(np.alltrue(
                    local_data_rank3 == get_local_data(data_2d, dist, procs, 3)))


if __name__ == '__main__':
        unittest.main()
