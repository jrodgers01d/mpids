import unittest
import numpy as np
from mpi4py import MPI
from mpids.MPInumpy.utils import distribution_to_dimensions, get_block_index, \
                                 get_cart_coords,  determine_local_data
from mpids.MPInumpy.errors import InvalidDistributionError

class UtilsTest(unittest.TestCase):

        def setUp(self):
                self.procs = 3
                self.ranks = [0, 1, 2]
                self.data = list(range(10))
                self.data_2d = np.array(list(range(20))).reshape(5,4)
                self.data_length = len(self.data)
                self.default_dist = 'b'


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

        def test_distribution_to_dimensions(self):
                self.assertEqual(1,
                    distribution_to_dimensions(self.default_dist, self.procs))
                self.assertEqual(1,
                    distribution_to_dimensions(('b','*'), self.procs))
                self.assertEqual(2, distribution_to_dimensions(('b', 'b'), self.procs))
                self.assertEqual(2, distribution_to_dimensions(['b', 'b'], self.procs))
                self.assertEqual([1, self.procs],
                    distribution_to_dimensions(('*','b'), self.procs))

                # Check unsupported distributions
                with self.assertRaises(InvalidDistributionError):
                        distribution_to_dimensions(('b','b','x'), self.procs)
                with self.assertRaises(InvalidDistributionError):
                        distribution_to_dimensions(('','b'), self.procs)
                with self.assertRaises(InvalidDistributionError):
                        distribution_to_dimensions(('u','u'), self.procs)

        def test_local_data_default_row_block_distribution(self):
                local_data_rank0 = self.data[0:4]
                local_data_rank1 = self.data[4:7]
                local_data_rank2 = self.data[7:10]

                self.assertEqual(local_data_rank0, determine_local_data(self.data,
                                                                  self.default_dist,
                                                                  self.procs,
                                                                  self.ranks[0]))

                self.assertEqual(local_data_rank1, determine_local_data(self.data,
                                                                  self.default_dist,
                                                                  self.procs,
                                                                  self.ranks[1]))

                self.assertEqual(local_data_rank2, determine_local_data(self.data,
                                                                  self.default_dist,
                                                                  self.procs,
                                                                  self.ranks[2]))


        def test_local_data_block_block_distribution(self):
                procs = 4
                dist = ('b', 'b')
                local_data_rank0 = self.data_2d[[slice(0, 3), slice(0, 2)]]
                local_data_rank1 = self.data_2d[[slice(0, 3), slice(2, 4)]]
                local_data_rank2 = self.data_2d[[slice(3, 5), slice(0, 2)]]
                local_data_rank3 = self.data_2d[[slice(3, 5), slice(2, 4)]]
                self.assertTrue(np.alltrue(
                    local_data_rank0 == determine_local_data(self.data_2d, dist, procs, 0)))
                self.assertTrue(np.alltrue(
                    local_data_rank1 == determine_local_data(self.data_2d, dist, procs, 1)))
                self.assertTrue(np.alltrue(
                    local_data_rank2 == determine_local_data(self.data_2d, dist, procs, 2)))
                self.assertTrue(np.alltrue(
                    local_data_rank3 == determine_local_data(self.data_2d, dist, procs, 3)))


        def test_local_data_col_block_distribution(self):
                procs = 4
                dist = ('*', 'b')
                local_data_rank0 = self.data_2d[:, slice(0, 1)]
                local_data_rank1 = self.data_2d[:, slice(1, 2)]
                local_data_rank2 = self.data_2d[:, slice(2, 3)]
                local_data_rank3 = self.data_2d[:, slice(4, 4)]
                self.assertTrue(np.alltrue(
                    local_data_rank0 == determine_local_data(self.data_2d, dist, procs, 0)))
                self.assertTrue(np.alltrue(
                    local_data_rank1 == determine_local_data(self.data_2d, dist, procs, 1)))
                self.assertTrue(np.alltrue(
                    local_data_rank2 == determine_local_data(self.data_2d, dist, procs, 2)))
                self.assertTrue(np.alltrue(
                    local_data_rank3 == determine_local_data(self.data_2d, dist, procs, 3)))

        def test_local_data_undistributed_distribution(self):
                procs = 4
                dist = 'u'

                # 1-D Data
                local_data_rank0 = self.data
                local_data_rank1 = self.data
                local_data_rank2 = self.data
                local_data_rank3 = self.data
                self.assertTrue(np.alltrue(
                    local_data_rank0 == determine_local_data(self.data, dist, procs, 0)))
                self.assertTrue(np.alltrue(
                    local_data_rank1 == determine_local_data(self.data, dist, procs, 1)))
                self.assertTrue(np.alltrue(
                    local_data_rank2 == determine_local_data(self.data, dist, procs, 2)))
                self.assertTrue(np.alltrue(
                    local_data_rank3 == determine_local_data(self.data, dist, procs, 3)))

                # 2-D Data
                local_data_rank0 = self.data_2d
                local_data_rank1 = self.data_2d
                local_data_rank2 = self.data_2d
                local_data_rank3 = self.data_2d
                self.assertTrue(np.alltrue(
                    local_data_rank0 == determine_local_data(self.data_2d, dist, procs, 0)))
                self.assertTrue(np.alltrue(
                    local_data_rank1 == determine_local_data(self.data_2d, dist, procs, 1)))
                self.assertTrue(np.alltrue(
                    local_data_rank2 == determine_local_data(self.data_2d, dist, procs, 2)))
                self.assertTrue(np.alltrue(
                    local_data_rank3 == determine_local_data(self.data_2d, dist, procs, 3)))


if __name__ == '__main__':
        unittest.main()
