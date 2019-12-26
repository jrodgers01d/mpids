import unittest
import numpy as np
from mpi4py import MPI
from mpids.MPInumpy.utils import *
from mpids.MPInumpy.errors import InvalidDistributionError


class UtilsDistributionIndependentTest(unittest.TestCase):

        def test_distribution_checks(self):
                undist = 'u'
                row_block = 'b'
                row_block_alt = ('b', '*')
                col_block = ('*', 'b')
                block_block = ('b', 'b')

                self.assertTrue(is_undistributed(undist))
                self.assertFalse(is_undistributed(row_block))
                self.assertFalse(is_undistributed(row_block_alt))
                self.assertFalse(is_undistributed(col_block))
                self.assertFalse(is_undistributed(block_block))

                self.assertTrue(is_row_block_distributed(row_block))
                self.assertTrue(is_row_block_distributed(row_block_alt))
                self.assertFalse(is_row_block_distributed(undist))
                self.assertFalse(is_row_block_distributed(col_block))
                self.assertFalse(is_row_block_distributed(block_block))

                self.assertTrue(is_column_block_distributed(col_block))
                self.assertFalse(is_column_block_distributed(undist))
                self.assertFalse(is_column_block_distributed(row_block))
                self.assertFalse(is_column_block_distributed(row_block_alt))
                self.assertFalse(is_column_block_distributed(block_block))

                self.assertTrue(is_block_block_distributed(block_block))
                self.assertFalse(is_block_block_distributed(undist))
                self.assertFalse(is_block_block_distributed(row_block))
                self.assertFalse(is_block_block_distributed(row_block_alt))
                self.assertFalse(is_block_block_distributed(col_block))


        def test_get_block_index(self):
                data_length = 10
                num_procs = 3
                rank_block_map = {0: (0, 4),
                                  1: (4, 7),
                                  2: (7, 10)}
                self.assertEqual(rank_block_map[0],
                                     get_block_index(data_length, num_procs, 0))
                self.assertEqual(rank_block_map[1],
                                     get_block_index(data_length, num_procs, 1))
                self.assertEqual(rank_block_map[2],
                                     get_block_index(data_length, num_procs, 2))


        def test_distribution_to_dimensions_with_invalid_distributions(self):
                procs = 4
                # Check unsupported distributions
                with self.assertRaises(InvalidDistributionError):
                        distribution_to_dimensions(('b','b','x'), procs)
                with self.assertRaises(InvalidDistributionError):
                        distribution_to_dimensions(('','b'), procs)
                with self.assertRaises(InvalidDistributionError):
                        distribution_to_dimensions(('u','u'), procs)


class UtilsDefaultTest(unittest.TestCase):

        def create_setUp_parms(self):
                parms = {}
                parms['procs'] = MPI.COMM_WORLD.Get_size()
                parms['rank'] = MPI.COMM_WORLD.Get_rank()
                parms['data'] = list(range(10))
                parms['data_2d'] = np.array(list(range(20))).reshape(5,4)
                # Default distribution
                parms['dist'] = 'b'
                parms['comm_dims'] = [parms['procs']]
                parms['comm_coord'] = [parms['rank']]
                parms['dist_to_dims'] = 1
                parms['single_dim_support'] = True
                rank_local_data_map = {0 : parms['data'][0:3],
                                       1 : parms['data'][3:6],
                                       2 : parms['data'][6:8],
                                       3 : parms['data'][8:10]}
                parms['local_data'] = rank_local_data_map[parms['rank']]
                rank_local_data_2d_map = {0 : parms['data_2d'][0:2],
                                          1 : parms['data_2d'][2:3],
                                          2 : parms['data_2d'][3:4],
                                          3 : parms['data_2d'][4:5]}
                parms['local_data_2d'] = rank_local_data_2d_map[parms['rank']]
                local_to_global_map = {0 : {0 : (0, 3)},
                                       1 : {0 : (3, 6)},
                                       2 : {0 : (6, 8)},
                                       3 : {0 : (8, 10)}}
                parms['local_to_global'] = local_to_global_map[parms['rank']]
                local_to_global_2d_map = {0 : {0 : (0, 2)},
                                          1 : {0 : (2, 3)},
                                          2 : {0 : (3, 4)},
                                          3 : {0 : (4, 5)}}
                parms['local_to_global_2d'] = local_to_global_2d_map[parms['rank']]
                return parms


        def setUp(self):
                parms = self.create_setUp_parms()
                self.procs = parms.get('procs')
                self.rank = parms.get('rank')
                self.data = parms.get('data')
                self.data_2d = parms.get('data_2d')
                self.dist = parms.get('dist')
                self.comm_dims = parms.get('comm_dims')
                self.comm_coord = parms.get('comm_coord')
                self.dist_to_dims = parms.get('dist_to_dims')
                self.single_dim_support = parms.get('single_dim_support')
                self.local_data = parms.get('local_data')
                self.local_data_2d = parms.get('local_data_2d')
                self.local_to_global = parms.get('local_to_global')
                self.local_to_global_2d = parms.get('local_to_global_2d')


        def test_get_comm_dims(self):
                self.assertEqual(self.comm_dims, get_comm_dims(self.procs, self.dist))


        def test_get_cart_coords(self):
                self.assertEqual(self.comm_coord,
                                 get_cart_coords(self.comm_dims, self.procs, self.rank))


        def test_distribution_to_dimensions(self):
                #Does not apply to undistributed
                if is_undistributed(self.dist):
                        return
                self.assertEqual(self.dist_to_dims,
                                 distribution_to_dimensions(self.dist, self.procs))


        def test_determine_local_data(self):
                # 1-D Data
                if self.single_dim_support:
                        self.assertEqual((self.local_data, self.local_to_global),
                                         determine_local_data(self.data,
                                                              self.dist,
                                                              self.comm_dims,
                                                              self.comm_coord))
                else:
                        # Test cases where dim input data != dim distribution
                        with self.assertRaises(InvalidDistributionError):
                                determine_local_data(self.data,
                                                     self.dist,
                                                     self.comm_dims,
                                                     self.comm_coord)

                # 2-D Data
                local_data_2d, local_to_global = \
                        determine_local_data(self.data_2d,
                                             self.dist,
                                             self.comm_dims,
                                             self.comm_coord)

                self.assertTrue(np.alltrue(self.local_data_2d == local_data_2d))
                self.assertEqual(self.local_to_global_2d, local_to_global)



class UtilsUndistributedTest(UtilsDefaultTest):

        def create_setUp_parms(self):
                parms = {}
                parms['procs'] = MPI.COMM_WORLD.Get_size()
                parms['rank'] = MPI.COMM_WORLD.Get_rank()
                parms['data'] = list(range(10))
                parms['data_2d'] = np.array(list(range(20))).reshape(5,4)
                # Undistributed distribution
                parms['dist'] = 'u'
                parms['comm_dims'] = None
                parms['comm_coord'] = None
                parms['single_dim_support'] = True
                parms['local_data'] = parms['data']
                parms['local_data_2d'] = parms['data_2d']
                parms['local_to_global'] = None
                parms['local_to_global_2d'] = None
                return parms


class UtilsAltRowBlockTest(UtilsDefaultTest):

        def create_setUp_parms(self):
                parms = {}
                parms['procs'] = MPI.COMM_WORLD.Get_size()
                parms['rank'] = MPI.COMM_WORLD.Get_rank()
                parms['data'] = list(range(10))
                parms['data_2d'] = np.array(list(range(20))).reshape(5,4)
                # Alternate row block distribution
                parms['dist'] = ('b', '*')
                parms['comm_dims'] = [parms['procs']]
                parms['comm_coord'] = [parms['rank']]
                parms['dist_to_dims'] = 1
                parms['single_dim_support'] = True
                rank_local_data_map = {0 : parms['data'][0:3],
                                       1 : parms['data'][3:6],
                                       2 : parms['data'][6:8],
                                       3 : parms['data'][8:10]}
                parms['local_data'] = rank_local_data_map[parms['rank']]
                rank_local_data_2d_map = {0 : parms['data_2d'][0:2],
                                          1 : parms['data_2d'][2:3],
                                          2 : parms['data_2d'][3:4],
                                          3 : parms['data_2d'][4:5]}
                parms['local_data_2d'] = rank_local_data_2d_map[parms['rank']]
                local_to_global_map = {0 : {0 : (0, 3)},
                                       1 : {0 : (3, 6)},
                                       2 : {0 : (6, 8)},
                                       3 : {0 : (8, 10)}}
                parms['local_to_global'] = local_to_global_map[parms['rank']]
                local_to_global_2d_map = {0 : {0 : (0, 2)},
                                          1 : {0 : (2, 3)},
                                          2 : {0 : (3, 4)},
                                          3 : {0 : (4, 5)}}
                parms['local_to_global_2d'] = local_to_global_2d_map[parms['rank']]
                return parms


class UtilsColBlockTest(UtilsDefaultTest):

        def create_setUp_parms(self):
                parms = {}
                parms['procs'] = MPI.COMM_WORLD.Get_size()
                parms['rank'] = MPI.COMM_WORLD.Get_rank()
                parms['data'] = list(range(10))
                parms['data_2d'] = np.array(list(range(20))).reshape(5,4)
                # Column block distribution
                parms['dist'] = ('*', 'b')
                parms['comm_dims'] = [1, parms['procs']]
                parms['comm_coord'] = [0, parms['rank']]
                parms['dist_to_dims'] = [1, parms['procs']]
                parms['single_dim_support'] = False
                rank_local_data_2d_map = {0 : parms['data_2d'][:, slice(0, 1)],
                                          1 : parms['data_2d'][:, slice(1, 2)],
                                          2 : parms['data_2d'][:, slice(2, 3)],
                                          3 : parms['data_2d'][:, slice(3, 4)]}
                parms['local_data_2d'] = rank_local_data_2d_map[parms['rank']]
                parms['local_to_global'] = None
                local_to_global_2d_map = {0 : {0 : (0, 5), 1 : (0, 1)},
                                          1 : {0 : (0, 5), 1 : (1, 2)},
                                          2 : {0 : (0, 5), 1 : (2, 3)},
                                          3 : {0 : (0, 5), 1 : (3, 4)}}
                parms['local_to_global_2d'] = local_to_global_2d_map[parms['rank']]

                return parms


class UtilsBlockBlockTest(UtilsDefaultTest):

        def create_setUp_parms(self):
                parms = {}
                parms['procs'] = MPI.COMM_WORLD.Get_size()
                parms['rank'] = MPI.COMM_WORLD.Get_rank()
                parms['data'] = list(range(10))
                parms['data_2d'] = np.array(list(range(20))).reshape(5,4)
                # Block block distribution
                parms['dist'] = ('b', 'b')
                parms['comm_dims'] = [2, 2]
                rank_coord_map = {0: [0, 0],
                                  1: [0, 1],
                                  2: [1, 0],
                                  3: [1, 1]}
                parms['comm_coord'] = rank_coord_map[parms['rank']]
                parms['dist_to_dims'] = 2
                parms['single_dim_support'] = False
                rank_local_data_2d_map = {0 : parms['data_2d'][0:3, 0:2],
                                          1 : parms['data_2d'][0:3, 2:4],
                                          2 : parms['data_2d'][3:5, 0:2],
                                          3 : parms['data_2d'][3:5, 2:4]}
                parms['local_data_2d'] = rank_local_data_2d_map[parms['rank']]
                parms['local_to_global'] = None
                local_to_global_2d_map = {0 : {0 : (0, 3), 1 : (0, 2)},
                                          1 : {0 : (0, 3), 1 : (2, 4)},
                                          2 : {0 : (3, 5), 1 : (0, 2)},
                                          3 : {0 : (3, 5), 1 : (2, 4)}}
                parms['local_to_global_2d'] = local_to_global_2d_map[parms['rank']]

                return parms


        def test_get_comm_non_square_result(self):
            procs = 3
            self.assertEqual([procs, 1], get_comm_dims(procs, self.dist))

            rank_coord_map = {0: [0, 0],
                              1: [1, 0],
                              2: [2, 0]}
            dims = get_comm_dims(procs, self.dist)
            if self.rank < procs:
                self.assertEqual(rank_coord_map[self.rank],
                                 get_cart_coords(dims, procs, self.rank))


if __name__ == '__main__':
        unittest.main()
