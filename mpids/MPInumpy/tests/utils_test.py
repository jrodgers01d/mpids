import unittest
import unittest.mock as mock
import numpy as np
from mpi4py import MPI
from mpids.MPInumpy.utils import *
from mpids.MPInumpy.utils import _format_indexed_result,\
                                 _global_to_local_key_int,  \
                                 _global_to_local_key_slice,\
                                 _global_to_local_key_tuple
from mpids.MPInumpy.errors import IndexError, InvalidDistributionError, \
                                  NotSupportedError


class UtilsDistributionIndependentTest(unittest.TestCase):

        def test_distribution_checks(self):
                undist = 'u'
                row_block = 'b'
                col_block = ('*', 'b')
                block_block = ('b', 'b')

                self.assertTrue(is_undistributed(undist))
                self.assertFalse(is_undistributed(row_block))
                self.assertFalse(is_undistributed(col_block))
                self.assertFalse(is_undistributed(block_block))

                self.assertTrue(is_row_block_distributed(row_block))
                self.assertFalse(is_row_block_distributed(undist))
                self.assertFalse(is_row_block_distributed(col_block))
                self.assertFalse(is_row_block_distributed(block_block))

                self.assertTrue(is_column_block_distributed(col_block))
                self.assertFalse(is_column_block_distributed(undist))
                self.assertFalse(is_column_block_distributed(row_block))
                self.assertFalse(is_column_block_distributed(block_block))

                self.assertTrue(is_block_block_distributed(block_block))
                self.assertFalse(is_block_block_distributed(undist))
                self.assertFalse(is_block_block_distributed(row_block))
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


        def test_global_to_local_key_int(self):
                globalshape = (5, 5)
                local_to_global = {0 : (1, 4), 1 : (1, 4)}

                #Inputs
                global_first_index = 1
                global_second_index = 2
                global_last_index = 3
                global_lower_outside_local_range = 0
                global_upper_outside_local_range = 4
                global_negative_last_index = -2
                global_negative_index_outside_range = -1
                #Expected Results
                local_first_index = 0
                local_second_index = 1
                local_last_index = 2
                local_negative_last_index = 2
                non_slice = slice(0, 0)

                self.assertEqual(local_first_index,
                                 _global_to_local_key_int(global_first_index, globalshape, local_to_global))
                self.assertEqual(local_second_index,
                                 _global_to_local_key_int(global_second_index, globalshape, local_to_global))
                self.assertEqual(local_last_index,
                                 _global_to_local_key_int(global_last_index, globalshape, local_to_global))
                self.assertEqual(non_slice,
                                 _global_to_local_key_int(global_lower_outside_local_range, globalshape, local_to_global))
                self.assertEqual(non_slice,
                                 _global_to_local_key_int(global_upper_outside_local_range, globalshape, local_to_global))
                self.assertEqual(local_negative_last_index,
                                 _global_to_local_key_int(global_negative_last_index, globalshape, local_to_global))
                self.assertEqual(non_slice,
                                 _global_to_local_key_int(global_negative_index_outside_range, globalshape, local_to_global))

                #Global result for local_to_global defined as None
                self.assertEqual(global_first_index,
                                 _global_to_local_key_int(global_first_index, globalshape, None))
                self.assertEqual(global_second_index,
                                 _global_to_local_key_int(global_second_index, globalshape, None))
                self.assertEqual(global_last_index,
                                 _global_to_local_key_int(global_last_index, globalshape, None))
                self.assertEqual(global_lower_outside_local_range,
                                 _global_to_local_key_int(global_lower_outside_local_range, globalshape, None))
                self.assertEqual(global_upper_outside_local_range,
                                 _global_to_local_key_int(global_upper_outside_local_range, globalshape, None))

                #Check for index errors
                index_out_of_global_range = 5
                negative_index_out_of_global_range = -6
                with self.assertRaises(IndexError):
                        _global_to_local_key_int(index_out_of_global_range, globalshape, local_to_global)
                with self.assertRaises(IndexError):
                        _global_to_local_key_int(negative_index_out_of_global_range, globalshape, local_to_global)
                with self.assertRaises(IndexError):
                        _global_to_local_key_int(index_out_of_global_range, globalshape, None)
                with self.assertRaises(IndexError):
                        _global_to_local_key_int(negative_index_out_of_global_range, globalshape, None)


        def test_global_to_local_key_slice(self):
                globalshape = (5, 5)
                local_to_global = {0 : (1, 4), 1 : (1, 4)}

                #Inputs
                global_first = slice(1, 2)
                global_second = slice(2, 3)
                global_last = slice(3, 4)
                global_lower_outside_local_range = slice(0, 1)
                global_upper_outside_local_range = slice(4, 5)
                #Expected Results
                local_first_index = slice(0, 1, 1)
                local_second_index = slice(1, 2, 1)
                local_last_index = slice(2, 3, 1)
                #Note: below would slice nothing as the start/stops are
                #out of the local range
                local_outside_min_range = slice(-1, 0, 1)
                local_outside_max_range = slice(3, 4, 1)
                #Equivalent to indexing arr[:]
                select_all = slice(None, None, None)

                self.assertEqual(select_all,
                                 _global_to_local_key_slice(select_all, globalshape, local_to_global))
                self.assertEqual(local_first_index,
                                 _global_to_local_key_slice(global_first, globalshape, local_to_global))
                self.assertEqual(local_second_index,
                                 _global_to_local_key_slice(global_second, globalshape, local_to_global))
                self.assertEqual(local_last_index,
                                 _global_to_local_key_slice(global_last, globalshape, local_to_global))
                self.assertEqual(local_outside_min_range,
                                 _global_to_local_key_slice(global_lower_outside_local_range, globalshape, local_to_global))
                self.assertEqual(local_outside_max_range,
                                 _global_to_local_key_slice(global_upper_outside_local_range, globalshape, local_to_global))


        def test_global_to_local_key_slice_with_steps(self):
                globalshape = (5, 5)
                local_to_global = {0 : (1, 4), 1 : (1, 4)}
                #Inputs
                global_first = slice(1, 5, 3)
                global_second = slice(0, 5, 2)
                global_last = slice(0, 5, 3)
                global_first_and_last = slice(1, 4, 2)
                #Expected Results
                local_first = slice(0, 4, 3)
                local_second = slice(-1, 4, 2)
                local_last = slice(-1, 4, 3)
                local_first_and_last = slice(0, 3, 2)

                self.assertEqual(local_first,
                                 _global_to_local_key_slice(global_first, globalshape, local_to_global))
                self.assertEqual(local_second,
                                 _global_to_local_key_slice(global_second, globalshape, local_to_global))
                self.assertEqual(local_last,
                                 _global_to_local_key_slice(global_last, globalshape, local_to_global))
                self.assertEqual(local_first_and_last,
                                 _global_to_local_key_slice(global_first_and_last, globalshape, local_to_global))


        def test_global_to_local_key_tuple(self):
                globalshape = (5, 5)
                local_to_global = {0 : (1, 4), 1 : (1, 4)}
                #Inputs
                int_tuple = (1, 1)
                slice_tuple = (slice(1, 2), slice(1, 2))
                mixed_tuple = (1, slice(1, 2))
                #Expected Results
                local_int_tuple = (0, 0)
                local_slice_tuple = (slice(0, 1, 1), slice(0, 1, 1))
                local_mixed_tuple = (0, slice(0, 1, 1))

                #Check that int/slice helper methods are called
                with mock.patch('mpids.MPInumpy.utils._global_to_local_key_int') as mock_obj_int:
                        _global_to_local_key_tuple(int_tuple, globalshape, local_to_global)
                calls = [mock.call(int_tuple[0], globalshape, local_to_global, 0),
                         mock.call(int_tuple[1], globalshape, local_to_global, 1)]
                mock_obj_int.assert_has_calls(calls)

                with mock.patch('mpids.MPInumpy.utils._global_to_local_key_slice') as mock_obj_slice:
                        _global_to_local_key_tuple(slice_tuple, globalshape, local_to_global)
                calls = [mock.call(slice_tuple[0], globalshape, local_to_global, 0),
                         mock.call(slice_tuple[1], globalshape, local_to_global, 1)]
                mock_obj_slice.assert_has_calls(calls)

                #Check return behavior
                self.assertEqual(local_int_tuple,
                                 _global_to_local_key_tuple(int_tuple, globalshape, local_to_global))
                self.assertEqual(local_slice_tuple,
                                 _global_to_local_key_tuple(slice_tuple, globalshape, local_to_global))
                self.assertEqual(local_mixed_tuple,
                                 _global_to_local_key_tuple(mixed_tuple, globalshape, local_to_global))

                #Global result for local_to_global defined as None
                self.assertEqual(int_tuple,
                                 _global_to_local_key_tuple(int_tuple, globalshape, None))
                self.assertEqual(slice_tuple,
                                 _global_to_local_key_tuple(slice_tuple, globalshape, None))
                self.assertEqual(mixed_tuple,
                                 _global_to_local_key_tuple(mixed_tuple, globalshape, None))

                #Check Index Error is propagated from _global_to_local_key_int
                with mock.patch('mpids.MPInumpy.utils._global_to_local_key_int',
                                side_effect = IndexError('Error')) as mock_obj_int:
                        with self.assertRaises(IndexError):
                                _global_to_local_key_tuple((6, 6), globalshape, local_to_global)
                with mock.patch('mpids.MPInumpy.utils._global_to_local_key_int',
                                side_effect = IndexError('Error')) as mock_obj_int:
                        with self.assertRaises(IndexError):
                                _global_to_local_key_tuple((6, 6), globalshape, None)

                #Check Index Error is thrown when key has more dimensions
                #than total array shape
                with self.assertRaises(IndexError):
                        _global_to_local_key_tuple((0, 1, 2), globalshape, local_to_global)
                with self.assertRaises(IndexError):
                        _global_to_local_key_tuple((0, 1, 2), globalshape, None)


        def test_global_to_local_key(self):
                globalshape = (5, 5)
                local_to_global = {0 : (1, 4), 1 : (1, 4)}


                #Check that int/slice/tuple helper methods are called
                with mock.patch('mpids.MPInumpy.utils._global_to_local_key_int') as mock_obj_int:
                        global_to_local_key(1, globalshape, local_to_global)
                mock_obj_int.assert_called_with(1, globalshape, local_to_global)

                with mock.patch('mpids.MPInumpy.utils._global_to_local_key_slice') as mock_obj_slice:
                        global_to_local_key(slice(1, 2), globalshape, local_to_global)
                mock_obj_slice.assert_called_with(slice(1, 2), globalshape, local_to_global)

                with mock.patch('mpids.MPInumpy.utils._global_to_local_key_tuple') as mock_obj_tuple:
                        global_to_local_key((1, 2), globalshape, local_to_global)
                mock_obj_tuple.assert_called_with((1, 2), globalshape, local_to_global)

                #Check Index Error is propagated from _global_to_local_key_int
                with mock.patch('mpids.MPInumpy.utils._global_to_local_key_int',
                                side_effect = IndexError('Error')) as mock_obj_int:
                        with self.assertRaises(IndexError):
                                global_to_local_key((6, 6), globalshape, local_to_global)

                #Check Index Error is propagated from _global_to_local_key_tuple
                with mock.patch('mpids.MPInumpy.utils._global_to_local_key_tuple',
                                side_effect = IndexError('Error')) as mock_obj_tuple:
                        with self.assertRaises(IndexError):
                                global_to_local_key((0, 1, 2), globalshape, local_to_global)

                #Check Not Supported Error is thrown when non int/slice/tuple key provided
                with self.assertRaises(NotSupportedError):
                        global_to_local_key([0, 1], globalshape, local_to_global)


        def test_format_indexed_result(self):
                test_matrix = np.array(list(range(16))).reshape(4,4)

                np_scalar = test_matrix[0,0]
                undesired_scalar_shape = np_scalar.shape
                self.assertEqual((), undesired_scalar_shape)

                desired_scalar_shape =(1,)
                formated_np_scalar = _format_indexed_result((0, 0), np_scalar)
                self.assertEqual(undesired_scalar_shape, np_scalar.shape)
                self.assertEqual(desired_scalar_shape, formated_np_scalar.shape)
                self.assertEqual(np_scalar, formated_np_scalar)

                empty_array = test_matrix[0:0]
                undesired_empty_shape = empty_array.shape
                self.assertEqual((0, 4), undesired_empty_shape)

                desired_empty_shape =(0,)
                formated_empty_array_slice = _format_indexed_result(slice(1,1), empty_array)
                self.assertEqual(undesired_empty_shape, empty_array.shape)
                self.assertEqual(desired_empty_shape, formated_empty_array_slice.shape)
                self.assertEqual(empty_array.data.tolist(), formated_empty_array_slice.data.tolist())

                #SPECIAL CASE, because a global index can exist outside of the
                ## local index space.
                #Result of slice
                empty_outside_index_array = np.array([]).reshape(0, 1)
                undesired_empty_outside_index_shape = empty_outside_index_array.shape
                self.assertEqual((0, 1), undesired_empty_outside_index_shape)

                desired_empty_outside_index_shape =(0, 0)
                formated_empty_outside_index_array = \
                        _format_indexed_result((slice(1,1), slice(1,1)), empty_outside_index_array)
                self.assertEqual(undesired_empty_outside_index_shape, empty_outside_index_array.shape)
                self.assertEqual(desired_empty_outside_index_shape, formated_empty_outside_index_array.shape)
                self.assertEqual(empty_outside_index_array.data.tolist(), formated_empty_outside_index_array.data.tolist())


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
                local_to_global_2d_map = {0 : {0 : (0, 2), 1 : (0, 4)},
                                          1 : {0 : (2, 3), 1 : (0, 4)},
                                          2 : {0 : (3, 4), 1 : (0, 4)},
                                          3 : {0 : (4, 5), 1 : (0, 4)}}
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
