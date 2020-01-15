import unittest
import unittest.mock as mock
import numpy as np
from mpi4py import MPI
from mpids.MPInumpy.utils import *
from mpids.MPInumpy.utils import format_indexed_result,     \
                                 _global_to_local_key_int,   \
                                 _global_to_local_key_slice, \
                                 _global_to_local_key_tuple
from mpids.MPInumpy.errors import IndexError, InvalidDistributionError, \
                                  NotSupportedError


class UtilsDistributionIndependentTest(unittest.TestCase):

    def setUp(self):
        class IndexKeyGenerator(object):
            def __getitem__(self, key):
                return key
        self.index_key_generator = IndexKeyGenerator()


    def test_distribution_checks(self):
        undist = 'u'
        row_block = 'b'

        self.assertTrue(is_undistributed(undist))
        self.assertFalse(is_undistributed(row_block))

        self.assertTrue(is_row_block_distributed(row_block))
        self.assertFalse(is_row_block_distributed(undist))


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
        global_first_index = self.index_key_generator[1]
        global_second_index = self.index_key_generator[2]
        global_last_index = self.index_key_generator[3]
        global_lower_outside_local_range = self.index_key_generator[0]
        global_upper_outside_local_range = self.index_key_generator[4]
        global_negative_last_index = self.index_key_generator[-2]
        global_negative_index_outside_range = self.index_key_generator[-1]
        #Check keys are what we expect
        self.assertEqual(1, global_first_index)
        self.assertEqual(2, global_second_index)
        self.assertEqual(3, global_last_index)
        self.assertEqual(0, global_lower_outside_local_range)
        self.assertEqual(4, global_upper_outside_local_range)
        self.assertEqual(-2, global_negative_last_index)
        self.assertEqual(-1, global_negative_index_outside_range)

        #Expected Results
        local_first_index = self.index_key_generator[0]
        local_second_index = self.index_key_generator[1]
        local_last_index = self.index_key_generator[2]
        local_negative_last_index = self.index_key_generator[2]
        non_slice = self.index_key_generator[0:0]
        #Check keyss are what we expect
        self.assertEqual(0, local_first_index)
        self.assertEqual(1, local_second_index)
        self.assertEqual(2, local_last_index)
        self.assertEqual(2, local_negative_last_index)
        self.assertEqual(slice(0, 0), non_slice)

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
        index_out_of_global_range = self.index_key_generator[5]
        negative_index_out_of_global_range = self.index_key_generator[-6]
        #Check keys are what we expect
        self.assertEqual(5, index_out_of_global_range)
        self.assertEqual(-6, negative_index_out_of_global_range)

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
        global_first = self.index_key_generator[1:2]
        global_second = self.index_key_generator[2:3]
        global_last = self.index_key_generator[3:4]
        global_lower_outside_local_range = self.index_key_generator[0:1]
        global_upper_outside_local_range = self.index_key_generator[4:5]
        #Check keys are what we expect
        self.assertEqual(slice(1, 2), global_first)
        self.assertEqual(slice(2, 3), global_second)
        self.assertEqual(slice(3, 4), global_last)
        self.assertEqual(slice(0, 1), global_lower_outside_local_range)
        self.assertEqual(slice(4, 5), global_upper_outside_local_range)

        #Expected Results
        local_first_index = self.index_key_generator[0:1:1]
        local_second_index = self.index_key_generator[1:2:1]
        local_last_index = self.index_key_generator[2:3:1]
        #Check keys are what we expect
        self.assertEqual(slice(0, 1, 1), local_first_index)
        self.assertEqual(slice(1, 2, 1), local_second_index)
        self.assertEqual(slice(2, 3, 1), local_last_index)

        #Note: below would slice nothing as the start/stops are
        #out of the local range
        local_outside_min_range = self.index_key_generator[-1:0:1]
        local_outside_max_range = self.index_key_generator[3:4:1]
        select_all = self.index_key_generator[:]
        #Check keys are what we expect
        self.assertEqual(slice(-1, 0, 1), local_outside_min_range)
        self.assertEqual(slice(3, 4, 1), local_outside_max_range)
        self.assertEqual(slice(None, None, None), select_all)

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
        global_first = self.index_key_generator[1:5:3]
        global_second = self.index_key_generator[0:5:2]
        global_last = self.index_key_generator[0:5:3]
        global_first_and_last = self.index_key_generator[1:4:2]
        #Check keys are what we expect
        self.assertEqual(slice(1, 5, 3), global_first)
        self.assertEqual(slice(0, 5, 2), global_second)
        self.assertEqual(slice(0, 5, 3), global_last)
        self.assertEqual(slice(1, 4, 2), global_first_and_last)

        #Expected Results
        local_first = self.index_key_generator[0:4:3]
        local_second = self.index_key_generator[-1:4:2]
        local_last = self.index_key_generator[-1:4:3]
        local_first_and_last = self.index_key_generator[0:3:2]
        #Check keys are what we expect
        self.assertEqual(slice(0, 4, 3), local_first)
        self.assertEqual(slice(-1, 4, 2), local_second)
        self.assertEqual(slice(-1, 4, 3), local_last)
        self.assertEqual(slice(0, 3, 2), local_first_and_last)

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
        int_tuple = self.index_key_generator[1, 1]
        slice_tuple = self.index_key_generator[1:2,1:2]
        mixed_tuple = self.index_key_generator[1,1:2]
        #Check keys are what we expect
        self.assertEqual((1, 1), int_tuple)
        self.assertEqual((slice(1, 2), slice(1, 2)), slice_tuple)
        self.assertEqual((1, slice(1, 2)), mixed_tuple)

        #Expected Results
        local_int_tuple = self.index_key_generator[0,0]
        local_slice_tuple = self.index_key_generator[0:1:1,0:1:1]
        local_mixed_tuple = self.index_key_generator[0,0:1:1]
        #Check keys are what we expect
        self.assertEqual((0, 0), local_int_tuple)
        self.assertEqual((slice(0, 1, 1), slice(0, 1, 1)), local_slice_tuple)
        self.assertEqual((0, slice(0, 1, 1)), local_mixed_tuple)

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


    def testformat_indexed_result(self):
        test_matrix = np.arange(16).reshape(4,4)

        np_scalar = test_matrix[0,0]
        undesired_scalar_shape = np_scalar.shape
        desired_scalar_shape =(1,)
        formated_np_scalar = format_indexed_result((0, 0), np_scalar)
        self.assertEqual((), undesired_scalar_shape)
        self.assertEqual(desired_scalar_shape, formated_np_scalar.shape)
        self.assertEqual(np_scalar, formated_np_scalar)

        empty_array = test_matrix[0:0]
        undesired_empty_shape = empty_array.shape
        desired_scalar_empty_shape =(0,)
        desired_empty_shape =(0, 0)
        self.assertEqual((0, 4), undesired_empty_shape)

        formated_empty_array_int = format_indexed_result(1, empty_array)
        self.assertEqual(desired_scalar_empty_shape, formated_empty_array_int.shape)
        self.assertEqual(empty_array.data.tolist(), formated_empty_array_int.data.tolist())

        formated_empty_array_slice = format_indexed_result(slice(1,1), empty_array)
        self.assertEqual(desired_empty_shape, formated_empty_array_slice.shape)
        self.assertEqual(empty_array.data.tolist(), formated_empty_array_slice.data.tolist())

        formated_empty_array_tuple_int = format_indexed_result((1, 1), empty_array)
        self.assertEqual(desired_scalar_empty_shape, formated_empty_array_tuple_int.shape)
        self.assertEqual(empty_array.data.tolist(), formated_empty_array_tuple_int.data.tolist())

        formated_empty_array_tuple_slice = format_indexed_result((slice(1,1), slice(1,1)), empty_array)
        self.assertEqual(desired_empty_shape, formated_empty_array_tuple_slice.shape)
        self.assertEqual(empty_array.data.tolist(), formated_empty_array_tuple_slice.data.tolist())


    def test_determine_global_offset(self):
        #1D
        global_shape = (10,)
        self.assertEqual(0, determine_global_offset([0], global_shape))
        self.assertEqual(5, determine_global_offset([5], global_shape))
        self.assertEqual(9, determine_global_offset([9], global_shape))

        # #2D
        global_shape = (3, 3)
        self.assertEqual(1, determine_global_offset([0, 1], global_shape))
        self.assertEqual(2, determine_global_offset([0, 2], global_shape))
        self.assertEqual(8, determine_global_offset([2, 2], global_shape))

        #3D
        global_shape = (3, 3, 3)
        self.assertEqual(9, determine_global_offset([1, 0, 0], global_shape))
        self.assertEqual(3, determine_global_offset([0, 1, 0], global_shape))
        self.assertEqual(1, determine_global_offset([0, 0, 1], global_shape))
        self.assertEqual(12, determine_global_offset([1, 1, 0], global_shape))
        self.assertEqual(4, determine_global_offset([0, 1, 1], global_shape))
        self.assertEqual(13, determine_global_offset([1, 1, 1], global_shape))

        #4D
        global_shape = (2, 2, 2, 2)
        self.assertEqual(0, determine_global_offset([0, 0, 0, 0], global_shape))
        self.assertEqual(8, determine_global_offset([1, 0, 0, 0], global_shape))
        self.assertEqual(4, determine_global_offset([0, 1, 0, 0], global_shape))
        self.assertEqual(2, determine_global_offset([0, 0, 1, 0], global_shape))
        self.assertEqual(1, determine_global_offset([0, 0, 0, 1], global_shape))
        self.assertEqual(12, determine_global_offset([1, 1, 0, 0], global_shape))
        self.assertEqual(6, determine_global_offset([0, 1, 1, 0], global_shape))
        self.assertEqual(3, determine_global_offset([0, 0, 1, 1], global_shape))
        self.assertEqual(9, determine_global_offset([1, 0, 0, 1], global_shape))
        self.assertEqual(10, determine_global_offset([1, 0, 1, 0], global_shape))
        self.assertEqual(5, determine_global_offset([0, 1, 0, 1], global_shape))

        #Index must be a list
        with self.assertRaises(TypeError):
            determine_global_offset(1, (2,3))
        with self.assertRaises(TypeError):
            determine_global_offset((2,), (2,3))

        #Index length must equal global shape length
        with self.assertRaises(ValueError):
            determine_global_offset([1], (2,3))


    def test_determine_redistribution_counts_from_shape(self):
        dist = 'b'
#TODO: Above does not really do anything, see utils
        rank = MPI.COMM_WORLD.rank
        #Emulate a 4x4 matrix row distributed on a 4 process comm.
        ##reshaping to a 2x8
        current_shape = (4,4)
        desired_shape = (2,8)
        send_counts, recv_counts = \
            determine_redistribution_counts_from_shape(current_shape,
                                                       desired_shape,
                                                       dist)
        self.assertTrue(isinstance(send_counts, np.ndarray))
        self.assertTrue(isinstance(recv_counts, np.ndarray))
        if rank == 0:
            self.assertTrue(np.alltrue(np.array([4,0,0,0]) == send_counts))
            self.assertTrue(np.alltrue(np.array([4,4,0,0]) == recv_counts))
        elif rank == 1:
            self.assertTrue(np.alltrue(np.array([4,0,0,0]) == send_counts))
            self.assertTrue(np.alltrue(np.array([0,0,4,4]) == recv_counts))
        else:
            self.assertTrue(np.alltrue(np.array([0,4,0,0]) == send_counts))
            self.assertTrue(np.alltrue(np.array([0,0,0,0]) == recv_counts))

        # #reshaping 5x5 to a 1x25
        current_shape = (5,5)
        desired_shape = (1,25)
        send_counts, recv_counts = \
            determine_redistribution_counts_from_shape(current_shape,
                                                       desired_shape,
                                                       dist)
        self.assertTrue(isinstance(send_counts, np.ndarray))
        self.assertTrue(isinstance(recv_counts, np.ndarray))
        if rank == 0:
            self.assertTrue(np.alltrue(np.array([10,0,0,0]) == send_counts))
            self.assertTrue(np.alltrue(np.array([10,5,5,5]) == recv_counts))
        else:
            self.assertTrue(np.alltrue(np.array([5,0,0,0]) == send_counts))
            self.assertTrue(np.alltrue(np.array([0,0,0,0]) == recv_counts))

        #reshaping 4x3 to a 3x4
        current_shape = (4,3)
        desired_shape = (3,4)
        send_counts, recv_counts = \
            determine_redistribution_counts_from_shape(current_shape,
                                                       desired_shape,
                                                       dist)
        self.assertTrue(isinstance(send_counts, np.ndarray))
        self.assertTrue(isinstance(recv_counts, np.ndarray))
        if rank == 0:
            self.assertTrue(np.alltrue(np.array([3,0,0,0]) == send_counts))
            self.assertTrue(np.alltrue(np.array([3,1,0,0]) == recv_counts))
        elif rank == 1:
            self.assertTrue(np.alltrue(np.array([1,2,0,0]) == send_counts))
            self.assertTrue(np.alltrue(np.array([0,2,2,0]) == recv_counts))
        elif rank == 2:
            self.assertTrue(np.alltrue(np.array([0,2,1,0]) == send_counts))
            self.assertTrue(np.alltrue(np.array([0,0,1,3]) == recv_counts))
        else:
            self.assertTrue(np.alltrue(np.array([0,0,3,0]) == send_counts))
            self.assertTrue(np.alltrue(np.array([0,0,0,0]) == recv_counts))


        #reshaping 3x3x3 to a 9x3
        current_shape = (3,3,3)
        desired_shape = (9,3)
        send_counts, recv_counts = \
            determine_redistribution_counts_from_shape(current_shape,
                                                       desired_shape,
                                                       dist)
        self.assertTrue(isinstance(send_counts, np.ndarray))
        self.assertTrue(isinstance(recv_counts, np.ndarray))
        if rank == 0:
            self.assertTrue(np.alltrue(np.array([9,0,0,0]) == send_counts))
            self.assertTrue(np.alltrue(np.array([9,0,0,0]) == recv_counts))
        elif rank == 1:
            self.assertTrue(np.alltrue(np.array([0,6,3,0]) == send_counts))
            self.assertTrue(np.alltrue(np.array([0,6,0,0]) == recv_counts))
        elif rank == 2:
            self.assertTrue(np.alltrue(np.array([0,0,3,6]) == send_counts))
            self.assertTrue(np.alltrue(np.array([0,3,3,0]) == recv_counts))
        else:
            self.assertTrue(np.alltrue(np.array([0,0,0,0]) == send_counts))
            self.assertTrue(np.alltrue(np.array([0,0,6,0]) == recv_counts))


class UtilsDefaultTest(unittest.TestCase):

    def create_setUp_parms(self):
        parms = {}
        parms['procs'] = MPI.COMM_WORLD.Get_size()
        parms['rank'] = MPI.COMM_WORLD.Get_rank()
        parms['data'] = list(range(10))
        parms['data_shape'] = np.shape(parms['data'])
        parms['data_2d'] = np.arange(20).reshape(5,4)
        parms['data_2d_shape'] = parms['data_2d'].shape
        # Default distribution
        parms['dist'] = 'b'
        parms['comm_dims'] = [parms['procs']]
        parms['comm_coord'] = [parms['rank']]
        parms['dist_to_dims'] = 1
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
        parms['local_data_shape'] = np.shape(parms['local_data'])
        parms['local_data_2d_shape'] = parms['local_data_2d'].shape
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
        self.data_shape = parms.get('data_shape')
        self.data_2d = parms.get('data_2d')
        self.data_2d_shape = parms.get('data_2d_shape')
        self.dist = parms.get('dist')
        self.comm_dims = parms.get('comm_dims')
        self.comm_coord = parms.get('comm_coord')
        self.dist_to_dims = parms.get('dist_to_dims')
        self.local_data = parms.get('local_data')
        self.local_data_2d = parms.get('local_data_2d')
        self.local_data_shape = parms.get('local_data_shape')
        self.local_data_2d_shape = parms.get('local_data_2d_shape')
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


    def test_distribute_array(self):
        # 1-D Data
        local_data, comm_dims, comm_coord, local_to_global = \
            distribute_array(self.data, self.dist)
        self.assertEqual((self.comm_dims, self.comm_coord, self.local_to_global),
                         (comm_dims, comm_coord, local_to_global))
        self.assertTrue(np.alltrue(self.local_data == local_data))

        # # 2-D Data
        local_data_2d, comm_dims, comm_coord, local_to_global_2d = \
            distribute_array(self.data_2d, self.dist)
        self.assertEqual((self.comm_dims, self.comm_coord, self.local_to_global_2d),
                         (comm_dims, comm_coord, local_to_global_2d))
        self.assertTrue(np.alltrue(self.local_data_2d == local_data_2d))


    def test_distribute_shape(self):
        # 1-D Data
        local_shape, comm_dims, comm_coord, local_to_global = \
            distribute_shape(self.data_shape, self.dist)
        self.assertEqual(
            (self.local_data_shape, self.comm_dims, self.comm_coord, self.local_to_global),
            (tuple(local_shape), comm_dims, comm_coord, local_to_global))


        # 2-D data
        local_shape_2d, comm_dims, comm_coord, local_to_global_2d = \
            distribute_shape(self.data_2d_shape, self.dist)
        self.assertEqual(
            (self.local_data_2d_shape, self.comm_dims, self.comm_coord, self.local_to_global_2d),
            (tuple(local_shape_2d), comm_dims, comm_coord, local_to_global_2d))


    def test_determine_local_shape_and_mapping(self):
        # 1-D Data
        self.assertEqual((self.local_data_shape, self.local_to_global),
                         determine_local_shape_and_mapping(self.data_shape,
                                                           self.dist,
                                                           self.comm_dims,
                                                           self.comm_coord))

        # 2-D data
        self.assertEqual((self.local_data_2d_shape, self.local_to_global_2d),
                         determine_local_shape_and_mapping(self.data_2d_shape,
                                                           self.dist,
                                                           self.comm_dims,
                                                           self.comm_coord))


    def test_slice_local_data_and_determine_mapping(self):
        # 1-D Data
        self.assertEqual((self.local_data, self.local_to_global),
                         slice_local_data_and_determine_mapping(self.data,
                                                                self.dist,
                                                                self.comm_dims,
                                                                self.comm_coord))

        # 2-D Data
        local_data_2d, local_to_global = \
            slice_local_data_and_determine_mapping(self.data_2d,
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
        parms['data_shape'] = np.shape(parms['data'])
        parms['data_2d'] = np.arange(20).reshape(5,4)
        parms['data_2d_shape'] = parms['data_2d'].shape
        # Undistributed distribution
        parms['dist'] = 'u'
        parms['comm_dims'] = None
        parms['comm_coord'] = None
        parms['local_data'] = parms['data']
        parms['local_data_2d'] = parms['data_2d']
        parms['local_data_shape'] = parms['data_shape']
        parms['local_data_2d_shape'] = parms['data_2d_shape']
        parms['local_to_global'] = None
        parms['local_to_global_2d'] = None
        return parms


if __name__ == '__main__':
    unittest.main()
