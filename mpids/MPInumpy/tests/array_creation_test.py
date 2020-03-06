import unittest
import unittest.mock as mock
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np
from mpids.MPInumpy.distributions import *
from mpids.MPInumpy.errors import InvalidDistributionError, \
                                  TypeError,                \
                                  ValueError
from mpids.MPInumpy.array_creation import _validate_shape

class ValidateShapeTest(unittest.TestCase):

    def test_int_as_shape(self):
        self.assertEqual((1,), _validate_shape(1))
        self.assertEqual((10,), _validate_shape(10))
        self.assertEqual((100,), _validate_shape(100))
        self.assertEqual((9999,), _validate_shape(9999))


    def test_non_int_raises_value_error(self):
        with self.assertRaises(ValueError):
            _validate_shape(1.0)


    def test_tuple_as_shape(self):
        self.assertEqual((1, 2), _validate_shape((1, 2)))
        self.assertEqual((1, 2, 3), _validate_shape((1, 2, 3)))
        self.assertEqual((1, 2, 3, 4), _validate_shape((1, 2, 3, 4)))


    def test_non_tuple_of_ints_raises_value_error(self):
        with self.assertRaises(ValueError):
            _validate_shape((1.0, 2))
        with self.assertRaises(ValueError):
            _validate_shape((1, 2.0))


    def test_series_of_ints_raises_type_error(self):
        with self.assertRaises(TypeError):
            _validate_shape(1, 2)
        with self.assertRaises(TypeError):
            _validate_shape(1, 2, 3)
        with self.assertRaises(TypeError):
            _validate_shape(1, 2, 3, 4)


    def test_series_of_tuples_raises_type_error(self):
        with self.assertRaises(TypeError):
            _validate_shape((1,), (2,))
        with self.assertRaises(TypeError):
            _validate_shape((1, 2), (3,))
        with self.assertRaises(TypeError):
            _validate_shape((1,), (2, 3))


    def test_special_case_of_None_provided_by_non_root_rank(self):
        rank = MPI.COMM_WORLD.Get_rank()
        shape = (1, 2) if rank == 0 else None
        self.assertEqual(shape, _validate_shape(shape))


class ArrayCreationErrorsPropegatedTest(unittest.TestCase):

    def test_unsupported_distribution(self):
        data = np.arange(10)
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
        parms['np_data'] = np.arange(20).reshape(5,4)
        parms['array_like_data'] = parms['np_data'].tolist()
        parms['comm'] = MPI.COMM_WORLD
        # Default distribution
        parms['dist'] = 'b'
        parms['dist_class'] = Block
        return parms


    def setUp(self):
        parms = self.create_setUp_parms()
        self.np_data = parms['np_data']
        self.array_like_data = parms['array_like_data']
        self.comm = parms['comm']
        self.dist = parms['dist']
        self.dist_class = parms['dist_class']
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()


    def test_return_behavior_with_np_data_from_all_ranks(self):
        for root in range(self.size):
            np_data = None
            self.assertTrue(np_data is None)
            if self.rank == root:
                np_data = self.np_data
            mpi_np_array = mpi_np.array(np_data,
                                        comm=self.comm,
                                        root=root,
                                        dist=self.dist)
            self.assertTrue(isinstance(mpi_np_array, mpi_np.MPIArray))
            self.assertTrue(isinstance(mpi_np_array, self.dist_class))
            self.assertEqual(mpi_np_array.comm, self.comm)
            self.assertEqual(mpi_np_array.dist, self.dist)


    def test_return_behavior_with_array_like_data_from_all_ranks(self):
        for root in range(self.size):
            array_like_data = None
            self.assertTrue(array_like_data is None)
            if self.rank == root:
                array_like_data = self.array_like_data
            mpi_np_array = mpi_np.array(array_like_data,
                                        comm=self.comm,
                                        root=root,
                                        dist=self.dist)
            self.assertTrue(isinstance(mpi_np_array, mpi_np.MPIArray))
            self.assertTrue(isinstance(mpi_np_array, self.dist_class))
            self.assertEqual(mpi_np_array.comm, self.comm)
            self.assertEqual(mpi_np_array.dist, self.dist)


class ArrayUndistributedTest(ArrayDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        parms['np_data'] = np.arange(20).reshape(5,4)
        parms['array_like_data'] = parms['np_data'].tolist()
        parms['comm'] = MPI.COMM_WORLD
        # Undistributed distribution
        parms['dist'] = 'u'
        parms['dist_class'] = Undistributed
        return parms


class ArangeDefaultTest(unittest.TestCase):

    def create_setUp_parms(self):
        parms = {}
        parms['comm'] = MPI.COMM_WORLD
        # Default distribution
        parms['dist'] = 'b'
        parms['dist_class'] = Block
        return parms


    def setUp(self):
        parms = self.create_setUp_parms()
        self.comm = parms['comm']
        self.dist = parms['dist']
        self.dist_class = parms['dist_class']
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()


    def test_return_behavior_from_all_ranks_int_stop(self):
        np_arange = np.arange(20)
        for root in range(self.size):
            stop = None
            self.assertTrue(stop is None)
            if self.rank == root:
                stop = 20
            mpi_np_arange = mpi_np.arange(stop,
                                          comm=self.comm,
                                          root=root,
                                          dist=self.dist)
            self.assertTrue(isinstance(mpi_np_arange, mpi_np.MPIArray))
            self.assertTrue(isinstance(mpi_np_arange, self.dist_class))
            self.assertEqual(mpi_np_arange.comm, self.comm)
            self.assertEqual(mpi_np_arange.dist, self.dist)
            self.assertTrue(np.alltrue(mpi_np_arange[:] == np_arange))


    def test_return_behavior_from_all_ranks_float_stop(self):
        np_arange = np.arange(20.0)
        for root in range(self.size):
            stop = None
            self.assertTrue(stop is None)
            if self.rank == root:
                stop = 20.0
            mpi_np_arange = mpi_np.arange(stop,
                                          comm=self.comm,
                                          root=root,
                                          dist=self.dist)
            self.assertTrue(isinstance(mpi_np_arange, mpi_np.MPIArray))
            self.assertTrue(isinstance(mpi_np_arange, self.dist_class))
            self.assertEqual(mpi_np_arange.comm, self.comm)
            self.assertEqual(mpi_np_arange.dist, self.dist)
            self.assertTrue(np.alltrue(mpi_np_arange[:] == np_arange))


    def test_return_behavior_from_all_ranks_int_start_stop(self):
        np_arange = np.arange(1, 20)
        for root in range(self.size):
            start = None
            stop = None
            self.assertTrue(start is None)
            self.assertTrue(stop is None)
            if self.rank == root:
                start = 1
                stop = 20
            mpi_np_arange = mpi_np.arange(start, stop,
                                          comm=self.comm,
                                          root=root,
                                          dist=self.dist)
            self.assertTrue(isinstance(mpi_np_arange, mpi_np.MPIArray))
            self.assertTrue(isinstance(mpi_np_arange, self.dist_class))
            self.assertEqual(mpi_np_arange.comm, self.comm)
            self.assertEqual(mpi_np_arange.dist, self.dist)
            self.assertTrue(np.alltrue(mpi_np_arange[:] == np_arange))


    def test_return_behavior_from_all_ranks_float_start_stop(self):
        np_arange = np.arange(1.0, 20.0)
        for root in range(self.size):
            start = None
            stop = None
            self.assertTrue(start is None)
            self.assertTrue(stop is None)
            if self.rank == root:
                start = 1.0
                stop = 20.0
            mpi_np_arange = mpi_np.arange(start, stop,
                                          comm=self.comm,
                                          root=root,
                                          dist=self.dist)
            self.assertTrue(isinstance(mpi_np_arange, mpi_np.MPIArray))
            self.assertTrue(isinstance(mpi_np_arange, self.dist_class))
            self.assertEqual(mpi_np_arange.comm, self.comm)
            self.assertEqual(mpi_np_arange.dist, self.dist)
            self.assertTrue(np.alltrue(mpi_np_arange[:] == np_arange))


    def test_return_behavior_from_all_ranks_int_start_stop_step(self):
        np_arange = np.arange(1, 20, 2)
        for root in range(self.size):
            start = None
            stop = None
            step = None
            self.assertTrue(start is None)
            self.assertTrue(stop is None)
            self.assertTrue(step is None)
            if self.rank == root:
                start = 1
                stop = 20
                step = 2
            mpi_np_arange = mpi_np.arange(start, stop, step,
                                          comm=self.comm,
                                          root=root,
                                          dist=self.dist)
            self.assertTrue(isinstance(mpi_np_arange, mpi_np.MPIArray))
            self.assertTrue(isinstance(mpi_np_arange, self.dist_class))
            self.assertEqual(mpi_np_arange.comm, self.comm)
            self.assertEqual(mpi_np_arange.dist, self.dist)
            self.assertTrue(np.alltrue(mpi_np_arange[:] == np_arange))


    def test_return_behavior_from_all_ranks_float_start_stop(self):
        np_arange = np.arange(1.0, 20.0, 2.0)
        for root in range(self.size):
            start = None
            stop = None
            step = None
            self.assertTrue(start is None)
            self.assertTrue(stop is None)
            self.assertTrue(step is None)
            if self.rank == root:
                start = 1.0
                stop = 20.0
                step = 2.0
            mpi_np_arange = mpi_np.arange(start, stop, step,
                                          comm=self.comm,
                                          root=root,
                                          dist=self.dist)
            self.assertTrue(isinstance(mpi_np_arange, mpi_np.MPIArray))
            self.assertTrue(isinstance(mpi_np_arange, self.dist_class))
            self.assertEqual(mpi_np_arange.comm, self.comm)
            self.assertEqual(mpi_np_arange.dist, self.dist)
            self.assertTrue(np.alltrue(mpi_np_arange[:] == np_arange))


class ArangeUndistributedTest(ArangeDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        parms['comm'] = MPI.COMM_WORLD
        # Undistributed distribution
        parms['dist'] = 'u'
        parms['dist_class'] = Undistributed
        return parms


class EmptyDefaultTest(unittest.TestCase):

    def create_setUp_parms(self):
        parms = {}
        parms['int_shape'] = 4
        parms['tuple_shape'] = (5, 4)
        parms['comm'] = MPI.COMM_WORLD
        # Default distribution
        parms['dist'] = 'b'
        parms['dist_class'] = Block
        return parms


    def setUp(self):
        parms = self.create_setUp_parms()
        self.int_shape = parms['int_shape']
        self.tuple_shape = parms['tuple_shape']
        self.comm = parms['comm']
        self.dist = parms['dist']
        self.dist_class = parms['dist_class']
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()


    def test_return_behavior_from_all_ranks_with_int_shape(self):
        for root in range(self.size):
            shape = None
            self.assertTrue(shape is None)
            if self.rank == root:
                shape = self.int_shape
            mpi_np_empty = mpi_np.empty(shape,
                                        comm=self.comm,
                                        root=root,
                                        dist=self.dist)
            self.assertTrue(isinstance(mpi_np_empty, mpi_np.MPIArray))
            self.assertTrue(isinstance(mpi_np_empty, self.dist_class))
            self.assertEqual(mpi_np_empty.comm, self.comm)
            self.assertEqual(mpi_np_empty.dist, self.dist)


    def test_return_behavior_from_all_ranks_with_tuple_shape(self):
        for root in range(self.size):
            shape = None
            self.assertTrue(shape is None)
            if self.rank == root:
                shape = self.tuple_shape
            mpi_np_empty = mpi_np.empty(shape,
                                        comm=self.comm,
                                        root=root,
                                        dist=self.dist)
            self.assertTrue(isinstance(mpi_np_empty, mpi_np.MPIArray))
            self.assertTrue(isinstance(mpi_np_empty, self.dist_class))
            self.assertEqual(mpi_np_empty.comm, self.comm)
            self.assertEqual(mpi_np_empty.dist, self.dist)


    def test_validate_shape_called(self):
        shape_int = 1
        with mock.patch('mpids.MPInumpy.array_creation._validate_shape') as mock_obj_int:
            mpi_np.empty(shape_int)
        mock_obj_int.assert_called_with(shape_int)

        shape_tuple = (1, 2)
        with mock.patch('mpids.MPInumpy.array_creation._validate_shape') as mock_obj_tuple:
            mpi_np.empty(shape_tuple)
        mock_obj_tuple.assert_called_with(shape_tuple)


    def test_validate_shape_errors_propegated(self):
        with mock.patch('mpids.MPInumpy.array_creation._validate_shape',
                        side_effect = Exception('Mock Execption')) as mock_obj:
            with self.assertRaises(Exception):
                mpi_np.empty(1)


class EmptyUndistributedTest(EmptyDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        parms['int_shape'] = 4
        parms['tuple_shape'] = (5, 4)
        parms['comm'] = MPI.COMM_WORLD
        # Undistributed distribution
        parms['dist'] = 'u'
        parms['dist_class'] = Undistributed
        return parms


class OnesDefaultTest(unittest.TestCase):

    def create_setUp_parms(self):
        parms = {}
        parms['int_shape'] = 4
        parms['tuple_shape'] = (5, 4)
        parms['comm'] = MPI.COMM_WORLD
        # Default distribution
        parms['dist'] = 'b'
        parms['dist_class'] = Block
        return parms


    def setUp(self):
        parms = self.create_setUp_parms()
        self.int_shape = parms['int_shape']
        self.tuple_shape = parms['tuple_shape']
        self.comm = parms['comm']
        self.dist = parms['dist']
        self.dist_class = parms['dist_class']
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()


    def test_return_behavior_from_all_ranks_with_int_shape(self):
        for root in range(self.size):
            shape = None
            self.assertTrue(shape is None)
            if self.rank == root:
                shape = self.int_shape
            mpi_np_ones = mpi_np.ones(shape,
                                      comm=self.comm,
                                      root=root,
                                      dist=self.dist)
            self.assertTrue(isinstance(mpi_np_ones, mpi_np.MPIArray))
            self.assertTrue(isinstance(mpi_np_ones, self.dist_class))
            self.assertEqual(mpi_np_ones.comm, self.comm)
            self.assertEqual(mpi_np_ones.dist, self.dist)
            self.assertTrue(np.alltrue((mpi_np_ones) == (1)))


    def test_return_behavior_from_all_ranks_with_tuple_shape(self):
        for root in range(self.size):
            shape = None
            self.assertTrue(shape is None)
            if self.rank == root:
                shape = self.tuple_shape
            mpi_np_ones = mpi_np.ones(shape,
                                      comm=self.comm,
                                      root=root,
                                      dist=self.dist)
            self.assertTrue(isinstance(mpi_np_ones, mpi_np.MPIArray))
            self.assertTrue(isinstance(mpi_np_ones, self.dist_class))
            self.assertEqual(mpi_np_ones.comm, self.comm)
            self.assertEqual(mpi_np_ones.dist, self.dist)
            self.assertTrue(np.alltrue((mpi_np_ones) == (1)))


    def test_validate_shape_called(self):
        shape_int = 1
        with mock.patch('mpids.MPInumpy.array_creation._validate_shape') as mock_obj_int:
            mpi_np.ones(shape_int)
        mock_obj_int.assert_called_with(shape_int)

        shape_tuple = (1, 2)
        with mock.patch('mpids.MPInumpy.array_creation._validate_shape') as mock_obj_tuple:
            mpi_np.ones(shape_tuple)
        mock_obj_tuple.assert_called_with(shape_tuple)


    def test_validate_shape_errors_propegated(self):
        with mock.patch('mpids.MPInumpy.array_creation._validate_shape',
                        side_effect = Exception('Mock Execption')) as mock_obj:
            with self.assertRaises(Exception):
                mpi_np.ones(1)


class OnesUndistributedTest(OnesDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        parms['int_shape'] = 4
        parms['tuple_shape'] = (5, 4)
        parms['comm'] = MPI.COMM_WORLD
        # Undistributed distribution
        parms['dist'] = 'u'
        parms['dist_class'] = Undistributed
        return parms


class ZerosDefaultTest(unittest.TestCase):

    def create_setUp_parms(self):
        parms = {}
        parms['int_shape'] = 4
        parms['tuple_shape'] = (5, 4)
        parms['comm'] = MPI.COMM_WORLD
        # Default distribution
        parms['dist'] = 'b'
        parms['dist_class'] = Block
        return parms


    def setUp(self):
        parms = self.create_setUp_parms()
        self.int_shape = parms['int_shape']
        self.tuple_shape = parms['tuple_shape']
        self.comm = parms['comm']
        self.dist = parms['dist']
        self.dist_class = parms['dist_class']
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()


    def test_return_behavior_from_all_ranks_with_int_shape(self):
        for root in range(self.size):
            shape = None
            self.assertTrue(shape is None)
            if self.rank == root:
                shape = self.int_shape
            mpi_np_zeros = mpi_np.zeros(shape,
                                        comm=self.comm,
                                        root=root,
                                        dist=self.dist)
            self.assertTrue(isinstance(mpi_np_zeros, mpi_np.MPIArray))
            self.assertTrue(isinstance(mpi_np_zeros, self.dist_class))
            self.assertEqual(mpi_np_zeros.comm, self.comm)
            self.assertEqual(mpi_np_zeros.dist, self.dist)
            self.assertTrue(np.alltrue((mpi_np_zeros) == (0)))


    def test_return_behavior_from_all_ranks_with_tuple_shape(self):
        for root in range(self.size):
            shape = None
            self.assertTrue(shape is None)
            if self.rank == root:
                shape = self.tuple_shape
            mpi_np_zeros = mpi_np.zeros(shape,
                                        comm=self.comm,
                                        root=root,
                                        dist=self.dist)
            self.assertTrue(isinstance(mpi_np_zeros, mpi_np.MPIArray))
            self.assertTrue(isinstance(mpi_np_zeros, self.dist_class))
            self.assertEqual(mpi_np_zeros.comm, self.comm)
            self.assertEqual(mpi_np_zeros.dist, self.dist)
            self.assertTrue(np.alltrue((mpi_np_zeros) == (0)))


    def test_validate_shape_called(self):
        shape_int = 1
        with mock.patch('mpids.MPInumpy.array_creation._validate_shape') as mock_obj_int:
            mpi_np.zeros(shape_int)
        mock_obj_int.assert_called_with(shape_int)

        shape_tuple = (1, 2)
        with mock.patch('mpids.MPInumpy.array_creation._validate_shape') as mock_obj_tuple:
            mpi_np.zeros(shape_tuple)
        mock_obj_tuple.assert_called_with(shape_tuple)


    def test_validate_shape_errors_propegated(self):
        with mock.patch('mpids.MPInumpy.array_creation._validate_shape',
                        side_effect = Exception('Mock Execption')) as mock_obj:
            with self.assertRaises(Exception):
                mpi_np.zeros(1)


class ZerosUndistributedTest(ZerosDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        parms['int_shape'] = 4
        parms['tuple_shape'] = (5, 4)
        parms['comm'] = MPI.COMM_WORLD
        # Undistributed distribution
        parms['dist'] = 'u'
        parms['dist_class'] = Undistributed
        return parms


if __name__ == '__main__':
    unittest.main()
