import unittest
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np

from mpids.MPInumpy.errors import NotSupportedError
from mpids.MPInumpy.distributions.Undistributed import Undistributed

class MPIArrayIndexingDefaultTest(unittest.TestCase):

    def create_setUp_parms(self):
        parms = {}
        parms['comm'] = MPI.COMM_WORLD
        # Default distribution
        parms['dist'] = 'b'
        parms['data'] = np.arange(25).reshape(5,5)
        return parms


    def setUp(self):
        parms = self.create_setUp_parms()
        self.comm = parms.get('comm')
        self.dist = parms.get('dist')
        self.data = parms.get('data')

        self.mpi_array = mpi_np.array(self.data, comm=self.comm, dist=self.dist)


    def test_custom_setitem_indexing_exceptions_behavior(self):
        #Check for value errors when trying to set a value to a type
        ## that can't be converted
        with self.assertRaises(ValueError):
            self.mpi_array[0] = 'A string'

        #Check Index Error for index outside global range
        with self.assertRaises(IndexError):
            shape_axis0, _ = self.mpi_array.globalshape
            self.mpi_array[shape_axis0 + 1] = 1

        with self.assertRaises(IndexError):
            _, shape_axis1 = self.mpi_array.globalshape
            self.mpi_array[shape_axis1 + 1] = 2

        #Check Index Error for index with more dims than distributed array
        with self.assertRaises(IndexError):
            ndims = len(self.mpi_array.globalshape)
            over_index = tuple([0] * (ndims + 1))
            self.mpi_array[over_index] = 3

        #Check Not Supported Error is thrown when non int/slice/tuple key provided
        with self.assertRaises(NotSupportedError):
            self.mpi_array[[0, 1]] = 4

        #Check that not supported is raised when trying to set value(s)
        ## with multiple elements for distributed arrays
        if not isinstance(self.mpi_array, Undistributed):
            with self.assertRaises(NotSupportedError):
                self.mpi_array[:,0] = [0] * self.mpi_array.globalshape[0]

            with self.assertRaises(NotSupportedError):
                self.mpi_array[:,0] = (1, 2, 3, 4, 5)

            with self.assertRaises(NotSupportedError):
                self.mpi_array[0] = [2, 3]


    def test_custom_setitem_indexing_modify_first_row(self):
        global_first_row = 0
        self.mpi_array[global_first_row] = 9999999
        self.data[global_first_row] = 9999999
        collected_array = self.mpi_array.collect_data()
        self.assertTrue(np.alltrue(collected_array[global_first_row] == 9999999))
        #Matches numpy result
        self.assertTrue(np.alltrue(collected_array == self.data))


    def test_custom_setitem_indexing_modify_last_row(self):
        global_last_row = self.mpi_array.globalshape[0] - 1
        self.mpi_array[global_last_row] = 9999999
        self.data[global_last_row] = 9999999
        collected_array = self.mpi_array.collect_data()
        self.assertTrue(np.alltrue(collected_array[global_last_row] == 9999999))
        #Matches numpy result
        self.assertTrue(np.alltrue(collected_array == self.data))


    def test_custom_setitem_indexing_modify_first_middle_last_row(self):
        global_first_row = 0
        global_middle_row = self.mpi_array.globalshape[0] // 2
        global_last_row = self.mpi_array.globalshape[0] - 1
        self.mpi_array[global_first_row] = 9999999
        self.data[global_first_row] = 9999999
        self.mpi_array[global_middle_row] = 6666666
        self.data[global_middle_row] = 6666666
        self.mpi_array[global_last_row] = 3333333
        self.data[global_last_row] = 3333333
        collected_array = self.mpi_array.collect_data()
        self.assertTrue(
            np.alltrue(collected_array[global_first_row] == 9999999))
        self.assertTrue(
            np.alltrue(collected_array[global_middle_row] == 6666666))
        self.assertTrue(
            np.alltrue(collected_array[global_last_row] == 3333333))
        #Matches numpy result
        self.assertTrue(np.alltrue(collected_array == self.data))


    def test_custom_setitem_indexing_modify_first_col(self):
        global_first_col = 0
        self.mpi_array[:, global_first_col] = 9999999
        self.data[:, global_first_col] = 9999999
        collected_array = self.mpi_array.collect_data()
        self.assertTrue(
            np.alltrue(collected_array[:, global_first_col] == 9999999))
        #Matches numpy result
        self.assertTrue(np.alltrue(collected_array == self.data))


    def test_custom_setitem_indexing_modify_last_col(self):
        global_last_col = self.mpi_array.globalshape[1] - 1
        self.mpi_array[:, global_last_col] = 9999999
        self.data[:, global_last_col] = 9999999
        collected_array = self.mpi_array.collect_data()
        self.assertTrue(
            np.alltrue(collected_array[:, global_last_col] == 9999999))
        #Matches numpy result
        self.assertTrue(np.alltrue(collected_array == self.data))


    def test_custom_setitem_indexing_modify_first_middle_last_col(self):
        global_first_col = 0
        global_middle_col = self.mpi_array.globalshape[1] // 2
        global_last_col = self.mpi_array.globalshape[1] - 1
        self.mpi_array[:, global_first_col] = 9999999
        self.data[:, global_first_col] = 9999999
        self.mpi_array[:, global_middle_col] = 6666666
        self.data[:, global_middle_col] = 6666666
        self.mpi_array[:, global_last_col] = 3333333
        self.data[:, global_last_col] = 3333333
        collected_array = self.mpi_array.collect_data()
        self.assertTrue(
            np.alltrue(collected_array[:, global_first_col] == 9999999))
        self.assertTrue(
            np.alltrue(collected_array[:, global_middle_col] == 6666666))
        self.assertTrue(
            np.alltrue(collected_array[:, global_last_col] == 3333333))
        #Matches numpy result
        self.assertTrue(np.alltrue(collected_array == self.data))


    def test_custom_setitem_indexing_modify_individual_locations(self):
        min_row, min_col = 0, 0
        mid_row = self.mpi_array.globalshape[0] // 2
        mid_col = self.mpi_array.globalshape[1] // 2
        max_row = self.mpi_array.globalshape[0] - 1
        max_col = self.mpi_array.globalshape[1] - 1
        self.mpi_array[min_row, min_col] = -1
        self.data[min_row, min_col] = -1
        self.mpi_array[min_row, mid_col] = -2
        self.data[min_row, mid_col] = -2
        self.mpi_array[min_row, max_col] = -3
        self.data[min_row, max_col] = -3
        self.mpi_array[mid_row, min_col] = -4
        self.data[mid_row, min_col] = -4
        self.mpi_array[mid_row, mid_col] = -5
        self.data[mid_row, mid_col] = -5
        self.mpi_array[mid_row, max_col] = -6
        self.data[mid_row, max_col] = -6
        self.mpi_array[max_row, min_col] = -7
        self.data[max_row, min_col] = -7
        self.mpi_array[max_row, mid_col] = -8
        self.data[max_row, mid_col] = -8
        self.mpi_array[max_row, max_col] = -9
        self.data[max_row, max_col] = -9

        collected_array = self.mpi_array.collect_data()
        self.assertEqual(-1, collected_array[min_row, min_col])
        self.assertEqual(-2, collected_array[min_row, mid_col])
        self.assertEqual(-3, collected_array[min_row, max_col])
        self.assertEqual(-4, collected_array[mid_row, min_col])
        self.assertEqual(-5, collected_array[mid_row, mid_col])
        self.assertEqual(-6, collected_array[mid_row, max_col])
        self.assertEqual(-7, collected_array[max_row, min_col])
        self.assertEqual(-8, collected_array[max_row, mid_col])
        self.assertEqual(-9, collected_array[max_row, max_col])
        #Matches numpy result
        self.assertTrue(np.alltrue(collected_array == self.data))


    def test_custom_getitem_indexing_exceptions_behavior(self):
        #Check Index Error for index outside global range
        with self.assertRaises(IndexError):
            shape_axis0, _ = self.mpi_array.globalshape
            self.mpi_array[shape_axis0 + 1]

        with self.assertRaises(IndexError):
            _, shape_axis1 = self.mpi_array.globalshape
            self.mpi_array[shape_axis1 + 1]

        #Check Index Error for index with more dims than distributed array
        with self.assertRaises(IndexError):
            ndims = len(self.mpi_array.globalshape)
            over_index = tuple([0] * (ndims + 1))
            self.mpi_array[over_index]

        #Check Not Supported Error is thrown when non int/slice/tuple key provided
        with self.assertRaises(NotSupportedError):
            self.mpi_array[[0, 1]]


    def test_custom_getitem_full_copy_return(self):
        returned_array = self.mpi_array[:]

        self.assertTrue(isinstance(returned_array, mpi_np.MPIArray))
        self.assertTrue(isinstance(returned_array, Undistributed))
        self.assertEqual(returned_array.comm, self.mpi_array.comm)
        self.assertEqual(returned_array.dist, 'u')
        self.assertTrue(returned_array.comm_dims is None)
        self.assertTrue(returned_array.comm_coord is None)
        self.assertTrue(returned_array.local_to_global is None)
        self.assertEqual(returned_array.globalshape, self.mpi_array.globalshape)
        self.assertEqual(returned_array.globalsize, self.mpi_array.globalsize)
        self.assertEqual(returned_array.globalnbytes, self.mpi_array.globalnbytes)
        self.assertEqual(returned_array.globalndim, self.mpi_array.globalndim)

        self.assertEqual(returned_array.shape, self.data.shape)
        self.assertEqual(returned_array.size, self.data.size)
        self.assertEqual(returned_array.nbytes, self.data.nbytes)
        self.assertEqual(returned_array.ndim, self.data.ndim)
        self.assertTrue(np.alltrue(returned_array == self.data))


    def test_custom_getitem_first_row_return(self):
        first_row = 0
        returned_array = self.mpi_array[first_row]

        self.assertTrue(isinstance(returned_array, mpi_np.MPIArray))
        self.assertTrue(isinstance(returned_array, Undistributed))
        self.assertEqual(returned_array.comm, self.mpi_array.comm)
        self.assertEqual(returned_array.dist, 'u')
        self.assertTrue(returned_array.comm_dims is None)
        self.assertTrue(returned_array.comm_coord is None)
        self.assertTrue(returned_array.local_to_global is None)
        self.assertEqual(returned_array.globalshape, self.data[first_row].shape)
        self.assertEqual(returned_array.globalsize, self.data[first_row].size)
        self.assertEqual(returned_array.globalnbytes, self.data[first_row].nbytes)
        self.assertEqual(returned_array.globalndim, self.data[first_row].ndim)

        self.assertEqual(returned_array.shape, self.data[first_row].shape)
        self.assertEqual(returned_array.size, self.data[first_row].size)
        self.assertEqual(returned_array.nbytes, self.data[first_row].nbytes)
        self.assertEqual(returned_array.ndim, self.data[first_row].ndim)
        self.assertTrue(np.alltrue(returned_array == self.data[first_row]))


    def test_custom_getitem_middle_row_slice_return(self):
        returned_array = self.mpi_array[2:3]

        self.assertTrue(isinstance(returned_array, mpi_np.MPIArray))
        self.assertTrue(isinstance(returned_array, Undistributed))
        self.assertEqual(returned_array.comm, self.mpi_array.comm)
        self.assertEqual(returned_array.dist, 'u')
        self.assertTrue(returned_array.comm_dims is None)
        self.assertTrue(returned_array.comm_coord is None)
        self.assertTrue(returned_array.local_to_global is None)
        self.assertEqual(returned_array.globalshape, self.data[2:3].shape)
        self.assertEqual(returned_array.globalsize, self.data[2:3].size)
        self.assertEqual(returned_array.globalndim, self.data[2:4].ndim)

        self.assertEqual(returned_array.shape, self.data[2:3].shape)
        self.assertEqual(returned_array.size, self.data[2:3].size)
        self.assertEqual(returned_array.nbytes, self.data[2:3].nbytes)
        self.assertEqual(returned_array.ndim, self.data[2:3].ndim)
        self.assertTrue(np.alltrue(returned_array == self.data[2:3]))


    def test_custom_getitem_last_row_return(self):
        last_row = self.mpi_array.globalshape[0] - 1
        returned_array = self.mpi_array[last_row]

        self.assertTrue(isinstance(returned_array, mpi_np.MPIArray))
        self.assertTrue(isinstance(returned_array, Undistributed))
        self.assertEqual(returned_array.comm, self.mpi_array.comm)
        self.assertEqual(returned_array.dist, 'u')
        self.assertTrue(returned_array.comm_dims is None)
        self.assertTrue(returned_array.comm_coord is None)
        self.assertTrue(returned_array.local_to_global is None)
        self.assertEqual(returned_array.globalshape, self.data[last_row].shape)
        self.assertEqual(returned_array.globalsize, self.data[last_row].size)
        self.assertEqual(returned_array.globalnbytes, self.data[last_row].nbytes)
        self.assertEqual(returned_array.globalndim, self.data[last_row].ndim)

        self.assertEqual(returned_array.shape, self.data[last_row].shape)
        self.assertEqual(returned_array.size, self.data[last_row].size)
        self.assertEqual(returned_array.nbytes, self.data[last_row].nbytes)
        self.assertEqual(returned_array.ndim, self.data[last_row].ndim)
        self.assertTrue(np.alltrue(returned_array == self.data[last_row]))


    def test_custom_getitem_first_half_rows_return(self):
        first_row = 0
        middle_row = self.mpi_array.globalshape[0] // 2
        returned_array = self.mpi_array[first_row:middle_row]

        self.assertTrue(isinstance(returned_array, mpi_np.MPIArray))
        self.assertTrue(isinstance(returned_array, Undistributed))
        self.assertEqual(returned_array.comm, self.mpi_array.comm)
        self.assertEqual(returned_array.dist, 'u')
        self.assertTrue(returned_array.comm_dims is None)
        self.assertTrue(returned_array.comm_coord is None)
        self.assertTrue(returned_array.local_to_global is None)
        self.assertEqual(returned_array.globalshape, self.data[first_row:middle_row].shape)
        self.assertEqual(returned_array.globalsize, self.data[first_row:middle_row].size)
        self.assertEqual(returned_array.globalnbytes, self.data[first_row:middle_row].nbytes)
        self.assertEqual(returned_array.globalndim, self.data[first_row:middle_row].ndim)

        self.assertEqual(returned_array.shape, self.data[first_row:middle_row].shape)
        self.assertEqual(returned_array.size, self.data[first_row:middle_row].size)
        self.assertEqual(returned_array.nbytes, self.data[first_row:middle_row].nbytes)
        self.assertEqual(returned_array.ndim, self.data[first_row:middle_row].ndim)
        self.assertTrue(np.alltrue(returned_array == self.data[first_row:middle_row]))


    def test_custom_getitem_last_half_rows_return(self):
        middle_row = self.mpi_array.globalshape[0] // 2
        last_row = self.mpi_array.globalshape[0] - 1
        returned_array = self.mpi_array[middle_row:last_row]

        self.assertTrue(isinstance(returned_array, mpi_np.MPIArray))
        self.assertTrue(isinstance(returned_array, Undistributed))
        self.assertEqual(returned_array.comm, self.mpi_array.comm)
        self.assertEqual(returned_array.dist, 'u')
        self.assertTrue(returned_array.comm_dims is None)
        self.assertTrue(returned_array.comm_coord is None)
        self.assertTrue(returned_array.local_to_global is None)
        self.assertEqual(returned_array.globalshape, self.data[middle_row:last_row].shape)
        self.assertEqual(returned_array.globalsize, self.data[middle_row:last_row].size)
        self.assertEqual(returned_array.globalnbytes, self.data[middle_row:last_row].nbytes)
        self.assertEqual(returned_array.globalndim, self.data[middle_row:last_row].ndim)

        self.assertEqual(returned_array.shape, self.data[middle_row:last_row].shape)
        self.assertEqual(returned_array.size, self.data[middle_row:last_row].size)
        self.assertEqual(returned_array.nbytes, self.data[middle_row:last_row].nbytes)
        self.assertEqual(returned_array.ndim, self.data[middle_row:last_row].ndim)
        self.assertTrue(np.alltrue(returned_array == self.data[middle_row:last_row]))


    def test_custom_getitem_first_column_return(self):
        first_col = 0
        returned_array = self.mpi_array[:, first_col]

        self.assertTrue(isinstance(returned_array, mpi_np.MPIArray))
        self.assertTrue(isinstance(returned_array, Undistributed))
        self.assertEqual(returned_array.comm, self.mpi_array.comm)
        self.assertEqual(returned_array.dist, 'u')
        self.assertTrue(returned_array.comm_dims is None)
        self.assertTrue(returned_array.comm_coord is None)
        self.assertTrue(returned_array.local_to_global is None)
        self.assertEqual(returned_array.globalshape, self.data[:, first_col].shape)
        self.assertEqual(returned_array.globalsize, self.data[:, first_col].size)
        self.assertEqual(returned_array.globalnbytes, self.data[:, first_col].nbytes)
        self.assertEqual(returned_array.globalndim, self.data[:, first_col].ndim)

        self.assertEqual(returned_array.shape, self.data[:, first_col].shape)
        self.assertEqual(returned_array.size, self.data[:, first_col].size)
        self.assertEqual(returned_array.nbytes, self.data[:, first_col].nbytes)
        self.assertEqual(returned_array.ndim, self.data[:, first_col].ndim)
        self.assertTrue(np.alltrue(returned_array == self.data[:, first_col]))


    def test_custom_getitem_middle_column_return(self):
        returned_array = self.mpi_array[:, 2:3]

        self.assertTrue(isinstance(returned_array, mpi_np.MPIArray))
        self.assertTrue(isinstance(returned_array, Undistributed))
        self.assertEqual(returned_array.comm, self.mpi_array.comm)
        self.assertEqual(returned_array.dist, 'u')
        self.assertTrue(returned_array.comm_dims is None)
        self.assertTrue(returned_array.comm_coord is None)
        self.assertTrue(returned_array.local_to_global is None)
        self.assertEqual(returned_array.globalshape, self.data[:, 2:3].shape)
        self.assertEqual(returned_array.globalsize, self.data[:, 2:3].size)
        self.assertEqual(returned_array.globalndim, self.data[:, 2:3].ndim)

        self.assertEqual(returned_array.shape, self.data[:, 2:3].shape)
        self.assertEqual(returned_array.size, self.data[:, 2:3].size)
        self.assertEqual(returned_array.nbytes, self.data[:, 2:3].nbytes)
        self.assertEqual(returned_array.ndim, self.data[:, 2:3].ndim)
        self.assertTrue(np.alltrue(returned_array == self.data[:, 2:3]))


    def test_custom_getitem_last_column_return(self):
        last_col = self.mpi_array.globalshape[1] - 1
        returned_array = self.mpi_array[:, last_col]

        self.assertTrue(isinstance(returned_array, mpi_np.MPIArray))
        self.assertTrue(isinstance(returned_array, Undistributed))
        self.assertEqual(returned_array.comm, self.mpi_array.comm)
        self.assertEqual(returned_array.dist, 'u')
        self.assertTrue(returned_array.comm_dims is None)
        self.assertTrue(returned_array.comm_coord is None)
        self.assertTrue(returned_array.local_to_global is None)
        self.assertEqual(returned_array.globalshape, self.data[:, last_col].shape)
        self.assertEqual(returned_array.globalsize, self.data[:, last_col].size)
        self.assertEqual(returned_array.globalndim, self.data[:, last_col].ndim)

        self.assertEqual(returned_array.shape, self.data[:, last_col].shape)
        self.assertEqual(returned_array.size, self.data[:, last_col].size)
        self.assertEqual(returned_array.nbytes, self.data[:, last_col].nbytes)
        self.assertEqual(returned_array.ndim, self.data[:, last_col].ndim)
        self.assertTrue(np.alltrue(returned_array == self.data[:, last_col]))


    def test_custom_getitem_first_half_columns_return(self):
        first_col = 0
        middle_col = (self.mpi_array.globalshape[1] - 1) // 2
        returned_array = self.mpi_array[:, first_col:middle_col]

        self.assertTrue(isinstance(returned_array, mpi_np.MPIArray))
        self.assertTrue(isinstance(returned_array, Undistributed))
        self.assertEqual(returned_array.comm, self.mpi_array.comm)
        self.assertEqual(returned_array.dist, 'u')
        self.assertTrue(returned_array.comm_dims is None)
        self.assertTrue(returned_array.comm_coord is None)
        self.assertTrue(returned_array.local_to_global is None)
        self.assertEqual(returned_array.globalshape, self.data[:, first_col:middle_col].shape)
        self.assertEqual(returned_array.globalsize, self.data[:, first_col:middle_col].size)
        self.assertEqual(returned_array.globalndim, self.data[:, first_col:middle_col].ndim)

        self.assertEqual(returned_array.shape, self.data[:, first_col:middle_col].shape)
        self.assertEqual(returned_array.size, self.data[:, first_col:middle_col].size)
        self.assertEqual(returned_array.nbytes, self.data[:, first_col:middle_col].nbytes)
        self.assertEqual(returned_array.ndim, self.data[:, first_col:middle_col].ndim)
        self.assertTrue(np.alltrue(returned_array == self.data[:, first_col:middle_col]))


    def test_custom_getitem_last_half_columns_column_return(self):
        middle_col = (self.mpi_array.globalshape[1] - 1) // 2
        last_col = self.mpi_array.globalshape[1] - 1
        returned_array = self.mpi_array[:, middle_col:last_col]

        self.assertTrue(isinstance(returned_array, mpi_np.MPIArray))
        self.assertTrue(isinstance(returned_array, Undistributed))
        self.assertEqual(returned_array.comm, self.mpi_array.comm)
        self.assertEqual(returned_array.dist, 'u')
        self.assertTrue(returned_array.comm_dims is None)
        self.assertTrue(returned_array.comm_coord is None)
        self.assertTrue(returned_array.local_to_global is None)
        self.assertEqual(returned_array.globalshape, self.data[:, middle_col:last_col].shape)
        self.assertEqual(returned_array.globalsize, self.data[:, middle_col:last_col].size)
        self.assertEqual(returned_array.globalndim, self.data[:, middle_col:last_col].ndim)

        self.assertEqual(returned_array.shape, self.data[:, middle_col:last_col].shape)
        self.assertEqual(returned_array.size, self.data[:, middle_col:last_col].size)
        self.assertEqual(returned_array.nbytes, self.data[:, middle_col:last_col].nbytes)
        self.assertEqual(returned_array.ndim, self.data[:, middle_col:last_col].ndim)
        self.assertTrue(np.alltrue(returned_array == self.data[:, middle_col:last_col]))


class MPIArrayIndexingUndistributedTest(MPIArrayIndexingDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        parms['comm'] = MPI.COMM_WORLD
        # Undistributed distribution
        parms['dist'] = 'u'
        parms['data'] = np.arange(25).reshape(5,5)
        return parms


if __name__ == '__main__':
    unittest.main()
