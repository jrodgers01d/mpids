import unittest
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np
from mpids.MPInumpy.errors import ValueError, NotSupportedError

class MPIArrayTest(unittest.TestCase):
#REMINDER!!!
#THESE TESTS ARE NOT DISTRIBUTED, DISTRIBUTION IS HANDLED BY THE ARRAY
#CREATION OBJECTS
        def setUp(self):
                self.comm = MPI.COMM_WORLD
                self.dist = 'u'
                self.comm_size = MPI.COMM_WORLD.Get_size()
                #Add 1 to avoid divide by zero errors/warnings
                self.data = (np.array(list(range(16))).reshape(4,4) + 1).tolist()
                self.np_array = np.array(self.data)
                self.mpi_array = mpi_np.MPIArray(self.data, comm=self.comm, dist=self.dist)
                self.scalar_data = 1
                self.np_scalar = np.array(self.scalar_data)
                self.mpi_scalar = mpi_np.MPIArray(self.scalar_data, comm=self.comm, dist='u')


        def test_properties(self):
                #Unique properties to MPIArray
                self.assertEqual(self.comm, self.mpi_array.comm)
                self.assertEqual(self.dist, self.mpi_array.dist)
                self.assertEqual(None, self.mpi_array.comm_dims)
                self.assertEqual(None, self.mpi_array.comm_coord)
                self.assertEqual(self.np_array.size, self.mpi_array.globalsize)
                self.assertEqual(self.np_array.nbytes, self.mpi_array.globalnbytes)
                self.assertEqual(self.np_array.shape,  self.mpi_array.globalshape)

                #Replicated numpy.ndarray properties
                self.assertTrue(np.alltrue(self.np_array.T == self.mpi_array.T))
                self.assertEqual(self.np_array.data, self.mpi_array.data)
                self.assertEqual(self.np_array.dtype, self.mpi_array.dtype)
                self.assertTrue(np.alltrue(self.np_array.imag == self.mpi_array.imag))
                self.assertTrue(np.alltrue(self.np_array.real == self.mpi_array.real))
                self.assertEqual(self.np_array.size, self.mpi_array.size)
                self.assertEqual(self.np_array.itemsize, self.mpi_array.itemsize)
                self.assertEqual(self.np_array.nbytes, self.mpi_array.nbytes)
                self.assertEqual(self.np_array.ndim, self.mpi_array.ndim)
                self.assertEqual(self.np_array.shape, self.mpi_array.shape)
                self.assertEqual(self.np_array.strides, self.mpi_array.strides)
                self.assertEqual('[[ 1  2  3  4]\n [ 5  6  7  8]\n [ 9 10 11 12]\n [13 14 15 16]]',
                                 str(self.mpi_array.base))


        def test_properties_under_different_distributions(self):
                row_block_dist = 'b'
                row_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=row_block_dist)
                self.assertEqual(row_block_dist, row_block_mpi_array.dist)
                self.assertEqual([self.comm_size], row_block_mpi_array.comm_dims)
                self.assertEqual([self.comm.Get_rank()], row_block_mpi_array.comm_coord)
                self.assertEqual(self.np_array.size, row_block_mpi_array.globalsize)
                self.assertEqual(self.np_array.nbytes, row_block_mpi_array.globalnbytes)
                self.assertEqual(self.np_array.shape, tuple(row_block_mpi_array.globalshape))

                alt_row_block_dist = ('b', '*')
                alt_row_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=alt_row_block_dist)
                self.assertEqual(alt_row_block_dist, alt_row_block_mpi_array.dist)
                self.assertEqual([self.comm_size], alt_row_block_mpi_array.comm_dims)
                self.assertEqual([self.comm.Get_rank()], alt_row_block_mpi_array.comm_coord)
                self.assertEqual(self.np_array.size, alt_row_block_mpi_array.globalsize)
                self.assertEqual(self.np_array.nbytes, alt_row_block_mpi_array.globalnbytes)
                self.assertEqual(self.np_array.shape, tuple(alt_row_block_mpi_array.globalshape))

                col_block_dist = ('*', 'b')
                col_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=col_block_dist)
                self.assertEqual(col_block_dist, col_block_mpi_array.dist)
                self.assertEqual([1, self.comm_size], col_block_mpi_array.comm_dims)
                self.assertEqual([0, self.comm.Get_rank()], col_block_mpi_array.comm_coord)
                self.assertEqual(self.np_array.size, col_block_mpi_array.globalsize)
                self.assertEqual(self.np_array.nbytes, col_block_mpi_array.globalnbytes)
                self.assertEqual(self.np_array.shape, tuple(col_block_mpi_array.globalshape))

                rank_coord_map = {0: [0, 0],
                                  1: [0, 1],
                                  2: [1, 0],
                                  3: [1, 1]}
                block_block_dist = ('b', 'b')
                block_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=block_block_dist)
                self.assertEqual(block_block_dist, block_block_mpi_array.dist)
                self.assertEqual([2, 2], block_block_mpi_array.comm_dims)
                self.assertEqual(rank_coord_map[self.comm.Get_rank()], block_block_mpi_array.comm_coord)
                self.assertEqual(self.np_array.size, block_block_mpi_array.globalsize)
                self.assertEqual(self.np_array.nbytes, block_block_mpi_array.globalnbytes)
                self.assertEqual(self.np_array.shape, tuple(block_block_mpi_array.globalshape))


        def test_dunder_methods(self):
                self.assertEqual('MPIArray(globalsize={}, globalshape={}, dist={}, dtype={})'\
                                    .format(self.mpi_array.globalsize, list(self.mpi_array.globalshape),
                                            self.dist, self.mpi_array.dtype)
                                 , self.mpi_array.__repr__())
                self.assertEqual(None, self.mpi_array.__array_finalize__(None))
                self.assertEqual('[[ 1  2  3  4]\n [ 5  6  7  8]\n [ 9 10 11 12]\n [13 14 15 16]]',
                                 self.mpi_array.__str__())
                self.assertTrue(np.alltrue(self.np_array == self.mpi_array.__array__()))


        def test_dunder_binary_operations(self):
                self.assertTrue(np.alltrue((self.np_array + 2) == (self.mpi_array + 2)))
                self.assertTrue(np.alltrue((3 + self.np_array) == (3 + self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_array - 2) == (self.mpi_array - 2)))
                self.assertTrue(np.alltrue((3 - self.np_array) == (3 - self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_array * 2) == (self.mpi_array * 2)))
                self.assertTrue(np.alltrue((3 * self.np_array) == (3 * self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_array // 2) == (self.mpi_array // 2)))
                self.assertTrue(np.alltrue((3 // self.np_array) == (3 // self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_array / 2) == (self.mpi_array / 2)))
                self.assertTrue(np.alltrue((3 / self.np_array) == (3 / self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_array % 2) == (self.mpi_array % 2)))
                self.assertTrue(np.alltrue((3 % self.np_array) == (3 % self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_array ** 2) == (self.mpi_array ** 2)))
                self.assertTrue(np.alltrue((3 ** self.np_array) == (3 ** self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_array << 2) == (self.mpi_array << 2)))
                self.assertTrue(np.alltrue((3 << self.np_array) == (3 << self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_array >> 2) == (self.mpi_array >> 2)))
                self.assertTrue(np.alltrue((3 >> self.np_array) == (3 >> self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_array & 2) == (self.mpi_array & 2)))
                self.assertTrue(np.alltrue((3 & self.np_array) == (3 & self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_array | 2) == (self.mpi_array | 2)))
                self.assertTrue(np.alltrue((3 | self.np_array) == (3 | self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_array ^ 2) == (self.mpi_array ^ 2)))
                self.assertTrue(np.alltrue((3 ^ self.np_array) == (3 ^ self.mpi_array)))


        def test_dunder_unary_operations(self):
                self.assertTrue(np.alltrue((-self.np_array) == (-self.mpi_array)))
                self.assertTrue(np.alltrue((+self.np_array) == (+self.mpi_array)))
                self.assertTrue(np.alltrue(abs(self.np_array) == abs(self.mpi_array)))
                self.assertTrue(np.alltrue((~self.np_array) == (~self.mpi_array)))
                self.assertEqual(complex(self.np_scalar), complex(self.mpi_scalar))
                self.assertEqual(int(self.np_scalar), int(self.mpi_scalar))
                self.assertEqual(float(self.np_scalar), float(self.mpi_scalar))
                self.assertEqual(oct(self.np_scalar), oct(self.mpi_scalar))
                self.assertEqual(hex(self.np_scalar), hex(self.mpi_scalar))


        def test_dunder_comparison_operations(self):
                self.assertTrue(np.alltrue((2 > self.np_array) == (2 > self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_array < 2) == (self.mpi_array < 2)))
                self.assertTrue(np.alltrue((2 >= self.np_array) == (2 >= self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_array <= 2) == (self.mpi_array <= 2)))
                self.assertTrue(np.alltrue((1 == self.np_scalar) == (1 == self.mpi_scalar)))
                self.assertTrue(np.alltrue((self.np_scalar == 1) == (self.mpi_scalar == 1)))
                self.assertTrue(np.alltrue((0 != self.np_scalar) == (0 != self.mpi_scalar)))
                self.assertTrue(np.alltrue((self.np_scalar != 0) == (self.mpi_scalar != 0)))
                self.assertTrue(np.alltrue((2 < self.np_array) == (2 < self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_array > 2) == (self.mpi_array > 2)))
                self.assertTrue(np.alltrue((2 <= self.np_array) == (2 <= self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_array >= 2) == (self.mpi_array >= 2)))


        def test_object_return_behavior(self):
                returned_array = self.mpi_array

                self.assertTrue(np.alltrue((returned_array) == (self.mpi_array)))
                self.assertTrue(returned_array is self.mpi_array)
                self.assertEqual(returned_array.comm, self.mpi_array.comm)
                self.assertEqual(returned_array.globalsize, self.mpi_array.globalsize)
                self.assertEqual(returned_array.globalnbytes, self.mpi_array.globalnbytes)


        def test_object_slicing_behavior(self):
                first_row = self.mpi_array[0]
                last_row = self.mpi_array[self.mpi_array.shape[0] - 1]

                self.assertTrue(first_row is not self.mpi_array)
                self.assertTrue(isinstance(first_row, mpi_np.MPIArray))
                self.assertTrue(first_row.base is self.mpi_array)
                self.assertEqual(self.np_array[0].size, first_row.globalsize)
                self.assertEqual(self.np_array[0].nbytes, first_row.globalnbytes)

                self.assertTrue(last_row is not self.mpi_array)
                self.assertTrue(last_row.base is self.mpi_array)
                self.assertTrue(isinstance(last_row, mpi_np.MPIArray))
                self.assertEqual(self.np_array[self.np_array.shape[0] - 1].size, last_row.globalsize)
                self.assertEqual(self.np_array[self.np_array.shape[0] - 1].nbytes, first_row.globalnbytes)

                first_half_first_row = first_row[:len(first_row) // 2]
                second_half_last_row = last_row[len(last_row) // 2:]

                self.assertTrue(first_half_first_row is not first_row)
                self.assertTrue(first_half_first_row.base is self.mpi_array)
                self.assertTrue(isinstance(first_half_first_row, mpi_np.MPIArray))
                self.assertEqual(self.np_array[0, :len(self.np_array[0]) // 2].size, first_half_first_row.globalsize)
                self.assertEqual(self.np_array[0, :len(self.np_array[0]) // 2].nbytes, first_half_first_row.globalnbytes)

                self.assertTrue(second_half_last_row is not last_row)
                self.assertTrue(second_half_last_row.base is self.mpi_array)
                self.assertTrue(isinstance(second_half_last_row, mpi_np.MPIArray))
                self.assertEqual(self.np_array[1, len(self.np_array[1]) // 2:].size, second_half_last_row.globalsize)
                self.assertEqual(self.np_array[1, len(self.np_array[1]) // 2:].nbytes, second_half_last_row.globalnbytes)

                first_column = self.mpi_array[:,[0]]
                last_column = self.mpi_array[:,[self.mpi_array.shape[1] - 1]]

                self.assertTrue(first_column is not self.mpi_array)
                self.assertTrue(first_column.base is not self.mpi_array)
                self.assertTrue(isinstance(first_column, mpi_np.MPIArray))
                self.assertEqual(self.np_array[:,[0]].size, first_column.globalsize)
                self.assertEqual(self.np_array[:,[0]].nbytes, first_column.globalnbytes)

                self.assertTrue(last_column is not self.mpi_array)
                self.assertTrue(last_column.base is not self.mpi_array)
                self.assertTrue(isinstance(last_column, mpi_np.MPIArray))
                self.assertEqual(self.np_array[:,[self.np_array.shape[1] - 1]].size, last_column.globalsize)
                self.assertEqual(self.np_array[:,[self.np_array.shape[1] - 1]].nbytes, last_column.globalnbytes)


# TODO REVIEW REDUCTIONS FOR UNDISTRIBUTED BEHAVIOR
        def test_custom_sum_method(self):
                #Default sum of entire array contents
                self.assertEqual(self.np_array.sum(), self.mpi_array.sum())

                #Modified output datatype
                self.assertEqual(self.np_array.sum(dtype=np.dtype(int)), self.mpi_array.sum(dtype=np.dtype(int)))
                self.assertEqual(self.np_array.sum(dtype=np.dtype(float)), self.mpi_array.sum(dtype=np.dtype(float)))
                self.assertEqual(self.np_array.sum(dtype=np.dtype(complex)), self.mpi_array.sum(dtype=np.dtype(complex)))
                self.assertEqual(self.np_array.sum(dtype=np.dtype('f8')), self.mpi_array.sum(dtype=np.dtype('f8')))
                self.assertEqual(self.np_array.sum(dtype=np.dtype('c16')), self.mpi_array.sum(dtype=np.dtype('c16')))

                #Sum along specified axies
                self.assertTrue(np.alltrue(self.np_array.sum(axis=0) == self.mpi_array.sum(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.sum(axis=1) == self.mpi_array.sum(axis=1)))
                with self.assertRaises(ValueError):
                        self.mpi_array.sum(axis=self.mpi_array.ndim)

                #Use of 'out' field
                mpi_out = np.zeros(())
                with self.assertRaises(NotSupportedError):
                        self.mpi_array.sum(out=mpi_out)

        def test_custom_sum_under_different_distributions(self):
                row_block_dist = 'b'
                row_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=row_block_dist)
                #Default sum of entire array contents
                self.assertEqual(self.np_array.sum(), row_block_mpi_array.sum())

                #Modified output datatype
                self.assertEqual(self.np_array.sum(dtype=np.dtype(int)), row_block_mpi_array.sum(dtype=np.dtype(int)))
                self.assertEqual(self.np_array.sum(dtype=np.dtype(float)), row_block_mpi_array.sum(dtype=np.dtype(float)))
                self.assertEqual(self.np_array.sum(dtype=np.dtype(complex)), row_block_mpi_array.sum(dtype=np.dtype(complex)))
                self.assertEqual(self.np_array.sum(dtype=np.dtype('f8')), row_block_mpi_array.sum(dtype=np.dtype('f8')))
                self.assertEqual(self.np_array.sum(dtype=np.dtype('c16')), row_block_mpi_array.sum(dtype=np.dtype('c16')))

                #Sum along specified axies
                self.assertTrue(np.alltrue(self.np_array.sum(axis=0) == row_block_mpi_array.sum(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.sum(axis=1) == row_block_mpi_array.sum(axis=1)))
                with self.assertRaises(ValueError):
                        row_block_mpi_array.sum(axis=row_block_mpi_array.ndim)

                alt_row_block_dist = ('b', '*')
                alt_row_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=alt_row_block_dist)
                #Default sum of entire array contents
                self.assertEqual(self.np_array.sum(), alt_row_block_mpi_array.sum())

                #Modified output datatype
                self.assertEqual(self.np_array.sum(dtype=np.dtype(int)), alt_row_block_mpi_array.sum(dtype=np.dtype(int)))
                self.assertEqual(self.np_array.sum(dtype=np.dtype(float)), alt_row_block_mpi_array.sum(dtype=np.dtype(float)))
                self.assertEqual(self.np_array.sum(dtype=np.dtype(complex)), alt_row_block_mpi_array.sum(dtype=np.dtype(complex)))
                self.assertEqual(self.np_array.sum(dtype=np.dtype('f8')), alt_row_block_mpi_array.sum(dtype=np.dtype('f8')))
                self.assertEqual(self.np_array.sum(dtype=np.dtype('c16')), alt_row_block_mpi_array.sum(dtype=np.dtype('c16')))

                #Sum along specified axies
                self.assertTrue(np.alltrue(self.np_array.sum(axis=0) == alt_row_block_mpi_array.sum(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.sum(axis=1) == alt_row_block_mpi_array.sum(axis=1)))
                with self.assertRaises(ValueError):
                        alt_row_block_mpi_array.sum(axis=alt_row_block_mpi_array.ndim)

                col_block_dist = ('*', 'b')
                col_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=col_block_dist)
                #Default sum of entire array contents
                self.assertEqual(self.np_array.sum(), alt_row_block_mpi_array.sum())

                #Modified output datatype
                self.assertEqual(self.np_array.sum(dtype=np.dtype(int)), col_block_mpi_array.sum(dtype=np.dtype(int)))
                self.assertEqual(self.np_array.sum(dtype=np.dtype(float)), col_block_mpi_array.sum(dtype=np.dtype(float)))
                self.assertEqual(self.np_array.sum(dtype=np.dtype(complex)), col_block_mpi_array.sum(dtype=np.dtype(complex)))
                self.assertEqual(self.np_array.sum(dtype=np.dtype('f8')), col_block_mpi_array.sum(dtype=np.dtype('f8')))
                self.assertEqual(self.np_array.sum(dtype=np.dtype('c16')), col_block_mpi_array.sum(dtype=np.dtype('c16')))

                #Sum along specified axies
                self.assertTrue(np.alltrue(self.np_array.sum(axis=0) == col_block_mpi_array.sum(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.sum(axis=1) == col_block_mpi_array.sum(axis=1)))
                with self.assertRaises(ValueError):
                        col_block_mpi_array.sum(axis=col_block_mpi_array.ndim)

                block_block_dist = ('b', 'b')
                block_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=block_block_dist)
                #Default sum of entire array contents
                self.assertEqual(self.np_array.sum(), block_block_mpi_array.sum())

                #Modified output datatype
                self.assertEqual(self.np_array.sum(dtype=np.dtype(int)), block_block_mpi_array.sum(dtype=np.dtype(int)))
                self.assertEqual(self.np_array.sum(dtype=np.dtype(float)), block_block_mpi_array.sum(dtype=np.dtype(float)))
                self.assertEqual(self.np_array.sum(dtype=np.dtype(complex)), block_block_mpi_array.sum(dtype=np.dtype(complex)))
                self.assertEqual(self.np_array.sum(dtype=np.dtype('f8')), block_block_mpi_array.sum(dtype=np.dtype('f8')))
                self.assertEqual(self.np_array.sum(dtype=np.dtype('c16')), block_block_mpi_array.sum(dtype=np.dtype('c16')))

                #Sum along specified axies
                self.assertTrue(np.alltrue(self.np_array.sum(axis=0) == block_block_mpi_array.sum(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.sum(axis=1) == block_block_mpi_array.sum(axis=1)))
                with self.assertRaises(ValueError):
                        block_block_mpi_array.sum(axis=block_block_mpi_array.ndim)


        def test_custom_min_method(self):
                #Default min of entire array contents
                self.assertEqual(self.np_array.min(), self.mpi_array.min())

                #Min along specified axies
                self.assertTrue(np.alltrue(self.np_array.min(axis=0) == self.mpi_array.min(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.min(axis=1) == self.mpi_array.min(axis=1)))
                with self.assertRaises(ValueError):
                        self.mpi_array.min(axis=self.mpi_array.ndim)

                #Use of 'out' field
                mpi_out = np.zeros(())
                with self.assertRaises(NotSupportedError):
                        self.mpi_array.min(out=mpi_out)


        def test_custom_min_under_different_distributions(self):
                row_block_dist = 'b'
                row_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=row_block_dist)
                #Default min of entire array contents
                self.assertEqual(self.np_array.min(), row_block_mpi_array.min())
                #Min along specified axies
                self.assertTrue(np.alltrue(self.np_array.min(axis=0) == row_block_mpi_array.min(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.min(axis=1) == row_block_mpi_array.min(axis=1)))
                with self.assertRaises(ValueError):
                        row_block_mpi_array.min(axis=row_block_mpi_array.ndim)

                alt_row_block_dist = ('b', '*')
                alt_row_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=alt_row_block_dist)
                #Default min of entire array contents
                self.assertEqual(self.np_array.min(), alt_row_block_mpi_array.min())
                #Min along specified axies
                self.assertTrue(np.alltrue(self.np_array.min(axis=0) == alt_row_block_mpi_array.min(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.min(axis=1) == alt_row_block_mpi_array.min(axis=1)))
                with self.assertRaises(ValueError):
                        alt_row_block_mpi_array.min(axis=alt_row_block_mpi_array.ndim)

                col_block_dist = ('*', 'b')
                col_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=col_block_dist)
                #Default min of entire array contents
                self.assertEqual(self.np_array.min(), alt_row_block_mpi_array.min())
                #Min along specified axies
                self.assertTrue(np.alltrue(self.np_array.min(axis=0) == col_block_mpi_array.min(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.min(axis=1) == col_block_mpi_array.min(axis=1)))
                with self.assertRaises(ValueError):
                        col_block_mpi_array.min(axis=col_block_mpi_array.ndim)

                block_block_dist = ('b', 'b')
                block_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=block_block_dist)
                #Default min of entire array contents
                self.assertEqual(self.np_array.min(), block_block_mpi_array.min())
                #Min along specified axies
                self.assertTrue(np.alltrue(self.np_array.min(axis=0) == block_block_mpi_array.min(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.min(axis=1) == block_block_mpi_array.min(axis=1)))
                with self.assertRaises(ValueError):
                        block_block_mpi_array.min(axis=block_block_mpi_array.ndim)


        def test_custom_max_method(self):
                #Default max of entire array contents
                self.assertEqual(self.np_array.max(), self.mpi_array.max())

                #Max along specified axies
                self.assertTrue(np.alltrue(self.np_array.max(axis=0) == self.mpi_array.max(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.max(axis=1) == self.mpi_array.max(axis=1)))
                with self.assertRaises(ValueError):
                        self.mpi_array.max(axis=self.mpi_array.ndim)

                #Use of 'out' field
                mpi_out = np.zeros(())
                with self.assertRaises(NotSupportedError):
                        self.mpi_array.max(out=mpi_out)

        def test_custom_max_under_different_distributions(self):
                row_block_dist = 'b'
                row_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=row_block_dist)
                #Default max of entire array contents
                self.assertEqual(self.np_array.max(), row_block_mpi_array.max())
                #Max along specified axies
                self.assertTrue(np.alltrue(self.np_array.max(axis=0) == row_block_mpi_array.max(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.max(axis=1) == row_block_mpi_array.max(axis=1)))
                with self.assertRaises(ValueError):
                        row_block_mpi_array.max(axis=row_block_mpi_array.ndim)

                alt_row_block_dist = ('b', '*')
                alt_row_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=alt_row_block_dist)
                #Default max of entire array contents
                self.assertEqual(self.np_array.max(), alt_row_block_mpi_array.max())
                #Max along specified axies
                self.assertTrue(np.alltrue(self.np_array.max(axis=0) == alt_row_block_mpi_array.max(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.max(axis=1) == alt_row_block_mpi_array.max(axis=1)))
                with self.assertRaises(ValueError):
                        alt_row_block_mpi_array.max(axis=alt_row_block_mpi_array.ndim)

                col_block_dist = ('*', 'b')
                col_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=col_block_dist)
                #Default max of entire array contents
                self.assertEqual(self.np_array.max(), alt_row_block_mpi_array.max())
                #Max along specified axies
                self.assertTrue(np.alltrue(self.np_array.max(axis=0) == col_block_mpi_array.max(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.max(axis=1) == col_block_mpi_array.max(axis=1)))
                with self.assertRaises(ValueError):
                        col_block_mpi_array.max(axis=col_block_mpi_array.ndim)

                block_block_dist = ('b', 'b')
                block_block_mpi_array = mpi_np.array(self.data, comm=self.comm, dist=block_block_dist)
                #Default max of entire array contents
                self.assertEqual(self.np_array.max(), block_block_mpi_array.max())
                #Max along specified axies
                self.assertTrue(np.alltrue(self.np_array.max(axis=0) == block_block_mpi_array.max(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.max(axis=1) == block_block_mpi_array.max(axis=1)))
                with self.assertRaises(ValueError):
                        block_block_mpi_array.max(axis=block_block_mpi_array.ndim)


if __name__ == '__main__':
        unittest.main()
