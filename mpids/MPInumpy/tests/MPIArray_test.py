import unittest
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np

class MPIArrayTest(unittest.TestCase):

        def setUp(self):
                self.comm = MPI.COMM_WORLD
                self.dist = 'b'
                self.comm_size = MPI.COMM_WORLD.Get_size()
                self.data = [list(range(1,5)), list(range(5,9))]
                self.np_array = np.array(self.data)
                self.mpi_array = mpi_np.MPIArray(self.data, comm=self.comm)
                self.scalar_data = [1]
                self.np_scalar = np.array(self.scalar_data)
                self.mpi_scalar = mpi_np.MPIArray(self.scalar_data, comm=self.comm)


        def test_properties(self):
                #Unique properties to MPIArray
                self.assertEqual(self.comm, self.mpi_array.comm)
                self.assertEqual(self.dist, self.mpi_array.dist)
                self.assertEqual(self.comm_size * self.np_array.size, self.mpi_array.globalsize)
                self.assertEqual(self.comm_size * self.np_array.nbytes, self.mpi_array.globalnbytes)

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
                self.assertEqual('[[1 2 3 4]\n [5 6 7 8]]', str(self.mpi_array.base))


        def test_dunder_methods(self):
                self.assertEqual('MPIArray(globalsize=[{}], dist={}, dtype={})'\
                                 .format(self.mpi_array.size * self.comm_size, self.dist,
                                         self.mpi_array.dtype), self.mpi_array.__repr__())
                self.assertEqual(None, self.mpi_array.__array_finalize__(None))
                self.assertEqual('[[1 2 3 4]\n [5 6 7 8]]', self.mpi_array.__str__())
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
                self.assertEqual(self.np_array[0].size * self.comm_size, first_row.globalsize)
                self.assertEqual(self.np_array[0].nbytes * self.comm_size, first_row.globalnbytes)

                self.assertTrue(last_row is not self.mpi_array)
                self.assertTrue(last_row.base is self.mpi_array)
                self.assertTrue(isinstance(last_row, mpi_np.MPIArray))
                self.assertEqual(self.np_array[self.np_array.shape[0] - 1].size * self.comm_size, last_row.globalsize)
                self.assertEqual(self.np_array[self.np_array.shape[0] - 1].nbytes * self.comm_size, first_row.globalnbytes)

                first_half_first_row = first_row[:len(first_row) / 2]
                second_half_last_row = last_row[len(last_row) / 2:]

                self.assertTrue(first_half_first_row is not first_row)
                self.assertTrue(first_half_first_row.base is self.mpi_array)
                self.assertTrue(isinstance(first_half_first_row, mpi_np.MPIArray))
                self.assertEqual(self.np_array[0, :len(self.np_array[0]) / 2].size * self.comm_size, first_half_first_row.globalsize)
                self.assertEqual(self.np_array[0, :len(self.np_array[0]) / 2].nbytes * self.comm_size, first_half_first_row.globalnbytes)

                self.assertTrue(second_half_last_row is not last_row)
                self.assertTrue(second_half_last_row.base is self.mpi_array)
                self.assertTrue(isinstance(second_half_last_row, mpi_np.MPIArray))
                self.assertEqual(self.np_array[1, len(self.np_array[1]) / 2:].size * self.comm_size, second_half_last_row.globalsize)
                self.assertEqual(self.np_array[1, len(self.np_array[1]) / 2:].nbytes * self.comm_size, second_half_last_row.globalnbytes)

                first_column = self.mpi_array[:,[0]]
                last_column = self.mpi_array[:,[self.mpi_array.shape[1] - 1]]

                self.assertTrue(first_column is not self.mpi_array)
                self.assertTrue(first_column.base is not self.mpi_array)
                self.assertTrue(isinstance(first_column, mpi_np.MPIArray))
                self.assertEqual(self.np_array[:,[0]].size * self.comm_size, first_column.globalsize)
                self.assertEqual(self.np_array[:,[0]].nbytes * self.comm_size, first_column.globalnbytes)

                self.assertTrue(last_column is not self.mpi_array)
                self.assertTrue(last_column.base is not self.mpi_array)
                self.assertTrue(isinstance(last_column, mpi_np.MPIArray))
                self.assertEqual(self.np_array[:,[self.np_array.shape[1] - 1]].size * self.comm_size, last_column.globalsize)
                self.assertEqual(self.np_array[:,[self.np_array.shape[1] - 1]].nbytes * self.comm_size, last_column.globalnbytes)


if __name__ == '__main__':
        unittest.main()
