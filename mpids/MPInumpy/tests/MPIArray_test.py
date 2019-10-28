import unittest
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np

class MPIArrayTest(unittest.TestCase):

        def setUp(self):
                self.comm = MPI.COMM_WORLD
                self.comm_size = MPI.COMM_WORLD.Get_size()
                self.data = [list(range(1,5)), list(range(5,9))]
                self.np_array = np.array(self.data)
                self.mpi_array = mpi_np.MPIArray(self.data, self.comm)
                self.scalar_data = [1]
                self.np_scalar = np.array(self.scalar_data)
                self.mpi_scalar = mpi_np.MPIArray(self.scalar_data, self.comm)

        def test_properties(self):
                #Unique properties to MPIArray
                self.assertEqual(self.comm, self.mpi_array.comm)
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
                self.assertEqual('MPIArray', self.mpi_array.__repr__())
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

## TODO: Result capture tests


if __name__ == '__main__':
        unittest.main()
