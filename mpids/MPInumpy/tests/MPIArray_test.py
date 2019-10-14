import unittest
import numpy as np
from mpi4py import MPI
from mpids.MPInumpy.MPIArray import MPIArray

class MPIArrayTest(unittest.TestCase):

        def setUp(self):
                self.comm = MPI.COMM_WORLD
                self.comm_size = MPI.COMM_WORLD.Get_size()
                self.data = list(range(1,5))
                self.np_array = np.array(self.data)
                self.mpi_array = MPIArray(self.data, self.comm)

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

        def test_dunder_methods(self):
                self.assertEqual('MPIArray', self.mpi_array.__repr__())
                self.assertEqual('[1 2 3 4]', self.mpi_array.__str__())
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
                np_scalar = np.array([1])
                mpi_scalar = MPIArray([1], self.comm)

                self.assertTrue(np.alltrue((-self.np_array) == (-self.mpi_array)))
                self.assertTrue(np.alltrue((+self.np_array) == (+self.mpi_array)))
                self.assertTrue(np.alltrue(abs(self.np_array) == abs(self.mpi_array)))
                self.assertTrue(np.alltrue((~self.np_array) == (~self.mpi_array)))
                # self.assertEqual(complex(np_scalar), complex(mpi_scalar))
                # self.assertEqual(int(np_scalar), int(mpi_scalar))
                # self.assertEqual(long(np_scalar), long(mpi_scalar))
                # self.assertEqual(float(np_scalar), float(mpi_scalar))
                # self.assertEqual(oct(np_scalar), oct(mpi_scalar))
                # self.assertEqual(hex(np_scalar), hex(mpi_scalar))


if __name__ == '__main__':
        unittest.main()
