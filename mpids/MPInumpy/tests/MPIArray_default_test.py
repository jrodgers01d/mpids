import unittest
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np
from mpids.MPInumpy.errors import ValueError, NotSupportedError

class MPIArrayDefaultTest(unittest.TestCase):

        def create_setUp_parms(self):
                parms = {}
                parms['comm'] = MPI.COMM_WORLD
                parms['rank'] = MPI.COMM_WORLD.Get_rank()
                parms['comm_size'] = MPI.COMM_WORLD.Get_size()
                # Default distribution
                parms['dist'] = 'b'
                #Add 1 to avoid divide by zero errors/warnings
                parms['data'] = (np.array(list(range(16))).reshape(4,4) + 1).tolist()
                parms['local_data'] = [parms['data'][parms['rank']]]
                parms['comm_dims'] = [parms['comm_size']]
                parms['comm_coords'] = [parms['rank']]
                return parms


        def setUp(self):
                parms = self.create_setUp_parms()
                self.comm = parms.get('comm')
                self.rank = parms.get('rank')
                self.comm_size = parms.get('comm_size')
                self.dist = parms.get('dist')
                self.data = parms.get('data')
                self.local_data = parms.get('local_data')
                self.comm_dims = parms.get('comm_dims')
                self.comm_coords = parms.get('comm_coords')

                self.np_array = np.array(self.data)
                self.np_local_array = np.array(self.local_data)
                self.mpi_array = mpi_np.array(self.data, comm=self.comm, dist=self.dist)


        def test_object_return_behavior(self):
                self.assertTrue(isinstance(self.mpi_array, mpi_np.MPIArray))

                returned_array = self.mpi_array
                self.assertTrue(np.alltrue((returned_array) == (self.mpi_array)))
                self.assertTrue(returned_array is self.mpi_array)
                self.assertEqual(returned_array.comm, self.mpi_array.comm)
                self.assertEqual(returned_array.globalsize, self.mpi_array.globalsize)
                self.assertEqual(returned_array.globalnbytes, self.mpi_array.globalnbytes)


        def test_properties(self):
                #Unique properties to MPIArray
                self.assertEqual(self.comm, self.mpi_array.comm)
                self.assertEqual(self.dist, self.mpi_array.dist)
                self.assertEqual(self.comm_dims, self.mpi_array.comm_dims)
                self.assertEqual(self.comm_coords, self.mpi_array.comm_coord)
                self.assertEqual(self.np_array.size, self.mpi_array.globalsize)
                self.assertEqual(self.np_array.nbytes, self.mpi_array.globalnbytes)
                self.assertEqual(self.np_array.shape,  tuple(self.mpi_array.globalshape))

                #Replicated numpy.ndarray properties
                self.assertTrue(np.alltrue(self.np_local_array.T == self.mpi_array.T))
                self.assertEqual(self.np_local_array.data, self.mpi_array.data)
                self.assertEqual(self.np_local_array.dtype, self.mpi_array.dtype)
                self.assertTrue(np.alltrue(self.np_local_array.imag == self.mpi_array.imag))
                self.assertTrue(np.alltrue(self.np_local_array.real == self.mpi_array.real))
                self.assertEqual(self.np_local_array.size, self.mpi_array.size)
                self.assertEqual(self.np_local_array.itemsize, self.mpi_array.itemsize)
                self.assertEqual(self.np_local_array.nbytes, self.mpi_array.nbytes)
                self.assertEqual(self.np_local_array.ndim, self.mpi_array.ndim)
                self.assertEqual(self.np_local_array.shape, self.mpi_array.shape)
                self.assertEqual(self.np_local_array.strides, self.mpi_array.strides)
                self.assertEqual(str(self.np_local_array), str(self.mpi_array.base))


        def test_dunder_methods(self):
                self.assertEqual('MPIArray(globalsize={}, globalshape={}, dist={}, dtype={})'\
                                    .format(self.mpi_array.globalsize, list(self.mpi_array.globalshape),
                                            self.dist, self.mpi_array.dtype)
                                 , self.mpi_array.__repr__())
                self.assertEqual(None, self.mpi_array.__array_finalize__(None))
                self.assertEqual(self.np_local_array.__str__(), self.mpi_array.__str__())
                self.assertTrue(np.alltrue(self.np_local_array == self.mpi_array.__array__()))


        def test_dunder_binary_operations(self):
                self.assertTrue(np.alltrue((self.np_local_array + 2) == (self.mpi_array + 2)))
                self.assertTrue(np.alltrue((3 + self.np_local_array) == (3 + self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array - 2) == (self.mpi_array - 2)))
                self.assertTrue(np.alltrue((3 - self.np_local_array) == (3 - self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array * 2) == (self.mpi_array * 2)))
                self.assertTrue(np.alltrue((3 * self.np_local_array) == (3 * self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array // 2) == (self.mpi_array // 2)))
                self.assertTrue(np.alltrue((3 // self.np_local_array) == (3 // self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array / 2) == (self.mpi_array / 2)))
                self.assertTrue(np.alltrue((3 / self.np_local_array) == (3 / self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array % 2) == (self.mpi_array % 2)))
                self.assertTrue(np.alltrue((3 % self.np_local_array) == (3 % self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array ** 2) == (self.mpi_array ** 2)))
                self.assertTrue(np.alltrue((3 ** self.np_local_array) == (3 ** self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array << 2) == (self.mpi_array << 2)))
                self.assertTrue(np.alltrue((3 << self.np_local_array) == (3 << self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array >> 2) == (self.mpi_array >> 2)))
                self.assertTrue(np.alltrue((3 >> self.np_local_array) == (3 >> self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array & 2) == (self.mpi_array & 2)))
                self.assertTrue(np.alltrue((3 & self.np_local_array) == (3 & self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array | 2) == (self.mpi_array | 2)))
                self.assertTrue(np.alltrue((3 | self.np_local_array) == (3 | self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array ^ 2) == (self.mpi_array ^ 2)))
                self.assertTrue(np.alltrue((3 ^ self.np_local_array) == (3 ^ self.mpi_array)))


        def test_dunder_unary_operations(self):
                self.assertTrue(np.alltrue((-self.np_local_array) == (-self.mpi_array)))
                self.assertTrue(np.alltrue((+self.np_local_array) == (+self.mpi_array)))
                self.assertTrue(np.alltrue(abs(self.np_local_array) == abs(self.mpi_array)))
                self.assertTrue(np.alltrue((~self.np_local_array) == (~self.mpi_array)))


        def test_dunder_comparison_operations(self):
                self.assertTrue(np.alltrue((2 > self.np_local_array) == (2 > self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array < 2) == (self.mpi_array < 2)))
                self.assertTrue(np.alltrue((2 >= self.np_local_array) == (2 >= self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array <= 2) == (self.mpi_array <= 2)))
                self.assertTrue(np.alltrue((1 == self.np_local_array) == (1 == self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array == 1) == (self.mpi_array == 1)))
                self.assertTrue(np.alltrue((0 != self.np_local_array) == (0 != self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array != 0) == (self.mpi_array != 0)))
                self.assertTrue(np.alltrue((2 < self.np_local_array) == (2 < self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array > 2) == (self.mpi_array > 2)))
                self.assertTrue(np.alltrue((2 <= self.np_local_array) == (2 <= self.mpi_array)))
                self.assertTrue(np.alltrue((self.np_local_array >= 2) == (self.mpi_array >= 2)))


        def test_object_slicing_behavior(self):
                first_row = self.mpi_array[0]
                np_first_row = self.np_local_array[0]
                last_row = self.mpi_array[self.mpi_array.shape[0] - 1]
                np_last_row = self.np_local_array[self.np_local_array.shape[0] - 1]

                self.assertTrue(first_row is not self.mpi_array)
                self.assertTrue(isinstance(first_row, mpi_np.MPIArray))
                self.assertTrue(first_row.base is self.mpi_array)
                self.assertEqual(np_first_row.size, first_row.size)
                self.assertEqual(np_first_row.nbytes, first_row.nbytes)

                self.assertTrue(last_row is not self.mpi_array)
                self.assertTrue(last_row.base is self.mpi_array)
                self.assertTrue(isinstance(last_row, mpi_np.MPIArray))
                self.assertEqual(np_last_row.size, last_row.size)
                self.assertEqual(np_last_row.nbytes, last_row.nbytes)

                first_half_first_row = first_row[:len(first_row) // 2]
                np_first_half_first_row = np_first_row[:len(np_first_row) // 2]
                second_half_last_row = last_row[len(last_row) // 2:]
                np_second_half_last_row = np_last_row[len(np_last_row) // 2:]

                self.assertTrue(first_half_first_row is not first_row)
                self.assertTrue(first_half_first_row.base is self.mpi_array)
                self.assertTrue(isinstance(first_half_first_row, mpi_np.MPIArray))
                self.assertEqual(np_first_half_first_row.size, first_half_first_row.size)
                self.assertEqual(np_first_half_first_row.nbytes, first_half_first_row.nbytes)

                self.assertTrue(second_half_last_row is not last_row)
                self.assertTrue(second_half_last_row.base is self.mpi_array)
                self.assertTrue(isinstance(second_half_last_row, mpi_np.MPIArray))
                self.assertEqual(np_second_half_last_row.size, second_half_last_row.size)
                self.assertEqual(np_second_half_last_row.nbytes, second_half_last_row.nbytes)

                first_column = self.mpi_array[:,[0]]
                np_first_column = self.np_local_array[:,[0]]
                last_column = self.mpi_array[:,[self.mpi_array.shape[1] - 1]]
                np_last_column = self.np_local_array[:,[self.np_local_array.shape[1] - 1]]

                self.assertTrue(first_column is not self.mpi_array)
                self.assertTrue(first_column.base is not self.mpi_array)
                self.assertTrue(isinstance(first_column, mpi_np.MPIArray))
                self.assertEqual(np_first_column.size, first_column.size)
                self.assertEqual(np_first_column.nbytes, first_column.nbytes)

                self.assertTrue(last_column is not self.mpi_array)
                self.assertTrue(last_column.base is not self.mpi_array)
                self.assertTrue(isinstance(last_column, mpi_np.MPIArray))
                self.assertEqual(np_last_column.size, last_column.size)
                self.assertEqual(np_last_column.nbytes, last_column.nbytes)


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


        def test_custom_mean_method(self):
                #Default mean of entire array contents
                self.assertEqual(self.np_array.mean(), self.mpi_array.mean())

                #Mean along specified axies
                print(self.np_array.mean(axis=0))
                print(self.mpi_array.mean(axis=0))
                self.assertTrue(np.alltrue(self.np_array.mean(axis=0) == self.mpi_array.mean(axis=0)))
                self.assertTrue(np.alltrue(self.np_array.mean(axis=1) == self.mpi_array.mean(axis=1)))
                with self.assertRaises(ValueError):
                        self.mpi_array.mean(axis=self.mpi_array.ndim)

                #Use of 'out' field
                mpi_out = np.zeros(())
                with self.assertRaises(NotSupportedError):
                        self.mpi_array.mean(out=mpi_out)


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


if __name__ == '__main__':
        unittest.main()
