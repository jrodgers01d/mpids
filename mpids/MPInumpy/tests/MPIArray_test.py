import unittest
import warnings
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np
from mpids.MPInumpy.errors import ValueError, NotSupportedError
from mpids.MPInumpy.distributions.Undistributed import Undistributed


class MPIArrayAbstractBaseClassTest(unittest.TestCase):

    def setUp(self):
        self.mpi_array = mpi_np.MPIArray([1])


    def test_abstract_dunder_methods_raise_not_implemented_errors(self):
        with self.assertRaises(NotImplementedError):
            self.mpi_array.__getitem__(0)


    def test_abstract_properties_raise_not_implemented_errors(self):
        with self.assertRaises(NotImplementedError):
            self.mpi_array.dist

        with self.assertRaises(NotImplementedError):
            self.mpi_array.globalshape

        with self.assertRaises(NotImplementedError):
            self.mpi_array.globalsize

        with self.assertRaises(NotImplementedError):
            self.mpi_array.globalnbytes

        with self.assertRaises(NotImplementedError):
            self.mpi_array.globalndim


    def test_abstract_methods_raise_not_implemented_errors(self):
        with self.assertRaises(NotImplementedError):
            self.mpi_array.max()

        with self.assertRaises(NotImplementedError):
            self.mpi_array.mean()

        with self.assertRaises(NotImplementedError):
            self.mpi_array.min()

        with self.assertRaises(NotImplementedError):
            self.mpi_array.std()

        with self.assertRaises(NotImplementedError):
            self.mpi_array.sum()

        with self.assertRaises(NotImplementedError):
            self.mpi_array.collect_data()

        with self.assertRaises(NotImplementedError):
            self.mpi_array.reshape()


class MPIArrayDefaultTest(unittest.TestCase):

    def create_setUp_parms(self):
        parms = {}
        parms['comm'] = MPI.COMM_WORLD
        parms['rank'] = MPI.COMM_WORLD.Get_rank()
        parms['comm_size'] = MPI.COMM_WORLD.Get_size()
        # Default distribution
        parms['dist'] = 'b'
        #Add 1 to avoid divide by zero errors/warnings
        np_data = np.arange(25).reshape(5,5) + 1
        parms['data'] = np_data
        local_data_map = {0: np_data[:2],
                          1: np_data[2:3],
                          2: np_data[3:4],
                          3: np_data[4:]}
        parms['local_data'] = local_data_map[parms['rank']].tolist()
        parms['comm_dims'] = [parms['comm_size']]
        parms['comm_coord'] = [parms['rank']]
        local_to_global_map = {0 : {0 : (0, 2), 1 : (0, 5)},
                               1 : {0 : (2, 3), 1 : (0, 5)},
                               2 : {0 : (3, 4), 1 : (0, 5)},
                               3 : {0 : (4, 5), 1 : (0, 5)}}
        parms['local_to_global'] = local_to_global_map[parms['rank']]
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
        self.comm_coord = parms.get('comm_coord')
        self.local_to_global = parms.get('local_to_global')

        self.np_array = np.array(self.data)
        self.np_local_array = np.array(self.local_data)
        self.mpi_array = mpi_np.array(self.data, comm=self.comm, dist=self.dist)


    def test_object_return_behavior(self):
        self.assertTrue(isinstance(self.mpi_array, mpi_np.MPIArray))

        returned_array = self.mpi_array
        self.assertTrue(np.alltrue((returned_array) == (self.mpi_array)))
        self.assertTrue(returned_array is self.mpi_array)
        self.assertEqual(returned_array.comm, self.mpi_array.comm)
        self.assertEqual(returned_array.globalshape, self.mpi_array.globalshape)
        self.assertEqual(returned_array.globalsize, self.mpi_array.globalsize)
        self.assertEqual(returned_array.globalnbytes, self.mpi_array.globalnbytes)
        self.assertEqual(returned_array.globalndim, self.mpi_array.globalndim)


    def test_properties(self):
        #Unique properties to MPIArray
        self.assertEqual(self.comm, self.mpi_array.comm)
        self.assertEqual(self.dist, self.mpi_array.dist)
        self.assertEqual(self.comm_dims, self.mpi_array.comm_dims)
        self.assertEqual(self.comm_coord, self.mpi_array.comm_coord)
        self.assertEqual(self.local_to_global, self.mpi_array.local_to_global)
        self.assertEqual(self.np_array.shape, self.mpi_array.globalshape)
        self.assertEqual(self.np_array.size, self.mpi_array.globalsize)
        self.assertEqual(self.np_array.nbytes, self.mpi_array.globalnbytes)
        self.assertEqual(self.np_array.ndim, self.mpi_array.globalndim)
#TODO: Re-evaluate these data types
        #Unique properties data types
        if isinstance(self.mpi_array, Undistributed):
            self.assertTrue(self.mpi_array.comm_dims is None)
            self.assertTrue(self.mpi_array.comm_coord is None)
            self.assertTrue(self.mpi_array.local_to_global is None)
        else:
            self.assertTrue(isinstance(self.mpi_array.comm_dims, list))
            self.assertTrue(isinstance(self.mpi_array.comm_dims[0], int))
            self.assertTrue(isinstance(self.mpi_array.comm_coord, list))
            self.assertTrue(isinstance(self.mpi_array.comm_coord[0], int))
            self.assertTrue(isinstance(self.mpi_array.local_to_global, dict))
            self.assertTrue(isinstance(self.mpi_array.local_to_global[0], tuple))
            self.assertTrue(isinstance(self.mpi_array.local_to_global[0][0], int))
        self.assertTrue(isinstance(self.mpi_array.globalsize, int))
        self.assertTrue(isinstance(self.mpi_array.globalnbytes, int))
        self.assertTrue(isinstance(self.mpi_array.globalndim, int))
        self.assertTrue(isinstance(self.mpi_array.globalshape, tuple))
        self.assertTrue(isinstance(self.mpi_array.globalshape[0], int))
        self.assertTrue(isinstance(self.mpi_array.globalshape[1], int))
        self.assertTrue(isinstance(self.mpi_array.local, np.ndarray))
        self.assertTrue(np.alltrue(self.mpi_array.local == self.np_local_array))

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
                    .format(self.mpi_array.globalsize, self.mpi_array.globalshape,
                            self.dist, self.mpi_array.dtype
                            ), self.mpi_array.__repr__())
        self.assertEqual(None, self.mpi_array.__array_finalize__(None))
        self.assertEqual(self.np_local_array.__str__(), self.mpi_array.__str__())
        self.assertTrue(np.alltrue(self.np_local_array == self.mpi_array.__array__()))


    def test_dunder_binary_operations(self):
        #Older versions of numpy will throw RuntimeWarning's for power operations
        warnings.simplefilter('ignore', category=RuntimeWarning)

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


    def test_custom_max_method(self):
        #Returned object is undistributed
        self.assertTrue(isinstance(self.mpi_array.max(), Undistributed))

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
        #Returned object is undistributed
        self.assertTrue(isinstance(self.mpi_array.mean(), Undistributed))

        #Default mean of entire array contents
        self.assertEqual(self.np_array.mean(), self.mpi_array.mean())

        #Mean along specified axies
        self.assertTrue(np.alltrue(self.np_array.mean(axis=0) == self.mpi_array.mean(axis=0)))
        self.assertTrue(np.alltrue(self.np_array.mean(axis=1) == self.mpi_array.mean(axis=1)))
        with self.assertRaises(ValueError):
            self.mpi_array.mean(axis=self.mpi_array.ndim)

        #Use of 'out' field
        mpi_out = np.zeros(())
        with self.assertRaises(NotSupportedError):
            self.mpi_array.mean(out=mpi_out)


    def test_custom_min_method(self):
        #Returned object is undistributed
        self.assertTrue(isinstance(self.mpi_array.min(), Undistributed))

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


    def test_custom_std_method(self):
        #Returned object is undistributed
        self.assertTrue(isinstance(self.mpi_array.std(), Undistributed))

        #Default std of entire array contents
        self.assertEqual(self.np_array.std(), self.mpi_array.std())

        #Std along specified axies
        self.assertTrue(np.alltrue(self.np_array.std(axis=0) == self.mpi_array.std(axis=0)))
        self.assertTrue(np.alltrue(self.np_array.std(axis=1) == self.mpi_array.std(axis=1)))
        with self.assertRaises(ValueError):
            self.mpi_array.std(axis=self.mpi_array.ndim)

        #Use of 'out' field
        mpi_out = np.zeros(())
        with self.assertRaises(NotSupportedError):
            self.mpi_array.std(out=mpi_out)


    def test_custom_sum_method(self):
        #Returned object is undistributed
        self.assertTrue(isinstance(self.mpi_array.sum(), Undistributed))

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


    def test_collect_data_method(self):
        collected_array = self.mpi_array.collect_data()

        #Returned object is properties
        self.assertTrue(isinstance(collected_array, mpi_np.MPIArray))
        self.assertTrue(isinstance(collected_array, Undistributed))
        self.assertTrue(collected_array is not self.mpi_array)
        self.assertEqual(collected_array.comm, self.mpi_array.comm)
        self.assertEqual(collected_array.globalshape, self.mpi_array.globalshape)
        self.assertEqual(collected_array.globalsize, self.mpi_array.globalsize)
        self.assertEqual(collected_array.globalnbytes, self.mpi_array.globalnbytes)
        self.assertEqual(collected_array.globalndim, self.mpi_array.globalndim)

        #Check collected values
        self.assertTrue(np.alltrue((collected_array) == (self.np_array)))


class MPIArrayUndistributedTest(MPIArrayDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        parms['comm'] = MPI.COMM_WORLD
        parms['rank'] = MPI.COMM_WORLD.Get_rank()
        parms['comm_size'] = MPI.COMM_WORLD.Get_size()
        # Undistributed distribution
        parms['dist'] = 'u'
        #Add 1 to avoid divide by zero errors/warnings
        parms['data'] = np.arange(25).reshape(5,5) + 1
        parms['local_data'] = parms['data']
        parms['comm_dims'] = None
        parms['comm_coord'] = None
        parms['local_to_global'] = None
        return parms


    def test_scalar_dunder_unary_operations(self):
        from mpids.MPInumpy.distributions.Undistributed import Undistributed

        scalar_data = 1
        np_scalar = np.array(scalar_data)
        mpi_scalar = Undistributed(scalar_data, comm=self.comm)

        self.assertEqual(complex(np_scalar), complex(mpi_scalar))
        self.assertEqual(int(np_scalar), int(mpi_scalar))
        self.assertEqual(float(np_scalar), float(mpi_scalar))
        self.assertEqual(oct(np_scalar), oct(mpi_scalar))
        self.assertEqual(hex(np_scalar), hex(mpi_scalar))


if __name__ == '__main__':
    unittest.main()
