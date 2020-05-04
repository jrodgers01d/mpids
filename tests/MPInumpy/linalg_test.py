import unittest
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np
from mpids.MPInumpy.errors import NotSupportedError

class LinAlgTest(unittest.TestCase):

    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.np_array_a = np.arange(16).reshape(4,4)
        self.np_array_b = np.arange(16).reshape(4,4) + 1

    def check_numpy_supported_version():
        #The numpy function 'matmul' was introduced in version 1.10...
        major, minor, _ = [int(x) for x in np.version.version.split('.')]
        return major > 0 and minor >= 10


    @unittest.skipIf(not check_numpy_supported_version(),
                     "Functionality missing from numpy version")
    def test_unsupported_functionality(self):
        #Use of 'out' field
        mpi_out = np.zeros(())
        with self.assertRaises(NotSupportedError):
            mpi_np.matmul(self.np_array_a, self.np_array_b, out=mpi_out)


    @unittest.skipIf(not check_numpy_supported_version(),
                     "Functionality missing from numpy version")
    def test_Replicated_matmul(self):
        mpi_array_a = mpi_np.array(self.np_array_a, dist='r')
        mpi_array_b = mpi_np.array(self.np_array_b, dist='r')

        #Check return type
        self.assertTrue(isinstance(
            mpi_np.matmul(self.np_array_a, self.np_array_b), mpi_np.MPIArray))
        self.assertTrue(isinstance(
            mpi_np.matmul(mpi_array_a, mpi_array_b), mpi_np.MPIArray))
        self.assertTrue(isinstance(
            mpi_np.matmul(mpi_array_a, self.np_array_b), mpi_np.MPIArray))
        self.assertTrue(isinstance(
            mpi_np.matmul(self.np_array_a, mpi_array_b), mpi_np.MPIArray))

        #Check result consistent with numpy
        self.assertTrue(np.alltrue(
            np.matmul(self.np_array_a, self.np_array_b) == \
            mpi_np.matmul(mpi_array_a, mpi_array_b)))
        self.assertTrue(np.alltrue(
            np.matmul(self.np_array_a, self.np_array_b) == \
            mpi_np.matmul(self.np_array_a, mpi_array_b)))
        self.assertTrue(np.alltrue(
            np.matmul(self.np_array_a, self.np_array_b) == \
            mpi_np.matmul(mpi_array_a, self.np_array_b)))


    @unittest.skipIf(not check_numpy_supported_version(),
                     "Functionality missing from numpy version")
    def test_block_distribution_matmul(self):
        rank = self.comm.Get_rank()
        mpi_array_a = mpi_np.array(self.np_array_a, dist='b')
        mpi_array_b = mpi_np.array(self.np_array_b, dist='b')

        #Check result consistent with numpy
        self.assertTrue(np.alltrue(
            np.matmul(self.np_array_a, self.np_array_b)[rank] == \
            mpi_np.matmul(mpi_array_a, mpi_array_b)))


    @unittest.skipIf(not check_numpy_supported_version(),
                     "Functionality missing from numpy version")
    def test_under_partitioned_block_distribution_matmul(self):
        #Current version of code will under partition a 2x8 matrix.
        #Want to make sure logic is sound with petsc4py.
        np_8x2_array = self.np_array_a.reshape(8,2)
        np_2x8_array = self.np_array_b.reshape(2,8)
        mpi_array_a = mpi_np.array(np_8x2_array, dist='b')
        mpi_array_b = mpi_np.array(np_2x8_array, dist='b')

        rank = self.comm.Get_rank()
        local_row_start = rank * 2
        local_row_stop = local_row_start + 2
        #Check result consistent with numpy
        self.assertTrue(np.alltrue(
            np.matmul(np_8x2_array, np_2x8_array)[local_row_start: local_row_stop] == \
            mpi_np.matmul(mpi_array_a, mpi_array_b)))


if __name__ == '__main__':
    unittest.main()
