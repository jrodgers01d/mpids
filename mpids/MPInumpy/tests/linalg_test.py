import unittest
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np
from mpids.MPInumpy.errors import NotSupportedError

class LinAlgTest(unittest.TestCase):

    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.np_array_a = np.array(list(range(16))).reshape(4,4)
        self.np_array_b = np.array(list(range(16))).reshape(4,4) + 1

    def test_unsupported_functionality(self):
        #Use of 'out' field
        mpi_out = np.zeros(())
        with self.assertRaises(NotSupportedError):
            mpi_np.matmul(self.np_array_a, self.np_array_b, out=mpi_out)


    def test_undistributed_matmul(self):
        mpi_array_a = mpi_np.array(self.np_array_a, dist='u')
        mpi_array_b = mpi_np.array(self.np_array_b, dist='u')

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

    def test_row_block_distribution_matmul(self):
        rank = self.comm.Get_rank()
        mpi_array_a = mpi_np.array(self.np_array_a, dist='b')
        mpi_array_b = mpi_np.array(self.np_array_b, dist='b')

        #Check result consistent with numpy
        self.assertTrue(np.alltrue(
            np.matmul(self.np_array_a, self.np_array_b)[rank] == \
            mpi_np.matmul(mpi_array_a, mpi_array_b)))

if __name__ == '__main__':
    unittest.main()
