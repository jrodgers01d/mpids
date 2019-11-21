import unittest
import numpy as np
import mpids.MPInumpy as mpi_np
from mpids.MPInumpy.errors import NotSupportedError

class LinAlgTest(unittest.TestCase):

        def test_undistributed_matmul(self):
                np_array_a = np.array(list(range(16))).reshape(4,4)
                mpi_array_a = mpi_np.array(np_array_a, dist='u')

                np_array_b = np.array(list(range(16))).reshape(4,4) + 1
                mpi_array_b = mpi_np.array(np_array_b, dist='u')

                #Check return type
                self.assertTrue(isinstance(
                        mpi_np.matmul(mpi_array_a, mpi_array_b), mpi_np.MPIArray))

                #Check result consistent with numpy
                self.assertTrue(np.alltrue(
                        np.matmul(np_array_a, np_array_b) == mpi_np.matmul(mpi_array_a, mpi_array_b)))

                #Use of 'out' field
                mpi_out = np.zeros(())
                with self.assertRaises(NotSupportedError):
                        mpi_np.matmul(mpi_array_a, mpi_array_b, out=mpi_out)


if __name__ == '__main__':
        unittest.main()
