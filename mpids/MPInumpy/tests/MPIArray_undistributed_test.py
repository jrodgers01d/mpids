import unittest
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np
import mpids.MPInumpy.tests.MPIArray_default_test as MPIArray_default_test

class MPIArrayUndistributedTest(MPIArray_default_test.MPIArrayDefaultTest):

        def create_setUp_parms(self):
                parms = {}
                parms['comm'] = MPI.COMM_WORLD
                parms['rank'] = MPI.COMM_WORLD.Get_rank()
                parms['comm_size'] = MPI.COMM_WORLD.Get_size()
                # Undistributed distribution
                parms['dist'] = 'u'
                #Add 1 to avoid divide by zero errors/warnings
                parms['data'] = (np.array(list(range(16))).reshape(4,4) + 1).tolist()
                parms['local_data'] = parms['data']
                parms['comm_dims'] = None
                parms['comm_coords'] = None
                return parms

        def test_scalar_dunder_unary_operations(self):
                scalar_data = 1
                np_scalar = np.array(scalar_data)
                mpi_scalar = mpi_np.MPIArray(scalar_data, comm=self.comm, dist=self.dist)

                self.assertEqual(complex(np_scalar), complex(mpi_scalar))
                self.assertEqual(int(np_scalar), int(mpi_scalar))
                self.assertEqual(float(np_scalar), float(mpi_scalar))
                self.assertEqual(oct(np_scalar), oct(mpi_scalar))
                self.assertEqual(hex(np_scalar), hex(mpi_scalar))


if __name__ == '__main__':
        unittest.main()
