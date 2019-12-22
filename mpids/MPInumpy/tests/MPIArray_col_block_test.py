import unittest
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np
import mpids.MPInumpy.tests.MPIArray_default_test as MPIArray_default_test

class MPIArrayColBlockTest(MPIArray_default_test.MPIArrayDefaultTest):

        def create_setUp_parms(self):
                parms = {}
                parms['comm'] = MPI.COMM_WORLD
                parms['rank'] = MPI.COMM_WORLD.Get_rank()
                parms['comm_size'] = MPI.COMM_WORLD.Get_size()
                # Column block distribution
                parms['dist'] = ('*', 'b')
                #Add 1 to avoid divide by zero errors/warnings
                parms['data'] = (np.array(list(range(16))).reshape(4,4) + 1).tolist()
                parms['local_data'] = \
                    (np.array(list(range(16))).reshape(4,4) + 1)[:,parms['rank']].reshape(4,1).tolist()
                parms['comm_dims'] = [1, parms['comm_size']]
                parms['comm_coords'] = [0, parms['rank']]
                return parms


if __name__ == '__main__':
        unittest.main()
