import unittest
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np
import mpids.MPInumpy.tests.MPIArray_default_test as MPIArray_default_test

class MPIArrayBlockBlockTest(MPIArray_default_test.MPIArrayDefaultTest):

        def create_setUp_parms(self):
                parms = {}
                parms['comm'] = MPI.COMM_WORLD
                parms['rank'] = MPI.COMM_WORLD.Get_rank()
                parms['comm_size'] = MPI.COMM_WORLD.Get_size()
                # Block block distribution
                parms['dist'] = ('b', 'b')
                #Add 1 to avoid divide by zero errors/warnings
                parms['data'] = (np.array(list(range(16))).reshape(4,4) + 1).tolist()
                local_data_map = {0: (np.array(list(range(16))).reshape(4,4) + 1)[:2,:2],
                                  1: (np.array(list(range(16))).reshape(4,4) + 1)[:2,2:],
                                  2: (np.array(list(range(16))).reshape(4,4) + 1)[2:,:2],
                                  3: (np.array(list(range(16))).reshape(4,4) + 1)[2:,2:]}
                parms['local_data'] = local_data_map[parms['rank']].tolist()
                parms['comm_dims'] = [2, 2]
                rank_coord_map = {0: [0, 0],
                                  1: [0, 1],
                                  2: [1, 0],
                                  3: [1, 1]}
                parms['comm_coords'] = rank_coord_map[parms['rank']]
                return parms


if __name__ == '__main__':
        unittest.main()
