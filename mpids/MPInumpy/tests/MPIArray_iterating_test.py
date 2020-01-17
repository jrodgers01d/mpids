import unittest
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np

class MPIArrayIteratorDefaultTest(unittest.TestCase):

    def create_setUp_parms(self):
        size = MPI.COMM_WORLD.size
        parms = {}
        # Default distribution
        parms['dist'] = 'b'
        parms['rank'] = MPI.COMM_WORLD.rank
        parms['data'] = np.arange(size)
        return parms

    def setUp(self):
        parms = self.create_setUp_parms()
        self.dist = parms.get('dist')
        self.data = parms.get('data')
        self.rank = parms.get('rank')

        self.mpi_array = mpi_np.array(self.data, dist=self.dist)


    def test_standard_for_iteration(self):
        for val in self.mpi_array:
            self.assertEqual(val, self.rank)

    
    def test_enumerate_iteration(self):
        for zero, val in enumerate(self.mpi_array):
            self.assertEqual(zero, 0)
            self.assertEqual(val, self.rank)


    def test_list_compreshension(self):
        val = [val for val in self.mpi_array]
        self.assertEqual(val, [self.rank])


class MPIArrayIteratorUndistributedTest(MPIArrayIteratorDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        #Being sneaky here, as a undistributed array will have the same value
        ##on all ranks
        parms['rank'] = MPI.COMM_WORLD.size
        # Undistributed distribution
        parms['dist'] = 'u'
        parms['data'] = np.array([parms['rank']])
        return parms
