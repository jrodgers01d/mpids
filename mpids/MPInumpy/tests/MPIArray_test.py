import unittest
import numpy as np
from mpi4py import MPI
from mpids.MPInumpy.MPIArray import MPIArray

class MPIArrayTest(unittest.TestCase):

        def setUp(self):
                self.comm = MPI.COMM_WORLD
                self.data = list(range(4))
                self.mpi_array = MPIArray(self.data, self.comm)

        def test_dunder_methods(self):
                self.assertEqual('MPIArray', self.mpi_array.__repr__())
                self.assertEqual(np.array(self.data).tolist(),
                                 self.mpi_array.__array__().tolist())

        def test_properties(self):
                self.assertEqual(self.data, self.mpi_array.data)
                self.assertEqual(self.comm, self.mpi_array.comm)


if __name__ == '__main__':
        unittest.main()
