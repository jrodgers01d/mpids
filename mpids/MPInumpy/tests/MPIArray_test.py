import unittest
import numpy as np
from mpi4py import MPI
from mpids.MPInumpy.MPIArray import MPIArray

class MPIArrayTest(unittest.TestCase):

        def setUp(self):
                self.comm = MPI.COMM_WORLD
                self.comm_size = MPI.COMM_WORLD.Get_size()
                self.data = list(range(4))
                self.np_array = np.array(self.data)
                self.mpi_array = MPIArray(self.data, self.comm)

        def test_dunder_methods(self):
                self.assertEqual('MPIArray', self.mpi_array.__repr__())
                self.assertEqual(self.np_array.tolist(),
                                 self.mpi_array.__array__().tolist())

        def test_properties(self):
                #Unique properties to MPIArray
                self.assertEqual(self.comm, self.mpi_array.comm)
                self.assertEqual(self.comm_size * self.np_array.size, self.mpi_array.globalsize)
                self.assertEqual(self.comm_size * self.np_array.nbytes, self.mpi_array.globalnbytes)

                #Replicated numpy.ndarray properties
                self.assertEqual(self.np_array.T.tolist(), self.mpi_array.T.tolist())
                self.assertEqual(self.np_array.data, self.mpi_array.data)
                self.assertEqual(self.np_array.dtype, self.mpi_array.dtype)
                self.assertEqual(self.np_array.imag.tolist(), self.mpi_array.imag.tolist())
                self.assertEqual(self.np_array.real.tolist(), self.mpi_array.real.tolist())
                self.assertEqual(self.np_array.size, self.mpi_array.size)
                self.assertEqual(self.np_array.itemsize, self.mpi_array.itemsize)
                self.assertEqual(self.np_array.nbytes, self.mpi_array.nbytes)
                self.assertEqual(self.np_array.ndim, self.mpi_array.ndim)
                self.assertEqual(self.np_array.shape, self.mpi_array.shape)
                self.assertEqual(self.np_array.strides, self.mpi_array.strides)


if __name__ == '__main__':
        unittest.main()
