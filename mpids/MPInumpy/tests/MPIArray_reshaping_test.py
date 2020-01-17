import unittest
import numpy as np
from mpi4py import MPI
import mpids.MPInumpy as mpi_np
from mpids.MPInumpy.errors import ValueError
from mpids.MPInumpy.distributions.Undistributed import Undistributed

class MPIArrayReshapeDefaultTest(unittest.TestCase):

    def create_setUp_parms(self):
        parms = {}
        parms['comm'] = MPI.COMM_WORLD
        parms['rank'] = MPI.COMM_WORLD.Get_rank()
        # Default distribution
        parms['dist'] = 'b'
        np_data = np.arange(16).reshape(4,4)
        parms['data'] = np_data
        return parms


    def setUp(self):
        parms = self.create_setUp_parms()
        self.comm = parms.get('comm')
        self.rank = parms.get('rank')
        self.dist = parms.get('dist')
        self.data = parms.get('data')

        self.np_array = np.array(self.data)
        self.mpi_array = mpi_np.array(self.data, comm=self.comm, dist=self.dist)


    def test_reshaping_to_shape_not_equal_to_array_size_raise_value_error(self):
        #Global shape that's twice as big
        with self.assertRaises(ValueError):
            shape = tuple([dim * 2 for dim in self.mpi_array.globalshape])
            self.mpi_array.reshape(shape)

        #Global shape that's half the size
        with self.assertRaises(ValueError):
            shape = tuple([dim // 2 for dim in self.mpi_array.globalshape])
            self.mpi_array.reshape(shape)

        #Global shape that's stretched along the x-axis
        with self.assertRaises(ValueError):
            self.mpi_array.reshape(1, self.mpi_array.globalsize + 1)

        #Global shape that's stretched along the y-axis
        with self.assertRaises(ValueError):
            self.mpi_array.reshape(self.mpi_array.globalsize + 1, 1)


    def test_flattening_along_axis_0(self):
        flatten_along_axis_0 = \
            self.mpi_array.reshape(1, self.mpi_array.globalsize)
        np_flatten_along_axis_0 = self.np_array.reshape(1, self.np_array.size)

        #Check updated global properties
        self.assertEqual(flatten_along_axis_0.dist, self.mpi_array.dist)
        self.assertEqual(flatten_along_axis_0.globalsize, self.mpi_array.globalsize)
        self.assertEqual(flatten_along_axis_0.globalshape, (1, self.mpi_array.globalsize))
        self.assertEqual(flatten_along_axis_0.globalndim, self.mpi_array.globalndim)
        self.assertEqual(flatten_along_axis_0.comm_dims, self.mpi_array.comm_dims)
        self.assertEqual(flatten_along_axis_0.comm_coord, self.mpi_array.comm_coord)
        if isinstance(flatten_along_axis_0, Undistributed):
            self.assertEqual(flatten_along_axis_0.local_to_global, self.mpi_array.local_to_global)
        else:
            #Test developed with Row Block as only existing distribution
            if self.rank == 0:
                self.assertEqual(flatten_along_axis_0.local_to_global,
                                {0: (0, 1), 1: (0, 16)})
            else: # Empty
                self.assertEqual(flatten_along_axis_0.local_to_global,
                                {0: (1, 1), 1: (0, 16)})

        #Check contents
        if isinstance(flatten_along_axis_0, Undistributed):
            self.assertTrue(np.alltrue(np_flatten_along_axis_0 == flatten_along_axis_0))
        else:
            #Test developed with Row Block as only existing distribution
            if self.rank == 0:
                self.assertTrue(np.alltrue(np_flatten_along_axis_0 == flatten_along_axis_0))
            else: # Empty
                self.assertEqual(0, flatten_along_axis_0.size)


    def test_flattening_along_axis_1(self):
        flatten_along_axis_1 = \
            self.mpi_array.reshape(self.mpi_array.globalsize, 1)
        np_flatten_along_axis_1 = self.np_array.reshape(self.np_array.size, 1)

        #Check updated global properties
        self.assertEqual(flatten_along_axis_1.dist, self.mpi_array.dist)
        self.assertEqual(flatten_along_axis_1.globalsize, self.mpi_array.globalsize)
        self.assertEqual(flatten_along_axis_1.globalshape, (self.mpi_array.globalsize, 1))
        self.assertEqual(flatten_along_axis_1.globalndim, self.mpi_array.globalndim)
        self.assertEqual(flatten_along_axis_1.comm_dims, self.mpi_array.comm_dims)
        self.assertEqual(flatten_along_axis_1.comm_coord, self.mpi_array.comm_coord)
        if isinstance(flatten_along_axis_1, Undistributed):
            self.assertEqual(flatten_along_axis_1.local_to_global, self.mpi_array.local_to_global)
        else:
            axis0_start = self.rank * 4
            axis0_stop = axis0_start + 4

            #Test developed with Row Block as only existing distribution
            self.assertEqual(flatten_along_axis_1.local_to_global,
                            {0: (axis0_start, axis0_stop ),
                             1: (0, 1)})

        #Check contents
        if isinstance(flatten_along_axis_1, Undistributed):
            self.assertTrue(np.alltrue(np_flatten_along_axis_1 == flatten_along_axis_1))
        else:
            #Test developed with Row Block as only existing distribution
            axis0_start = self.rank * 4
            axis0_stop = axis0_start + 4

            self.assertTrue(
                np.alltrue(np_flatten_along_axis_1[axis0_start: axis0_stop] == \
                           flatten_along_axis_1))


    def test_change_shape_from_4x4_to_2x8(self):
        mpi_array_2x8 = \
            self.mpi_array.reshape(2, 8)
        np_2x8 = self.np_array.reshape(2, 8)

        #Check updated global properties
        self.assertEqual(mpi_array_2x8.dist, self.mpi_array.dist)
        self.assertEqual(mpi_array_2x8.globalsize, self.mpi_array.globalsize)
        self.assertEqual(mpi_array_2x8.globalshape, (2, 8))
        self.assertEqual(mpi_array_2x8.globalndim, self.mpi_array.globalndim)
        self.assertEqual(mpi_array_2x8.comm_dims, self.mpi_array.comm_dims)
        self.assertEqual(mpi_array_2x8.comm_coord, self.mpi_array.comm_coord)
        if isinstance(mpi_array_2x8, Undistributed):
            self.assertEqual(mpi_array_2x8.local_to_global, self.mpi_array.local_to_global)
        else:
            #Test developed with Row Block as only existing distribution
            if self.rank < 2:
                axis0_start = self.rank
                axis0_stop = axis0_start + 1

                self.assertEqual(mpi_array_2x8.local_to_global,
                                {0: (axis0_start, axis0_stop ),
                                 1: (0, 8)})
            else: # Empty
                self.assertEqual(mpi_array_2x8.local_to_global,
                                {0: (2, 2), 1: (0, 8)})

        #Check contents
        if isinstance(mpi_array_2x8, Undistributed):
            self.assertTrue(np.alltrue(np_2x8 == mpi_array_2x8))
        else:
            #Test developed with Row Block as only existing distribution
            if self.rank < 2:
                self.assertTrue(np.alltrue(np_2x8[self.rank] == mpi_array_2x8))
            else: # Empty
                self.assertEqual(0, mpi_array_2x8.size)


    def test_change_shape_from_4x4_to_8x2(self):
        mpi_array_8x2 = \
            self.mpi_array.reshape(8, 2)
        np_8x2 = self.np_array.reshape(8, 2)

        #Check updated global properties
        self.assertEqual(mpi_array_8x2.dist, self.mpi_array.dist)
        self.assertEqual(mpi_array_8x2.globalsize, self.mpi_array.globalsize)
        self.assertEqual(mpi_array_8x2.globalshape, (8, 2))
        self.assertEqual(mpi_array_8x2.globalndim, self.mpi_array.globalndim)
        self.assertEqual(mpi_array_8x2.comm_dims, self.mpi_array.comm_dims)
        self.assertEqual(mpi_array_8x2.comm_coord, self.mpi_array.comm_coord)
        if isinstance(mpi_array_8x2, Undistributed):
            self.assertEqual(mpi_array_8x2.local_to_global, self.mpi_array.local_to_global)
        else:
            #Test developed with Row Block as only existing distribution
            axis0_start = self.rank * 2
            axis0_stop = axis0_start + 2

            self.assertEqual(mpi_array_8x2.local_to_global,
                            {0: (axis0_start, axis0_stop ),
                             1: (0, 2)})
        #Check contents
        if isinstance(mpi_array_8x2, Undistributed):
            self.assertTrue(np.alltrue(np_8x2 == mpi_array_8x2))
        else:
            axis0_start = self.rank * 2
            axis0_stop = axis0_start + 2
            #Test developed with Row Block as only existing distribution
            self.assertTrue(np.alltrue(np_8x2[axis0_start: axis0_stop] == mpi_array_8x2))


    def test_change_shape_from_4x4_to_4x2x2(self):
        mpi_array_4x2x2 = \
            self.mpi_array.reshape(4, 2, 2)
        np_4x2x2 = self.np_array.reshape(4, 2, 2)

        #Check updated global properties
        self.assertEqual(mpi_array_4x2x2.dist, self.mpi_array.dist)
        self.assertEqual(mpi_array_4x2x2.globalsize, self.mpi_array.globalsize)
        self.assertEqual(mpi_array_4x2x2.globalshape, (4, 2, 2))
        self.assertEqual(mpi_array_4x2x2.globalndim, 3)
        self.assertEqual(mpi_array_4x2x2.comm_dims, self.mpi_array.comm_dims)
        self.assertEqual(mpi_array_4x2x2.comm_coord, self.mpi_array.comm_coord)
        if isinstance(mpi_array_4x2x2, Undistributed):
            self.assertEqual(mpi_array_4x2x2.local_to_global, self.mpi_array.local_to_global)
        else:
            #Test developed with Row Block as only existing distribution
            axis0_start = self.rank
            axis0_stop = axis0_start + 1

            self.assertEqual(mpi_array_4x2x2.local_to_global,
                            {0: (axis0_start, axis0_stop ),
                             1: (0, 2),
                             2: (0, 2)})
        #Check contents
        if isinstance(mpi_array_4x2x2, Undistributed):
            self.assertTrue(np.alltrue(np_4x2x2 == mpi_array_4x2x2))
        else:
            axis0_start = self.rank
            axis0_stop = axis0_start + 1
            #Test developed with Row Block as only existing distribution
            self.assertTrue(np.alltrue(np_4x2x2[axis0_start: axis0_stop] == mpi_array_4x2x2))


    def test_change_shape_from_4x4_to_2x2x2x2(self):
        mpi_array_2x2x2x2 = \
            self.mpi_array.reshape(2, 2, 2, 2)
        np_2x2x2x2 = self.np_array.reshape(2, 2, 2, 2)

        #Check updated global properties
        self.assertEqual(mpi_array_2x2x2x2.dist, self.mpi_array.dist)
        self.assertEqual(mpi_array_2x2x2x2.globalsize, self.mpi_array.globalsize)
        self.assertEqual(mpi_array_2x2x2x2.globalshape, (2, 2, 2, 2))
        self.assertEqual(mpi_array_2x2x2x2.globalndim, 4)
        self.assertEqual(mpi_array_2x2x2x2.comm_dims, self.mpi_array.comm_dims)
        self.assertEqual(mpi_array_2x2x2x2.comm_coord, self.mpi_array.comm_coord)
        if isinstance(mpi_array_2x2x2x2, Undistributed):
            self.assertEqual(mpi_array_2x2x2x2.local_to_global, self.mpi_array.local_to_global)
        else:
            #Test developed with Row Block as only existing distribution
            if self.rank < 2:
                axis0_start = self.rank
                axis0_stop = axis0_start + 1

                self.assertEqual(mpi_array_2x2x2x2.local_to_global,
                                {0: (axis0_start, axis0_stop ),
                                 1: (0, 2),
                                 2: (0, 2),
                                 3: (0, 2)})
            else: # Empty
                self.assertEqual(mpi_array_2x2x2x2.local_to_global,
                                {0: (2, 2), 1: (0, 2), 2: (0, 2), 3: (0, 2)})

        #Check contents
        if isinstance(mpi_array_2x2x2x2, Undistributed):
            self.assertTrue(np.alltrue(np_2x2x2x2 == mpi_array_2x2x2x2))
        else:
            #Test developed with Row Block as only existing distribution
            if self.rank < 2:
                self.assertTrue(np.alltrue(np_2x2x2x2[self.rank] == mpi_array_2x2x2x2))
            else: # Empty
                self.assertEqual(0, mpi_array_2x2x2x2.size)


class MPIArrayReshapeUndistributedTest(MPIArrayReshapeDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        parms['comm'] = MPI.COMM_WORLD
        parms['rank'] = MPI.COMM_WORLD.Get_rank()
        # Undistributed distribution
        parms['dist'] = 'u'
        parms['data'] = np.arange(16).reshape(4,4)
        return parms


if __name__ == '__main__':
    unittest.main()
