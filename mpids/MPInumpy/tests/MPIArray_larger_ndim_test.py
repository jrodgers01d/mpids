import numpy as np
from mpi4py import MPI
from mpids.MPInumpy.tests.MPIArray_test import MPIArrayDefaultTest


class MPIArray3DDefaultTest(MPIArrayDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        parms['comm'] = MPI.COMM_WORLD
        parms['rank'] = MPI.COMM_WORLD.Get_rank()
        parms['comm_size'] = MPI.COMM_WORLD.Get_size()
        # Default distribution
        parms['dist'] = 'b'
        #Add 1 to avoid divide by zero errors/warnings
        np_data = np.arange(16).reshape(4,2,2) + 1
        parms['data'] = np_data
        local_data_map = {0: np_data[:1],
                          1: np_data[1:2],
                          2: np_data[2:3],
                          3: np_data[3:]}
        parms['local_data'] = local_data_map[parms['rank']].tolist()
        parms['comm_dims'] = [parms['comm_size']]
        parms['comm_coord'] = [parms['rank']]
        local_to_global_map = {0 : {0 : (0, 1), 1 : (0, 2), 2 : (0, 2)},
                               1 : {0 : (1, 2), 1 : (0, 2), 2 : (0, 2)},
                               2 : {0 : (2, 3), 1 : (0, 2), 2 : (0, 2)},
                               3 : {0 : (3, 4), 1 : (0, 2), 2 : (0, 2)}}
        parms['local_to_global'] = local_to_global_map[parms['rank']]
        return parms


    def test_custom_max_higher_dim_method(self):
        #Max along specified axies
        self.assertTrue(np.alltrue(self.np_array.max(axis=2) == self.mpi_array.max(axis=2)))


    def test_custom_mean_higher_dim_method(self):
        #Mean along specified axies
        self.assertTrue(np.alltrue(self.np_array.mean(axis=2) == self.mpi_array.mean(axis=2)))


    def test_custom_min_higher_dim_method(self):
        #Min along specified axies
        self.assertTrue(np.alltrue(self.np_array.min(axis=2) == self.mpi_array.min(axis=2)))


    def test_custom_std_higher_dim_method(self):
        #Std along specified axies
        self.assertTrue(np.alltrue(self.np_array.std(axis=2) == self.mpi_array.std(axis=2)))


    def test_custom_sum_higher_dim_method(self):
        #Sum along specified axies
        self.assertTrue(np.alltrue(self.np_array.sum(axis=2) == self.mpi_array.sum(axis=2)))


class MPIArray3DUndistributedTest(MPIArray3DDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        parms['comm'] = MPI.COMM_WORLD
        parms['rank'] = MPI.COMM_WORLD.Get_rank()
        parms['comm_size'] = MPI.COMM_WORLD.Get_size()
        # Undistributed distribution
        parms['dist'] = 'u'
        #Add 1 to avoid divide by zero errors/warnings
        parms['data'] = np.arange(16).reshape(4,2,2) + 1
        parms['local_data'] = parms['data']
        parms['comm_dims'] = None
        parms['comm_coord'] = None
        parms['local_to_global'] = None
        return parms


class MPIArray4DDefaultTest(MPIArrayDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        parms['comm'] = MPI.COMM_WORLD
        parms['rank'] = MPI.COMM_WORLD.Get_rank()
        parms['comm_size'] = MPI.COMM_WORLD.Get_size()
        # Default distribution
        parms['dist'] = 'b'
        #Add 1 to avoid divide by zero errors/warnings
        np_data = np.arange(32).reshape(4,2,2,2) + 1
        parms['data'] = np_data
        local_data_map = {0: np_data[:1],
                          1: np_data[1:2],
                          2: np_data[2:3],
                          3: np_data[3:]}
        parms['local_data'] = local_data_map[parms['rank']].tolist()
        parms['comm_dims'] = [parms['comm_size']]
        parms['comm_coord'] = [parms['rank']]
        local_to_global_map = {0 : {0 : (0, 1), 1 : (0, 2), 2 : (0, 2), 3 : (0, 2)},
                               1 : {0 : (1, 2), 1 : (0, 2), 2 : (0, 2), 3 : (0, 2)},
                               2 : {0 : (2, 3), 1 : (0, 2), 2 : (0, 2), 3 : (0, 2)},
                               3 : {0 : (3, 4), 1 : (0, 2), 2 : (0, 2), 3 : (0, 2)}}
        parms['local_to_global'] = local_to_global_map[parms['rank']]
        return parms


    def test_custom_max_higher_dim_method(self):
        #Max along specified axies
        self.assertTrue(np.alltrue(self.np_array.max(axis=2) == self.mpi_array.max(axis=2)))
        self.assertTrue(np.alltrue(self.np_array.max(axis=3) == self.mpi_array.max(axis=3)))


    def test_custom_mean_higher_dim_method(self):
        #Mean along specified axies
        self.assertTrue(np.alltrue(self.np_array.mean(axis=2) == self.mpi_array.mean(axis=2)))
        self.assertTrue(np.alltrue(self.np_array.mean(axis=3) == self.mpi_array.mean(axis=3)))


    def test_custom_min_higher_dim_method(self):
        #Min along specified axies
        self.assertTrue(np.alltrue(self.np_array.min(axis=2) == self.mpi_array.min(axis=2)))
        self.assertTrue(np.alltrue(self.np_array.min(axis=3) == self.mpi_array.min(axis=3)))


    def test_custom_std_higher_dim_method(self):
        pass
#TODO: Need to revisit for higher dim
        #Std along specified axies
        # self.assertTrue(np.alltrue(self.np_array.std(axis=2) == self.mpi_array.std(axis=2)))
        # self.assertTrue(np.alltrue(self.np_array.std(axis=3) == self.mpi_array.std(axis=3)))


    def test_custom_sum_higher_dim_method(self):
        #Sum along specified axies
        self.assertTrue(np.alltrue(self.np_array.sum(axis=2) == self.mpi_array.sum(axis=2)))
        self.assertTrue(np.alltrue(self.np_array.sum(axis=3) == self.mpi_array.sum(axis=3)))


class MPIArray4DUndistributedTest(MPIArray4DDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        parms['comm'] = MPI.COMM_WORLD
        parms['rank'] = MPI.COMM_WORLD.Get_rank()
        parms['comm_size'] = MPI.COMM_WORLD.Get_size()
        # Undistributed distribution
        parms['dist'] = 'u'
        #Add 1 to avoid divide by zero errors/warnings
        parms['data'] = np.arange(32).reshape(4,2,2,2) + 1
        parms['local_data'] = parms['data']
        parms['comm_dims'] = None
        parms['comm_coord'] = None
        parms['local_to_global'] = None
        return parms


class MPIArray5DDefaultTest(MPIArrayDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        parms['comm'] = MPI.COMM_WORLD
        parms['rank'] = MPI.COMM_WORLD.Get_rank()
        parms['comm_size'] = MPI.COMM_WORLD.Get_size()
        # Default distribution
        parms['dist'] = 'b'
        #Add 1 to avoid divide by zero errors/warnings
        np_data = np.arange(64).reshape(4,2,2,2,2) + 1
        parms['data'] = np_data
        local_data_map = {0: np_data[:1],
                          1: np_data[1:2],
                          2: np_data[2:3],
                          3: np_data[3:]}
        parms['local_data'] = local_data_map[parms['rank']].tolist()
        parms['comm_dims'] = [parms['comm_size']]
        parms['comm_coord'] = [parms['rank']]
        local_to_global_map = {0 : {0 : (0, 1), 1 : (0, 2), 2 : (0, 2), 3 : (0, 2), 4 : (0, 2)},
                               1 : {0 : (1, 2), 1 : (0, 2), 2 : (0, 2), 3 : (0, 2), 4 : (0, 2)},
                               2 : {0 : (2, 3), 1 : (0, 2), 2 : (0, 2), 3 : (0, 2), 4 : (0, 2)},
                               3 : {0 : (3, 4), 1 : (0, 2), 2 : (0, 2), 3 : (0, 2), 4 : (0, 2)}}
        parms['local_to_global'] = local_to_global_map[parms['rank']]
        return parms


    def test_custom_max_higher_dim_method(self):
        #Max along specified axies
        self.assertTrue(np.alltrue(self.np_array.max(axis=2) == self.mpi_array.max(axis=2)))
        self.assertTrue(np.alltrue(self.np_array.max(axis=3) == self.mpi_array.max(axis=3)))
        self.assertTrue(np.alltrue(self.np_array.max(axis=4) == self.mpi_array.max(axis=4)))


    def test_custom_mean_higher_dim_method(self):
        #Mean along specified axies
        self.assertTrue(np.alltrue(self.np_array.mean(axis=2) == self.mpi_array.mean(axis=2)))
        self.assertTrue(np.alltrue(self.np_array.mean(axis=3) == self.mpi_array.mean(axis=3)))
        self.assertTrue(np.alltrue(self.np_array.mean(axis=4) == self.mpi_array.mean(axis=4)))


    def test_custom_min_higher_dim_method(self):
        #Min along specified axies
        self.assertTrue(np.alltrue(self.np_array.min(axis=2) == self.mpi_array.min(axis=2)))
        self.assertTrue(np.alltrue(self.np_array.min(axis=3) == self.mpi_array.min(axis=3)))
        self.assertTrue(np.alltrue(self.np_array.min(axis=4) == self.mpi_array.min(axis=4)))


    def test_custom_std_higher_dim_method(self):
        pass
#TODO: Need to revisit for higher dim
        #Std along specified axies
        # self.assertTrue(np.alltrue(self.np_array.std(axis=2) == self.mpi_array.std(axis=2)))
        # self.assertTrue(np.alltrue(self.np_array.std(axis=3) == self.mpi_array.std(axis=3)))
        # self.assertTrue(np.alltrue(self.np_array.std(axis=4) == self.mpi_array.std(axis=4)))


    def test_custom_sum_higher_dim_method(self):
        #Sum along specified axies
        self.assertTrue(np.alltrue(self.np_array.sum(axis=2) == self.mpi_array.sum(axis=2)))
        self.assertTrue(np.alltrue(self.np_array.sum(axis=3) == self.mpi_array.sum(axis=3)))
        self.assertTrue(np.alltrue(self.np_array.sum(axis=4) == self.mpi_array.sum(axis=4)))


class MPIArray5DUndistributedTest(MPIArray5DDefaultTest):

    def create_setUp_parms(self):
        parms = {}
        parms['comm'] = MPI.COMM_WORLD
        parms['rank'] = MPI.COMM_WORLD.Get_rank()
        parms['comm_size'] = MPI.COMM_WORLD.Get_size()
        # Undistributed distribution
        parms['dist'] = 'u'
        #Add 1 to avoid divide by zero errors/warnings
        parms['data'] = np.arange(64).reshape(4,2,2,2,2) + 1
        parms['local_data'] = parms['data']
        parms['comm_dims'] = None
        parms['comm_coord'] = None
        parms['local_to_global'] = None
        return parms


if __name__ == '__main__':
    unittest.main()
