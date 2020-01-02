import unittest
from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.mpi_utils import *
from mpids.MPInumpy.errors import TypeError

class AllGatherVTest(unittest.TestCase):

        def setUp(self):
                self.num_procs = MPI.COMM_WORLD.Get_size()
                self.rank = MPI.COMM_WORLD.Get_rank()
                self.local_scalar = np.array(self.rank)
                self.local_data_1d = np.array([self.rank] * self.num_procs)
                self.local_data_2d = \
                        np.array([self.rank] * self.num_procs**2).reshape(self.num_procs,
                                                                          self.num_procs)
                self.global_scalar = np.array(list(range(self.num_procs)))
                self.global_data_1d = \
                    np.array([[proc] * self.num_procs for proc in range(self.num_procs)]
                            ).reshape(self.num_procs, self.num_procs)
                self.global_data_2d = \
                    np.array([[proc] * self.num_procs**2 for proc in range(self.num_procs)]
                            ).reshape(self.num_procs**2, self.num_procs)
                self.empty = np.array([], dtype=self.local_scalar.dtype)


        def test_supplying_a_non_numpy_array_raise_type_error(self):
                data_int = 1
                data_list = [1,2]
                data_tuple = (1,2)
                data_dict = {0: 1, 2: 3}

                with self.assertRaises(TypeError):
                        all_gather_v(data_int)
                with self.assertRaises(TypeError):
                        all_gather_v(data_list)
                with self.assertRaises(TypeError):
                        all_gather_v(data_tuple)
                with self.assertRaises(TypeError):
                        all_gather_v(data_dict)


        def test_gather_empty_value_found_on_all_procs(self):
                local_data = np.array([])
                expected_gathered_data = np.array([] * self.num_procs)

                gathered_data = all_gather_v(local_data)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_single_value_found_on_all_procs(self):
                local_data = self.local_scalar
                expected_gathered_data = self.global_scalar

                gathered_data = all_gather_v(local_data)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_single_value_found_only_on_first_proc_empty_on_rest(self):
                if self.rank == 0:
                        local_data = self.local_scalar
                else:
                        local_data = self.empty
                expected_gathered_data = self.global_scalar[0]

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_single_value_found_only_on_last_proc_empty_on_rest(self):
                last_proc = self.num_procs - 1
                if self.rank == last_proc:
                        local_data = self.local_scalar
                else:
                        local_data = self.empty
                expected_gathered_data = self.global_scalar[last_proc]

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_single_value_found_only_on_middle_proc_empty_on_rest(self):
                middle_proc = self.num_procs // 2
                if self.rank == middle_proc:
                        local_data = self.local_scalar
                else:
                        local_data = self.empty
                expected_gathered_data = self.global_scalar[middle_proc]

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_single_value_found_only_on_first_last_proc_empty_on_rest(self):
                last_proc = self.num_procs - 1
                if self.rank == 0 or self.rank == last_proc:
                        local_data = self.local_scalar
                else:
                        local_data = self.empty

                expected_gathered_data = np.array([0, last_proc])

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_1d_array_found_on_all_procs(self):
                local_data = self.local_data_1d
                expected_gathered_data = self.global_data_1d

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_1d_array_found_only_on_first_proc_empty_on_rest(self):
                if self.rank == 0:
                        local_data = self.local_data_1d
                else:
                        local_data = self.empty
                expected_gathered_data = self.global_data_1d[0]

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_1d_array_found_only_on_last_proc_empty_on_rest(self):
                last_proc = self.num_procs - 1
                if self.rank == last_proc:
                        local_data = self.local_data_1d
                else:
                        local_data = self.empty
                expected_gathered_data = self.global_data_1d[last_proc]

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_1d_array_found_only_on_middle_proc_empty_on_rest(self):
                middle_proc = self.num_procs // 2
                if self.rank == middle_proc:
                        local_data = self.local_data_1d
                else:
                        local_data = self.empty
                expected_gathered_data = self.global_data_1d[middle_proc]

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_1d_array_found_only_on_first_last_proc_empty_on_rest(self):
                last_proc = self.num_procs - 1
                if self.rank == 0  or self.rank == last_proc:
                        local_data = self.local_data_1d
                else:
                        local_data = self.empty
                expected_gathered_data = np.concatenate((self.global_data_1d[0],
                                                         self.global_data_1d[last_proc]))

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_2d_array_found_on_all_procs(self):
                local_data = self.local_data_2d
                expected_gathered_data = self.global_data_2d

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_2d_array_found_on_first_proc_empty_on_rest(self):
                if self.rank == 0:
                        local_data = self.local_data_2d
                else:
                        local_data = self.empty
                expected_gathered_data = self.global_data_2d[self.global_data_2d == 0]

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_2d_array_found_on_last_proc_empty_on_rest(self):
                last_proc = self.num_procs - 1
                if self.rank == last_proc:
                        local_data = self.local_data_2d
                else:
                        local_data = self.empty
                expected_gathered_data = self.global_data_2d[self.global_data_2d == last_proc]

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_2d_array_found_on_middle_proc_empty_on_rest(self):
                middle_proc = self.num_procs // 2
                if self.rank == middle_proc:
                        local_data = self.local_data_2d
                else:
                        local_data = self.empty
                expected_gathered_data = self.global_data_2d[self.global_data_2d == middle_proc]

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_2d_array_found_on_first_last_proc_empty_on_rest(self):
                last_proc = self.num_procs - 1
                if self.rank == 0 or self.rank == last_proc:
                        local_data = self.local_data_2d
                else:
                        local_data = self.empty
                expected_gathered_data = \
                        np.concatenate((self.global_data_2d[self.global_data_2d == 0],
                                        self.global_data_2d[self.global_data_2d == last_proc]))

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_1d_array_found_on_first_proc_2d_array_found_on_rest(self):
                if self.rank == 0:
                        local_data = self.local_data_1d
                else:
                        local_data = self.local_data_2d
                global_data_2d_no_first = \
                        self.global_data_2d[self.global_data_2d != 0].reshape(-1,
                                                                              self.num_procs)
                global_data_1d_w_first = self.global_data_1d[0].reshape(-1,
                                                                       self.num_procs)
                expected_gathered_data = np.concatenate((global_data_1d_w_first,
                                                        global_data_2d_no_first))

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_1d_array_found_on_last_proc_2d_array_found_on_rest(self):
                last_proc = self.num_procs - 1
                if self.rank == last_proc:
                        local_data = self.local_data_1d
                else:
                        local_data = self.local_data_2d
                global_data_2d_no_last = \
                        self.global_data_2d[self.global_data_2d != last_proc].reshape(-1,
                                                                                      self.num_procs)
                global_data_1d_w_last = self.global_data_1d[last_proc].reshape(-1,
                                                                               self.num_procs)
                expected_gathered_data = np.concatenate((global_data_2d_no_last,
                                                         global_data_1d_w_last))

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


        def test_gather_1d_array_found_on_middle_proc_2d_array_found_on_rest(self):
                middle_proc = self.num_procs // 2
                if self.rank == middle_proc:
                        local_data = self.local_data_1d
                else:
                        local_data = self.local_data_2d
                global_data_2d_up_to_mid = \
                        self.global_data_2d[self.global_data_2d < middle_proc].reshape(-1,
                                                                                       self.num_procs)
                global_data_2d_past_mid = \
                        self.global_data_2d[self.global_data_2d > middle_proc].reshape(-1,
                                                                                       self.num_procs)
                global_data_1d_w_mid = self.global_data_1d[middle_proc].reshape(-1,
                                                                                self.num_procs)
                expected_gathered_data = np.concatenate((global_data_2d_up_to_mid,
                                                         global_data_1d_w_mid,
                                                         global_data_2d_past_mid))

                gathered_data = all_gather_v(local_data, shape=expected_gathered_data.shape)
                self.assertEqual(gathered_data.dtype, expected_gathered_data.dtype)
                self.assertEqual(gathered_data.shape, expected_gathered_data.shape)
                self.assertEqual(gathered_data.size, expected_gathered_data.size)
                self.assertTrue(
                        np.alltrue((gathered_data) == (expected_gathered_data)))


if __name__ == '__main__':
        unittest.main()
