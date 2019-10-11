import unittest
from mpids.MPInumpy.utils import get_local_data, low_block, high_block

class UtilsTest(unittest.TestCase):

        def setUp(self):
                self.procs = 3
                self.ranks = [0, 1, 2]
                self.data = list(range(10))
                self.data_length = len(self.data)

        def test_low_block(self):
                self.assertEqual(0, low_block(self.data_length,
                                              self.procs,
                                              self.ranks[0]))

                self.assertEqual(3, low_block(self.data_length,
                                              self.procs,
                                              self.ranks[1]))

                self.assertEqual(6, low_block(self.data_length,
                                              self.procs,
                                              self.ranks[2]))

        def test_high_block(self):
                self.assertEqual(3, high_block(self.data_length,
                                              self.procs,
                                              self.ranks[0]))

                self.assertEqual(6, high_block(self.data_length,
                                              self.procs,
                                              self.ranks[1]))

                self.assertEqual(10, high_block(self.data_length,
                                              self.procs,
                                              self.ranks[2]))

        def test_local_data_default_distribution(self):
                local_data_rank0 = self.data[0:3]
                local_data_rank1 = self.data[3:6]
                local_data_rank2 = self.data[6:10]

                self.assertEqual(local_data_rank0, get_local_data(self.data,
                                                                  'DUMMY',
                                                                  self.procs,
                                                                  self.ranks[0]))

                self.assertEqual(local_data_rank1, get_local_data(self.data,
                                                                  'DUMMY',
                                                                  self.procs,
                                                                  self.ranks[1]))

                self.assertEqual(local_data_rank2, get_local_data(self.data,
                                                                  'DUMMY',
                                                                  self.procs,
                                                                  self.ranks[2]))


if __name__ == '__main__':
        unittest.main()
