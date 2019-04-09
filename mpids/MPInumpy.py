from mpi4py import MPI
import numpy as np

class MPINpArray:
    def __init__(self, np_array, comm = MPI.COMM_WORLD)
        self.data = np_array.data
        self.dtype = np_array.dtype
        self.shape = np_array.shape
        self.shape_global = None
        self.strides = np_array.strides
        self.comm = comm

def dummy_function_for_commit_test():
        """Still configuring environment"""
        pass
