from mpi4py import MPI
import numpy as np

from mpids.MPInumpy.MPIArray import MPIArray
from mpids.MPInumpy.utils import _format_indexed_result, global_to_local_key
from mpids.MPInumpy.distributions.Undistributed import Undistributed


"""
    Block-Block implementation of MPIArray abstract base class.
"""
class BlockBlock(MPIArray):

#TODO: Resolve this namespace requirement
        def __getitem__(self, key):
                local_key = global_to_local_key(key,
                                                self.globalshape,
                                                self.local_to_global)
                indexed_result = self.base.__getitem__(local_key)
                indexed_result = _format_indexed_result(key, indexed_result)

                distributed_result = \
                        self.__class__(indexed_result,
                                       dtype=self.dtype,
                                       comm=self.comm,
                                       comm_dims=self.comm_dims,
                                       comm_coord=self.comm_coord,
                                       local_to_global=self.local_to_global)
                #Return undistributed copy of data
                return distributed_result.collect_data()


        #Unique properties to MPIArray
        @property
        def dist(self):
                return ('b', 'b')


        @property
        def globalshape(self):
                if self._globalshape is None:
                        self.__globalshape()
                return self._globalshape

        def __globalshape(self):
                local_shape = self.shape
                comm_shape = []
                axis = 0
                for axis_dim in local_shape:
                    axis_length = self.custom_reduction(MPI.SUM,
                                                        np.asarray(local_shape[axis]),
                                                        axis = axis)
                    comm_shape.append(int(axis_length[0]))
                    axis += 1

                self._globalshape = tuple(comm_shape)


        @property
        def globalsize(self):
                if self._globalsize is None:
                        self.__globalsize()
                return self._globalsize

        def __globalsize(self):
                comm_size = np.zeros(1, dtype='int')
                self.comm.Allreduce(np.array(self.size), comm_size, op=MPI.SUM)
                self._globalsize = int(comm_size)


        @property
        def globalnbytes(self):
                if self._globalnbytes is None:
                        self.__globalnbytes()
                return self._globalnbytes

        def __globalnbytes(self):
                comm_nbytes = np.zeros(1, dtype='int')
                self.comm.Allreduce(np.array(self.nbytes), comm_nbytes, op=MPI.SUM)
                self._globalnbytes = int(comm_nbytes)


        @property
        def globalndim(self):
                if self._globalndim is None:
                        self.__globalndim()
                return self._globalndim

        def __globalndim(self):
                self._globalndim = int(len(self.globalshape))


        #Custom reduction method implementations
        def max(self, **kwargs):
                self.check_reduction_parms(**kwargs)
                local_max = np.asarray(self.base.max(**kwargs))
                global_max = self.custom_reduction(MPI.MAX, local_max, **kwargs)
                return Undistributed(global_max,
                                     dtype=global_max.dtype,
                                     comm=self.comm)

        def mean(self, **kwargs):
                global_sum = self.sum(**kwargs)
                axis = kwargs.get('axis')
                if axis is not None:
                        global_mean = global_sum * 1. / self.globalshape[axis]
                else:
                        global_mean = global_sum * 1. / self.globalsize

                return Undistributed(global_mean,
                                     dtype=global_mean.dtype,
                                     comm=self.comm)


        def min(self, **kwargs):
                self.check_reduction_parms(**kwargs)
                local_min = np.asarray(self.base.min(**kwargs))
                global_min = self.custom_reduction(MPI.MIN, local_min, **kwargs)
                return Undistributed(global_min,
                                     dtype=global_min.dtype,
                                     comm=self.comm)


        def std(self, **kwargs):
                axis = kwargs.get('axis')
                local_mean = self.mean(**kwargs)

                if axis == 0:
                        col_min, col_max = self.local_to_global[1]
                        local_mean = local_mean[col_min: col_max]
                if axis == 1:
                        row_min, row_max = self.local_to_global[0]
                        local_mean = local_mean[row_min: row_max]
#TODO: Explore np kwarg 'keepdims' to avoid force transpose
                        #Force a transpose
                        local_mean = local_mean.reshape(self.shape[0], 1)

                local_square_diff = (self - local_mean)**2
                local_sum_square_diff = \
                        np.asarray(local_square_diff.base.sum(**kwargs))
                global_sum_square_diff = \
                        self.custom_reduction(MPI.SUM,
                                              local_sum_square_diff,
                                              dtype = local_sum_square_diff.dtype,
                                              **kwargs)
                if axis is not None:
                        global_std = np.sqrt(
                                global_sum_square_diff * 1. / self.globalshape[axis])
                else:
                        global_std = np.sqrt(
                                global_sum_square_diff * 1. / self.globalsize)

                return Undistributed(global_std,
                                     dtype=global_std.dtype,
                                     comm=self.comm)


        def sum(self, **kwargs):
                self.check_reduction_parms(**kwargs)
                local_sum = np.asarray(self.base.sum(**kwargs))
                global_sum = self.custom_reduction(MPI.SUM, local_sum, **kwargs)
                return Undistributed(global_sum,
                                     dtype=global_sum.dtype,
                                     comm=self.comm)


        def custom_reduction(self, operation, local_red, axis=None, dtype=None,
                             out=None):
                if dtype is None: dtype = self.dtype

                if axis is None:
                        global_red = np.zeros(local_red.size, dtype=dtype)
                        self.comm.Allreduce(local_red, global_red, op=operation)

                else:
                        row_comm = self.comm.Split(color = self.comm_coord[0],
                                                   key = self.comm.Get_rank())
                        col_comm = self.comm.Split(color = self.comm_coord[1],
                                                   key = self.comm.Get_rank())

                        if axis == 0:
                                col_red = np.zeros(local_red.size, dtype=dtype)
                                col_comm.Allreduce(local_red, col_red, op=operation)

                                local_displacement = np.zeros(1, dtype='int')
                                local_count = np.asarray(col_red.size, dtype='int')
                                displacements = np.zeros(self.comm_dims[1],
                                                         dtype=local_displacement.dtype)
                                counts = np.zeros(self.comm_dims[1],
                                                  dtype=local_count.dtype)
                                total_count = np.zeros(1, dtype=local_count.dtype)

                                #Exclusive scan to determine displacements
                                row_comm.Exscan(local_count, local_displacement, op=MPI.SUM)
                                row_comm.Allreduce(local_count, total_count, op=MPI.SUM)
                                row_comm.Allgather(local_displacement, displacements)
                                row_comm.Allgather(local_count, counts)

                                global_red = np.zeros(total_count, dtype=dtype)
                                # Final conditioning of displacements list
                                displacements[0] = 0

                                mpi_dtype = MPI._typedict[np.sctype2char(local_red.dtype)]
                                row_comm.Allgatherv(col_red,
                                        [global_red, (counts, displacements), mpi_dtype])

                        if axis == 1:
                                row_red = np.zeros(local_red.size, dtype=dtype)
                                row_comm.Allreduce(local_red, row_red, op=operation)

                                local_displacement = np.zeros(1, dtype='int')
                                local_count = np.asarray(row_red.size, dtype='int')
                                displacements = np.zeros(self.comm_dims[0],
                                                         dtype=local_displacement.dtype)
                                counts = np.zeros(self.comm_dims[0],
                                                  dtype=local_count.dtype)
                                total_count = np.zeros(1, dtype=local_count.dtype)

                                #Exclusive scan to determine displacements
                                col_comm.Exscan(local_count, local_displacement, op=MPI.SUM)
                                col_comm.Allreduce(local_count, total_count, op=MPI.SUM)
                                col_comm.Allgather(local_displacement, displacements)
                                col_comm.Allgather(local_count, counts)

                                global_red = np.zeros(total_count, dtype=dtype)
                                # Final conditioning of displacements list
                                displacements[0] = 0

                                mpi_dtype = MPI._typedict[np.sctype2char(local_red.dtype)]
                                col_comm.Allgatherv(row_red,
                                        [global_red, (counts, displacements), mpi_dtype])

                return global_red


        def collect_data(self):
                #First combine the columns(transpose is expensive)
                row_comm = self.comm.Split(color = self.comm_coord[0],
                                           key = self.comm.Get_rank())

                local_displacement = np.zeros(1, dtype='int')
                displacements = np.zeros(self.comm_dims[0],
                                         dtype=local_displacement.dtype)
                counts = np.zeros(self.comm_dims[0], dtype='int')

                #Transpose prior to send to have consistent tranversal
                local_transpose = \
                        np.zeros((self.shape[1], self.shape[0]), dtype=self.dtype)
                local_transpose[:,:] = np.transpose(self.base)
                local_count = np.asarray(local_transpose.size, dtype='int')

                #Exclusive scan to determine displacements
                row_comm.Exscan(local_count, local_displacement, op=MPI.SUM)
                row_comm.Allgather(local_displacement, displacements)
                row_comm.Allgather(local_count, counts)

                #Number of transposed 'rows' will be global number of columns
                ## while the number of columns will be the locals number of rows
                global_row_data_transpose = \
                        np.zeros((self.globalshape[1], self.shape[0]),
                                 dtype=self.dtype)
                # Final conditioning of displacements list
                displacements[0] = 0

                mpi_dtype = MPI._typedict[np.sctype2char(self.dtype)]
                row_comm.Allgatherv(local_transpose,
                        [global_row_data_transpose, (counts, displacements), mpi_dtype])

                #Final transpose to recover original ordering
                global_row_data = \
                        np.zeros((global_row_data_transpose.shape[1],
                                  global_row_data_transpose.shape[0]),
                                  dtype=self.dtype)
                global_row_data[:,:] = np.transpose(global_row_data_transpose)

                #Now let's combin those rows!
                col_comm = self.comm.Split(color = self.comm_coord[1],
                                           key = self.comm.Get_rank())

                displacements = np.zeros(self.comm_dims[1],
                                         dtype=local_displacement.dtype)
                counts = np.zeros(self.comm_dims[1], dtype='int')
                local_count = np.asarray(global_row_data.size, dtype='int')

                #Exclusive scan to determine displacements
                col_comm.Exscan(local_count, local_displacement, op=MPI.SUM)
                col_comm.Allgather(local_displacement, displacements)
                col_comm.Allgather(local_count, counts)

                global_data = np.zeros(self.globalshape, dtype=self.dtype)

                # Final conditioning of displacements list
                displacements[0] = 0

                mpi_dtype = MPI._typedict[np.sctype2char(self.dtype)]
                col_comm.Allgatherv(global_row_data,
                        [global_data, (counts, displacements), mpi_dtype])

                return Undistributed(global_data,
                                     dtype=global_data.dtype,
                                     comm=self.comm)
