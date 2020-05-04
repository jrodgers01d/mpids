from mpi4py import MPI
import pandas as pd
import numpy as np

import itertools
from mpids.utils.PandasUtils import get_pandas_version

import mpids.MPIcollections.MPICounter as MPICounter

import collections

class ParallelSeries(pd.Series):
  """ DistributedSeries subclass of pd.Series """
  _metadata = ['comm', 'dist', 'global_to_local']
  @property
  def __constructor(self):
    from .ParallelSeries import ParallelSeries
    return ParallelSeries

  @property
  def _constructor_expanddim(self):
    from .ParallelDataFrame import ParallelDataFrame
    return ParallelDataFrame

  @property
  def global_to_local(self):
    if self._global_to_local is None:
      self.find_global_to_local()
    return self._global_to_local

  @global_to_local.setter
  def global_to_local(self, something):
    self._global_to_local = something

  @property
  def globalIndex(self):
    if(self.dist == 'replicated'):
      return self.index
      #raise Exception('Operation not applicable')
    else:
      return pd.Index(list(self.global_to_local.keys()))


  def __init__(self,data=None, index=None, dtype=None, name=None, copy=False, fastpath=False,
               comm=MPI.COMM_WORLD, dist='distributed', dist_data= True):

    if((dist =='distributed' and dist_data == True) or dist =='replicated'):
      super().__init__(data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath)
    else:#creating empty objects
      super().__init__(None, index = index, dtype=dtype, name=name, copy=copy, fastpath=fastpath)

    self.comm = comm
    self.dist = dist
    self.global_to_local = None

  #collects a distributed series and returns a replicated series
  def collect(self):
    if(self.dist == 'replicated'):
      raise Exception('Operation not applicable!')

    comm = self.comm
    rank = comm.Get_rank()
    number_processors = comm.Get_size()

    values = self.values.tolist()
    indices = self.index.tolist()

    number_values = np.array(len(values))
    number_max = np.array(0)
    comm.Allreduce ([number_values,1,MPI.LONG], number_max, op=MPI.MAX)

    #get all values
    all_values = []
    values_to_send = ['###' for i in range(int(number_max))]

    if(number_max > number_values):
      values_to_send[0:number_values] = np.array(values)
    else:
      values_to_send[0:number_max] = np.array(values)

    all_values = comm.allgather(values_to_send)

    del values_to_send

    #flatten the list of lists
    all_values_flattened = list(itertools.chain.from_iterable(all_values))
    #get rid of all the fill values
    all_values_list = [x for x in all_values_flattened if x != '###']


    #send and gather all indices
    indices_combined = []
    indices_to_send = ['###' for i in range(int(number_max))]

    indices_to_send[:int(number_values)] = indices

    indices_combined = comm.allgather(indices_to_send)

    #flatten the list of lists
    indices_flattened = list(itertools.chain.from_iterable(indices_combined))
    #get rid of all the fill values
    indices_list = [x for x in indices_flattened if x != '###']

    return self.__constructor(all_values_list, index = indices_list, dist='replicated', name = self.name)

  def value_counts(self, normalize = False, sort = True, ascending = False, bins = None, dropna = True):
    """
    Function that returns a replicated series containing counts of unique values
    """
    if(self.dist == 'distributed'):
      comm = self.comm
      rank = comm.Get_rank()
      number_processors = comm.Get_size()

      #create a word count dictionary distributed among all the processes
      words_count = MPICounter.Counter_all(self.values.tolist(), tracing = False, tokens_per_iter = 1000000)

      #collect all data to all processors-------------

      #define tuple datatype
      np_word_tuple_dtype = [('word', '|S20'), ('count', np.int32)]

      test_array = np.zeros(2, dtype = np_word_tuple_dtype)
      displacements = [test_array.dtype.fields[field][1] for field in ['word', 'count']]

      mpi_word_tuple_dtype = MPI.Datatype.Create_struct([20,1],displacements,[MPI.CHAR, MPI.INTEGER4])
      mpi_word_tuple_dtype.Commit()

      to_send_list = list(words_count.items())

      number_tuples = np.array(len(to_send_list))
      number_max = np.array(0)
      comm.Allreduce ([number_tuples,1,MPI.LONG], number_max, op=MPI.MAX)

      combined = np.full((number_max*number_processors), dtype = np_word_tuple_dtype, fill_value = 0)
      tuples_to_send = np.full((number_max), dtype = np_word_tuple_dtype, fill_value = 0)


      tuples_to_send[0:number_tuples] = to_send_list


      comm.Allgather([tuples_to_send, number_max, mpi_word_tuple_dtype],
                     [combined, number_max, mpi_word_tuple_dtype])

      combined_cleaned = [(x.decode(),y) for (x,y) in combined if x.decode() != '0']
      combined_dict = dict(combined_cleaned)

      value_counts = self.__constructor(list(combined_dict.values()), index = list(combined_dict.keys()), dist='replicated', name = self.name)
      if(sort):
        value_counts.__sort_series_by_value(ascending=ascending, inplace = True)

      return value_counts
    else:
      super_return = super().value_counts(normalize = normalize, sort = sort, ascending = ascending, bins = bins, dropna = dropna)
      return self.__constructor(super_return)

  def find_global_to_local(self):
    if(self.dist == 'distributed'):
      #everyone sends their index names
      number_processors = self.comm.Get_size()

      max_local_index = np.array(0)
      local_indices = list(self.index.values)
      local_index_number = np.array(len(local_indices))

      self.comm.Allreduce([local_index_number, 1, MPI.LONG], max_local_index, op=MPI.MAX)

      recv_data = []

      if(max_local_index > local_index_number):
        for i in range(0, max_local_index - local_index_number):
          local_indices.append('###')

      recv_data = self.comm.allgather(local_indices)

      #build the global to local mapping
      global_to_local = {}
      processor = 0
      for processor_partitions in recv_data:
        for one_partition in processor_partitions:
          if one_partition != '###':
            global_to_local[one_partition] = processor
        processor += 1
      self._global_to_local = global_to_local # set the attr without executing setter
    else:
      raise ValueError("This operation is inapplicable")
      raise #Exception("This operation is inapplicable!")
    return


  def __sort_series_by_value(self,ascending, inplace):
    pandas_version = get_pandas_version()
    if(pandas_version >= 0.17):
      return self.sort_values(axis = 0, ascending = ascending, inplace = inplace)#
    else:
      return self.sort(axis = 0, ascending = ascending, inplace = inplace)# for pandas 0.16
