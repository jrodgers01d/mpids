from mpi4py import MPI
import pandas as pd
import numpy as np

from mpids.utils.PandasUtils import get_pandas_version


class ParallelDataFrame(pd.DataFrame):
  """ DistributedDataFrame subclass of pd.DataFrame """
  _metadata = ['comm', 'dist','globalShape','global_to_local','loc', 'orient']

  @property
  def __constructor(self):
    return ParallelDataFrame
  @property
  def __constructor_sliced(self):
    from .ParallelSeries import ParallelSeries
    return ParallelSeries
  @property
  def global_to_local(self):
    if self._global_to_local is None:
      self.find_global_to_local()
    return self._global_to_local #_ to call attr without executing getter

  @global_to_local.setter
  def global_to_local(self, something):
    self._global_to_local = something

  @property
  def globalShape(self):
    if self._globalShape is None:
      self.find_globalShape()
      return self._globalShape


  @globalShape.setter
  def globalShape(self, shape):
    self._globalShape = shape

  @property
  def globalColumns(self):
    if(self.dist == 'replicated'):
      return self.columns
    elif(self.orient == 'columns'):
      return pd.Index(list(self.global_to_local.keys()))
    elif(self.orient == 'index'):
      return self.columns

  @property
  def globalIndex(self):
    if(self.dist == 'replicated'):
      return self.index
    elif(self.orient == 'index'):
      return pd.Index(list(self.global_to_local.keys()))
    elif(self.orient == 'columns'):
      return self.index

  @property
  def loc(self):
    from .ParallelPandasUtils import _CustomLocIndexer
    pandas_version = get_pandas_version()
    if (pandas_version >= 0.25):
      return _CustomLocIndexer("loc", self)
    else:
      return _CustomLocIndexer(self, name='loc')

  def __init__(self, data= None, index = None, columns = None, dtype = None, copy = False,
               dist = 'distributed', dist_data= True, orient = 'columns', comm = MPI.COMM_WORLD):
    """
      Class Initializer
      Parameters:
      -----------
      data, index, columns, dtype, and copy are parameter that are in the Pandas dataframe contructor
      Additional Parameters:
      ----------------------
      dist: str, value can be 'distributed' or 'replicated'
      dist_data: Boolean, only applicable for 'distributed' parallel dataframe
        True when the data being passed is already distributed
      orient: str, value can be 'columns' or 'index'
        represents if the data is distributed by columns or by rows
      comm: MPI comm object
    """

    if((dist =='distributed' and dist_data == True) or dist == 'replicated'):
      super().__init__(data, index=index, columns=columns, dtype=dtype, copy=copy)

    elif(dist_data == False and isinstance(data, (pd.DataFrame, dict))):
      if (isinstance(data, pd.DataFrame)):
        dictionary = data.to_dict()
      else:
        dictionary = data
      distributed_data = ParallelDataFrame._get_distributed_data(dictionary, orient='columns', columns=columns, dtype=dtype, comm=comm)
      super().__init__(distributed_data, index=index, columns=columns, dtype=dtype, copy=copy)

    #only have support for a replicated numpy array for now
    if(isinstance(data, np.ndarray)):
      super().__init__(data, index=index, columns=columns, dtype=dtype, copy=copy)
      dist = 'replicated'
      #raise NotImplementedError

    self.comm = comm
    self.dist = dist
    self.orient = orient
    self.globalShape = None
    self.global_to_local = None

    if(dist == 'replicated'):
      self.orient = None

  def get_global_to_local(self):
    return self._global_to_local

  def find_global_to_local(self):
    if(self.dist == 'distributed'):
      #everyone sends their column/or row names (depending on the distribution orientation)
      #partition refers to row or column depending on the distribution orientation
      number_processors = self.comm.Get_size()

      max_local_partitions = np.array(0)
      if(self.orient == 'columns'):
        local_partitions = list(self.columns.values)
      else:
        local_partitions = list(self.index.values)
      local_partition_number = np.array(len(local_partitions))
      self.comm.Allreduce([local_partition_number, 1, MPI.LONG], max_local_partitions, op=MPI.MAX)

      recv_data = []

      if(max_local_partitions > local_partition_number):
        for i in range(0, max_local_partitions - local_partition_number):
          local_partitions.append('###')

      recv_data = self.comm.allgather(local_partitions)

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
      raise Exception("This operation is inapplicable!")
    return

  def find_globalShape(self):
    if(self.dist == 'replicated'):
      (rows, cols) = self.shape
      #raise Exception('Operation not applicable')
    elif(self.orient == 'columns'):
      cols = len(self.global_to_local)
      rows = self.shape[0]
    else:
      rows = len(self.global_to_local)
      cols = self.shape[1]
    self._globalShape = (rows, cols)
    return

  #columns and index introduced in 0.21
  ##TODO need to implement index and columns
  #if inplace = True is used, it will update the DistributedDataFrame
  #if inplace = Flase is used, it will return local part of dataframe with column dropped
  def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
    pandas_version = get_pandas_version()
    if(axis == 'columns'): axis = 1
    if(axis == 'index'): axis = 0

    super_return = self
    if(self.dist == 'distributed' and
        ((axis == 1 or columns != None) and self.orient == 'columns')  or
        ((axis == 0 or index!=None) and self.orient == 'index')):

      if(index != None):
        axis = 0
        labels = index
        index = None
      elif(columns != None):
        axis = 1
        labels = columns
        columns = None

      col_or_row_names = self.global_to_local.keys()
      local_labels = None

      if isinstance(labels, list):
        local_labels = []
        for a_label in labels:
          if(a_label not in col_or_row_names):
            raise Exception("Column/Row does not exist!")
          elif((axis == 1 and a_label in self.columns.values) or
               (axis == 0 and a_label in self.index.values)):
            local_labels.append(a_label)
      elif(labels not in col_or_row_names):#incase labels is not a list
        raise Exception("Column does not exist!")
      elif((axis ==1 and labels in self.columns.values) or
           (axis ==0 and labels in self.index.values)):
        local_labels = labels

      if(inplace == True and self.get_global_to_local() != None):
        self._global_to_local=None

      #perform drop
      if(local_labels is not None and len(local_labels)!=0):
        if(pandas_version >= 0.21):
          super_return = super().drop(local_labels, axis, index, columns, level, inplace, errors)
        else:
          super_return = super().drop(local_labels, axis, level, inplace, errors)
      elif(inplace == True): # when some other node had the item to be dropped
        return self
      else:
        self.__constructor(data = self, dist = self.dist, comm = self.comm, orient = self.orient, dist_data = True)

    # for replicated distribution OR
    # for dropping a row in a column-distribution OR
    # a column in row-distribution
    else:
      if(pandas_version >= 0.21):
        super_return = super().drop(labels,axis,index,columns,level,inplace,errors)
      else:
        super_return = super().drop(labels,axis,level,inplace,errors)

    if(inplace == True):
      return self
    else:
      return self.__constructor(data = super_return, dist = self.dist, comm = self.comm, orient = self.orient)

  def apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwds):
    """
    Function to apply a function along an axis of the ParallelDataFrame
    Return: Returns a parallel dataframe or paralle series which is a result of applying the function
    """
    pandas_version = get_pandas_version()
    if(pandas_version >= 0.23):
      super_return = super().apply(func, axis=axis, raw=raw, result_type=result_type, args=args)
    else:
      super_return = super().apply(func, axis=axis, raw=raw, args=args)
    #dist_data = True if self.dist == 'distributed' else False

    if(isinstance(super_return, pd.Series)):
      return self.__constructor_sliced(data = super_return, dist = self.dist, comm = self.comm)#, dist_data = dist_data)
    elif(isinstance(super_return, pd.DataFrame)):
      return self.__constructor(data = super_return, dist = self.dist, comm = self.comm, orient = self.orient)#, dist_data = dist_data)


  def corr(self, method = 'pearson', min_periods = 1):
    """
    Function to compute correlation amongst the columns of a ParallelDataFrame
    excluding NA/null values

    Assumes columns distribution since that is the only one that is supported currently

    Return: Returns a parallel dataframe which is a corelation matrix
    """
    if(self.dist == 'distributed' and self.orient == 'columns'):
      df = self
      comm = self.comm
      rank = comm.Get_rank()
      number_processors = comm.Get_size()

      output = super().corr(method = method, min_periods = min_periods)

      to_send = np.array(df.values, order='C', dtype=np.float32)#sending contiguous array
      to_send_shape = np.array([df.shape[0],df.shape[1]])

      req = []
      #each rank will 1st send to one up and below, then 2 up and below and so on alternatively...until the last one
      #creates a distributed corr dataframe where data is divided by columns of dataframe
      for step in range (1, number_processors):
        #send
        destination1 = rank + step
        destination2 = rank - step

        #send to one below
        req = self.__send_for_corr(to_send, df.columns, to_send_shape, destination1, number_processors, req, df.shape, rank)
        #send to one up
        req = self.__send_for_corr(to_send, df.columns, to_send_shape, destination2, number_processors, req, df.shape, rank)

        #receive
        source1 = rank - step
        source2 = rank + step

        #block receive and process from one above
        output = self.__recv_and_process_for_corr(source1, number_processors, rank, df, method, min_periods, output)
        #block receive and process from one below
        output = self.__recv_and_process_for_corr(source2, number_processors, rank, df, method, min_periods, output)

        #wait for all the sends to complete
        if(len(req) != 0):
          MPI.Request.Waitall(req)
          req = []
      del to_send
      del to_send_shape

      output = output.transpose() # to have column distribution
      #sort so that all the rows are arragend similarly in various processors

      output.sort_index(inplace = True)

      return self.__constructor(output, dist = self.dist, comm= comm, dist_data = True, orient = self.orient)
    elif(self.dist == 'distributed' and self.orient=='index'):
      raise NotImplementedError
    else:
      super_return = super().corr(method=method, min_periods=min_periods)
      return self.__constructor(data = super_return, dist = self.dist, comm = self.comm, orient = self.orient)

  def div(self, other, axis='columns', level=None, fill_value=None):
    super_return = super().div(other, axis=axis, level=level, fill_value=fill_value)
    return self.__constructor(data = super_return, dist = self.dist, comm = self.comm, orient=self.orient)

  #experimental, should work (not tested rigorously), not present in Pandas version 0.16.2 (used on crill)
  def round(self, decimals=0, *args, **kwargs):
    super_return = super().round(decimals)
    return self.__constructor(data = super_return, dist = self.dist, comm = self.comm, orient=self.orient)

  @classmethod
  def from_dict(cls, data, orient = "columns", columns = None, dtype = None,
                comm = MPI.COMM_WORLD, dist = 'distributed') -> "ParallelDataFrame":
    """
    Class method that can create a dataframe from a dictionary based on the specified orientation (index or columns)
    Both columns and index orientation is supported
    """
    if(dist == 'distributed'):
      distributed_data = ParallelDataFrame._get_distributed_data(data, orient, columns, dtype, comm)
      return cls(data= distributed_data, comm= comm, dtype = dtype, dist_data = True, orient = orient)
    else:
      pandas_version = get_pandas_version()
      if pandas_version >= 0.23:
        dataFrame = pd.DataFrame.from_dict(data, orient = orient, columns = columns, dtype = dtype)
      else:
        dataFrame = pd.DataFrame.from_dict(data, orient = orient, dtype = dtype)
      return cls(data=dataFrame, comm= comm, dtype = dtype, dist = 'replicated')

  #returns distributed local dataFrame and takes in a serial dictionary
  @staticmethod
  def _get_distributed_data(data, orient, columns, dtype, comm):
    """
    Helper routine to distribute dictionary data uniformly
    """
    no_of_rows, keys, indices = ParallelDataFrame.__dictionary_info(data, comm)
    key_and_value = {}
    for i in indices:
      key_and_value[keys[i]] = data[keys[i]]#[)]?
    pandas_version = get_pandas_version()
    if pandas_version >= 0.23:
      dataFrame = pd.DataFrame.from_dict(key_and_value, orient = orient, columns = columns, dtype = dtype)
    else:
      dataFrame = pd.DataFrame.from_dict(key_and_value, orient = orient, dtype = dtype)
    return dataFrame

  #can be outside of the class instead of staticmethod
  @staticmethod
  def __dictionary_info(dict, comm):
    """
    Helper routine to return the number of rows,
    keys, and a set of indices assigned to each
    process.
    """

    rank = comm.Get_rank()
    size = comm.Get_size()

    nr_of_rows = len(dict)
    keys = []

    if rank == 0:
      keys = sorted(dict, key = lambda key: len(dict[key]))

    keys = comm.bcast(keys, root=0)

    indices=[]

    for i in range(0, ((nr_of_rows//size) + 1), 2):
      thisindex = (i*size + rank)
      if (thisindex < nr_of_rows):
        indices.append(thisindex)
      j = i+2
      thisindex = (j*size - rank -1)
      if ( thisindex < nr_of_rows ):
        indices.append(thisindex )

    return nr_of_rows, keys, indices

  def __send_for_corr(self, to_send, columns, to_send_shape, destination, number_processors, req, shape, rank):
    """
    Helper routine for the corelation function, performs a non-blocking send
    Returns: A request object which contains all the send requests
    """
    if destination >= 0 and destination < number_processors:
      tag = rank + destination
      req.append(self.comm.Isend([to_send_shape, 2, MPI.LONG], dest = destination, tag = tag + 100))
      req.append(self.comm.isend(columns, dest = destination, tag = tag + 10))
      req.append(self.comm.Isend([to_send, shape[0]*shape[1], MPI.FLOAT], dest = destination, tag = tag))
    return req

  def __recv_and_process_for_corr(self, source, number_processors, rank, df, method, min_periods, output):
    """
    Helper routine for the corelation function, performs a blocking receive, processes the received data,
    and appends ot the dataframe that it receives as an input parameter
    Returns: a dataframe
    """
    pandas_version = get_pandas_version()
    if source >= 0 and source < number_processors:
      tag = source + rank

      recv_shape = np.zeros(2, dtype = np.int)
      self.comm.Recv([recv_shape, 2, MPI.LONG], source = source, tag = tag + 100)

      data_labels = [] #file names
      data_labels = self.comm.recv(source = source, tag = tag + 10)

      recv_data = np.zeros([recv_shape[0], recv_shape[1]], dtype = np.float32)
      self.comm.Recv([recv_data, recv_shape[0]*recv_shape[1],MPI.FLOAT], source = source, tag = tag)

     #create a temp_df and find corelation with new files and then concat with output (corelation matrix)
      temp_df = pd.DataFrame()
      i = 0
      # create a df from the data received
      for a_label in data_labels:
        temp_df[a_label] = recv_data[:,i]
        i += 1
      del recv_data

      #add to local df
      for column in df.columns:
        temp_df[column] = df[column].values
      #-----can be done without the 2 lines below
      temp_df['index'] = df.index.values
      temp_df.set_index('index', inplace = True)

      temp_output = temp_df.corr(method = method, min_periods = min_periods)

      if(pandas_version >= 0.23):
        output=output.append(temp_output, sort=False)
        #merge duplicate rows if any
        output = output.groupby(output.index,sort=False,axis=0).min(axis = 1, skipna = True)
      else:
        output = output.append(temp_output)
        # merge duplicate rows if any
        output = output.groupby(output.index,sort=False,axis=0).min()
      del temp_output
      #drop duplicate-data tht exists on other processors (comparison of docs in a processor)
      for a_label in data_labels:
        if a_label in output.index:
          output.drop(a_label, axis=0, inplace = True)

    return output

  def _dict_depth(self,d):
    if isinstance(d, dict):
      return 1 + (max(map(self._dict_depth, d.values())) if d else 0)
    return 0

 
