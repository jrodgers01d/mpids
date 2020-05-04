#The examples of these tests are setup up in such a way that the tests can be performed for 3 processors or less.

# TO RUN: mpiexec -n 2 python3 ./tests/ParallelSeries_test.py

import sys
#sys.path.append(".") # Adds higher directory to python modules path.
sys.path.append("..") # Adds higher directory to python modules path.

import unittest
import numpy as np
import pandas as pd

#from PandasUtils import get_pandas_version
from mpids.MPIpandas.src.ParallelDataFrame import ParallelDataFrame
from mpids.MPIpandas.src.ParallelSeries import ParallelSeries

class ParallelSeriesTest(unittest.TestCase):
  
  def setUp(self):
    self.d = {'col1': [1, 2], 'col2': [3, 4], 'col3':[5, 6]}
    self.dict1 = {'a': 1, 'b':3, 'c':5, 'd': 6, 'e':7}
    self.dict2 = {'a': [1,2,0], 'b':[3,4,2], 'c':[5,6,4], 'd':[6,7,5], 'e':[7,8,6],'f': [4,2,3],'g':[5,4,6]}
    self.dict3 = {'a': [1.4,2.4,0.4], 'b':[3.4,4.4,2.4], 'c':[5.4,6.4,4.4], 'd':[6.4,7.4,5.4], 'e':[7.4,8.4,6.4],'f': [4.4,2.4,3.4],'g':[5.4,4.4,6.4]}
    self.np_array1 = np.array(['a','b', 'c', 'd'])
    self.np_array2 = np.array([['a','b', 'c', 'd', 'a', 'c', 'f', 'g', 'd', 'g', 'c', 'f']])
    self.np_array3 = np.array(['a','b', 'c', 'd', 'a', 'c', 'f', 'g', 'd', 'g', 'c', 'f'])
  
  #Testing constructor-------------------------------
  def test_creation_of_empty_parallel_series(self):
    parallel_series = ParallelSeries()
    self.assertTrue(isinstance(parallel_series, ParallelSeries))
    self.assertTrue(parallel_series.empty)
  
  def test_creation_of_replicated_series_from_dictionary(self):
    series = pd.Series(self.dict1)
    rep_series = ParallelSeries(self.dict1, dist = 'replicated')
    
    self.assertTrue(isinstance(rep_series, ParallelSeries))
    self.assertTrue(rep_series.equals(series))
  
  def test_creation_of_replicated_series_from_np_array(self):
    series = pd.Series(self.np_array1)
    rep_series = ParallelSeries(self.np_array1, dist = 'replicated')
    
    self.assertTrue(isinstance(rep_series, ParallelSeries))
    self.assertTrue(rep_series.equals(series))
  
  def test_creating_distributed_series_by_gettting_row_from_column_distributed_dataframe(self):  
    dist_df = ParallelDataFrame(data=self.d, dist_data = False)
    parallel_series = dist_df.loc[1]
    
    pd_df = pd.DataFrame(self.d)
    pd_series = pd_df.loc[1]
    
    self.assertTrue(isinstance(parallel_series, ParallelSeries))
    self.assertEqual(parallel_series.dist, "distributed")
    
    self.assertTrue(parallel_series.collect().sort_index().equals(pd_series.sort_index()))
    
  #Testing value_counts----------------------------------------------------------
  def test_value_count_with_distributed_series_and_int_data(self):
    pd_df = pd.DataFrame(self.dict2)
    pd_series = pd_df.loc[1]
    
    dist_df = ParallelDataFrame(data= self.dict2, dist_data = False)
    dist_series = dist_df.loc[1]
    
    pd_series_vc = pd_series.value_counts()
    dist_series_vc = dist_series.value_counts()
    #convert the indices to string (that is what the parallelPandas returns)
    pd_series_vc.index = pd_series_vc.index.map(str)
   
    self.assertTrue(dist_series_vc.dist, 'replicated')
    self.assertTrue(dist_series_vc.sort_index().equals(pd_series_vc.sort_index()))
  
  def test_value_count_with_distributed_series_and_string_data(self):
    pd_df = pd.DataFrame(self.np_array2)
    pd_series = pd_df.loc[0]
    
    dist_df = ParallelDataFrame(pd_df, dist_data = False)
    dist_series = dist_df.loc[0]
    pd_series_vc = pd_series.value_counts()
    dist_series_vc = dist_series.value_counts()
    
    self.assertTrue(dist_series_vc.dist, 'replicated')
    self.assertTrue(dist_series_vc.sort_index().equals(pd_series_vc.sort_index()))
  
  def test_value_count_with_distributed_series_and_float_data(self):
    pd_df = pd.DataFrame(self.dict3)
    pd_series = pd_df.loc[1]
    
    dist_df = ParallelDataFrame(data= self.dict3, dist_data = False)
    dist_series = dist_df.loc[1]
    
    pd_series_vc = pd_series.value_counts()
    dist_series_vc = dist_series.value_counts()
    #convert the indices to string (that is what the parallelPandas returns)
    pd_series_vc.index = pd_series_vc.index.map(str)
    
    self.assertTrue(dist_series_vc.dist, 'replicated')
    self.assertTrue(dist_series_vc.sort_index().equals(pd_series_vc.sort_index()))
    
    
  def test_value_count_with_replicated_series_and_string_data(self):
    pd_series = pd.Series(self.np_array3)
    rep_series = ParallelSeries(self.np_array3, dist = 'replicated')
    
    pd_series_vc = pd_series.value_counts()
    rep_series_vc = rep_series.value_counts()
  
    self.assertTrue(rep_series_vc.dist, 'replicated')
    self.assertTrue(rep_series_vc.sort_index().equals(pd_series_vc.sort_index()))
    
  #testing global_to_local-------------------------------------------------------
  def test_global_to_local_distributed_series(self):
    pd_df = pd.DataFrame(self.dict3)
    pd_series = pd_df.loc[1]
    
    dist_df = ParallelDataFrame(data= self.dict3, dist_data = False)
    dist_series = dist_df.loc[1]
   
    self.assertEqual(set(dist_series.global_to_local.keys()), set(pd_series.index))
    
  def test_global_to_local_replicated_series(self):
    dist_df = ParallelDataFrame(data= self.dict3, dist = 'replicated')
    
    dist_series = dist_df.loc[1]
   
    self.assertRaises(ValueError, dist_series.find_global_to_local, )
  
  #testing globalIndex-------------------------------------------------------
  def test_globalIndex_distributed_series(self):
    pd_df = pd.DataFrame(self.dict3)
    pd_series = pd_df.loc[1]
    
    dist_df = ParallelDataFrame(data= self.dict3, dist_data = False)
    dist_series = dist_df.loc[1]
    
    self.assertEqual(set(dist_series.globalIndex), set(pd_series.index))
  
  def test_globalIndex_replicated_series(self):
    pd_df = pd.DataFrame(self.dict3)
    pd_series = pd_df.loc[1]
    
    dist_df = ParallelDataFrame(data= self.dict3, dist = 'replicated')
    dist_series = dist_df.loc[1]
    
    self.assertEqual(set(dist_series.globalIndex), set(pd_series.index))
  
 
if __name__ == "__main__":
  unittest.main()