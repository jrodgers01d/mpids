#The examples of these tests are setup up in such a way that the tests can be performed for 3 processors or less.

# RUN from the : mpiexec -n 2 python3 ./tests/ParallelDataFrame_test.py

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import unittest
import numpy as np
import pandas as pd

from mpids.utils.PandasUtils import get_pandas_version
from mpids.MPIpandas.ParallelDataFrame import ParallelDataFrame
from mpids.MPIpandas.ParallelSeries import ParallelSeries

class ParallelDataFrameTest(unittest.TestCase):

  def setUp(self):
    self.dict1 = {'key1':[10, 11, 22], 'key2': [23,34,56],'key3':[1, 2, 3]}
    self.dict2 = {'key1':[10, 11, 22], 'key2': [23,34,56],'key3':[1, 2, 3], 'key4': [29, 38, 47]}
    self.dict3 = {'key1':[10, 11, 22], 'key2': [23,34,56],'key3':[1, 2, 3], 'key4': [29, 38, 47],
                  'key5':[10, 11, 22], 'key6': [23,34,56],'key7':[1, 2, 3], 'key8': [29, 38, 47]}
    self.pd_df1 = pd.DataFrame([[4.0, 9.0, 16.0, 25.0, 36.0]] * 5, columns=['A', 'B', 'C', 'D', 'E'])

    self.pd_df2 = pd.DataFrame({'angles':[0, 3, 4],
                                'degrees':[360, 180, 360],
                                'equalsides':[0, 3, 2]},
                               index=['circle', 'triangle', 'rectangle'])

    self.df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
                                      'degrees': [360, 180, 360, 360, 540, 720],
                                      'equalsides': [0, 3, 2, 4, 5, 6]},
                                    index=[['A', 'A', 'A', 'B', 'B', 'B'],
                                          ['circle', 'triangle', 'rectangle',
                                          'square', 'pentagon', 'hexagon']])

  def test_canary(self):
    self.assertTrue(True)

  #Testing constructor-------------------------------
  def test_creation_of_empty_parallel_dataframe(self):
    df = ParallelDataFrame()

    self.assertTrue(isinstance(df,ParallelDataFrame))
    self.assertTrue(df.empty)

  def test_replicated_df_creation_with_constructor_input_dictionary(self):
    df = pd.DataFrame(self.dict1)
    rep_df = ParallelDataFrame(self.dict1, dist = 'replicated')

    self.assertEqual(df.shape, rep_df.shape)
    self.assertTrue(isinstance(rep_df, ParallelDataFrame))
    self.assertEqual(rep_df.dist, 'replicated')

  def test_distributed_df_creation_with_constructor_input_dictionary(self):
    df = ParallelDataFrame(self.dict2, dist_data = False)

    self.assertEqual(df.globalShape, (3,4))
    self.assertEqual(df.dist, "distributed")

  def test_distributed_df_creation_with_constructor_input_dataframe(self):
    df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]))
    dist_df = ParallelDataFrame(df, dist_data = False)

    self.assertTrue(isinstance(dist_df, ParallelDataFrame))
    self.assertEqual(dist_df.dist, 'distributed')
    self.assertEqual(dist_df.globalShape, df.shape)
    self.assertNotEqual(dist_df.shape, dist_df.globalShape)

  #Testing from_dict function----------------------------------
  def test_distributed_df_creation_with_from_dict_function_orient_index(self):
    df = ParallelDataFrame.from_dict(self.dict2, orient = 'index')
    self.assertEqual(df.globalShape, (4,3))
    self.assertEqual(set(list(df.globalIndex)), set(['key1','key2','key3','key4']))
    self.assertEqual(list(df.globalColumns), [0, 1, 2])

  def test_distributed_df_creation_with_from_dict_function_orient_columns(self):
    df = ParallelDataFrame.from_dict(self.dict2, orient = 'columns')
    self.assertEqual(df.globalShape, (3,4))
    self.assertEqual(set(list(df.globalColumns)), set(['key1','key2','key3','key4']))
    self.assertEqual(list(df.globalIndex), [0, 1, 2])

  def test_replicated_df_creation_with_from_dict_function_orient_index(self):
    pd_df = pd.DataFrame.from_dict(self.dict2, orient = 'index')
    df = ParallelDataFrame.from_dict(self.dict2, orient = 'index', dist = 'replicated')

    self.assertTrue(df.equals(pd_df))

  def test_replicated_df_creation_with_from_dict_function_orient_columns(self):
    pd_df = pd.DataFrame.from_dict(self.dict2, orient = 'columns')
    df = ParallelDataFrame.from_dict(self.dict2, orient = 'columns', dist = 'replicated')

    self.assertTrue(df.equals(pd_df))

  #Testing global_to_local property---------------------------------
  def test_global_to_local_functionality_with_column_distribution(self):
    df = ParallelDataFrame(self.dict2, dist_data = False)
    self.assertTrue(isinstance(df.global_to_local, dict))
    self.assertEqual(len(df.global_to_local), 4)
    self.assertEqual(set(list(df.global_to_local.keys())), set(['key1','key2','key3','key4']))

  def test_global_to_local_functionality_with_index_distribution(self):
    df = ParallelDataFrame.from_dict(self.dict2, orient = 'index')
    self.assertTrue(isinstance(df.global_to_local, dict))
    self.assertEqual(len(df.global_to_local), 4)
    self.assertEqual(set(list(df.global_to_local.keys())), set(['key1','key2','key3','key4']))

  #Testing 'drop' function---------------------------------------
  def test_inplace_dropping_multiple_columns_in_column_distributed_dataframe(self):
    df = ParallelDataFrame(self.dict3, dist_data = False)
    self.assertEqual(df.globalShape, (3,8))
    df.drop(['key4','key8'], axis = 1, inplace = True)
    self.assertEqual(set(list(df.globalColumns)), set(['key1', 'key2', 'key3', 'key5', 'key6', 'key7']))
    self.assertEqual(list(df.globalIndex), [0, 1, 2])

  if(get_pandas_version() >= 0.21):
    def test_inplace_dropping_multiple_columns_in_column_distributed_dataframe_specifying_columns(self):
      df = ParallelDataFrame(self.dict3, dist_data = False)
      self.assertEqual(df.globalShape, (3,8))
      df.drop(columns=['key4','key8'], inplace = True)
      self.assertEqual(set(list(df.globalColumns)), set(['key1', 'key2', 'key3', 'key5', 'key6', 'key7']))
      self.assertEqual(list(df.globalIndex), [0, 1, 2])

  def test_non_inplace_dropping_single_column_in_column_distributed_dataframe(self):
    df = ParallelDataFrame(self.dict2, dist_data = False)
    self.assertEqual(df.globalShape, (3,4))
    new_df = df.drop('key4', axis = 1, inplace = False)
    self.assertEqual(set(list(new_df.globalColumns)), set(['key1', 'key2', 'key3']))
    self.assertEqual(list(new_df.globalIndex), [0, 1, 2])

  def test_inplace_dropping_single_row_in_index_distributed_dataframe(self):
    df = ParallelDataFrame.from_dict(self.dict2, orient = 'index')
    self.assertEqual(df.globalShape, (4,3))
    df.drop('key4', axis = 0, inplace = True)
    self.assertEqual(set(list(df.globalIndex)), set(['key1', 'key2', 'key3']))
    self.assertEqual(list(df.globalColumns), [0, 1, 2])

  def test_non_inplace_dropping_single_row_in_index_distributed_dataframe(self):
    df = ParallelDataFrame.from_dict(self.dict2, orient = 'index')
    self.assertEqual(df.globalShape, (4,3))
    new_df = df.drop('key4', axis = 0, inplace = False)
    self.assertEqual(set(list(new_df.globalIndex)), set(['key1', 'key2', 'key3']))
    self.assertEqual(list(new_df.globalColumns), [0, 1, 2])

  def test_inplace_dropping_single_column_in_index_distributed_dataframe(self):
    df = ParallelDataFrame.from_dict(self.dict2, orient = 'index')
    self.assertEqual(df.globalShape, (4,3))
    df.drop(1, axis = 1, inplace = True)
    self.assertEqual(set(list(df.globalIndex)), set(['key1', 'key2', 'key3', 'key4']))
    self.assertEqual(list(df.globalColumns), [0,2])

  def test_inplace_dropping_single_row_in_column_distributed_dataframe(self):
    df = ParallelDataFrame(self.dict2, dist_data = False)
    self.assertEqual(df.globalShape, (3,4))
    df.drop(2, axis = 0, inplace = True)
    self.assertEqual(set(list(df.globalColumns)), set(['key1', 'key2', 'key3', 'key4']))
    self.assertEqual(list(df.globalIndex), [0, 1])

  if(get_pandas_version() >= 0.21):
    def test_inplace_dropping_single_row_in_column_distributed_dataframe_specifying_index(self):
      df = ParallelDataFrame(self.dict2, dist_data = False)
      self.assertEqual(df.globalShape, (3,4))
      df.drop(index=2, inplace = True)
      self.assertEqual(set(list(df.globalColumns)), set(['key1', 'key2', 'key3', 'key4']))
      self.assertEqual(list(df.globalIndex), [0, 1])

  def test_inplace_dropping_single_row_replicated_dataframe(self):
    df = ParallelDataFrame(self.dict2, dist = 'replicated')
    df.drop(2, axis = 0, inplace = True)
    self.assertEqual(set(list(df.globalColumns)), set(['key1', 'key2', 'key3', 'key4']))
    self.assertEqual(list(df.globalIndex), [0, 1])

  def test_non_inplace_dropping_single_column_replicated_dataframe(self):
    df = ParallelDataFrame(self.dict2, dist = 'replicated')
    new_df = df.drop('key4', axis = 1, inplace = False)
    self.assertEqual(set(list(new_df.globalColumns)), set(['key1', 'key2', 'key3']))
    self.assertEqual(list(new_df.globalIndex), [0, 1, 2])

  #new index/column introduced in Pandas version 0.21
  if(get_pandas_version()>=0.21):
    def test_non_inplace_dropping_multiple_columns_replicated_dataframe(self):
      df = ParallelDataFrame(self.dict3, dist = 'replicated')
      new_df = df.drop(columns=['key4','key7'], inplace = False)
      self.assertEqual(set(list(new_df.globalColumns)), set(['key1', 'key2', 'key3', 'key5', 'key6', 'key8']))
      self.assertEqual(list(new_df.globalIndex), [0, 1, 2])

    def test_non_inplace_dropping_multiple_columns_and_row_in_same_call_replicated_dataframe(self):
      df = ParallelDataFrame(self.dict3, dist = 'replicated')
      new_df = df.drop(columns=['key4','key7'],index=1, inplace = False)
      self.assertEqual(set(list(new_df.globalColumns)), set(['key1', 'key2', 'key3', 'key5', 'key6', 'key8']))
      self.assertEqual(list(new_df.globalIndex), [0, 2])


  #Testing apply function----------------------------------------------------
  #The examples below have been inspired by the examples given in the Pandas documentation
  def test_column_distributed_df_apply_function_sqrt_returns_distributed_df(self):
    df1 = ParallelDataFrame(self.pd_df1, dist_data = False)
    result = df1.apply(np.sqrt)
    df3 = result.apply(np.square)

    self.assertTrue(isinstance(result,ParallelDataFrame))
    self.assertEqual(result.dist, 'distributed')
    self.assertFalse(result.equals(df1))
    self.assertTrue(df1.equals(df3))

  def test_column_distributed_df_apply_function_sum_returns_distributed_series_raw_True(self):
    df1 = ParallelDataFrame(self.pd_df1, dist_data = False)

    pd_result=self.pd_df1.apply(np.sum, axis=0, raw = True)
    result=df1.apply(np.sum, axis=0, raw= True)

    self.assertTrue(isinstance(result, ParallelSeries))
    self.assertEqual(result.dist, 'distributed')
    self.assertEqual(set(list(result.globalIndex)), set(list(pd_result.index)))
    self.assertTrue(result.collect().sort_index().equals(pd_result.sort_index()))


  def test_column_distributed_df_apply_function_sum_returns_distributed_series_raw_False(self):
    df1 = ParallelDataFrame(self.pd_df1, dist_data = False)

    pd_result=self.pd_df1.apply(np.sum, axis=0, raw = False)
    result=df1.apply(np.sum, axis=0, raw= False)

    self.assertTrue(isinstance(result, ParallelSeries))
    self.assertEqual(result.dist, 'distributed')
    self.assertEqual(set(list(result.globalIndex)), set(list(pd_result.index)))
    self.assertTrue(result.collect().sort_index().equals(pd_result.sort_index()))

  def test_replicated_df_apply_function_sqrt_returns_replicated_df(self):
    df1 = ParallelDataFrame(self.pd_df1, dist = 'replicated')

    pd_result = self.pd_df1.apply(np.sqrt)
    result = df1.apply(np.sqrt)

    self.assertTrue(result.equals(pd_result))
    self.assertEqual(result.dist, 'replicated')

  def test_replicated_df_apply_function_sum_axis0_returns_replicated_series(self):
    df1 = ParallelDataFrame(self.pd_df1, dist = 'replicated')

    pd_result=self.pd_df1.apply(np.sum, axis=0)
    result=df1.apply(np.sum, axis=0)

    self.assertTrue(isinstance(result, ParallelSeries))
    self.assertEqual(result.dist, 'replicated')
    self.assertTrue(result.equals(pd_result))

  def test_replicated_df_apply_function_sum_axis1_returns_replicated_series(self):
    df1 = ParallelDataFrame(self.pd_df1, dist = 'replicated')

    pd_result=self.pd_df1.apply(np.sum, axis=1)
    result=df1.apply(np.sum, axis=1)

    self.assertTrue(isinstance(result, ParallelSeries))
    self.assertEqual(result.dist, 'replicated')
    self.assertTrue(result.equals(pd_result))


  def test_replicated_df_apply_function_list_like_result_returns_replicated_series(self):
    df = ParallelDataFrame(self.pd_df1, dist = 'replicated')

    pd_result=self.pd_df1.apply(lambda x: [1, 2], axis=1)
    result=df.apply(lambda x: [1, 2], axis=1)

    self.assertTrue(isinstance(result, ParallelSeries))
    self.assertEqual(result.dist, 'replicated')
    self.assertTrue(result.equals(pd_result))

  if(get_pandas_version()>=0.23):
    def test_replicated_df_apply_function_list_like_result_expand_returns_replicated_df(self):
      df = ParallelDataFrame(self.pd_df1, dist = 'replicated')

      pd_result=self.pd_df1.apply(lambda x: [1, 2], axis=1, result_type='expand')
      result=df.apply(lambda x: [1, 2], axis=1, result_type='expand')

      self.assertTrue(isinstance(result, ParallelDataFrame))
      self.assertEqual(result.dist, 'replicated')
      self.assertTrue(result.equals(pd_result))

  #Testing 'div' function---------------------------------------------
  #The examples below have been inspired by the examples from the Pandas documentation

  def test_div_constant_replicated_df(self):
    df = ParallelDataFrame(self.pd_df1, dist = 'replicated')

    result = df.div(10)
    pd_result = self.pd_df1.div(10)
    self.assertTrue(result.equals(pd_result))

  def test_div_constant_distributed_df(self):
    df1 = ParallelDataFrame(self.pd_df1, dist_data = False)
    pd_df2 = self.pd_df1.div(10)
    df2 = ParallelDataFrame(pd_df2, dist_data = False)

    result = df1.div(10)
    self.assertTrue(result.equals(df2))

  def test_div_by_multiIndex_by_level_replicated_df(self):
    df = ParallelDataFrame(self.pd_df2, dist = 'replicated')
    rep_multindex = ParallelDataFrame(self.df_multindex, dist = 'replicated')

    result = df.div(rep_multindex, level=1, fill_value=0)

    pd_result = self.pd_df2.div(self.df_multindex, level=1, fill_value=0)
    self.assertTrue(result.equals(pd_result))


  #Testing slicing--------------------------------------------------
  def test_slicing_with_single_label_getting_dist_series_from_column_distributed_df(self):
    d = {'col1': [1, 2], 'col2': [3, 4], 'col3':[5, 6]}
    pd_df = pd.DataFrame(data=d)
    dist_df = ParallelDataFrame(data=d, dist_data = False)

    dist_series = dist_df.loc[1]
    pd_series = pd_df.loc[1]

    self.assertTrue(isinstance(dist_series, ParallelSeries))
    self.assertEqual(dist_series.dist, 'distributed')
    self.assertTrue(dist_series.collect().sort_index().equals(pd_series.sort_index()))

  def test_slicing_with_slice_object_getting_dist_df_in_column_distributed_df(self):
    d = {'col1': [1, 2], 'col2': [3, 4], 'col3':[5, 6]}
    pd_df = pd.DataFrame(data=d)
    dist_df = ParallelDataFrame(data=d, dist_data = False)

    dist_slice = dist_df.loc[0:1]
    pd_slice = pd_df.loc[0:1]
    pd_slice_dist = ParallelDataFrame(data=pd_slice, dist_data = False)

    self.assertTrue(isinstance(dist_slice, ParallelDataFrame))
    self.assertEqual(dist_slice.dist, 'distributed')
    self.assertTrue(dist_slice.equals(pd_slice_dist))

  def test_slicing_with_single_label_getting_rep_series_from_replicated_df(self):
    d = {'col1': [1, 2], 'col2': [3, 4], 'col3':[5, 6]}
    pd_df = pd.DataFrame(data=d)
    rep_df = ParallelDataFrame(data=d, dist = 'replicated')

    rep_series = rep_df.loc[1]
    pd_series = pd_df.loc[1]

    self.assertTrue(isinstance(rep_series, ParallelSeries))
    self.assertEqual(rep_series.dist, 'replicated')
    self.assertTrue(rep_series.sort_index().equals(pd_series.sort_index()))

  def test_slicing_with_list_of_labels_getting_rep_df_from_replicated_df(self):
    d = {'col1': [1, 2, 4, 5], 'col2': [3, 4, 6, 7], 'col3':[5, 6, 1, 3]}
    pd_df = pd.DataFrame(data=d)
    rep_df = ParallelDataFrame(data=d, dist = 'replicated')

    rep_slice = rep_df.loc[[0,3]]
    pd_slice = pd_df.loc[[0,3]]

    self.assertTrue(isinstance(rep_slice, ParallelDataFrame))
    self.assertEqual(rep_slice.dist, 'replicated')
    self.assertTrue(rep_slice.sort_index().equals(pd_slice.sort_index()))

  def test_slicing_getting_cell_value_in_replicated_df(self):
    d = {'col1': [1, 2], 'col2': [3, 4], 'col3':[5, 6]}
    pd_df = pd.DataFrame(data=d)
    rep_df = ParallelDataFrame(data=d, dist = 'replicated')

    rep_series = rep_df.loc[1, 'col2']
    pd_series = pd_df.loc[1, 'col2']

    self.assertEqual(rep_series, pd_series)

  def test_slicing_with_boolean_array_getting_rep_df_from_replicated_df(self):
    d = {'col1': [1, 2, 4, 5], 'col2': [3, 4, 6, 7], 'col3':[5, 6, 1, 3]}
    pd_df = pd.DataFrame(data=d)
    rep_df = ParallelDataFrame(data=d, dist = 'replicated')

    rep_series = rep_df.loc[[True, False, False, True]]
    pd_series = pd_df.loc[[True, False, False, True]]

    self.assertTrue(rep_series.sort_index().equals(pd_series.sort_index()))

  #testing corr----------------------------------------------------------------------
  def test_corr_with_col_distributed_dataframe(self):
    pd_df = pd.DataFrame([(.2, .3, .4), (.0, .6, .9), (.6, .0, .6), (.2, .1, .1)],
                      columns=['dogs', 'cats', 'rats'])
    dist_df = ParallelDataFrame(pd_df, dist_data = False)

    dist_corr = dist_df.corr()
    pd_corr = pd_df.corr()

    #compare values of each row (rounded to 6 digits)
    for row in dist_corr.globalIndex:
      self.assertEqual(list(dist_corr.loc[row].collect().sort_index().round(6)),list(pd_corr.loc[row].sort_index().round(6)))

  def test_corr_with_replicated_dataframe(self):
    pd_df = pd.DataFrame([(.2, .3, .4), (.0, .6, .9), (.6, .0, .6), (.2, .1, .1)],
                      columns=['dogs', 'cats', 'rats'])
    rep_df = ParallelDataFrame(pd_df, dist = 'replicated')

    rep_corr = rep_df.corr()
    pd_corr = pd_df.corr()

    self.assertTrue(rep_corr.equals(pd_corr))


if __name__ == "__main__":
  unittest.main()
