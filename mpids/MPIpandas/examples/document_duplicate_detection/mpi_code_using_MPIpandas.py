import time
from mpi4py import MPI
import sys
import re
import os

import pandas as pd
import numpy as np


import mpids.MPIpandas.src.ParallelDataFrame as dpd
from  mpids.MPIpandas.src.ParallelSeries import ParallelSeries

from sys import argv
import gc
from itertools import islice

import serial_code_using_pandas as serial
from mpids.utils.PandasUtils import get_pandas_version
  
def sort_df(df, col_name):
  pandas_version = get_pandas_version()
  if(pandas_version >= 0.17):
    return df.sort_values(by=col_name, ascending = False)
  else:
    return df.sort(col_name, ascending = False)# for pandas 0.16
 
def get_cutoff(rank,all_counts, number_top_words):
  temp = np.sort(all_counts, axis = 0)[::-1]
  return temp[number_top_words-1]

def pre_process(x):
 
  return serial.pre_process(x)

def read_files(path, comm = MPI.COMM_WORLD):
  file_name_and_text = {}
  for filename in os.listdir(path):
    with open(path+filename, "rb") as myfile:
      file_name_and_text[filename] = [str(myfile.read())]
  #dataFrame = dpd.ParallelDataFrame(file_name_and_text, dist_data = False)
  dataFrame = dpd.ParallelDataFrame.from_dict(file_name_and_text, orient = 'columns')
  if(dataFrame is None or dataFrame.empty):
    print("Reduce number of processors!")
    exit_program()
  else:
    dataFrame.index = ['text']

  return dataFrame

def get_top_words(dataFrame, number_top_words):
  top_words_series = ParallelSeries(' '.join(dataFrame.loc['text']).split(), dist="distributed").value_counts()[:number_top_words]
  top_words = dpd.ParallelDataFrame(dist_data=top_words_series.values, index = top_words_series.index.values, columns = ['counts'], dist = 'replicated')
  top_words.index.name = "words"

  return top_words
  
def remove_non_top_words(dataFrame, top_words):
  return dpd.ParallelDataFrame(dataFrame.loc['text'].apply(lambda x: ' '.join([word for word in x.split() if word in top_words])))

def create_inverted_index(dataFrame, total_words, top_words):
  inverted_index = dpd.ParallelDataFrame()
  pandas_version = get_pandas_version()
  
  i=0
  for onefile in dataFrame.columns:
    
    if(dataFrame[onefile].isnull().all()):
      word_weights_per_file = dpd.ParallelDataFrame(np.nan, index = ['##'], columns = [onefile])
    else:
      word_weights_per_file = dpd.ParallelDataFrame(dataFrame[onefile].value_counts())
     
    word_weights_per_file.index.name = "words"
   
    if(total_words[onefile] != 0):
      word_weights_per_file = word_weights_per_file.div(total_words[onefile])
    #else:
    #  word_weights_per_file = dpd.DistributedDataFrame(np.nan, index = ['##'], columns = [onefile])
    
    #if (inverted_index.empty):
    #  inverted_index = word_weights_per_file
    
    #else:
    if(pandas_version >= 0.23):
      inverted_index = pd.concat([inverted_index, word_weights_per_file], axis = 1, sort = False)
    else:
      inverted_index = pd.concat([inverted_index, word_weights_per_file], axis = 1)#, sort = False)
    i += 1
    
    del word_weights_per_file
  
  inverted_index.columns =  dataFrame.columns
  
  if '##' in inverted_index.index:
    inverted_index.drop('##', axis = 0, inplace = True)
  
  inverted_index.fillna(np.float32(0), inplace = True)
 
  pandas_version = get_pandas_version()
  
  #makes sure to include all top words in inverted index
  if(pandas_version >= 0.23):
    inverted_index = pd.concat([top_words, inverted_index], axis = 1, sort = False)
  else:
    inverted_index = pd.concat([top_words, inverted_index], axis = 1)
    
  inverted_index.fillna(np.float32(0), inplace = True)
  
  return inverted_index

def create_similarity_matrix(inverted_index, comm = MPI.COMM_WORLD):
  similarity_matrix = inverted_index.corr('pearson')
  
  #removing self comparison for files
  for i in similarity_matrix.columns.values:
    similarity_matrix[i].loc[i] = np.nan
 
  similarity_matrix.dropna(axis = 1, how = 'all', inplace = True)
  similarity_matrix.dropna(axis = 0, how = 'all', inplace = True)
  
  return similarity_matrix
 
def  print_similar_documents(similarity_matrix, threshold, file_results, comm = MPI.COMM_WORLD):
  
  similarity_matrix = similarity_matrix[similarity_matrix>=threshold]
  similarity_matrix.dropna(inplace=True, axis = 0, how = 'all')
  
  rank = comm.Get_rank()
  number_processors = comm.Get_size()
  pair_number = 1
  if rank ==0: 
    printer_rank = number_processors-1
  else:
    printer_rank = None
  for i in range(number_processors):
    printer_rank = comm.bcast(printer_rank, root = 0)
    if(printer_rank == rank):
      pair_number = serial.print_pairs(similarity_matrix, file_results, pair_number)
    pair_number = comm.bcast(pair_number, root = printer_rank)
    if(rank == 0):
      printer_rank -= 1 
  
  return
  
def print_to_file(x, file_path, comm = MPI.COMM_WORLD):
  rank = comm.Get_rank()
  if(rank == 0 and file_path != ''):
    file = open(file_path, 'a')
    file.write(x)
    file.close()
  return
  
def get_processors(comm = MPI.COMM_WORLD):
  return comm.Get_size()

def exit_program(comm = MPI.COMM_WORLD):
  sys.stdout.flush()
  comm.Abort()