import sys
sys.path.append("..") # Adds higher directory to python modules path.
#sys.path.append("..") # Adds higher directory to python modules path.

import time
from mpi4py import MPI
import sys
import re
import os
import numpy as np
import pandas as pd
from sys import argv
import gc
from itertools import islice

import mpids.utils.ParallelIO as parallelIO
import mpids.MPIcollections.src.MPICounter as MPICounter
import serial_code_using_pandas as serial
from mpids.utils.PandasUtils import get_pandas_version
  
def sort_df(df, col_name):
  pandas_version = get_pandas_version()
  if(pandas_version >= 0.17):
    return df.sort_values(by=col_name, ascending = False)
  else:
    return df.sort(col_name, ascending = False)
 
def get_cutoff(rank,all_counts, number_top_words):
  temp = np.sort(all_counts, axis = 0)[::-1]
  if number_top_words > all_counts.size:
    return temp[-1]
  else:
    return temp[number_top_words-1]

def pre_process(x):
  return serial.pre_process(x)

def read_files(path_input):
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  
  file_name_and_text = parallelIO.read_all (path_input, return_type = 'dict')
  pandas_version = get_pandas_version()
  #create df 
  if (pandas_version >= 0.23):
    dataFrame = pd.DataFrame.from_dict(file_name_and_text, orient = 'index', columns = ['text'])#.rename(columns ={0:'text'})#\
  else:
    dataFrame = pd.DataFrame.from_dict(file_name_and_text, orient = 'index').rename(columns ={0:'text'})#\
  dataFrame.index.name = "filename"
  #dataFrame.index = ['text']
  
  if(dataFrame is None or dataFrame.empty):
    print("ERROR: Either the directory does not exist!\nOR no documents in the directory!\nOR reduce the number of processors!")
    exit_program()
  
  return dataFrame


def get_top_words(dataFrame, number_top_words, comm = MPI.COMM_WORLD):
  rank = comm.Get_rank()
  number_processors = comm.Get_size()
  
  #get all words in each processor
  words_per_processor = serial.get_all_words(dataFrame)
  #create a word count dictionary distributed among all the processes
  temp_words_count = MPICounter.Counter_all(words_per_processor, tracing = False, tokens_per_iter = 1000000)
  #sort by counts in descending order
  words_count = {k:v for k,v in sorted(temp_words_count.items(), key = lambda item: item[1], reverse = True)}
  
  ###changed
  #only keep number_top_words and not more
  #words_count = {k:v for k,v in islice(words_count.items(),0,number_top_words-1, None)}
  
  temp_words_count.clear()
  
  #find top words------------------
  #every processor sends their top number_top_words counts to everyone to find cuttoff
  number_words = np.array(len(words_count))#(words_count.shape[0])
  number_max = np.array(0)
  comm.Allreduce ([number_words,1,MPI.LONG], number_max, op=MPI.MAX)
  
  
  all_counts = np.full((number_max*number_processors), dtype = np.int32, fill_value = 0)
  
  counts_to_send = np.full((number_max), dtype = np.int32, fill_value = 0)
  if(number_max > number_words):
    counts_to_send[0:number_words] = np.array(list(words_count.values()))
  else:
    counts_to_send[0:number_max] = np.array(list(words_count.values()))
  
  comm.Allgather([counts_to_send, number_max, MPI.INTEGER4],
                 [all_counts, number_max, MPI.INTEGER4])#
  
  
  cutoff = get_cutoff(rank,all_counts, number_top_words)
 
  if cutoff==0: cutoff = 1
  
  #all_counts = [x for x in all_counts if x >= cutoff]
  all_counts = all_counts[all_counts >= cutoff]
  del counts_to_send
  
  #take out all the words less than cutoff, if any
  local_top_words_count = {k:v for k,v in words_count.items() if v >= cutoff}
  
  words_count.clear()#del words_count
  
  #define a datatype for sending words
  STRING20 = MPI.Datatype.Create_contiguous(MPI.CHAR, 20) 
  STRING20.Commit()
  
  number_words = np.array(len(local_top_words_count))
  number_max = np.array(0)
  comm.Allreduce ([number_words,1,MPI.LONG], number_max, op=MPI.MAX)
  
  #send and gather all top words
  top_words_numpy = np.full((number_max*number_processors), dtype = '|S20', fill_value = np.nan)
  words_to_send = np.full((number_max), dtype = '|S20', fill_value = '###')
  
  words_to_send[:int(number_words)] = list(local_top_words_count.keys())
  
  comm.Allgather([words_to_send, number_max, STRING20],
                 [top_words_numpy, number_max, STRING20])

  #get rid of all the fill values
  top_words_list = [x.decode() for x in top_words_numpy if x.decode() != '###']
  
  top_words = pd.DataFrame(data = all_counts, index = top_words_list, columns = ['counts'])
  
  del top_words_numpy

  if(len(top_words_list) > number_top_words):
    top_words = sort_df(top_words, 'counts')
    top_words = top_words.head(number_top_words)
    
  return top_words
  
def remove_non_top_words(dataFrame, top_words):
  return pd.DataFrame(dataFrame['text'].apply(lambda x: ' '.join([word for word in x.split() if word in top_words])))
  #return serial.remove_non_top_words(dataFrame, top_words)
 
def create_inverted_index(dataFrame, total_words, top_words):
  
  #create inverted index (row labels = words, column labels = filenames)
  inverted_index = serial.create_inverted_index(dataFrame, total_words, top_words)
  
  pandas_version = get_pandas_version()
  #makes sure to include all top words in inverted index
  if(pandas_version >= 0.23):
    inverted_index = pd.concat([top_words, inverted_index], axis = 1, sort = False)
  else:
    inverted_index = pd.concat([top_words, inverted_index], axis = 1)
  
  inverted_index.fillna(np.float32(0), inplace = True)
  
  return inverted_index

def create_similarity_matrix(inverted_index):
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  number_processors = comm.Get_size()
  pandas_version = get_pandas_version()
  
  similarity_matrix = serial.create_similarity_matrix(inverted_index)
  to_send = np.array(inverted_index.values, order='C', dtype=np.float32)#sending contiguous array
  to_send_shape = np.array([inverted_index.shape[0],inverted_index.shape[1]])
 
  req = []
  #each rank will 1st send to one up and below, then 2 up and below and so on alternatively...until the last one
  #creates a distributed corr dataframe where data is divided by columns of dataframe
  for step in range (1, number_processors): 
    #send
    destination1 = rank + step
    destination2 = rank - step
    
    #send to one below
    req = _send_for_simi_matrix(to_send, inverted_index.columns, to_send_shape, destination1, number_processors, req, inverted_index.shape, rank, comm)
    #send to one up
    req = _send_for_simi_matrix(to_send, inverted_index.columns, to_send_shape, destination2, number_processors, req, inverted_index.shape, rank, comm)

    #receive
    source1 = rank - step
    source2 = rank + step
    
    method = 'pearson'
    min_periods = 1
    #block receive and process from one above
    similarity_matrix = _recv_and_process_for_simi_matrix(source1, number_processors, rank, inverted_index, method, min_periods, similarity_matrix, comm)
    #block receive and process from one below
    similarity_matrix = _recv_and_process_for_simi_matrix(source2, number_processors, rank, inverted_index, method, min_periods, similarity_matrix, comm)
   
    #wait for all the sends to complete
    if(len(req) != 0):
      MPI.Request.Waitall(req)
      req = []  
  del to_send
  del to_send_shape
  
  #output = output.transpose() # to have column distribution
  #sort so that all the rows are arragend similarly in various processors

  #similarity_matrix.sort_index(inplace = True)
  similarity_matrix = similarity_matrix.transpose()
  
  #removing self comparison for files
  for i in similarity_matrix.columns.values:
    similarity_matrix[i].loc[i] = np.nan
 
  similarity_matrix.dropna(axis = 1, how = 'all', inplace = True)
  similarity_matrix.dropna(axis = 0, how = 'all', inplace = True)
  
  return similarity_matrix
  
def _send_for_simi_matrix(to_send, columns, to_send_shape, destination, number_processors, req, shape, rank, comm):
  if destination >= 0 and destination < number_processors:
    tag = rank + destination
    req.append(comm.Isend([to_send_shape, 2, MPI.LONG], dest = destination, tag = tag + 100))
    req.append(comm.isend(columns, dest = destination, tag = tag + 10))
    req.append(comm.Isend([to_send, shape[0]*shape[1], MPI.FLOAT], dest = destination, tag = tag))
  return req
  
def _recv_and_process_for_simi_matrix(source, number_processors, rank, df, method, min_periods, output, comm):
  pandas_version = get_pandas_version()
  if source >= 0 and source < number_processors:
    tag = source + rank
      
    recv_shape = np.zeros(2, dtype = np.int)
    comm.Recv([recv_shape, 2, MPI.LONG], source = source, tag = tag + 100)
    
    data_labels = [] #file names
    data_labels = comm.recv(source = source, tag = tag + 10)
    
    recv_data = np.zeros([recv_shape[0], recv_shape[1]], dtype = np.float32)
    comm.Recv([recv_data, recv_shape[0]*recv_shape[1],MPI.FLOAT], source = source, tag = tag)
    
   #create a temp df and find corelation with new files and then concat with similarity matrix
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
    
def get_similar_documents(similarity_matrix):
  return serial.get_similar_documents(similarity_matrix)
 
def  print_similar_documents(similarity_matrix, threshold, file_results, comm= MPI.COMM_WORLD):
  
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
  
def print_to_file(x, file_path):
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  if(rank == 0 and file_path != ''):
    file = open(file_path, 'a')
    file.write(x)
    file.close()
  return
  
def get_processors():
  comm = MPI.COMM_WORLD
  return comm.Get_size()

def exit_program():
  sys.stdout.flush()
  comm = MPI.COMM_WORLD
  comm.Abort()