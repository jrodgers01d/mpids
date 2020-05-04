from mpi4py import MPI
import string
import sys
import re
import os
import numpy as np
from collections import Counter, OrderedDict, defaultdict
from itertools import groupby
import pandas as pd
import gc


__lastrank = -1
__comm     = MPI.COMM_NULL
__startletters =[]

def __init_bib ():
  global __startletters

  for digit in range (ord('0'), ord('9') + 1 ):
    __startletters.append(chr(digit))
  for first in range(ord('a'), ord('z') + 1):
    __startletters.append(chr(first))
    for second in range(ord('a'), ord('z') + 1):
      __startletters.append(chr(first) + chr(second))
  
  return
  
def __get_index(dict_len, procs, rank):
  """ return start and end index for each process"""
  num = int(dict_len/procs)
  rem = int(dict_len%procs)
  if rank < rem:
    my_len = num+1
    start_idx=rank*my_len
  else:
    my_len=num
    start_idx=rem*(num+1) + (rank-rem)*num

  end_idx = start_idx+my_len
  return start_idx, end_idx

def __groupfunction( item):
  global __comm, __lastrank 
  size = __comm.Get_size()
          
  for i in range ( __lastrank, size ):
    s, e = __get_index(float(712), size, i)#26^2+26+10
    for j in range ( s, e):
      itemstring = str(item[0])
      if itemstring.startswith(tuple('([{')):
       itemstring = itemstring[2:]

      if  len(itemstring) == 1:
       starting = itemstring
      elif itemstring.startswith(tuple('0123456789')):
       starting = itemstring[0]
      else:
        starting = itemstring[:2]
      #print(starting)
      if starting == __startletters[j]:
        __lastrank=i
        return i


def Counter_all (tokens, comm=MPI.COMM_WORLD, tokens_per_iter=10000, tracing=False):

  #define tuple datatype
  np_word_tuple_dtype = [('word', '|S20'), ('count', np.int32)]
  
  test_array = np.zeros(2, dtype = np_word_tuple_dtype)
  displacements = [test_array.dtype.fields[field][1] for field in ['word', 'count']]

  mpi_word_tuple_dtype = MPI.Datatype.Create_struct([20,1],displacements,[MPI.CHAR, MPI.INTEGER4])
  mpi_word_tuple_dtype.Commit()
  
  
  __init_bib()
  global __comm, __lastrank
  __comm = comm
  size = comm.Get_size()
  rank = comm.Get_rank()

  length = len(tokens)
  nm = np.asarray(length)
  nm_max = np.asarray (int(0))
  comm.Allreduce (nm, nm_max, op=MPI.MAX)

  num_iterations = int(nm_max / (tokens_per_iter))###
  if ( nm_max % tokens_per_iter ):
    num_iterations = num_iterations + 1

  #if tracing and rank == 0: 
  #  print("Processing documents in ", num_iterations," iterations")

  lastindex = 0  
  final_wcount={}

  for iter  in range (0, num_iterations):  
    #if tracing:# and rank == 0:
    #  print('rank'+str(rank)+"  Processing iteration", iter)

    firstindex = lastindex
    if firstindex < length:
      lastindex = (iter+1) * tokens_per_iter
      if lastindex > length:
        lastindex = length

    partial_tokens = tokens[firstindex:lastindex]
    word_count = Counter(partial_tokens)
    
    od = OrderedDict(sorted(word_count.items(), key=lambda t: t[0]))
    __lastrank = 0
   
    odd = sorted(od.items(), key=__groupfunction)
   
    
    wordcount_per_reducer={}
    __lastrank=0
    for ranks, words in groupby(odd, lambda s: __groupfunction(s)):
      wordcount_per_reducer[ranks]=list(words)
    
    od.clear()
    odd.clear()
    partial_tokens.clear()
    
    
    for step in range(0, size):
      sendto = ( rank + step ) % size
      recvfrom = ( rank + size - step) % size
      reqs=[]
      
      df = pd.DataFrame()
      recv_dict = {}
      size_s = np.array(0)# if nothing to send, send a zero so that the other processor knows to expect nothing
      size_r = np.array(0)#
      #first step is send to self:
      if(step == 0 and sendto in wordcount_per_reducer and len(wordcount_per_reducer[sendto]) != 0):
        for key,value in wordcount_per_reducer[sendto]:
            if key not in final_wcount:
              key = str(key)
              final_wcount[key] = value 
            else:
              final_wcount[key] += value
              
      #make sure that we have something to send
      if(sendto in wordcount_per_reducer and len(wordcount_per_reducer[sendto]) != 0):
        to_send = np.array(list((wordcount_per_reducer[sendto])), dtype = np_word_tuple_dtype)
      
        size_s = np.array(to_send.size)
      
        reqs.append(comm.Isend ([to_send, int(size_s), mpi_word_tuple_dtype], dest=sendto, tag=478))
      reqs.append(comm.Isend([size_s, 1, MPI.INTEGER8], dest = sendto, tag = 22))
    
      #receive--------------------------
      if(step != 0):
        size_r = np.zeros(1, dtype = np.int64)
        comm.Recv([size_r, 1, MPI.INTEGER8], source = recvfrom, tag = 22)
        if(int(size_r) != 0):
          recv_data = np.zeros(int(size_r), dtype = np_word_tuple_dtype)
          comm.Recv([recv_data, int(size_r), mpi_word_tuple_dtype], source = recvfrom, tag =478)
          
          recv_dict = {k.decode():v for k,v in recv_data}
          
          del recv_data
     
      if((step == 0 and int(size_s) != 0) or (int(size_r) != 0)):    
       
        for key,value in recv_dict.items():
          if key not in final_wcount:
            final_wcount[key] = value 
          else:
            final_wcount[key] += value
        recv_dict.clear()
       
      del df
   
      if(step != 0 and int(size_s) != 0):
        MPI.Request.Waitall(reqs)
        del to_send
        
  return final_wcount
