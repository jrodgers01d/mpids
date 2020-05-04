'''
On the cluster it is run by the job file located at:
  sbatch jobs/job_microbenchmarks
On local system, it is run by the following:
  mpiexec -n 1 python3 ./tests/microbenchmarks.py --stats_file ./microbenchmarks_stats.results
'''
'''
Details on how to run this are located in the ReadMe file
'''
#!/usr/bin/env python3
import time
from mpi4py import MPI
import argparse
import os.path


import sys
sys.path.append(".") # Adds higher directory to python modules path.
sys.path.append("..") # Adds higher directory to python modules path.

import pandas as pd
from  mpids.MPIpandas.src.ParallelDataFrame import ParallelDataFrame

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--stats_file', help = "Name and location of file where the stats should be saved")
  args = parser.parse_args()
  if args.stats_file:
    file_stats = args.stats_file
  else:
    file_stats = "./microbenchmark_stats.txt"
  
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  number_processors = comm.Get_size()
  
  data_folder1 = './MPIpandas/examples/microbenchmarks/test_data1_microbenchmarks'
  data_folder2 = './MPIpandas/examples/microbenchmarks/test_data2_microbenchmarks'
  
  #if file does not exist add header
  if rank == 0 and os.path.isfile(file_stats) != True:
    with open(file_stats, 'a') as file:
      file.write("#Processors, Function, Time(ms)\n")
      
  #---------------------------------------------------------------------------------------
  #value_counts function------------------------------------------------------------------
  #---------------------------------------------------------------------------------------
  pd_data = pd.read_csv(data_folder1+"/adult_vc_transpose_40K.data", low_memory = False)
  #pd_data = pd.read_csv(data_folder2+"/adult_vc_transpose_80K.data", low_memory = False)
  pd_data.index = ['workclass', 'education', 'educationNum']
  dist_data = ParallelDataFrame(pd_data, dist_data = False)
  
  dist_data = dist_data.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
  dist_data = dist_data.apply(lambda x: x.str.replace('-','') if x.dtype == 'object' else x)
  dist_data = dist_data.apply(lambda x: x.str.replace(' ','') if x.dtype == 'object' else x)
  dist_series = dist_data.loc['education']
  comm.Barrier() 
  t0 = time.time()
  dist_series.value_counts()
  comm.barrier()
  if(rank==0):
    with open(file_stats, 'a') as file:
      file.write('{}, value_counts, {}\n'.format(number_processors,(time.time() - t0)*1000))
  
  #---------------------------------------------------------------------------------------
  #apply function------------------------------------------------------------------
  #---------------------------------------------------------------------------------------
  pd_data = pd.read_csv(data_folder1+"/adult_80K.data")
  #pd_data = pd.read_csv(data_folder2+"/adult_163K.data")
  pd_data.columns = ['age','workclass','fnlwgt','education','educationNum','maritalStatus','occupation','relationship','race','sex','capitalGain','capitalLoss','hoursPerWeek','nativeCountry','payPerYear']
  dist_data = ParallelDataFrame(pd_data, dist_data = False)
  
  comm.Barrier() 
  t0 = time.time()
  new_dist_data = dist_data.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
  comm.barrier()
  if(rank==0):
    with open(file_stats, 'a') as file:
      file.write('{}, apply, {}\n'.format(number_processors,(time.time() - t0)*1000))
  
  #---------------------------------------------------------------------------------------
  #from_dict function------------------------------------------------------------------
  #---------------------------------------------------------------------------------------
  pd_data = pd.read_csv(data_folder1+"/adult_80K.data")
  #pd_data = pd.read_csv(data_folder2+"/adult_163K.data")
  pd_data.columns = ['age','workclass','fnlwgt','education','educationNum','maritalStatus','occupation','relationship','race','sex','capitalGain','capitalLoss','hoursPerWeek','nativeCountry','payPerYear']
  dict = pd_data.to_dict()
  
  comm.Barrier() 
  t0 = time.time()
  dist_data = ParallelDataFrame.from_dict(dict, orient='index')
  comm.barrier()
  if(rank==0):
    with open(file_stats, 'a') as file:
      file.write('{}, from_dict, {}\n'.format(number_processors,(time.time() - t0)*1000))
  
  #---------------------------------------------------------------------------------------
  #corr function------------------------------------------------------------------
  #---------------------------------------------------------------------------------------
  pd_data = pd.read_csv(data_folder1+"/adult_discretized_82K.data")
  #pd_data = pd.read_csv(data_folder2+"/adult_discretized_164K.data")
  dist_data = ParallelDataFrame(pd_data)
  
  comm.Barrier() 
  t0 = time.time()
  dist_data.corr()
  comm.barrier()
  if(rank==0):
    with open(file_stats, 'a') as file:
      file.write('{}, corr, {}\n'.format(number_processors,(time.time() - t0)*1000))
  
  
if __name__ == "__main__":
  main()