'''
Details on how to run this are located in the ReadMe file
'''

#!/usr/bin/env python3
import time
import argparse
import os.path

import sys
sys.path.append(".") # Adds higher directory to python modules path.
sys.path.append("..") # Adds higher directory to python modules path.

import pandas as pd

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--stats_file', help = "Name and location of file where the stats should be saved")
  args = parser.parse_args()
  if args.stats_file:
    file_stats = args.stats_file
  else:
    file_stats = "./microbenchmark_stats_serial.txt"
  
  data_folder1 = './MPIpandas/examples/microbenchmarks/test_data1_microbenchmarks'
  data_folder2 = './MPIpandas/examples/microbenchmarks/test_data2_microbenchmarks'
  
  #if file does not exist add header
  if os.path.isfile(file_stats) != True:
    with open(file_stats, 'a') as file:
      file.write("Function, Time(ms)\n")
      
  #---------------------------------------------------------------------------------------
  #value_counts function------------------------------------------------------------------
  #---------------------------------------------------------------------------------------
  pd_data = pd.read_csv(data_folder1+"/adult_vc_transpose_40K.data", low_memory = False)
  #pd_data = pd.read_csv(data_folder2+"/adult_vc_transpose_80K.data", low_memory = False)
  pd_data.index = ['workclass', 'education', 'educationNum']
  
  pd_data = pd_data.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
  pd_data = pd_data.apply(lambda x: x.str.replace('-','') if x.dtype == 'object' else x)
  pd_data = pd_data.apply(lambda x: x.str.replace(' ','') if x.dtype == 'object' else x)
  pd_series = pd_data.loc['education']

  t0 = time.time()
  pd_series.value_counts()
  with open(file_stats, 'a') as file:
    file.write('value_counts, {}\n'.format((time.time() - t0)*1000))
  
  #---------------------------------------------------------------------------------------
  #apply function------------------------------------------------------------------
  #---------------------------------------------------------------------------------------
  pd_data = pd.read_csv(data_folder1+"/adult_80K.data")
  #pd_data = pd.read_csv(data_folder2+"/adult_163K.data")
  pd_data.columns = ['age','workclass','fnlwgt','education','educationNum','maritalStatus','occupation','relationship','race','sex','capitalGain','capitalLoss','hoursPerWeek','nativeCountry','payPerYear']
  
  t0 = time.time()
  new_pd_data = pd_data.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
  with open(file_stats, 'a') as file:
    file.write('apply, {}\n'.format((time.time() - t0)*1000))
  
  #---------------------------------------------------------------------------------------
  #from_dict function------------------------------------------------------------------
  #---------------------------------------------------------------------------------------
  pd_data = pd.read_csv(data_folder1+"/adult_80K.data")
  #pd_data = pd.read_csv(data_folder2+"/adult_163K.data")
  pd_data.columns = ['age','workclass','fnlwgt','education','educationNum','maritalStatus','occupation','relationship','race','sex','capitalGain','capitalLoss','hoursPerWeek','nativeCountry','payPerYear']
  dict = pd_data.to_dict()
   
  t0 = time.time()
  pd_dataframe = pd.DataFrame.from_dict(dict, orient='index')
  with open(file_stats, 'a') as file:
    file.write('from_dict, {}\n'.format((time.time() - t0)*1000))
  
  #---------------------------------------------------------------------------------------
  #corr function------------------------------------------------------------------
  #---------------------------------------------------------------------------------------
  pd_data = pd.read_csv(data_folder1+"/adult_discretized_82K.data")
  #pd_data = pd.read_csv(data_folder2+"/adult_discretized_164K.data")
  
  t0 = time.time()
  pd_data.corr()
  with open(file_stats, 'a') as file:
    file.write('corr, {}\n'.format((time.time() - t0)*1000))
  
if __name__ == "__main__":
  main()