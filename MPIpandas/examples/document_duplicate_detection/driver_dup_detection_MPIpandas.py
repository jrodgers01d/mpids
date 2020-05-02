'''
Details on how to run this are located in the ReadMe file
'''
#!/usr/bin/env python3
import sys
sys.path.append("..") # Adds higher directory to python modules path.

import mpids.MPIpandas.src.ParallelDataFrame as dpd
#import pandas as pd

from mpi_code_using_MPIpandas import *

import argparse
import time
import gc
import os.path


def main():# and from_dict
  parser = argparse.ArgumentParser()
  parser.add_argument('dir', help = "Directory where the data-set is located")
  parser.add_argument('threshold', help = "Threshold for a similarity score", type = float)
  parser.add_argument('stats_file', help = "Name and location of file where the stats should be saved")
  parser.add_argument('results_file', help = "Name and location of file where the results should be saved")
  parser.add_argument('--debug_file', help = "Name and location where the debug info is printed")
  args = parser.parse_args()
  
  t0 = time.time()
  number_top_words = 1000

  directory_dataset = args.dir
  file_stats = args.stats_file
  file_results = args.results_file
  threshold = float(args.threshold)
  file_debug = ""
  if args.debug_file:
    file_debug = args.debug_file
  
  if os.path.isfile(file_stats) != True:
    print_to_file("#Processors\t\t\tTime(seconds)\n",file_stats)
  #===========================================================================================
  #read-in files
  print_to_file("\nReading files.........\n", file_debug)#-------------------------------------------------------------
  dataFrame = read_files(directory_dataset)
  
  # remove non-alphanmeric characters,numbers, stopwords, & do stemming 
  print_to_file("\nCleaning data.........\n", file_debug)#-------------------------------------------------------------
  
  dataFrame.update(dataFrame.apply(lambda x: pre_process(x), axis = "columns"))#.rename('text'))

  print_to_file("\nGetting top words.........\n", file_debug)#-------------------------------------------------------------
  #get top 1000 words in all files, create a dataframe
  top_words = get_top_words(dataFrame, number_top_words)
  top_words.drop(['counts'], axis = 1, inplace = True)
  
  total_words = dataFrame.apply(lambda x: len(x['text'].split()))
  
  print_to_file("\nRemoving everything but top words\n", file_debug)
  
  #remove everything but top words
  dataFrame.update(remove_non_top_words(dataFrame, top_words.index.values).transpose())#.rename('text'))

  
  # columns titles = filenames, rows = words
  dataFrame = dataFrame.loc['text'].str.split(expand=True).transpose()
  
  
  #create inverted index (row labels = words, column labels = filenames)
  print_to_file("\nCreating inverted index........\n", file_debug)#-------------------------------------------------------------
  inverted_index = create_inverted_index(dataFrame, total_words, top_words)
  
  del top_words
  del dataFrame
  gc.collect()
  
  #create similarity matrix
  print_to_file("\nCreating similarity matrix........\n", file_debug)#-------------------------------------------------------------
  similarity_matrix = create_similarity_matrix(inverted_index)

  del inverted_index
  
  #get most similar documents------------------------------------------------------------------------------
  print_to_file("\n#Most similar documents in '{}' using the threshold {} with the corresponding similarity scores are listed below:\n".format(directory_dataset, threshold), file_results)
  #max_sorted = get_similar_documents(similarity_matrix)

  print_similar_documents(similarity_matrix, threshold, file_results)
  
  processors = get_processors()
  print_to_file("{}\t\t\t{}\n".format(processors,(time.time() - t0)), file_stats)
  print_to_file("\nFinished!!!!\n", file_debug)
  

if __name__ == "__main__":
  main()