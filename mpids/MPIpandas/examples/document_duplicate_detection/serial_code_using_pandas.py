import re
import os
import pandas as pd
import numpy as np
import gc


import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from mpids.utils.PandasUtils import get_pandas_version

DEBUG = 0


def pre_process(series):
  for index in series.index:
    text = series[index]
    #remove all the symbols or numbers
    text = re.sub(r'</*\w*/*>|&\w+;|\W|[0-9]|_',' ',text.lower()) 
  
  #stemming
    ps = PorterStemmer()
    text = ' '.join([ps.stem(y) for y in word_tokenize(text)])
  
  #remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
  
  #remove single letters
    text = re.sub(r'\b\w\b',' ',text)
    series[index] = text
  
  return series

def read_files(path):
  file_name_and_text = {}
  for filename in os.listdir(path):
    with open(path+filename, "rb") as myfile:
      file_name_and_text[filename] = [str(myfile.read())]
     
  pandas_version = get_pandas_version()
  if pandas_version >= 0.23:
    dataFrame = pd.DataFrame.from_dict(file_name_and_text, orient = 'columns')
  else:
    dataFrame = pd.DataFrame.from_dict(file_name_and_text, orient = 'columns')
  dataFrame.index = ['text']

  
  return dataFrame     

# get words and freq of the top words
def get_top_words(dataFrame, number_top_words):
  top_words_series = pd.Series(' '.join(dataFrame.loc['text']).split()).value_counts()[:number_top_words]
  top_words = pd.DataFrame(top_words_series, columns = ['counts'])
  top_words.index.name = "words"
  return top_words

def get_word_count(dataFrame):
  return pd.DataFrame(pd.Series(' '.join(dataFrame['text']).split()).value_counts(), columns = ['freq'])
  
def get_all_words(dataFrame):
  return (' '.join(dataFrame['text']).split())
  
def clean_top_words(top_words, top_words_to_remove):
  top_words = top_words.tail(top_words.size - top_words_to_remove)
  return top_words
  
def remove_non_top_words(dataFrame, top_words):
  return pd.DataFrame(dataFrame.loc['text'].apply(lambda x: ' '.join([word for word in x.split() if word in top_words])))

def create_inverted_index(dataFrame, total_words, top_words):
  inverted_index = pd.DataFrame()
  pandas_version = get_pandas_version()
  
  i=0
  for onefile in dataFrame.columns:
    if(dataFrame[onefile].isnull().all()):
      word_weights_per_file = pd.DataFrame(np.nan, index = ['##'], columns = [i])
    else:
      word_weights_per_file = pd.DataFrame(dataFrame[onefile].value_counts())
    
    word_weights_per_file.index.name = "words"
    
    if(total_words[onefile] != 0):
      word_weights_per_file = word_weights_per_file.div(total_words[onefile])
    #else:
    #  word_weights_per_file = pd.DataFrame(np.nan, index = ['##'], columns = [i])
     
    #if (inverted_index.empty):
    #  inverted_index = word_weights_per_file  
    #else:
    if(pandas_version >= 0.23):
      inverted_index = pd.concat([inverted_index, word_weights_per_file], axis = 1, sort = False)
    else:
      inverted_index = pd.concat([inverted_index, word_weights_per_file], axis = 1)
    i += 1
    del word_weights_per_file
  
  inverted_index.columns =  dataFrame.columns
 
  if '##' in inverted_index.index:
    inverted_index.drop('##', axis = 0, inplace = True)
  
  inverted_index.fillna(np.float32(0), inplace = True)
 
  return inverted_index

def create_similarity_matrix(inverted_index):
  
  similarity_matrix = inverted_index.corr(method = 'pearson')
 
  for i in similarity_matrix.columns.values:
    similarity_matrix[i].loc[i] = np.nan
  
  similarity_matrix.dropna(axis = 1, how = 'all', inplace = True)
  similarity_matrix.dropna(axis = 0, how = 'all', inplace = True)

  return similarity_matrix

def get_similar_documents(similarity_matrix):
  
  pandas_version = get_pandas_version()
  if(pandas_version >= 0.17):
    max_sorted = pd.DataFrame(similarity_matrix.max(skipna = True)).rename(columns = {0:"score"}).sort_values(by = "score", ascending = False, inplace = False)#sort_values for 0.17 and higher pandas, sot_index otherwise
  else:
    max_sorted = pd.DataFrame(similarity_matrix.max(skipna = True)).rename(columns = {0:"score"}).sort_index(by = "score", ascending = False, inplace = False)#sort_index otherwise
  max_sorted.index.name = "filename"
  
  max_sorted.fillna(inplace = True, value = -10)  
  
  return max_sorted
 
# prints similar documents based on threshold
def print_similar_documents(similarity_matrix, threshold, file_path, pair_number = 1):
  
  similarity_matrix = similarity_matrix[similarity_matrix>=threshold]
  similarity_matrix.dropna(inplace=True, axis = 0, how = 'all')
  
  print_pairs(similarity_matrix, file_path, pair_number)
  
  return

def print_pairs(similarity_matrix, file_path, pair_number):
  pandas_version = get_pandas_version()
  for col_label,row in similarity_matrix.items():
    row.dropna(inplace=True)
    for row_label,content in row.items():
      to_print = "{} {} {}\t\t\t\t {} \n".format(pair_number,col_label, row_label, content)
      print_to_file(to_print, file_path)
      pair_number += 1
  return pair_number
  
  
def print_to_file(x, file_path):
  if file_path != '':
    file = open(file_path, 'a')
    file.write(x)
    file.close()
  return
  
def get_processors():
  return 1


 

 

  
