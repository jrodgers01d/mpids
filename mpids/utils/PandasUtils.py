import pandas as pd

def get_pandas_version():
  pd_version = pd.__version__.split('.')
  pd_version_final = pd_version[0]+'.'+pd_version[1]
  return float(pd_version_final)  