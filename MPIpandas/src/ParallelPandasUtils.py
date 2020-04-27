#from pandas.core.indexing import IndexingMixin
from pandas.core.indexing import _LocIndexer

#from pandas.core.strings import StringMethods
#import numpy as np

class _CustomLocIndexer(_LocIndexer):
  def _getitem_axis(self, key, axis: int):
    from_super = super()._getitem_axis(key, axis)
    from .ParallelSeries import ParallelSeries
    from .ParallelDataFrame import ParallelDataFrame
    import pandas as pd
    if(isinstance(from_super, pd.Series)):
      if(self.obj.dist == 'distributed'):
        return ParallelSeries(data=from_super, dist = 'distributed', dist_data=True)
      else:
        return ParallelSeries(data=from_super, dist = 'replicated')
    elif(isinstance(from_super, pd.DataFrame)):
      if(self.obj.dist == 'distributed'):
        return ParallelDataFrame(data=from_super, dist = 'distributed', dist_data=True)
      else:
        return ParallelDataFrame(data=from_super, dist = 'replicated')
    else:
      return from_super
  
# class CustomIndexingMixin(IndexingMixin):
  # @property
  # def loc(self):
    # return _CustomLocIndexer("loc", self)
# class CustomStringMethods(StringMethods):
  # def _wrap_result(
      # self,
      # result,
      # use_codes = True,
      # name = None,
      # expand=None,
      # fill_value=np.nan,
      # returns_string=True,
  # ):
    # # return_value = None
    # return_value = super()._wrap_result(result,use_codes,name,expand,fill_value,return_strings)
    # print('boo')
    # # if(isinstance(self._orig, DistributedSerier):
      # # #cannot do anything
    # # if(return_value is not None):
      # # return return_value