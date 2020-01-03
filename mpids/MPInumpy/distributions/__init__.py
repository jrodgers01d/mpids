from .Undistributed import Undistributed
from .RowBlock import RowBlock

__all__ = ['Undistributed', 'RowBlock']


Distribution_Dict = {'b' : RowBlock,
                     'u' : Undistributed}
