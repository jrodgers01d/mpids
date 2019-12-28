from .Undistributed import Undistributed
from .RowBlock import RowBlock
from .ColumnBlock import ColumnBlock
from .BlockBlock import BlockBlock

__all__ = ['Undistributed', 'RowBlock', 'ColumnBlock', 'BlockBlock']


Distribution_Dict = { 'b' : RowBlock,
                      ('b', '*') : RowBlock,
                      ('*', 'b') : ColumnBlock,
                      ('b', 'b') : BlockBlock,
                      'u' : Undistributed}
