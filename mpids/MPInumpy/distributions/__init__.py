from .Replicated import Replicated
from .Block import Block

__all__ = ['Replicated', 'Block']


Distribution_Dict = {'b' : Block,
                     'r' : Replicated}
