import warnings
import random
from .attributes import FloatAttribute, BoolAttribute, StringAttribute

class BaseGene:
  def __init__(self, key):
    self.key = key
    
  def __str__(self):
    attrib = ["key"] + [a.name for a in self._gene_attributes]
    attrib = ["{0}={1}".format(a, getattr(self, a)) for a in attrib]
    return "{0}({1})".format(self.__class__.__name__, ", ".join(attrib))
  
  def __lt__(self, other):
    assert isinstance(self.key, type(other.key)), "Can't compare keys {0!r} and {1!r}".format(self.key, other.key)
    return self.key < other.key
  
  
