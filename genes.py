import warnings
import random
from .attributes import FloatAttribute, BoolAttribute, StringAttribute

class BaseGene:
  def __init__(self, key):
    self.key = key
    
  def __str__(self):
    """
    Overloads 'str()' to internally (within the BaseGene class) or explicitly (by calling the 'str()' method) 
    create strings describing gene attributes and printing these strings when calling 'print(BaseGene_object)' instead
    of printing the object's location.
    """
    attrib = ["key"] + [a.name for a in self._gene_attributes]
    attrib = ["{0}={1}".format(a, getattr(self, a)) for a in attrib]
    return "{0}({1})".format(self.__class__.__name__, ", ".join(attrib))
  
  def __lt__(self, other):
    """
    Overloads the '<' comparison operator to include 'Can't compare' warning.
    """
    assert isinstance(self.key, type(other.key)), "Can't compare keys {0!r} and {1!r}".format(self.key, other.key)
    return self.key < other.key
  
  @classmethod
  def parse_config(cls, config, param_dict):
    pass
  
  
