import warnings
import random
from .attributes import FloatAttribute, BoolAttribute, StringAttribute

class BaseGene:
  def __init__(self, key):
    self.key = key
    
  def __str__(self):
    """
    Overloads 'str()' to internally (within the BaseGene class) or explicitly (by calling the 'str()' method) 
    create strings describing gene attributes and printing these strings when calling 'print(BaseGene_object)' 
    instead of printing the object's location.
    """
    attrib = ["key"] + [a.name for a in self._gene_attributes]
    attrib = ["{0}={1}".format(a, getattr(self, a)) for a in attrib]
    return "{0}({1})".format(self.__class__.__name__, ", ".join(attrib))
  
  def __lt__(self, other):
    """
    Overloads the '<' comparison operator to compare genes' keys and include 'Can't compare keys' warning.
    """
    assert isinstance(self.key, type(other.key)), "Can't compare keys {0!r} and {1!r}".format(self.key, other.key)
    return self.key < other.key
  
  @classmethod
  def parse_config(cls, config, param_dict):
    pass
  
  @classmethod
  def get_config_params(cls):
    """
    Gets and returns the configuration parameters from each BaseGene attributes.
    """
    params = []
    if not hasattr(cls, "_gene_attributes"):
      setattr(cls, "_gene_attributes", getattr(cls, "__gene_attributes__"))
      warnings.warn("Class '{!s}' {!r} needs '_gene_attributes' not '__gene_attributes__'".format(cls.name, cls),
                   DeprecationWarning)
    for a in cls._gene_attributes:
      params += a.get_config_params()
    return params
  
  def init_attributes(self, config):
    """
    Sets gene attributes to initial configuration parameters.
    """
    for a in self._gene_attributes:
      setattr(self, a.name, a.init_value(config))
      
  def mutate(self, config):
    """
    Mutates genes' values according to configuration parameters.
    """
    for a in self._gene_attributes:
      value = geattr(self, a.name)
      setattr(self, a.name, a.mutate_value(v, config))
      
  def copy(self):
    """
    Creates copy of the BaseGene object.
    """
    new_gene = self.__clas__(self.key)
    for a in self._gene_attributes:
      setattr(new_gene, a.name, getattr(self, a.name))
    return new_gene
  
  def cross_over(self, gene2):
    """
    """
    
