"""Implementation based on NEAT-Python neat-python/neat/genes.py:
https://github.com/CodeReclaimers/neat-python/blob/master/neat/genes.py"""

import warnings
import random

from neuroevolution.attributes import FloatAttribute, BoolAttribute, StringAttribute




class BaseGene:
    def __init__(self, key):
        self.key = key

    def __str__(self):
        """
        Overloads 'str()' to describe gene attributes intstead of the object's location.
        """
        attrib = ["key"] + [a.name for a in self._gene_attributes]
        attrib = ["{0}={1}".format(a, getattr(self, a)) for a in attrib]
        return "{0}({1})".format(self.__class__.__name__, ", ".join(attrib))

    def __lt__(self, other):
        """
        Overloads the '<' operator to compare genes' keys and include 'Can't compare keys' warning.
        """
        assert isinstance(self.key, type(other.key)), "Can't compare keys {0!r} and {1!r}".format(self.key, other.key)
        return self.key < other.key

    @classmethod
    def parse_config(cls, config, param_dict):
        pass

    @classmethod
    def get_config_params(cls):
        params = []
        if not hasattr(cls, "_gene_attributes"):
            setattr(cls, "_gene_attributes", getattr(cls, "__gene_attributes__"))
            warnings.warn(
                "Class '{!s}' {!r} needs '_gene_attributes' not '__gene_attributes__'".format(cls.__name__, cls),
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
        Core implementation of evolution in NEAT where genes' values are mutated according to configuration parameters.
        """
        for a in self._gene_attributes:
            value = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(value, config))

    def copy(self):
        """
        Creates copy of the BaseGene object.
        """
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            setattr(new_gene, a.name, getattr(self, a.name))
        return new_gene

    def crossover(self, gene2):
        """
        Core implementation of evolution in NEAT where homologous genes' (descendants from same parent-have same key)
        attributes are randomly mixed to create a new gene.
        """
        assert self.key == gene2.key
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            if random.random() > 0.5:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(gene2, a.name))
        return new_gene




class DefaultNodeGene(BaseGene):
    _gene_attributes = [FloatAttribute("bias"),
                        FloatAttribute("response"),
                        StringAttribute("activation", options="sigmoid"),
                        StringAttribute("aggregation", options="sum")]

    def __init__(self, key):
        assert isinstance(key, int), "DefaultNodeGene key must be an int, not a {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        """
        Computes how different node genes are for the compatibility distance measure.
        """
        distance = abs(self.bias - other.bias) + abs(self.response - other.response)
        if self.activation != other.activation:
            distance += 1
        if self.aggregation != other.aggregation:
            distance += 1
        return distance * config.compatibility_weight_coefficient




class DefaultConnectionGene(BaseGene):
    _gene_attributes = [FloatAttribute("weight"), BoolAttribute("enabled")]

    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not a {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        """
        Computes how different connection genes are for the compatibility distance measure.
        """
        distance = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            distance += 1
        return distance * config.compatibility_weight_coefficient
