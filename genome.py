"""Implementation based on NEAT-Python genome.py"""

import random
from itertools import count

import .activations
import .aggregations
import .genes
import .graphs
from .config import ConfigParameter, write_pretty_params

class DefaultGenomeConfig:
  allowed_connectivity = ["unconnnected", "fs_neat_nohidden", "fs_neat", "fs_neat_hidden", "full_nodirect", "full",
                          "full_direct", "partial_nodirect", "partial", "partial_direct"]
  
  def __init__(self, params):
    self.activation_defs = activations.str_to_activation
    self.aggregation_defs = aggregations.str_to_aggregation
    self._params = [ConfigParameter("n_in", int), ConfigParameter("n_out", int), ConfigParameter("n_hid", int),
                    ConfigParameter("feed_forward", bool), ConfigParameter("compatibility_disjoint_coefficient", float),
                    ConfigParameter("compatibility_weight_coefficient", float), ConfigParameter("conn_add_prob", float),
                    ConfigParameter("conn_delete_prob", float), ConfigParameter("node_add_prob", float),
                    ConfigParameter("node_delete_prob", float), ConfigParameter("single_structural_mutation", bool, "false"),
                    ConfigParameter("single_structural_surer", str, "default"), ConfigParameter("initial_connection", str,
                                                                                                "unconnected")]
    
    self.node_gene_type = params["node_gene_type"]
    self._params += self.node_gene_type.get_config_params()
    self.connection_gene_type = params["connection_gene_type"]
    self._params += self.connection_gene_type.get_config_params()
    
    for p in self._params:
      setattr(self, p.name, p.interpret(params))
      
    self.input_keys = [-i - 1 for i in range(self.n_in)]
    self.output_keys = [i for i in range(self.n_out)]
    
    self.connection_fraction = None
    
    if "partial" in self.initial_connection:
      c, p = self.initial_connection.split()
      self.initial_connection = c
      self.connection_fraction = float(p)
      if not (0 <= self.connection_fraction <= 1):
        raise RuntimeError("'partial' connection must between [0.0, 1.0].")
    
    assert self.initial_connection in self.allowed_connectivity
      
    if self.structural_mutation_surer.lower() in ["true", "1", "yes", "on"]:
      self.structural_mutation_surer = "true"
    elif self.structural_mutation_surer.lower() in ["false", "0", "no", "off"]:
      self.structural_mutation_surer = "false"
    elif self.structural_mutation_surer.lower() == "default":
      self.structural_mutation_surer = "default"
    else:
      raise RuntimeError("Invalid structural_mutation_surer {!r}".format(self.structural_mutation_surer))
    
    self.node_idexer = None
    
  def add_activation(self, name, func):
    self.activation_defs[name] = func
    
  def add_aggregation(self, name, func):
    self.aggregation_defs[name] = func
    
  def save(self, f):
    if 'partial' in self.initial_connection:
      if not (0 <= self.connection_fraction <= 1):
        raise RuntimeError("'partial' connection value must be between [0.0, 1.0]")
      f.write("initial_connection = {0} {1} \n".format(self.initial_connection, self.connection_fraction))
    else:
      f.write("initial_connection = {0} {1} \n".format(self.initial_connection))
    assert self.initial_connection in self.allowed_connectivity
    write_pretty_params(f, self, [p for p in self._params if not "initial_connection" in p.name])
    
  def get_new_node_key(self, node_dict):
    if self.node_indexer is None:
      self.node_indexer = count(max(list(node_dict.keys())) + 1)
    new_id = next(self.node_indexer)
    assert new_id not in node_dict
    return new_id
  
  def check_structural_mutation_surer(self):
    if self.structural_mutation_surer == "true":
      return True
    elif self.structural_mutation_surer == "false":
      return False
    elif self.strcutural_mutation_surer == "default":
      return self.single_structural_surer
    else:
      raise RuntimeError("Invalid structural_mutation_surer {!r}".format(self.strcutural_mutation_surer))
      
class DefaultGenome:
  
