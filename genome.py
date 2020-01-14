"""Implementation based on NEAT-Python genome.py"""

import sys
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
  def __init__(self, key):
    self.key = key
    self.connections = {}
    self.nodes = {}
    self.fitness = None
    
  @classmethod
  def parse_config(cls, param_dict):
    param_dict['node_gene_type'] = genes.DefaultNodeGene
    param_dict['connection_gene_type'] = genes.DefaultConnectionGene
    return DefaultGenomeConfig(param_dict)
  
  @classmethod
  def write_config(cls, f, config):
    config.save(f)
    
  def configure_new(self, config):
    for node_key in config.output_keys:
      self.nodes[node_key] = self.create_node(config, node_key)
      
    if config.n_hid > 0:
      for i in range(config.n_hid):
        node_key = config.get_new_node_key(self.nodes)
        assert node_key not in self.nodes
        node = sefl.create_node(config, node_key)
        self.nodes[node_key] = node
        
    if "fs_neat" in config.initial_connection:
      if config.initial_connection == "fs_neat_nohidden":
        self.connect_fs_neat_nohidden(config)
      elif config.initial_connection == "fs_neat_hidden":
        self.connect_fs_neat_hidden(config)
      else:
        if config.n_hid > 0:
          print("Warning: initial_connection = fs_neat will not connect to hidden nodes;",
                "\tif this is desired, set initial_connection = fs_nohidden;",
                "\tif if not, set initial_connection = fs_neat_hidden", sep="\n", file=sys.stderr)
          
        self.connect_partial_nodirect(config)
        
  def configure_crossover(self, genome1, genome2, config):
    assert isinstance(genome1.fitness, (int, float))
    assert isinstance(genome2.fitness, (int, float))
    if genome1.fitness > genome2.fitness:
      parent1, parent2 = genome1, genome2
    else:
      parent1, parent2 = genome2, genome1
      
    for key, cg1 in parent1.connections.items():
      cg2 = parent2.connections.get(key) #'.get(key)' instead of '[key]' cause it returns None if '[key]' doesn't exist
      if cg2 is None: # ---> excess or disjoint gene, copy from fittest parent
        self.connections[key] = cg1.copy()
      else: #homologous gene, combine parents' genes
        self.connections[key] = cg1.crossover(cg2)
        
    parent1_set = parent1.nodes
    parent2_set = parent2.nodes
    
    for key, ng1 in parent1_set.items():
      ng2 = parent2_set.get(key)
      assert key not in self.nodes
      if ng2 is None: #extra gene, copy from fittest parent
        self.nodes[key] = ng1.copy()
      else: #homologous gene, combine parents' genes
        self.nodes[key] = ng1.crossover(ng2)
        
  def mutate(self, config):
    if config.single_structural_mutation:
      div = max(1, (config.node_add_prob + config.node_delete_prob + config.conn_add_prob + config.conn_delete_prob))
      r = random.random()
      if r < (config.node_add_prob / div):
        self.mutate_add_node(config)
      elif r < ((config.node_add_prob + config.node_delete_prob) / div):
        self.mutate_delete_node(config)
      elif r < ((config.node_add_prob + config.node_delete_prob + config.conn_add_prob) / div):
        self.mutate_add_connection(config)
      elif r < 1:#sum of probs/div can yield a result >1, but random.random() always produces [0.0,1.0), so same but simpler
        self.mutate_delete_connection(config)
    else:
      if random.random() < config.node_add_prob:
        self.mutate_add_node(config)
      if random.random() < config.node_delete_prob:
        self.mutate_delete_node(config)
      if random.random() < config.conn_add_prob:
        self.mutate_add_connection(config)
      if random.random() < config.conn_delete_prob:
        self.mutate_delete_connection(config)
    
    for cg in self.connections.values():#mutate connection genes
      cg.mutate(config)
      
    for ng in self.nodes.values():#mutate node genes
      ng.mutate(config)
      
      
