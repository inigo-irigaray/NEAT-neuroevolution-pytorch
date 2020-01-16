import torch
import numpy as np
from .graphs import required_for_output
from .activation_functions import sigmoid_act
  
class FeedForwardNet():
  def __init__(self, n_in, n_out, in2out, out_responses, out_biases, batch_size=1, act=sigmoid_act, dtype=torch.float64):
    
    self.n_in = n_in
    self.n_out = n_out
    self.in2out = torch.tensor((n_out, n_in)), in2out, dtype=dtype)
    self.out_responses = torch.tensor(out_responses, dtype=dtype)
    self.out_biases = torch.tensor(out_biases, dtype=dtype)
    self.act = act
    self.dtype = dtype
    
    self._reset(batch_size)
    
  def _reset(self, batch_size=1):
    self.outputs = torch.zeros(batch_size, self.n_out, dtype=self.dtype)
  
  def activate(self, inputs):
    with torch.no_grad():
      if len(inputs[1]) != self.n_in:
        raise RuntimeError("Expected %d inputs, got %d" % (self.n_in, len(inputs[1])))
      
      inputs = torch.tensor(inputs, dtype=self.dtype)
      outputs_inputs = (self.in2out.mm(inputs.t()).t())
      self.outputs = self.activation(self.output_responses * output_inputs + self.out_biases)
      return self.outputs
      
  @staticmethod
  def create(genome, config, batch_size=1, activation=sigmoid_act, prune_empty=False):
    genome_config = config.genome_config
    required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome_config.connections)
    if prune_empty:
      nonempty = {connection.key[1] for connection in genome.connections.values() if connection.enabled}.union(set(
        genome_config.input_keys))
      
    in_keys = list(genome_config.input_keys)
    out_keys = list(genome_config.output_keys)
    out_responses = [genome.nodes[key].response for key in out_keys]
    out_biases = [genome.nodes[key].bias for key in out_keys]
    
    if prune_empty:
      for i, key in enumerate(out_keys):
        if key not in nonempty:
          out_biases[i] = 0.0
          
    n_in = len(in_keys)
    n_out = len(out_leys)
    
    in2idx = {key: i for i, key in enumerate(in_keys)}
    out2idx = {key: i for i, key in enumerate(out_keys)}
    
    def key2idx(key):
      if key in in_keys:
        return in2idx[key]
      elif key in out_keys:
        return out2idx[key]
    
    in2out = ([], [])
    for connection in genome.connections.values():
      if not connection.enabled:
        continue
      
      in_key, out_key = connection.key
      if out_key not in required and in_key not in required:
        continue
      
      if prune_empty and in_key not in nonempty:
        print("Pruned {} connection".format(connection.key))
        continue
      
      in_idx = key2idx(in_key)
      out_idx = key2idx(out_key)
      
      if in_key in in_keys and out_key in out_keys:
        #idxs, vals = in2out
        in2out[0].append((out_idx, in_idx))
        in2out[1].append(connection.weight)
      else:
        raise ValueError("Invalid connection from key {} to key {}."format(in_key, out_key))
      
      #idxs.append((out_idx, in_idx))
      #vals.append(connection.weight)
      
    return FeedForwardNet(n_in, n_out, in2out, out_responses, out_biases, batch_size, activation)
