import torch
import numpy as np
from .graphs import feed_forward
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
      
