import torch
import numpy as np
import utils
from .graphs import feed_forward
from .activation_functions import sigmoid_act
  
class FeedForwardNet():
  def __init__(self, n_in, n_hid, n_out, in2hid, hid2hid, out2hid, in2out, hid2out, out2out, hid_responses, 
               out_responses, hid_biases, out_biases, batch_size=1, use_current_acts=False, act=sigmoid_act, 
               n_internal_steps=1, dtype=torch.float64):
    
    self.n_in = n_in
    self.n_hid = n_hid
    self.n_out = n_out
    
    if n_hidden > 0:
      self.in2hid = utils.make_dense((n_hid, n_in), in2hid, dtype=dtype)
      self.hid2hid = utils.make_dense((n_hid, n_hid), hid2hid, dtype=dtype)
      self.out2hid = utils.make_dense((n_hid, n_out), out2hid, dtype=dtype)
      self.hid2out = utils.make_dense((n_out, n_hid), hid2out, dtype=dtype)
      self.hid_responses = torch.tensor(hid_responses, dtype=dtype)
      self.hid_biases = torch.tensor(hid_biases, dtype=dtype)
      
    self.in2out = torch.tensor((n_out, n_in)), in2out, dtype=dtype)
    self.out2out = torch.tensor((n_out, n_out), out2out, dtype=dtype)
    self.out_responses = torch.tensor(out_responses, dtype=dtype)
    self.out_biases = torch.tensor(out_biases, dtype=dtype)
    self.use_current_acts = use_current_acts
    self.act = act
    self.n_internal_steps = n_internal_steps
    self.dtype = dtype
    
    self.reset(batch_size)
    
  
    
    
    
