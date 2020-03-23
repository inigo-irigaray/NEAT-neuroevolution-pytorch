"""Implementation based on Uber's AI Lab Pytorch-NEAT/activations.py repository:
https://github.com/uber-research/PyTorch-NEAT/blob/master/pytorch_neat/activations.py"""

import math

import torch
import torch.nn.functional as F

"""Selected scalars modify functions to have interesting properties around values -1, 0, 1 for x and y"""

def sigmoid_act(x):
  return torch.sigmoid(5.0 * x)
  
def tanh_act(x):
  return torch.tanh(2.5 * x)

def sin_act(x):
  return torch.sin(math.pi * x)
  
def relu_act(x):
  return F.relu(x)

def gaussian_act(x):
  return torch.exp(-5.0 * x**2)

def identity_act(x):
  return x

def absolute_act(x):
  return torch.abs(x)

str_to_activation = {
    'sigmoid': sigmoid_act,
    'tanh': tanh_act,
    'sin': sin_act,
    'relu': relu_act,
    'gaussian': gaussian_act,
    'identity': identity_act,
    'abs': absolute_act,
}
