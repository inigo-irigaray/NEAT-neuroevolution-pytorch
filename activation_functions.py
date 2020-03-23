import torch
import torch.nn.functional as F

def sigmoid_act(x):
  return torch.sigmoid(x)
  
def tanh_act(x):
  return torch.tanh(x)

def sin_act(x):
  return torch.sin(x)
  
def relu_act(x):
  return F.relu(x)

def gaussian_act(x):
  return torch.exp(- x**2)

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
    'identity': identity_activation,
    'abs': absolute_act,
}
