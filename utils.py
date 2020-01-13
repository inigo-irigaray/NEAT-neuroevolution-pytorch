import torch
import numpy as np

def make_dense(shape, connections, dtype=torch.float64):
  matrix = torch.zeros(shape, dtype=dtype)
  idxs, weights = connections
  if len(idxs)==0:
    return matrix
  rows, cols = np.array(idxs).T
  rows = torch.tensor(rows)
  cols=torch.tensor(cols)
  matrix[rows, cols] = torch.tensor(wieghts, dtype=dtype)
  return matrix

def make_sparse(shape, connections):
  if len(idxs)==0:
    matrix = torch.sparse.FloatTensor(shape[0], shape[1])
    return matrix
  idxs, weights = connections
  idxs = torch.LongTensor(idxs).t()
  weights = torch.FloatTensor(weights)
  matrix = torch.sparse.FloatTensor(idxs, weights, shape)
  return matrix
