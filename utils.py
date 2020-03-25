from math import sqrt, exp
import numpy as np
import torch




def make_dense(shape, connections, dtype=torch.float64):
    matrix = torch.zeros(shape, dtype=dtype)
    idxs, weights = connections
    if len(idxs)==0:
        return matrix
    rows, cols = np.array(idxs).T
    rows = torch.tensor(rows)
    cols = torch.tensor(cols)
    matrix[rows, cols] = torch.tensor(weights, dtype=dtype)
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

def mean(values):
    values = list(values)
    return sum(map(float, values)) / len(values)


def median(values):
    values = list(values)
    values.sort()
    return values[len(values) // 2]


def median2(values):
    values = list(values)
    n = len(values)
    if n <= 2:
        return mean(values)
    values.sort()
    if (n % 2) == 1:
        return values[n//2]
    i = n//2
    return (values[i - 1] + values[i])/2.0


def variance(values):
    values = list(values)
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def std(values):
    return sqrt(variance(values))


def softmax(values):
    """
    Compute the softmax of the given value set, v_i = exp(v_i) / s,
    where s = sum(exp(v_0), exp(v_1), ..)."""
    e_values = list(map(exp, values))
    s = sum(e_values)
    inv_s = 1.0 / s
    return [ev * inv_s for ev in e_values]

stat_functions = {'min': min, 'max': max, 'mean': mean, 'median': median,
                  'median2': median2}
