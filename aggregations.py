"""Implementation based on NEAT-Python neat-python/neat/aggregations.py:
https://github.com/CodeReclaimers/neat-python/blob/master/neat/aggregations.py"""

from functools import reduce
from operator import mul

def sum_agg(inputs):
    return sum(inputs)

def mul_agg(inputs):
    return reduce(mul, inputs, 1)

def max_agg(inputs):
    return max(inputs)

def min_agg(inputs):
    return min(inputs)

def mean_agg(inputs):
    return mean(inputs)

def median_agg(inputs):
    return median(inputs)

str_to_aggregation = {
    'sum': sum_agg,
    'mul': mul_agg,
    'max': max_agg,
    'min': min_agg,
    'mean': mean_agg,
    'median': median,
}
