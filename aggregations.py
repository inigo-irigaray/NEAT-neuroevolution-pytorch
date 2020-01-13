from functools import reduce
from operator import mul

def sum_agg(inputs):
  '''returns the sum of all input elements'''
  return sum(inputs)

def mul_agg(inputs):
  '''returns the product of all input elements in the iterable'''
  return reduce(mul, inputs, 1)

str_to_aggregation = {
    'sum': sum_agg,
    'prod': prod_agg,
}
