"""Implementation based on NEAT-Python graphs.py"""

def required_for_outputs(inputs, outputs, connections):
  required = set(outputs)
  s = set(outputs)
  while True:
    t = set(a for (a,b) in connections if b in s and a not in s)
    if not t:
      break
      
    layer_nodes = set(x for x in t if x not in inputs)
    if not layer_nodes:
      break
      
    required = required.union(layer_nodes)
    s = s.union(t)
  
  return required

def creates_cycle(connections, test):
  inp, out = test
  if inp == out:
    return True
  
  visited = {out}
  while True:
    n_added = 0
    for a, b in connections:
      if a in visited and b not in visited:
        if b == inp:
          return True
        visited.add(b)
        n_added += 1
    if n_added == 0:
      return False
    
"""def feed_forward(inputs, outputs, connections):
  required = required_for_outputs(inputs, outputs, connections)
  layers = []
  s = set(inputs)
  while True:
    c = set(b for (a,b) in connections if a in s and b not in s)
    t = set()
    for n in c:
      if n is required and all(a in s for (a,b) in connections if b==n):
        t.add(n)
        
    if not t:
      break
      
    layers.append(t)
    s = s.union(t)
    
  return layers"""
