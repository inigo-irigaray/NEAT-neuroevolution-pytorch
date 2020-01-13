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
      
    requires = required.union(layer_nodes)
    s = s.union(t)
  
  return required
  
def feed_forward(inputs, outputs, connections):
  
