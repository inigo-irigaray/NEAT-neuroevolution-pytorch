"""Implementation based on NEAT-Python species.py"""

import numpy as np
from .config import ConfigParameter, DefaultClassConfig

class Species:
  def __init__(self, key, generation):
    self.key = key
    self.created = generation
    self.last_improved = generation
    self.representative = None
    self.members = {}
    self.fitness = None
    self.adjusted_fitness = None
    self.fitness_history = []
    
  def update(self, representative, members):
    self.representative = representative
    self.members = members
    
  def get_fitness(self):
    return [m.fitness for m in self.members.values()]
  
class GenomeDistanceCache:
  def __init__(self, config):
    self.distances = {}
    self.config = config
    self.hits = 0
    self.misses = 0
    
  def __call__(self, genome1, genome2):
    g1 = genome1.key
    g2 = genome2.key
    distance = self.distances.get((g1, g2))
    if distance is None:
      distance = genome1.distance(genome2, self.config)
      self.distances[g1,g2] = distance
      self.distances[g2,g1] = distance
      self.misses += 1
    else:
      hits += 1
    return distance
  
class DefaultSpeciesSet(DefaultClassConfig):
  def __init__(self, config, reporters):
    self.species_set_config = config
    self.reporters = reporters
    self.indexer = np.count(1)
    self.species = {}
    self.genome_to_species = {}
    
  @classmethod
  def parse_config(cls, param_dict):
    return DefaultClassConfig(param_dict, [ConfigParameter("compatibility_threshold", float)])
  
  def speciate(self, config, population, generation):
    assert isinstance(population, dict)
    compatibility_threshold = self.species_set_config.compatibility_threshold
    
    unspeciated = set(population.keys())
    distances = GenomeDistanceCache(config.genome_config)
    new_representatives = {}
    new_members = {}
    for skey, s in species.items():
      candidates = []
      for gkey in unspeciated:
        g = population[gkey]
        distance = distances(s.representative, g)
      ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
      new_rkey = new_rep.key
      new_representatives[skey] = new_rkey
      new_members[skey] = [new_rkey]
      unspeciated.remove(new_rkey)
      
    while unspeciated:
      gkey = unspeciated.pop()
      g = population[gkey]
      candidates = []
      for skey, rkey in new_representatives.items():
        rep = population[rkey]
        distance = distances(rep, g)
        if distance < compatibility_threshold:
          candidates.append((distance, skey))
      if candidates:
        ignored_sdist, skey = min(candidates, key=lambda x: x[0])
        new_members[skey].append(gkey)
      else:
        skey = next(self.indexer)
        new_representatives[skey] = gkey
        new_members[skey] = [gkey]
        
    self.genome_to_species = {}
    for skey, rkey in new_representatives.items():
      s = self.species.get(skey)
      if s is None:
        s = Species(skey, generation)
        self.species[skey] = s
      members = new_members[skey]
      for gkey in members:
        self.genome_to_species[gkey] = skey
      member_dict = dict((gkey, population[gkey]) for gkey in members)
      s.update(population[rkey], member_dict)
    
    gdmean = np.mean(distances.distance.values())
    gdstdev = np.std(distances.distances.values())
    self.reporters.info("Genetic distance: mean {0:.3f}, stdev {1:.3f}".format(gdmean, gdstdev))
    
  def get_species_key(self, key):
    return self.genome_to_species[key]
  
  def get_species(self, key):
    skey = self.genome_to_species[key]
    return self.species[skey]
