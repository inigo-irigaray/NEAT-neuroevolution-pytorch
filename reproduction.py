"""Implementation based on NEAT-Python reproduction.py"""

import random
import numpy as np
from math import ceil
from itertools import count
from .config import ConfigParameter, DefaultClassConfig

class DefaultReproduction(DefaultClassConfig):
  def __init__(self, config, reporters, stagnation):
    self.reproduction_config = config
    self.reporters = reporters
    self.stagnation = stagnation
    self.genome_indexer = count(1)
    self.ancestors = {}
    
  @classmethod
  def parse_config(cls, params):
    return DefaultClassConfig(params, [ConfigParameter("elitism", int, 0), ConfigParameter("survival_threshold", float, 0.2),
                                       ConfigParameter("min_species_size", int, 2)])
  
  @staticmethod
  def compute_spawn(adj_fitness, prev_sizes, pop_size, min_size):
    adj_fit_sum = sum(adj_fitness)
    spawn_amounts = []
    for adj_fit, prev_size in zip(adj_fitness, prev_sizes):
      if adj_fit_sum > 0:
        size = max(min_size, adj_fit / adj_fit_sum * pop_size)
      else:
        size = min_size
      
      diff = (size - prev_size) * 0.5
      species_extra = int(round(diff))
      spawn = prev_size
      if abs(species_extra) > 0:
        spawn += species_extra
      elif diff > 0:
        spawn += 1
      elif diff < 0:
        spawn -= 1
      spawn_amounts.append(spawn)
    
    total_spawn = sum(spawn_amounts)
    norm_coeff = pop_size / total_spawn
    spawn_amounts = [max(min_species_size, int(round(spawn * norm_coeff)) for spawn in spawn_amounts)]
    return spawn_amounts
  
  def create_new(self, genome_type, genome_config, n_genomes):
    new_genomes = {}
    for i in range(n_genomes):
      key = next(self.genome_indexer)
      genome = genome_type(key)
      genome.configure_new(genome_config)
      new_genomes[key] = genome
      self.ancestors[key] = tuple()
    return new_genomes
  
  def reproduce(self, config, species, pop_size, generation):
    all_fitnesses = []
    remaining_species = []
    for stag_key, stag_species, stagnant in self.stagnation.update(species, generation): #removes stagnant species
      if stagnant:
        self.reporters.species_stagnant(stag_key, stag_species)
      else:
        all_fitnesses.extend(member.fitness for member in stag_species.members.values())
        remaining_species.append(stag_species)
    
    if not remaining_species: #all species were stagnant & therefore eliminated
      species.species = {}
      return {}
    
    min_fitness = min(all_fitnesses)
    max_fitness = max(all_fitnesses)
    fitness_range = max(1.0, max_fitness - min_fitness)
    for spec in remaining_species:
      mean_species_fit = np.mean([member.fitness for member in spec.members.values()])
      adj_fit = (mean_species_fit - min_fitness) / fitness_range
      spec.adjusted_fitness = adj_fit
    
    adj_fitnesses = [spec.adjusted_fitness for spec in remaining_species]
    mean_adj_fit = np.mean(adj_fitnesses)
    self.reporters.info("Average adjusted fitness: {:.3f}".format(mean_adj_fit))
    
    prev_sizes = [len(spec.members for spec in remaining_species)]
    min_species_size = max(self.reproduction_config.min_species_size, self.reproduction_config.elitism)
    spawn_amounts = self.compute_spawn(adj_fitnesses, prev_sizes, pop_size, min_species_size)
    
    new_population = {}
    species.species = {}
    for spawn, spec in zip(spawn_amounts, remaining_species):
      spawn = max(spawn, self.reproduction_config.elitism)
      assert spawn > 0
      old_members = list(spec.members.items())
      old_members.sort(reverse=True, key=lambda x: x[1].fitness)
      spec.members = {}
      species.species[spec.key] = spec
      if self.reproduction_config.elitism > 0:
        for key, member in old_members[:self.reproduction_configuration.elitism]:
          new_population[key] = member
          spawn -= 1
      if spawn <= 0:
        continue
        
      cutoff = int(ceil(self.reproduction_config.survival_threshold * len(old_members)))
      cutoff = max(cutoff, 2)
      old_members = old_members[:cutoff]
      while spawn > 0:
        spawn -= 1
        parent1_key, parent1 = random.choice(old_members)
        parent2_key, parent2 = random.choice(old_members)
        gkey = next(self.genome_indexer)
        child = config.genome_type(gkey)
        child.configure_crossover(parent1, parent2, config.genome_config)
        child.mutate(config.genome_config)
        new_population[gkey] = child
        self.ancestors[gkey] = (parent1_key, parent2_key)
        
    return new_population 
