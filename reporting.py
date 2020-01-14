"""Implementation based on NEAT-Python reporting.py and Pytorch-Neat neat_reporter.py"""
import json
import time
from pprint import pprint

import numpy as np

class ReporterSet:
  def __init__(self):
    self.reporters = []
    
  def add(self, reporter):
    self.reporters.append(reporter)
  
  def remove(self, reporter):
    self.reporter.remove(reporter)
    
  def start_generation(self, gen):
    for reporter in self.reporters:
      reporter.start_generation(gen)
      
  def end_generation(self, config, population, species_set):
    for reporter in self.reporters:
      reporter.end_generation(config, population, species_set)
  
  def post_evaluate(self, config, population, species, best_genome):
    for reporter in self.reporters:
      reporter.post_evaluate(config, population, species, best_genome)
  
  def post_reproduction(self, config, population, species):
    for reporter in self.reporters:
      reporter.post_reproduction(config, population, species)
  
  def complete_extinction(self):
    for reporter in self.reporters:
      reporter.complete_extinction()
      
  def found_solution(self, config, generation, best):
    for reporter in self.reporters:
      reporter.found_solution(config, generation, best)
      
  def species_stagnant(self, skey, species):
    for reporter in self.reporters:
      reporter.species_stagnant(skey, species)
  
  def info(self, message):
    for reporter in self.reporters:
      reporter.info(message)

     
class BaseReporter:
  def start_generation(self, generation):
    pass
  
  def end_generation(self, config, population, species, best_genome):
    pass
  
  def post_evaluate(self, config, population, species):
    pass
  
  def post_reproduction(self, config, population, species):
    pass
  
  def complete_extinction(self):
    pass
  
  def found_solution(self, config, generation, best):
    pass
  
  def species_stagnant(self, skey, species):
    pass
  
  def info(self, message):
    pass
  
  
class StdOut(BaseReporter):
  def __init__(self, show_species_detail):
    self.show_species_detail = show_species_detail
    self.generation = None
    self.generation_start_time = None
    self.generation_times = []
    self.num_extinctions = 0
    
  def start_generation(self, generation):
    self.generation = generation
    print("\n RUNNING GENERATION {0} \n".format(generation))
    self.generation_start_time = time.time()
    
  def end_generation(self, config, population, species_set):
    ng = len(population)
    ns = len(species_set.species)
    if self.show_species_detail:
      print("Population of {0:d} members in {1:d} species:".format(generation))
      skeys = list(species_set.species.keys())
      skeys.sort()
      print("   ID   age  size  fitness  adj fit  stag")
      print("  ====  ===  ====  =======  =======  ====")
      for skey in skeys:
        s = species_set.species[skey]
        a = self.generation - s.created
        n = len(s.members)
        f = "--" if s.fitness is None else "{:.f}".format(s.fitness)
        af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
        st = self.generation - s.last_improved
        print(" {: >4} {: >3} {: >4} {: >7} {: >7} {: >4}".format(skey, a, n, f, af, st))
    else:
      print("Population of {0:d} members in {1:d} species".format(ng, ns))
      
    elapsed = time.time() - self.generation_start_time
    self.generation_times.append(elapsed)
    self.generation_times = self.generation_times[-10:]
    average = sum(self.generation_times) / len(self.generation_times)
    print("Total extinctions: {0:d}".format(self.num_extinctions))
    if len(self.generation_times) > 1:
      print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
    else:
      print("Generation time: {0:.3f} sec".format(elapsed))
      
  def post_evaluate(self, config, population, species, best_genome):
    fitnesses = [c.fitness for c in population.values()]
    fit_mean = np.mean(fitnesses)
    fit_stdev = np.std(fitnesses)
    best_species_key = species.get_species_key(best_genome.key)
    print("Population's average fitness: {0:3.5f} stdev: {1:3.5f}".format(fit_mean, fit_stdev))
    print("Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}".format(best_genome.fitness, best_genome.size(),
                                                                               best_species_key, best_genome.key))
    
  def complete_extinction(self):
    self.num_extinctions += 1
    print("All species extinct.")
    
  def found_solution(self, config, generation, best):
    print("\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}".format(self.generation,
                                                                                                   best.size()))
    
  def species_stagnant(self, skey, species):
    if self.show_species_detail:
      print("\nSpecies {0} with {1} members is stagnated: removing it".format(skey, len(species.members)))
      
  def info(self, message):
    print(message)

 
class LogReporter(BaseReporter):
  def __init__(self, filename, eval_best, eval_debug=False):
    self.log = open(filename, "a")
    self.generation = None
    self.generation_start_time = None
    self.generation_times = []
    self.num_extinctions = 0
    self.eval_best = eval_best
    self.eval_debug = eval_debug
    self.log_dict = {}
    
  def start_generation(self, generation):
    self.log_dict["generation"] = generation
    self.generation_start_time = time.time()
    
  def end_generation(self, config, population, species_set):
    ng = len(population)
    self.log_dict["pop_size"] = ng
    ns = len(species_set.species)
    self.log_dict["n_species"] = ns
    elapsed = time.time() - self.generation_start_time
    self.log_dict["time_elapsed"] = elapsed
    self.generation_times.append(elapsed)
    self.generation_times = self.generaton_times[-10:]
    average = np.mean(self.generation_times)
    self.log_dict["time_elapsed_avg"] = average
    self.log_dict["n_extinctions"] = self.num_extinctions
    pprint(self.log_dict)
    self.log.write(json.dumps(self.log_dict) + "\n")
    
  def post_evaluate(self, config, population, species, best_genome):
    fitnesses = [c.fitness for c in population.values()]
    fit_mean = np.mean(fitnesses)
    fit_stdev = np.std(fitnesses)
    
    self.log_dict["fitness_avg"] = fit_mean
    self.log_dict["fitness_std"] = fit_std
    self.log_dict["fitness_best"] = best_genome.fitness
    
    print("=" * 50 + " Best Genome: " + "=" * 50)
    if self.eval_debug:
      print(best_genome)
      
    best_fitness_val = self.eval_best(best_genome, config, debug=self.eval_debug)
    self.log_dict["fitness_best_val"] = best_fitness_val
    
    n_neurons_best, n_conns_best = best_genome.size()
    self.log_dict["n_neurons_best"] = n_neurons_best
    self.log_dict["n_conns_best"] = n_conns_best
    
  def complete_extinction(self):
    self.num_extinctions += 1
    
  def found_solution(self, config, generation, best):
    pass
  
  def species_stagnant(self, skey, species):
    pass
