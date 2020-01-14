"""Implementation based on NEAT-Python reporting.py"""

import time

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
      
  def species_stagnant(self, sid, species):
    for reporter in self.reporters:
      reporter.species_stagnant(sid, species)
  
  def info(self, message):
    for reporter in self.reporters:
      reporter.info(message)
