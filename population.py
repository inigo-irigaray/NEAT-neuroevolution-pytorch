"""Implementation based on NEAT-Python neat-python/neat/population.py:
https://github.com/CodeReclaimers/neat-python/blob/master/neat/population.py"""

import numpy.mean as mean

import .reporting




class CompleteExtinctionException(Exception):
    pass

class Population:
    def __init__(self, config, initial_state=None):
        self.reporters = 
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_type, self.reporters, stagnation)
    
        if config.fitness_criterion == "max":
            self.fitness_criterion = max
        elif config.fitness_criterion == "min":
            self.fitness_criterion = min
        elif config.fitness_criterion == "mean":
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError("Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))
      
        if initial_state is None:
            self.population = self.reproduction.create_new(config.genome_type, config.genome_config, config.pop_size,)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generaton = initial_state
      
        self.best_genome = None
    
    def add_reporter(self, reporter):
        self.reporters.add(reporter)
    
    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)
    
    def run(self, fitness_function, n=None):
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination.")
  
        k = 0
        while n is None or k < n:
            k += 1
            self.reporters.start_generation(self.generation)
            fitness_function(list(self.population.items()), self.config)
            best = None
            for g in self.population.values():
                if g.fitness is None:
                    raise RuntimeError("Fitness not assigned to genome {}".format(g.key))
                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)
      
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best
      
            if not self.config.no_fitness_termination:
                fv = self.fitness_criterion(g.fitness for g in self.population.values())
            if fv >= self.config.fitness_threshold:
                self.reporters.found_solution(self.config, self.generation, best)
                break
          
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)
            if not self.species.species:
                self.reporters.complete_extinction()
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type, self.config.genome_config, 
                                                                   self.config.genome.pop_size)
                else:
                    raise CompleteExtinctionException()
      
            self.species.speciate(self.config, self.population, self.generation)
            self.reporters.end_generation(self.config, self.population, self.species)
            self.generation += 1
      
        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)
        return self.best_genome
