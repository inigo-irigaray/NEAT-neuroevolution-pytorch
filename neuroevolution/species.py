"""Implementation based on NEAT-Python neat-python/neat/species.py:
https://github.com/CodeReclaimers/neat-python/blob/master/neat/species.py"""

from itertools import count

from neuroevolution.config import ConfigParameter, DefaultClassConfig
from neuroevolution.utils import mean, std




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

    def get_fitnesses(self):
        return [member.fitness for member in self.members.values()]




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
            self.hits += 1
        return distance




class DefaultSpeciesSet(DefaultClassConfig):
    def __init__(self, config, reporters):
        self.species_set_config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species = {}
        self.genome_to_species = {}

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict, [ConfigParameter("compatibility_threshold", float)])

    def speciate(self, config, population, generation):
        assert isinstance(population, dict)
        compatibility_threshold = self.species_set_config.compatibility_threshold

        # find the best representative for each species
        unspeciated = set(population.keys())
        distances = GenomeDistanceCache(config.genome_config)
        new_representatives = {}
        new_members = {}
        for skey, s in self.species.items():
            candidates = []
            for gkey in unspeciated:
                g = population[gkey]
                distance = distances(s.representative, g)
                candidates.append((distance, g))
            # new representative is the closest to current representative
            ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
            new_rep_key = new_rep.key
            new_representatives[skey] = new_rep_key
            new_members[skey] = [new_rep_key]
            unspeciated.remove(new_rep_key)
        # break population into species niches based on genetic similarity
        while unspeciated:
            gkey = unspeciated.pop()
            g = population[gkey]
            # find species with the most similar representative
            candidates = []
            for skey, rep_key in new_representatives.items():
                rep = population[rep_key]
                distance = distances(rep, g)
                if distance < compatibility_threshold:
                    candidates.append((distance, skey))
            if candidates:
                ignored_sdist, skey = min(candidates, key=lambda x: x[0])
                new_members[skey].append(gkey)
            else: # if no species is similar enough, create new one
                skey = next(self.indexer)
                new_representatives[skey] = gkey
                new_members[skey] = [gkey]

        self.genome_to_species = {}
        for skey, rep_key in new_representatives.items():
            s = self.species.get(skey)
            if s is None:
                s = Species(skey, generation)
                self.species[skey] = s
            members = new_members[skey]
            for gkey in members:
                self.genome_to_species[gkey] = skey
            member_dict = dict((gkey, population[gkey]) for gkey in members)
            s.update(population[rep_key], member_dict)

        gdmean = mean(distances.distances.values())
        gdstdev = std(distances.distances.values())
        self.reporters.info("Genetic distance: mean {0:.3f}, stdev {1:.3f}".format(gdmean, gdstdev))

    def get_species_key(self, key):
        return self.genome_to_species[key]

    def get_species(self, key):
        skey = self.genome_to_species[key]
        return self.species[skey]
