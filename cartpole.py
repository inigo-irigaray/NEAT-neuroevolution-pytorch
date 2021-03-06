import os
import click

import gym

from neuroevolution.config import Config
from neuroevolution.genome import DefaultGenome
from neuroevolution.multienv_eval import MultiEnvEvaluator
from neuroevolution.population import Population
from neuroevolution.recurrent import RecurrentNetwork
from neuroevolution.reporting import LogReporter, StdOutReporter, StatisticsReporter
from neuroevolution.reproduction import DefaultReproduction
from neuroevolution.species import DefaultSpeciesSet
from neuroevolution.stagnation import DefaultStagnation




max_env_steps = 200

def make_env():
    return gym.make("CartPole-v0")

def make_net(genome, config, bs):
    return RecurrentNetwork.create(genome, config, bs)

def activate_net(net, states):
    outputs = net.activate(states).numpy()
    return outputs[:, 0] > 0.5

@click.command()
@click.option("--n_generations", type=int, default=100)
def run(n_generations):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), "cartpole.cfg")
    config = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, config_path)

    evaluator = MultiEnvEvaluator(make_net, activate_net, make_env=make_env, max_env_steps=max_env_steps)

    def eval_genomes(genomes, config):
        for _, genome in genomes:
            genome.fitness = evaluator.eval_genome(genome, config)

    pop = Population(config)
    stats = StatisticsReporter()
    pop.add_reporter(stats)
    reporter = StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("cartpole.log", evaluator.eval_genome)
    pop.add_reporter(logger)

    pop.run(eval_genomes, n_generations)




if __name__ == "__main__":
    run()
