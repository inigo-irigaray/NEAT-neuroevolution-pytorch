"""Implementation based on """

import os

import click
import numpy as np

import flappybird_env
from neuroevolution.config import Config
from neuroevolution.genome import DefaultGenome
from neuroevolution.population import Population
from neuroevolution.recurrent import RecurrentNetwork
from neuroevolution.reporting import LogReporter, StdOutReporter, StatisticsReporter
from neuroevolution.reproduction import DefaultReproduction
from neuroevolution.species import DefaultSpeciesSet
from neuroevolution.stagnation import DefaultStagnation

MAX_ENV_STEPS = 200




class MultiEnvEvaluator:
    def __init__(self, make_net, activate_net, batch_size=1,
                 max_env_steps=None, make_env=None, envs=None):
        if envs is None:
            self.envs = [make_env() for _ in range(batch_size)]
        else:
            self.envs = envs

        self.make_net = make_net
        self.activate_net = activate_net
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps
        self.clock = pygame.time.Clock()

    def eval_genome(self, genome, config, debug=False):
        net = self.make_net(genome, config, self.batch_size)
        fitnesses = np.zeros(self.batch_size)
        states = [env.state for env in self.envs]
        dones = [env.done for env in self.envs]
        for env in self.envs:
            env.gen += 1

        step_num = 0
        while True:
            step_num += 1
            if self.max_env_steps is not None and step_num == self.max_env_steps:
                break
            if debug:
                actions = self.activate_net(net, states, debug=True, step_num=step_num)
            else:
                actions = self.activate_net(net, states)
            assert len(actions) == len(self.envs)
            for i, (env, action, done) in enumerate(zip(self.envs, actions, dones)):
                if not done:
                    state, reward, done = env.step(action)
                    fitnesses[i] += reward
                    if not done:
                        states[i] = state
                    dones[i] = done
            if all(dones):
                break

        return sum(fitnesses) / len(fitnesses)





def make_env():
    return flappybird_env.Env()

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
    config_path = os.path.join(os.path.dirname(__file__), "flappybird.cfg")
    config = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet,
                    DefaultStagnation, config_path)

    evaluator = MultiEnvEvaluator(make_net, activate_net, make_env=make_env,
                                  max_env_steps=MAX_ENV_STEPS)

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
