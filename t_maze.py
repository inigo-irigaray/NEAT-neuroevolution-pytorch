import click
import multiprocessing
import os

import numpy as np

import t_maze_env
from neuroevolution.activation_functions import tanh_act
from neuroevolution.adaptive_linear import AdaptiveLinearNetwork
from neuroevolution.config import Config
from neuroevolution.genome import DefaultGenome
from neuroevolution.multienv_eval import MultiEnvEvaluator
from neuroevolution.population import Population
from neuroevolution.reporting import LogReporter, StatisticsReporter, StdOutReporter
from neuroevolution.reproduction import DefaultReproduction
from neuroevolution.species import DefaultSpeciesSet
from neuroevolution.stagnation import DefaultStagnation

batch_size = 4
DEBUG = True


def make_net(genome, config, _batch_size):
    input_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, -1.0]]
    output_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    return AdaptiveLinearNetwork.create(
        config,
        genome,
        in_coords=input_coords,
        out_coords=output_coords,
        weights_threshold=0.4,
        batch_size=_batch_size,
        activation=tanh_act,
        out_activation=tanh_act,
        device="cpu",
    )

def activate_net(net, states, debug=False, step_num=0):
    if debug and step_num == 1:
        print("\n" + "=" * 20 + " DEBUG " + "=" * 20)
        print(net.delta_w_node)
        print("W init: ", net.in2out[0])
    outputs = net.activate(states).numpy()
    if debug and (step_num - 1) % 100 == 0:
        print("\nStep {}".format(step_num - 1))
        print("Outputs: ", outputs[0])
        print("Delta W: ", net.delta_w[0])
        print("W: ", net.in2out[0])
    return np.argmax(outputs, axis=1)


@click.command()
@click.option("--n_generations", type=int, default=10000)
@click.option("--n_processes", type=int, default=1)
def run(n_generations, n_processes):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), "t_maze.cfg")
    config = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet,
                    DefaultStagnation, config_path)

    envs = [t_maze_env.TMazeEnv(init_reward_side=i, n_trials=100) for i in [1, 0, 1, 0]]

    evaluator = MultiEnvEvaluator(make_net, activate_net, envs=envs,
                                 batch_size=batch_size, max_env_steps=1000
                                 )

    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)

        def eval_genomes(genomes, config):
            fitnesses = pool.starmap(evaluator.eval_genome,
                                    ((genome, config) for _, genome in genomes))
            for (_, genome), fitness in zip(genomes, fitnesses):
                genome.fitness = fitness

    else:

        def eval_genomes(genomes, config):
            for i, (_, genome) in enumerate(genomes):
                try:
                    genome.fitness = evaluator.eval_genome(genome, config,
                                    debug=DEBUG and i % 100 == 0)
                except Exception as e:
                    print(genome)
                    raise e

    pop = Population(config)
    stats = StatisticsReporter()
    pop.add_reporter(stats)
    reporter = StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("log.json", evaluator.eval_genome)
    pop.add_reporter(logger)

    winner = pop.run(eval_genomes, n_generations)

    print(winner)
    final_performance = evaluator.eval_genome(winner, config)
    print("Final performance: {}".format(final_performance))
    generations = reporter.generation + 1
    return generations


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
