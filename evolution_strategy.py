###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from evolution_strategy_controller import ESController

# imports other libs
import time
import numpy as np
from math import fabs, sqrt
import glob, os
import multiprocessing
import pickle

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools

# dont display the screen
os.environ['SDL_VIDEODRIVER'] = 'dummy'

experiment_name = 'evolution_strategy'

# initializes simulation in individual evolution mode, for single static enemy.
controller = ESController()

env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=controller,
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  sound="off",
                  logs="off")

# genetic algorithm params
n_vars = (env.get_num_sensors() + 1) * sum(
    controller.n_hidden) + (sum(controller.n_hidden) + 1) * 5  # multilayer with 10 hidden neurons

creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


# runnign the evoman game
def evaluate(x):
    f, p, e, t = env.play(pcont=x)
    return f,


toolbox.register("evaluate", evaluate)


def timeit(*args, **kwargs):
    global gen_start_time
    gen_end_time = time.time()
    gen_time = gen_end_time - gen_start_time
    gen_start_time = gen_end_time
    return gen_time


if __name__ == '__main__':
    # The cma module uses the np random number generator
    np.random.seed(128)

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # default environment fitness is assumed for experiment
    env.state_to_log()  # checks environment state

    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES
    strategy = cma.Strategy(centroid=[0] * n_vars, sigma=1 / controller.n_hidden[0], lambda_=100)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    # multi processing
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    hof = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("time", timeit)

    ini = time.time()  # sets total time marker
    gen_start_time = time.time()  # sets gen time marker

    # The CMA-ES algorithm converge with good probability with those settings
    population, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=30, stats=stats, halloffame=hof, verbose=True)

    # print "Best individual is %s, %s" % (hof[0], hof[0].fitness.values)
    print(hof[0].fitness.values[0])

    fim = time.time()  # prints total execution time for experiment
    print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')

    with open(experiment_name + 'logbook.pkl', 'wb') as file:
        pickle.dump((population, logbook, hof), file)

    env.state_to_log()  # checks environment state
    pool.close()
