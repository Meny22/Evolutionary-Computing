# imports framework
# import sys
# sys.path.insert(0, 'evoman')
# from environment import Environment
# from evolution_strategy_controller import ESController

# imports other libs
import time
import numpy as np
import glob, os
import pickle
from matplotlib import pyplot as plt

from deap import creator
from deap import base

creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

experiment_name = 'evolution_strategy'

# # dont display the screen
# os.environ['SDL_VIDEODRIVER'] = 'dummy'

# # initializes simulation in individual evolution mode, for single static enemy.
# controller = ESController()

avg = np.zeros((3, 10, 30))
energy = np.zeros((3, 10, 30))
fs = np.ndarray((10, 3))

for enemy in range(1, 4):
    # env = Environment(experiment_name=experiment_name,
    #                   enemies=[enemy],
    #                   playermode="ai",
    #                   player_controller=controller,
    #                   enemymode="static",
    #                   level=2,
    #                   speed="fastest",
    #                   sound="off",
    #                   logs="on")

    # env.state_to_log()  # checks environment state
    for i in range(10):
        with open('{}/{}/logbook_{}.pkl'.format(experiment_name, enemy, i), 'rb') as file:
            population, log, hof = pickle.load(file)

        avg[enemy - 1, i] = log.chapters['fitness'].select('avg')
        energy[enemy - 1, i] = log.chapters['energy'].select('avg')
        # f, p, e, t = env.play(pcont=hof[-1])
        fs[i, enemy - 1] = hof[0].fitness.values[0]

    # print(avg.mean(1))
    # print(avg.std(1))
    # print(energy.mean(1))
    # print(energy.std(1))

    # print(fs.mean())
    # print(fs.std())
    # print(fs.min())
    # print(fs.max())

plt.boxplot(fs)
plt.show()