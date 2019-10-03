################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from deap import base
from deap import creator
from deap import tools
import random
import numpy as np
from deap import algorithms
import multiprocessing

os.environ['SDL_VIDEODRIVER'] = 'dummy'

experiment_name = 'GA_experiment_final'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name, player_controller=player_controller(), level=2)
# print(env.play())

toolbox = base.Toolbox()

n_hidden = 10
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 # multilayer with 10 hidden neurons
population_size = 20

def evaluate(individual):
    return env.play(pcont=individual)

def init():
    creator.create("FitnessMax",base.Fitness,weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox.register("sensors",np.random.uniform,-1,1)
    toolbox.register("Individual",tools.initRepeat,creator.Individual,toolbox.sensors, n_vars)
    toolbox.register("Population", tools.initRepeat, list , toolbox.Individual)

    toolbox.register("evaluate",evaluate)
    toolbox.register("crossover",tools.cxTwoPoint)
    toolbox.register("mutate",tools.mutFlipBit,indpb=0.05)
    toolbox.register("select",tools.selTournament,tournsize=3)

def create_population():
    population = toolbox.Population(n=population_size)
    return population

def evaluate_population(population,run,file):
    fitness = list(map(toolbox.evaluate,population))
    for ind,fit in zip(population,fitness):
        ind.fitness.values = fit
    evolution(population,run,file)

def evolution(population,run,file):
    fits = [ind.fitness.values[0] for ind in population]
    # Variable keeping track of the number of generations
    g = 0
    crossover_prob = 0.5
    mutation_prob = 0.2

    # Begin the evolution
    while g < 30:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.crossover(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        energy = []
        for ind, fit in zip(invalid_ind, fitnesses):
            energy.append(fit[1])
            ind.fitness.values = fit

        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]

        length = len(population)
        mean = sum(fits) / length
        mean_energy = sum(energy)/length
        std_energy = np.std(energy)
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        best = np.argmax(fits)

        # saves results

        # print('\n GENERATION ' + str(g) + ' ' + str(round(fits[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
        #     round(std, 6)))
        # file_aux.write(
        #     '\n' + str(g) + ' ' + str(round(fits[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
        file.write("%f,%f,%f,%f,%d,%f \n" % (round(mean,4),round(std,4),round(max(fits),4),round(min(fits),4),mean_energy,round(std_energy,4)))

        # saves generation number
        # file_aux = open(experiment_name + '/gen.txt', 'w')
        # file_aux.write(str(g))
        # file_aux.close()

        # saves file with the best solution
        np.savetxt(experiment_name + '/best'+str(run)+'.txt', population[best])

        # saves simulation state
        solutions = [population, fits]
        env.update_solutions(solutions)
        env.save_state()


def start(i,file):
    print("Starting run: " + str(i))
    pop = create_population()
    evaluate_population(pop,i,file)

init()
enemies = [2]
for x in enemies:
    env.update_parameter('enemies', [x])
    experiment_name = 'GA_experiment_final_enemy_test'+str(x)
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    for i in range(10):
        file_aux = open(experiment_name + '/results'+str(i)+'.csv', 'a')
        file_aux.write('Avg,Std,Max,Min,Energy,Std_energy \n')
        start(i,file_aux)