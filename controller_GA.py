#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np

experiment_name = 'GA_experiment_final'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=player_controller(),
			  	  speed="fastest",
				  enemymode="static",
				  level=2)

enemies = [1,2,3]
fits = []
for i in enemies:
    fit = []
    #Update the enemy
    env.update_parameter('enemies',[i])
    file = "GA_experiment_final_enemy"+str(i)
    for x in range(10):
        sol = np.loadtxt(file+'/best'+str(x)+'.txt')
        f,p,e,t = env.play(pcont=sol)
        fit.append(f)
    fits.append(fit)
print(fits)

# tests saved demo solutions for each enemy
# for en in range(1, 9):
# 	# Update the number of neurons for this specific example
# 	env.player_controller.n_hidden = [0]
#
# 	#Update the enemy
# 	env.update_parameter('enemies',[1])
#
# 	# Load specialist controller
# 	sol = np.loadtxt('solutions_demo/demo_'+str(1)+'.txt')
# 	print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY '+str(en)+' \n')
# 	env.play(sol)
