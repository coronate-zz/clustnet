import pandas as pd 
import numpy as np 
import random 
from pprint import pprint 
import copy
import pdb, traceback, sys
import multiprocessing
import random
import time

from utils_genetic import generate_genoma, parallel_solve, generate_population, look_solutions, \
                          score_genetics, save_solutions, sort_population, population_summary, select_topn, \
                          cross_mutate, score_test, solve_genetic_algorithm, report_genetic_results

#random.seed( 1010 )



model_type = "test"
GENETIC_OPTIMIZATION_SOLUTIONS = dict()



GENLONG = 10
FIRST_ITERATION = True
N = 5
PC =.5   #Crossover probability
PM = .25 #Mutation probability
MAX_ITERATIONS = 1000
GENOMA_TEST = 50
N_WORKERS = 16
"""
For a particular genetic algorithm with GENLONG = len_population and 
"""

MODELS_dict = {
"genetic": 
   {
   "function": score_genetics,
   "model_type": "genetic",
   "params":
       {
       #-------------------------Must Sum GENLONG--------------------------------------------
       "len_population": 4,  #<---- test GA of POPULATION 0 - (len_population)^2
       "len_pc": 3,          #<-----test GA of Crossover probability 0-1 in steps of 1/len_pc
       "len_pm": 3,          #<-----test GA of Mutation probability  0-1 in steps of 1/len_pm 

       #-------------------------Long of genetic algorithm to test ----------------------------
       "len_genoma": GENOMA_TEST,
       "n_workers": 4,
       "max_iterations": 1000
       }
   },

"test": 
    {
        "function": score_test,
        "model_type":"test",
        "params":
        {
        "n_workers": 4
        }
    }
}

POPULATION_X, SOLUTIONS = solve_genetic_algorithm( GENLONG,  N, PC, PM, N_WORKERS, MAX_ITERATIONS, MODELS_dict["genetic"])

report_genetic_results( POPULATION_X[0]["GENOMA"], MODELS_dict["genetic"])

