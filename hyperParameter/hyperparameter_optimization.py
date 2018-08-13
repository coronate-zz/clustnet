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



GENLONG = 15
FIRST_ITERATION = True
N = 20
PC =.25   #Crossover probability
PM = .1   #Mutation probability
MAX_ITERATIONS = 1000
GENOMA_TEST = 783
N_WORKERS = 16
"""
For a particular genetic algorithm with GENLONG = len_population and 
"""

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


MODELS_dict = {
"genetic": 
   {
   "function": score_genetics,
   "model_type": "genetic",
   "params":
     {
     #-------------------------Must Sum GENLONG--------------------------------------------
     "len_population": 5,  #<---- test GA of POPULATION 0 - (len_population)^2
     "len_pc": 5,          #<-----test GA of Crossover probability 0-1 in steps of 1/len_pc
     "len_pm": 5,          #<-----test GA of Mutation probability  0-1 in steps of 1/len_pm 

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
    },

"xgboost":
    {
      "function":  "score_xgboost",
      "model_type":"xgboost",
      "params":
      {}

    }



}



POPULATION_X, SOLUTIONS = solve_genetic_algorithm( GENLONG,  N, PC, PM, N_WORKERS, MAX_ITERATIONS, MODELS_dict["genetic"])

report_genetic_results( POPULATION_X[0]["GENOMA"], MODELS_dict["genetic"])

SOLUTIONS
pc_total =0
pm_total =0 
pop_tot = 0
cont = 0 

#save_obj(SOLUTIONS, "test2")

MODEL = MODELS_dict["genetic"]
for genoma in SOLUTIONS.keys():
  score = SOLUTIONS[genoma]
  if score >-200:
    cont+=1
    end_population = MODEL["params"]["len_population"]
    end_pc         = end_population + MODEL["params"]["len_pc"]
    end_pm         = end_pc         + MODEL["params"]["len_pm"]

    POPULATION_SIZE  = int(genoma[:end_population], 2) +1
    PC               = 1/(int(genoma[end_population: end_pc], 2) +.0001)
    PM               = 1/(int(genoma[end_pc: end_pm], 2) +.0001)
    if PC >1:
      PC =1
    if PM>1:
      PM =1
    pc_total += PC
    pm_total += PM
    pop_tot += POPULATION_SIZE

    MAX_ITERATIONS = MODEL["params"]["max_iterations"]
    GENLONG        = MODEL["params"]["len_genoma"] + 1
    print("\n\n\nEXECUTE SCORE GENETICS: {} \n\POPULATION_SIZE: {}\n\tpc: {} \n\tPM: {}".format(score, POPULATION_SIZE, PC, PM ))

print("AVG PM", pm_total/cont)
print("AVG PC", pc_total/cont)
print("AVG POP", pop_tot/cont)