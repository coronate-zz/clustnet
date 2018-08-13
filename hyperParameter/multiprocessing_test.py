import pandas as pd 
import numpy as np 
import random 
from pprint import pprint 
import copy
import pdb, traceback, sys
import multiprocessing
import random
import time


N_WORKERS = 16

manager = multiprocessing.Manager()
SOLUTIONS = manager.dict()
lock = multiprocessing.Lock()

POPULATION_SIZE = 200
POPULATION_X = dict()

def solve_individual( genoma, SOLUTIONS, lock):
	#In this part we expect an XGBOOST model to be fitted. Every XGBOOST require N number of cores
	#If we have a total of T Cores and N_WORKERS then to every thread we need to assign T/N_WORKERS
	#This means that each XGBOOST model will be fitted with T/N_WORKERS Cores.
	suma = 0
	for i in genoma:
		suma += int(i)
	#random_sleep =  random.randint(1, 10) * .05
	#time.sleep(.005)
	#time.sleep(random_sleep)
	#lock.acquire()
	SOLUTIONS[genoma] = suma
	#lock.release()
	print("\n\n", SOLUTIONS)
 
def generate_genoma(GENLONG):
    """
    Generates a Random secuence of 1,0 of length GENLONG
    """
    genoma = ""
    for i in range(GENLONG):
        genoma += str(random.choice([1,0]))
    return genoma

for s in range(POPULATION_SIZE): 
	POPULATION_X[s] = {"GENOMA" : str(s*100) , "SCORE": np.nan , "PROCESS": 0}

s = 0 

t = time.time()

while s < POPULATION_SIZE-1: 
	#EN esta parte antes de meterlo a procesamiento deberia buscar la solucion...
	started_process = list()
	active_workers = 0

	while active_workers < N_WORKERS:
		if s >= POPULATION_SIZE:
			break
		else:
			if POPULATION_X[s]["GENOMA"] in SOLUTIONS.keys():
				print("\n\tSolucion encontrada! ", s)
				s+=1
			else:
				#<------SOLVE
				POPULATION_X[s]["PROCESS"] = multiprocessing.Process( target= solve_individual, args=(POPULATION_X[s]["GENOMA"], SOLUTIONS, lock,  ))
				POPULATION_X[s]["PROCESS"].start()
				started_process.append(s)
				s+=1

				active_workers +=1


	print("TRABAJOS ASIGNADOS: \n\t", started_process)
	for sp in started_process:  # <---CUANTOS CORES TENGO 
		POPULATION_X[sp]["PROCESS"].join()


#NOW ALL THE SCORES ARE IN SOLUTIONS DICTIONARY...
for s in POPULATION_X.keys():
	POPULATION_X[s]["SCORE"] = SOLUTIONS[ POPULATION_X[s]["GENOMA"] ]


print("\n\n\nTIEMPO PROCESAMIENTO: ", time.time() - t)
print("\n\nTEST   LEN(SOLUTIONS): {}  POPULATION_SIZE: {}".format(len(SOLUTIONS), POPULATION_SIZE))






