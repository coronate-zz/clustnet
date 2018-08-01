import pandas as pd 
import numpy as np 
import random 
from pprint import pprint 
import copy
import pdb, traceback, sys
import multiprocessing
import random
import time


#MODEL= "test"; len_population = 0; len_pc = 0; len_pm= 0 ; N_WORKERS =16


def generate_genoma(GENLONG):
    """
    Generates a Random secuence of 1,0 of len GENLONG
    """
    genoma = ""
    for i in range(GENLONG):
        genoma += str(random.choice([1,0]))
    return genoma


def solve_genetic_algorithm( GENLONG, N, PC, PM, N_WORKERS, MAX_ITERATIONS, MODEL ):
    """
    Solve a Genetic Algorithm with 
        len(genoma)               == GENLONG, 
        POPULATION                == N individuals
        Mutation probability      == PM
        Crossover probability     == PC
        Max number of iterations  == MAX_ITERATIONS
        Cost function             == MODEL

    """
    STILL_CHANGE = True 
    still_change_count = 0 

    LOCK = multiprocessing.Lock()
    manager   = multiprocessing.Manager()
    manager2  = multiprocessing.Manager()
    manager3  = multiprocessing.Manager()

    SOLUTIONS = manager.dict()
    MODEL_P = manager2.dict()

    #Transform model to multiprocessing object:
    for key in MODEL.keys():
        if key == "params":
            MODEL_P[key] = manager3.dict()
            for key_params in MODEL["params"].keys():
                MODEL_P["params"][key_params] = MODEL["params"][key_params]
        else:
            MODEL_P[key] = MODEL[key]


    OBJECTIVE = generate_genoma(GENLONG)
    POPULATION_X = generate_population( N, GENLONG)

    POPULATION_X = parallel_solve(POPULATION_X, SOLUTIONS, LOCK, N_WORKERS, MODEL_P  )
    POPULATION_X = sort_population(POPULATION_X)


    n_iter = 0 
    while (n_iter <= MAX_ITERATIONS) & STILL_CHANGE:

        POPULATION_Y = cross_mutate(POPULATION_X, N, PC, PM, GENLONG)

        #RESELECT parallel_solve
        POPULATION_X = parallel_solve(POPULATION_X, SOLUTIONS, LOCK, N_WORKERS, MODEL)
        POPULATION_Y = parallel_solve(POPULATION_Y, SOLUTIONS, LOCK, N_WORKERS, MODEL)

        POPULATION_X = sort_population(POPULATION_X)
        POPULATION_Y = sort_population(POPULATION_Y)

        POPULATION_X = select_topn(POPULATION_X, POPULATION_Y, N)
        
        equal_individuals, max_score = population_summary(POPULATION_X)
        if MODEL["model_type"] == "genetic":
            print("\n\n\n\tequal_individuals: {}\t max_score: {}".format(equal_individuals, max_score))
        
        n_iter += 1
        if equal_individuals >= len(POPULATION_X.keys())*.8:
            still_change_count +=1 
            #print("SAME ** {} ".format(still_change_count))
            if still_change_count >= 50:
                print("\n\n\nGA Solved: \n\tGENLONG: {} \n\tN: {}\n\tPC:{}, \n\tPM: {}\n\tN_WORKERS: {} \n\tMAX_ITERATIONS: {}\n\tMODE: {}".format( GENLONG, N, PC, PM, N_WORKERS, MAX_ITERATIONS, MODEL ))
                STILL_CHANGE = False

    return POPULATION_X, SOLUTIONS



def parallel_solve(POPULATION_X, SOLUTIONS, LOCK, N_WORKERS, MODEL ):
    """
    Each individual in POPULATION_X is assigned to a different process to score it's own
    genoma. Each genoma is scored based on the scoring function and parameters saved individual_score
    MODEL dictionary.

    """

    s = 0
    POPULATION_SIZE = len(POPULATION_X)
    while s < POPULATION_SIZE-1: 
        #EN esta parte antes de meterlo a procesamiento deberia buscar la solucion...
        started_process = list()
        active_workers = 0

        while active_workers < N_WORKERS:
            if s >= POPULATION_SIZE:
                break
            else:
                if POPULATION_X[s]["GENOMA"] in SOLUTIONS.keys():
                    if MODEL["model_type"] in ["genetic", "other"]:
                        print("Modelo encontrado en ", MODEL["model_type"])
                    s+=1
                else:
                    #<------SOLVE
                    #TO DO: tenemos Z started_processes, estos deberian repartire los cores
                    started_process.append(s)
                    s+=1
                    active_workers +=1

        for s in started_process:
            CORES_PER_SESION = MODEL["params"]["n_workers"]
            time.sleep( random.randint(0,len(started_process)) * .05)               
            POPULATION_X[s]["PROCESS"] = multiprocessing.Process( target= MODEL["function"] , args=(POPULATION_X[s]["GENOMA"], SOLUTIONS, CORES_PER_SESION, LOCK, MODEL, ))
            POPULATION_X[s]["PROCESS"].start()

        #print("STARTED PROCESSES: \n\t", started_process)

        for sp in started_process:  # <---CUANTOS CORES TENGO 
            POPULATION_X[sp]["PROCESS"].join()
            del POPULATION_X[sp]["PROCESS"]

    #All genoma in SOLUTIONS
    for s in POPULATION_X.keys():
        if  POPULATION_X[s]["GENOMA"] in SOLUTIONS.keys():
            POPULATION_X[s]["SCORE"] = SOLUTIONS[POPULATION_X[s]["GENOMA"]]
            POPULATION_X[s]["PROB"] = POPULATION_X[s]["SCORE"] / POPULATION_SIZE
        else:
            print(POPULATION_X[s])
            print("**WARNING: GENOMA not found in SOLUTIONS after scoring")

    return POPULATION_X

def generate_population(N ,GENLONG):
    POPULATION_X = dict()
    for i in range(N):
        genoma = generate_genoma(GENLONG)
        POPULATION_X[i] = {"GENOMA":genoma, "SCORE": np.nan, "PROB": np.nan}
        #print("New individual generated: {}".format(genoma))
    return POPULATION_X

def look_solutions(genoma, SOLUTIONS):
    try:
        return SOLUTIONS[genoma]
    except Exception as e:
        print("genoma is not save in SOLUTIONS")
        return False

def score_genetics(genoma, SOLUTIONS, CORES_PER_SESION, LOCK, MODEL ):
    """
    This scoring fucntion is use to test how good is the configuration of a 
    genetic algorithm to solve a problem of GENLONG 
    """
    LOCK.acquire()
    end_population = MODEL["params"]["len_population"]
    end_pc         = end_population + MODEL["params"]["len_pc"]
    end_pm         = end_pc         + MODEL["params"]["len_pm"]

    POPULATION_SIZE  = int(genoma[:end_population], 2) +1
    PC               = 1/(int(genoma[end_population: end_pc], 2) +.0001)
    PM               = 1/(int(genoma[end_pc: end_pm], 2) +.0001)

    MAX_ITERATIONS = MODEL["params"]["max_iterations"]
    GENLONG        = MODEL["params"]["len_genoma"] + 1
    N_WORKERS      = MODEL["params"]["n_workers"]
    LOCK.release()

    print("\n\n\nEXECUTE SCORE GENETICS: \n\POPULATION_SIZE: {}\n\tpc: {} \n\tPM: {}".format(POPULATION_SIZE, PC, PM ))

    time_start = time.time()
    test ={"model_type": "test",  "function": score_test, "params": { "n_workers": 4}}
    POPULATION_X, SOLUTIONS_TEST = solve_genetic_algorithm(GENLONG, POPULATION_SIZE, PC, PM,  N_WORKERS, MAX_ITERATIONS, test)
    time_end = time.time()
    x, max_score = population_summary(POPULATION_X)


    TOTAL_TIME = time_end - time_start
    SCORE = GENLONG- max_score
    final_score = -(SCORE) -(.1 *TOTALTIME)
    print("\t\tTIME: {}  MAX_SCORE: {} FINAL_SCORE: {}".format(TOTAL_TIME, max_score, final_score))

    LOCK.acquire()
    SOLUTIONS[genoma]  = final_score
    LOCK.release()




def score_test(genoma, SOLUTIONS,  CORES_PER_SESION, LOCK, MODEL):
    #print("\n\n\n MODEL TYPE: {} , \n\tGENOMA: {}, ".format( MODEL["model_type"], genoma, SOLUTIONS))
    suma = 0 
    for i in genoma:
        suma += int(i)
    time.sleep(.1)
    LOCK.acquire()
    SOLUTIONS[genoma] = suma
    LOCK.release()


def save_solutions(genoma, new_score, SOLUTIONS):
    SOLUTIONS[genoma] = new_score



def sort_population(POPULATION_X):
    POPULATION_Y = sorted(POPULATION_X.items(), key=lambda x: x[1]["SCORE"], reverse = True)
    POPULATION_NEW = dict()
    result = list()
    cont = 0
    for i in POPULATION_Y:
        POPULATION_NEW[cont] = i[1]
        cont += 1
    return POPULATION_NEW

def population_summary(POPULATION_X):
    suma = 0
    min_score = 10000
    max_score = -10000 
    lista_genomas = list()

    for key in POPULATION_X.keys():
        individual_score =  POPULATION_X[key]["SCORE"]
        lista_genomas.append(POPULATION_X[key]["GENOMA"])
        suma += individual_score
        if max_score < individual_score:
            max_score = individual_score
        if min_score > individual_score:
            min_score = individual_score
    promedio = suma/len(POPULATION_X.keys())
    equal_individuals = len(lista_genomas) - len(set(lista_genomas))
    #print("\n\nTOTAL SCORE: {} MEAN SCORE: {} \n\t MAX_SCORE: {} MIN_SCORE: {} ".format(suma, promedio, max_score, min_score))
    return equal_individuals, max_score


def select_topn( POPULATION_X, POPULATION_Y, N):
    for key in POPULATION_X.keys():
        new_key = key  + N
        POPULATION_Y[new_key] = POPULATION_X[key]

    POPULATION_Y = sort_population(POPULATION_Y)
    POPULATION_NEW =dict()

    for key in range(N):
        POPULATION_NEW[key] = POPULATION_Y[key]

    return POPULATION_NEW


def cross_mutate( POPULATION_X, N, PC, PM, GENLONG):
    POPULATION_Y = copy.deepcopy(POPULATION_X)
    #pprint(POPULATION_X)
    for j in range(int(N/2)):
        pc = random.uniform(1,0)

        #CROSSOVER
        if pc < PC:
            best = POPULATION_Y[j]["GENOMA"]
            worst = POPULATION_Y[N -j-1]["GENOMA"]
            startBest =  best
            startWorst = worst

            genoma_crossover = random.randint(0, GENLONG)
            best_partA  = best[:genoma_crossover]
            worst_partA = worst[:genoma_crossover]

            best_partB  =  best[genoma_crossover:]
            worst_partB = worst[genoma_crossover:]

            new_best    = best_partA  + worst_partB
            new_worst   = worst_partA + best_partB

            endBest = new_best
            endWorst =  new_worst

            POPULATION_Y[j]["GENOMA"]    = new_best
            POPULATION_Y[N-j-1]["GENOMA"]  = new_worst

            #print("\n\nCrossover Performed on individual {}-{}: \n\t StartBest: {} \n\t EndBest: {} \n\t StartWorst: {} \n\t EndWorst: {}".format(j,genoma_crossover, startBest, endBest, startWorst, endWorst))

    for j in range(N):
        #MUTATION

        pm = random.uniform(1,0)
        mutation_gen = random.randint(0, GENLONG-1)

        if pm < PM:
            #PERFORM MUTATION

            mutated_genoma = POPULATION_Y[j]["GENOMA"]
            start = mutated_genoma
            if mutated_genoma[mutation_gen] =="1":
                mutated_genoma = list(mutated_genoma)
                mutated_genoma[mutation_gen] = "0"
                mutated_genoma = "".join(mutated_genoma)
            else:
                mutated_genoma = list(mutated_genoma)
                mutated_genoma[mutation_gen] = "1"
                mutated_genoma = "".join(mutated_genoma)

            end = mutated_genoma

            POPULATION_Y[j]["GENOMA"] = mutated_genoma

            #print("\nMutation Performed on individual {}-{}: \n\t Start: {} \n\t End: {}".format(j, mutation_gen, start, end))

    return POPULATION_Y

def report_genetic_results(genoma, MODEL):
    end_population = MODEL["params"]["len_population"]
    end_pc         = end_population + MODEL["params"]["len_pc"]
    end_pm         = end_pc         + MODEL["params"]["len_pm"]

    POPULATION_SIZE  = int(genoma[:end_population], 2) +1
    PC               = 1/(int(genoma[end_population: end_pc], 2) +.01)
    PM               = 1/(int(genoma[end_pc: end_pm], 2) +.01)

    MAX_ITERATIONS = MODEL["params"]["max_iterations"]
    GENLONG        = MODEL["params"]["len_genoma"] + 1
    N_WORKERS      = MODEL["params"]["n_workers"]
    print("\n\n ** BEST GA **: \n\tgenlong: {}\n\tpc: {} \n\tPM: {}".format(POPULATION_SIZE, PC, PM ))



