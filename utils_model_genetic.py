import pandas as pd 
import numpy as np 
import random 
from   pprint import pprint 
import copy
import pdb, traceback, sys
import multiprocessing
import random
import time
from sklearn import linear_model
import pickle
import threading

#MODEL= "test"; len_population = 0; len_pc = 0; len_pm= 0 ; N_WORKERS =16




def solve_genetic_algorithm(N, PC, PM, N_WORKERS, MAX_ITERATIONS, MODEL, NEURONAL_SOLUTIONS, n_col, df, df_kfolded, df_exmodel, round_prediction ):
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
    manager_solutions   = multiprocessing.Manager()
    manager_model       = multiprocessing.Manager()
    manager_params      = multiprocessing.Manager()

    SOLUTIONS = manager_solutions.dict()
    MODEL_P   = manager_model.dict()

    #Transform model to multiprocessing object:
    for key in MODEL.keys():
        if key == "params":
            MODEL_P[key] = manager_params.dict()
            for key_params in MODEL["params"].keys():
                MODEL_P["params"][key_params] = MODEL["params"][key_params]
        else:
            MODEL_P[key] = MODEL[key]

    POPULATION_X = generate_model_population(df_kfolded, NEURONAL_SOLUTIONS, n_col, N)
    POPULATION_X = parallel_solve(POPULATION_X, SOLUTIONS, LOCK, N_WORKERS, MODEL_P,
                                  df_kfolded, df_exmodel, "MSE", round_prediction)

    POPULATION_X = sort_population(POPULATION_X)


    n_iter = 0 
    while (n_iter <= MAX_ITERATIONS) & STILL_CHANGE:

        POPULATION_Y = cross_mutate(POPULATION_X, N, PC, PM)

        #RESELECT parallel_solve
        POPULATION_X = parallel_solve(POPULATION_X, SOLUTIONS, LOCK, N_WORKERS, MODEL, 
                       df_kfolded, df_exmodel, "MSE", round_prediction)
        POPULATION_Y = parallel_solve(POPULATION_Y, SOLUTIONS, LOCK, N_WORKERS, MODEL,
                       df_kfolded, df_exmodel, "MSE", round_prediction)

        POPULATION_X = sort_population(POPULATION_X)
        POPULATION_Y = sort_population(POPULATION_Y)

        POPULATION_X = select_topn(POPULATION_X, POPULATION_Y, N)
        
        equal_individuals, max_score = population_summary(POPULATION_X)
        if MODEL["model_name"] == "LR":
            print("\n\n\n\tequal_individuals: {}\t max_score: {}".format(equal_individuals, max_score))
        
        n_iter += 1
        if equal_individuals >= len(POPULATION_X.keys())*.8:
            still_change_count +=1 
            #print("SAME ** {} ".format(still_change_count))
            if still_change_count >= 50:
                print("\n\n\nGA Solved: \n\tN: {}\n\tPC:{}, \n\tPM: {}\n\tN_WORKERS: {} \n\tMAX_ITERATIONS: {}\n\tMODE: {}".format( N, PC, PM, N_WORKERS, MAX_ITERATIONS, MODEL ))
                STILL_CHANGE = False

    return POPULATION_X, SOLUTIONS


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


def parallel_solve(POPULATION_X, SOLUTIONS, LOCK, N_WORKERS, MODEL, df_kfolded, df_exmodel, error_type, round_prediction):
    """
    Each individual in POPULATION_X is assigned to a different process to score it's own
    genoma. Each genoma is scored based on the scoring function and parameters saved individual_score
    MODEL dictionary.

    """
    print("\n\n parallel_solve: \n\tMODEL: {} \n\t ERROR: {}".format(MODEL, error_type))
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
                    if MODEL["model_name"] in ["LR"]:
                        print("Modelo encontrado en ", MODEL["model_name"])
                    s+=1
                else:
                    #<------SOLVE
                    #TO DO: tenemos Z started_processes, estos deberian repartire los cores
                    started_process.append(s)
                    s+=1
                    active_workers +=1
        model_name = MODEL["model_name"]
        for s in started_process:
            CORES_PER_SESION = MODEL["params"]["n_workers"]
            time.sleep( random.randint(0,len(started_process)) * .05)               
            POPULATION_X[s]["PROCESS"] = multiprocessing.Process( target= MODEL["function"] , 
                                         args=(POPULATION_X[s], model_name, SOLUTIONS, 
                                         CORES_PER_SESION, LOCK, MODEL, df_kfolded, df_exmodel, error_type, round_prediction, ))

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

def generate_model_population(df_kfolded, NEURONAL_SOLUTIONS, n_col, N):
    POPULATION = dict()
    df = df_kfolded["all_data"]["data"]

    for i in range(N):
        POPULATION[i] = dict()
        baseline_features = df.columns
        exmodel_features = get_exmodel_features( NEURONAL_SOLUTIONS, n_col )
        baseline_features_chromosome = ""
        baseline_features_selected = list()

        for feature in baseline_features:
            isincluded = random.choice([True,False])
            if isincluded:
                baseline_features_selected.append(feature)
                baseline_features_chromosome += "1"
            else:
                baseline_features_chromosome += "0" 
        POPULATION[i]["baseline_features"] =  baseline_features_selected
        POPULATION[i]["baseline_features_chromosome"] = baseline_features_chromosome

        if "VentaUnidades" in baseline_features:
            print("ERROR: Target Variable in baseline_features")


        exmodel_features_chromosome = ""
        exmodel_features_selected = list()
        for feature in exmodel_features:
            isincluded = random.choice([True,False])
            if isincluded:
                exmodel_features_selected.append(feature)
                exmodel_features_chromosome += "1"
            else:
                exmodel_features_chromosome += "0"  

        POPULATION[i]["exmodel_features"] =  exmodel_features_selected
        POPULATION[i]["exmodel_features_chromosome"] = exmodel_features_chromosome
        POPULATION[i]["GENOMA"] = POPULATION[i]["baseline_features_chromosome"] +""+ POPULATION[i]["exmodel_features_chromosome"]
        POPULATION[i]["SCORE"]  = np.nan
    return POPULATION



def score_model(INDIVIDUAL, model_name, SOLUTIONS, CORES_PER_SESION, LOCK, MODEL, df_kfolded, df_exmodel, error_type, round_prediction = False):
    """
    Scores the model inside a neuron given a particular set of variables and calculates
    the ERROR_TYPES[error_type] for each fold in df_kfolded.
    """
    ERROR_TYPES  = load_obj("ERROR_TYPES")
    genoma =  INDIVIDUAL["GENOMA"]
    baseline_features = INDIVIDUAL["baseline_features"]
    exmodel_features  = INDIVIDUAL["exmodel_features"]
    #print("\n\nTEST SKLEARN_MODELS: {} \n ERROR_TYPES: {} ".format(SKLEARN_MODELS, ERROR_TYPES))

    model = MODEL["model_class"]
    error_function = ERROR_TYPES[error_type]
    total_error = 0
    for test_fold in df_kfolded.keys():
        #print("test_fold: ", test_fold)
        if test_fold == "all_data":
            continue
        else:
            X_test_baseline  = df_kfolded[test_fold]["data"].copy()
            X_test_baseline  = X_test_baseline[baseline_features]
            X_test_exmodel   = df_exmodel[test_fold]
            X_test_exmodel   = X_test_exmodel[exmodel_features]


            X_test = X_test_baseline.merge(X_test_exmodel, right_index = True, left_index =  True)
            y_test  = df_kfolded[test_fold]["y"]

            X_train_baseline = pd.DataFrame()
            y_train = pd.DataFrame()
            for train_fold in df_kfolded.keys():
                #print("train_fold: ", train_fold)
                if train_fold == "all_data" or train_fold == test_fold:
                    continue 
                else:
                    if len(X_train_baseline)==0:
                        X_train_baseline = df_kfolded[train_fold]["data"]
                        X_train_baseline = X_train_baseline[baseline_features]

                        X_train_exmodel  = df_exmodel[train_fold]
                        X_train_exmodel  = X_train_exmodel[exmodel_features]

                        X_train = X_train_baseline.merge(X_train_exmodel, right_index = True, left_index = True)
                        y_train = df_kfolded[train_fold]["y"] 
                    else:
                        X_train_baseline_append = df_kfolded[train_fold]["data"]
                        X_train_baseline_append = X_train_baseline_append[baseline_features]

                        X_train_exmodel_append = df_exmodel[train_fold]
                        X_train_exmodel_append = X_train_exmodel_append[exmodel_features]

                        X_train_append = X_train_baseline_append.merge(X_train_exmodel_append, right_index = True, left_index=True)

                        X_train = X_train.append(X_train_append)
                        y_train = y_train.append(df_kfolded[train_fold]["y"]) 

                    #print("\n\nTEST 2:\t\ty_train {} \n\t\tX_train_baseline: {} ".format(y_train.shape, X_train_baseline.shape))
            
            model.fit(X_train, y_train, X_test, y_test)
            prediction   = model.predict(X_test)
            if round_prediction:
                prediction = np.round(prediction)

            error = error_function(y_test, prediction)
            total_error += error

    score =  - total_error/len(df_kfolded.keys())-1
    SOLUTIONS[genoma]  = score
    print("\n\nTEST 3 {} : \n\tSCORE: {}".format(multiprocessing.current_process(), score))





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
    final_score = -(SCORE) -(.1 *TOTAL_TIME)
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


def get_exmodel_features(NEURONAL_SOLUTIONS, eval_layer):
    """
    Append all exmodel output features names to call them from the df_exmodel dataset.
    """
    exmodel_features =list()
    for n_col in NEURONAL_SOLUTIONS.keys():
        for m_row in NEURONAL_SOLUTIONS[n_col].keys():
            if int(eval_layer.replace("N_", "")) > int(n_col.replace("N_", "")):
                #A model can use an other neuron output iff the layer of the model is lower
                #than the neurons layer, i.e it can only used output from already trained 
                #models.
                exmodel_features.append(NEURONAL_SOLUTIONS[n_col][m_row]["name"])
    return exmodel_features


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
    min_score =  100000000000000000000000
    max_score = -100000000000000000000000
    lista_genomas = list()

    for key in POPULATION_X.keys():
        individual_score =  POPULATION_X[key]["SCORE"]
        lista_genomas.append(POPULATION_X[key]["GENOMA"])
        suma += individual_score
        if max_score < individual_score:
            max_score = individual_score
            best_baseline_features = POPULATION_X[key]["baseline_features_chromosome"]
            best_exmodel_features  = POPULATION_X[key]["exmodel_features_chromosome"]
        if min_score > individual_score:
            min_score = individual_score
    promedio = suma/len(POPULATION_X.keys())
    equal_individuals = len(lista_genomas) - len(set(lista_genomas))
    print("\n\nTEST 4: \n\tTOTAL SCORE: {} \n\tMEAN SCORE: {} \n\t MAX_SCORE: {} \n\tMIN_SCORE: {} \n\t BASELINE FEAT: {} \n\t EXMODEL FEAT: {}".format(suma, promedio, max_score, min_score, best_baseline_features, best_exmodel_features))
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


def cross_mutate( POPULATION_X, N, PC, PM):
    POPULATION_Y = copy.deepcopy(POPULATION_X)
    #pprint(POPULATION_X)
    for j in range(int(N/2)):
        pc = random.uniform(1,0)
        GENLONG =  len(POPULATION_X[0]["GENOMA"])
        #CROSSOVER
        if pc < PC:
            best = POPULATION_Y[j]["GENOMA"]
            worst = POPULATION_Y[N -j-1]["GENOMA"]
            startBest =  best
            startWorst = worst

            genoma_crossover = random.randint(0, GENLONG)
            genoma_crossover_final = int(genoma_crossover + np.round(GENLONG/2))
            if genoma_crossover_final > GENLONG:
                # To perform a roulette  cross over we need to fill the
                extra_genes = genoma_crossover_final - GENLONG 
                genoma_crossover_final =  GENLONG

            else: 
                extra_genes = 0

            best_partA  = best[:extra_genes]
            worst_partA = worst[:extra_genes]

            best_partB  = best[extra_genes:genoma_crossover]
            worst_partB = worst[extra_genes:genoma_crossover]

            best_partC  = best[genoma_crossover:genoma_crossover_final]
            worst_partC = worst[genoma_crossover: genoma_crossover_final]

            best_partD  = best[genoma_crossover_final:GENLONG]
            worst_partD = worst[genoma_crossover_final: GENLONG]

            #print("TEST CROSSOVER:   {} -> {} -> {} -> {} \npart A {} \npartB {} \npart C{} \npartD {}".format(extra_genes, genoma_crossover, genoma_crossover_final, GENLONG, best_partA, best_partB, best_partC, best_partD) ) 
            new_best    = worst_partA + best_partB + worst_partC + best_partD
            new_worst   = best_partA + worst_partB + best_partC + worst_partD

            endBest = new_best
            endWorst =  new_worst

            POPULATION_Y[j]["GENOMA"]     = new_best
            POPULATION_Y[N-j-1]["GENOMA"] = new_worst

            #print("\n\nCrossover Performed on individual {}-{}: \n\t StartBest: {} \n\t NewBest: {} \n\t StartWorst: {} \n\t NewWorst: {}".format(j,genoma_crossover, startBest, endBest, startWorst, endWorst))

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



