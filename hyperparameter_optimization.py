import pandas as pd 
import numpy as np 
import random 

random.seed( 1010 )

GENLONG = 100 
def generate_genoma(GENLONG):
    """
    Generates a Random secuence of 1,0 of length GENLONG
    """
    genoma = ""
    for i in range(GENLONG):
        genoma += str(random.choice([1,0]))
    return genoma


def generate_population( N , GENLONG):
    POPULATION_X = dict()
    for i in range(N):
        individual = generate_genoma(GENLONG)
        POPULATION_X[individual] = {"SCORE": np.nan, "PROB": np.nan}
        print("New individual generated: {}".format(individual))
    return POPULATION_X

def look_solutions(genoma, SOLUTIONS):
    try:
        return SOLUTIONS[genoma]
    except Exception as e:
        print("genoma is not save in SOLUTIONS")
        return False

def score_genoma(genoma, OBJECTIVE):
    new_score = 0 
    for bit_genoma, bit_objective in zip(genoma, OBJECTIVE):
        if bit_objective == bit_genoma:
            new_score +=1
    return new_score

def save_solutions(genoma, new_score, SOLUTIONS):
    SOLUTIONS[genoma] = new_score

def score_population(POPULATION_X, SOLUTIONS):
    for genoma in POPULATION_X.keys():
        if look_solutions(genoma, SOLUTIONS):
            new_score = look_solutions
            POPULATION_X[genoma] = new_score
        else:
            new_score = score_genoma(genoma, OBJECTIVE)
            save_solutions(genoma, new_score, SOLUTIONS)
            POPULATION_X[genoma] = new_score
    return POPULATION_X



FIRST_ITERATION = True
N = 100
PC =.1   #Crossover probability
PM = .05 #Mutation probability
SOLUTIONS = dict()
OBJECTIVE = generate_genoma(GENLONG)
POPULATION_X = generate_population( N, GENLONG)
POPULATION_X = score_population(POPULATION_X, SOLUTIONS)

def sort_population(POPULATION_X):
    sorted_population = sorted(POPULATION_X.items(), key=lambda x: x[1], reverse = True)
    result = list()
    for i in sorted_population:
        result.append( list(i) )

    return result

def sorted_to_dict(sorted_population):
    POPULATION_X = dict()
    for i in range(len(sorted_population)):
        values = sorted_population[i] 
        print(values)
        POPULATION_X[values[0]] = values[1]
    return POPULATION_X

POPULATION_X = sorted_to_dict(sorted_population)

sorted_population = sort_population(POPULATION_X)

POPULATION_Y = sorted_population.copy()
for j in range(int(N/2)):
    pc = random.uniform(1,0)

    #CROSSOVER
    if pc < PC:
        best = sorted_population[j][0]
        worst = sorted_population[N -j][0]
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


        sorted_population[j] = new_best
        sorted_population[N-j]  = new_worst

        print("\n\nCrossover Performed on gen {}: \n\t StartBest: {} \n\t EndBest: {} \n\t StartWorst: {} \n\t EndWorst: {}".format(genoma_crossover, startBest, endBest, startWorst, endWorst))

for j in range(N):
    #MUTATION

    pm = random.uniform(1,0)
    mutation_gen = random.randint(0, GENLONG)

    if pm < PM:
        #PERFORM MUTATION

        mutated_genoma = sorted_population[j][0]
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

        sorted_population[j][0] = mutated_genoma

        print("\nMutation Performed at gen {}: \n\t Start: {} \n\t End: {}".format(mutation_gen, start, end))

#RESELECT
POPULATION_Y
POPULATION_X = sorted_to_dict(sorted_population)
POPULATION_X = score_population(POPULATION_X, SOLUTIONS)






