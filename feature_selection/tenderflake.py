import pandas as pd
import numpy as np
import random
import time
from xgboost import XGBRegressor
from xgboost import plot_importance
import utils_model_genetic 
import pickle
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error 
import importlib 
from sklearn import linear_model
import utils_exomodel
import intramodel_hyperparameters_regressor 

importlib.reload(utils_exomodel)
importlib.reload(intramodel_hyperparameters_regressor)
importlib.reload(utils_model_genetic)

SKLEARN_MODELS = intramodel_hyperparameters_regressor.load_models()
models_list =  list(SKLEARN_MODELS.keys())


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)



#-----------------------------Upload INFO and get training sets-----------------------------------
import pandas as pd 
import numpy as np
from  aiqutils.data_preparation import interact_categorical_numerical

import importlib





if True:
    df = pd.read_csv('/home/alejandro/Dropbox/Github/TenderFlakes/Weekly/data_regresion_brand_substitution.csv')
    df = df.reset_index()

    df =  df.drop_duplicates()
    df["DESCRIPTION"] = df.DESCRIPTION.apply(lambda x:"producto_" + str(x))
    df["MARKET_MEANING"] = df.MARKET_MEANING.apply(lambda x:"agencia_" + str(x))

    df["DATE"] = pd.to_datetime(df.DATE, format = "%Y-%m-%d")
    df["MONTH"] = df.DATE.apply(lambda x:  x.month)
    df["YEAR"] = df.DATE.apply(lambda x:  x.year)

    df = df.drop(['Unnamed: 0', 'BRAND', "index",
       'CATEGORY_SUBGROUP', 'CB_PRODUCT_SEGMENT', 'DAY',
       'MANUFACTURER', 'MARKET', 'MARKET_LEVEL', 'UPC',
       '_Vol', 'HOLIDAY_DUMMY',
       'RETAILER', 'COMPLIMENTS', 'SELECTION', 'TENDERFLAKE', 'WESTERN_FAMILY'], axis =1 )



    date_col = "DATE"
    id_columns = ['DESCRIPTION', 'MARKET_MEANING']
    fillmethod = "zeros"


    #df_test = aiqutils.data_preparation.fill_timeseries(df, id_columns, date_col, freq = "D", fillmethod = "zeros")
    df["Avg_Retail_Unit_Price"] = (df.Avg_Retail_Unit_Price - df.Avg_Retail_Unit_Price.mean())/df.Avg_Retail_Unit_Price.std()

    lag_col          = "DATE"
    numerical_cols   = ["Avg_Retail_Unit_Price"]
    categorical_cols = ["MARKET_MEANING", "DESCRIPTION"]
    lag_list         = [1,2,3]#[1,2,3,4,8]
    rolling_list     = [1,2,3]#[1,2,3,4,6,8,12,16,20,24,28,32]

    df_ewm = interact_categorical_numerical(
                                       df, lag_col, numerical_cols,
                                       categorical_cols, lag_list,
                                       rolling_list, agg_funct="sum",
                                       rolling_function = "ewm", freq=None,
                                       group_name=None, store_name=False)
    df_rolling = interact_categorical_numerical(
                                       df, lag_col, numerical_cols,
                                       categorical_cols, lag_list,
                                       rolling_list, agg_funct="sum",
                                       rolling_function = "rolling", freq=None,
                                       group_name=None, store_name=False)
    df_expansion = interact_categorical_numerical(
                                       df, lag_col, numerical_cols,
                                       categorical_cols, lag_list,
                                       rolling_list, agg_funct="sum",
                                       rolling_function = "expanding", freq=None,
                                       group_name=None, store_name=False)


    id_columns = ['DESCRIPTION', 'MARKET_MEANING', "DATE"]
    df= df.merge(df_ewm, on = id_columns )
    df= df.merge(df_expansion, on = id_columns )
    df= df.merge(df_rolling, on = id_columns )


    df["Unit_Vol"] = (df.Unit_Vol - df.Unit_Vol.mean())/df.Unit_Vol.std()

    df_kfolded = utils_exomodel.transform_KFold(df, "Unit_Vol", 5)

    for id_col in ['DESCRIPTION', "MONTH", "YEAR", "HOLIDAY_NAME"]:
      df_dummies = pd.get_dummies(df[id_col])
      df = df.merge(df_dummies, right_index = True, left_index = True)
      del df[id_col]

    del df["DATE"]
    #del df["DESCRIPTION"]
    del df["MARKET_MEANING"]

    df = df.dropna()
    df =  df.drop_duplicates()

    df_kfolded = utils_exomodel.transform_KFold(df, "Unit_Vol", 5)


    M= 4     
    N= 4

    neuronal_system, chromosome = utils_exomodel.build_neuronal_system(M, N, models_list)
    NEURONAL_SOLUTIONS          = utils_exomodel.decode_neuronal_system(neuronal_system, df, SKLEARN_MODELS)
    utils_exomodel.get_intramodel_chromosomes(NEURONAL_SOLUTIONS)
    #utils_exomodel.simulate_output(NEURONAL_SOLUTIONS, N, df_kfolded)
    #df_exmodel = utils_exomodel.get_models_output(NEURONAL_SOLUTIONS, df_kfolded) #runs everytime a layer is finished.



    #-------------------GENETIC ALGORITM PER MODEL ----------------------------------

    for n_col in NEURONAL_SOLUTIONS.keys():
        for m_row in NEURONAL_SOLUTIONS[n_col].keys():
            model_name = NEURONAL_SOLUTIONS[n_col][m_row]["generate_model_populationme"]
            model = NEURONAL_SOLUTIONS[n_col][m_row]["model"]
            neuron_name = NEURONAL_SOLUTIONS[n_col][m_row]["name"]




FIRST_ITERATION = True
N = 50
PC =.22   #Crossover probability
PM =.1  #Mutation probability
MAX_ITERATIONS = 15

N_WORKERS = 4
round_prediction = False
max_features = 1000

n_col = "N_0"


print("\n\nTEST1 KFOLDED MEANS: ")
for fold in df_kfolded.keys():
    print("FOLD", np.mean(df_kfolded[fold]["y"]))



ERROR_TYPES = {"MAE": mean_absolute_error, "MSE": mean_squared_error, "R2": r2_score}
POPULATION_test = utils_model_genetic.generate_model_population(df_kfolded, NEURONAL_SOLUTIONS, n_col, N, max_features)
INDIVIDUAL = POPULATION_test[0] 
error_type = "MAE"
MODEL = SKLEARN_MODELS["xgboost"]


print("\n\nTEST2 KFOLDED MEANS: ")
for fold in df_kfolded.keys():
    print("FOLD", np.mean(df_kfolded[fold]["y"]))


t1 = time.time()
POPULATION_X, SOLUTIONS =  utils_model_genetic.solve_genetic_algorithm( N, PC, PM, N_WORKERS, 
                            MAX_ITERATIONS, MODEL, NEURONAL_SOLUTIONS, n_col, df,df_kfolded, 
                            df_exmodel = None , max_features = max_features, round_prediction= round_prediction,
                            parallel_execution = True)
t2 = time.time()

utils_model_genetic.print_SOLUTIONS(SOLUTIONS)
equal_scores_keys = utils_model_genetic.test_equal_squares(SOLUTIONS)
utils_model_genetic.test_equal_squares_likelyness(equal_scores_keys, SOLUTIONS)
print("\n\nTIME: {} {}  == {} ".format(t1,t2, t2-t1))

SOLUTIONS[SOLUTIONS.keys()[len(SOLUTIONS)-1]]["baseline_features"] == SOLUTIONS[SOLUTIONS.keys()[len(SOLUTIONS)-2]]["baseline_features"]
SOLUTIONS.keys()[len(SOLUTIONS)-1]== SOLUTIONS.keys()[len(SOLUTIONS)-2]
