import pandas as pd
import numpy as np
import random
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

df = pd.read_csv('/home/coronado/Desktop/clustnet/Data/df_elglobo_sample.csv')
df = df.groupby(["Agencia_ID","Product_ID","FechaCal"]).sum()
df = df.reset_index()

df =  df.drop_duplicates()
df["Product_ID"] = df.Product_ID.apply(lambda x:"producto_" + str(x))
df["Agencia_ID"] = df.Agencia_ID.apply(lambda x:"agencia_" + str(x))
df["FechaCal"] = pd.to_datetime(df.FechaCal, format = "%Y-%m-%d")


date_col = "FechaCal"
id_columns = ['Product_ID', 'Agencia_ID']
fillmethod = "zeros"


#df_test = aiqutils.data_preparation.fill_timeseries(df, id_columns, date_col, freq = "D", fillmethod = "zeros")

lag_col          = "FechaCal"
numerical_cols   = ["PrecioBruto"]
categorical_cols = ["Agencia_ID", "Product_ID"]
lag_list         = [1,2,3]
rolling_list     = [1,2,3,4,8]

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


id_columns = ['Product_ID', 'Agencia_ID', "FechaCal"]
df= df.merge(df_ewm, on = id_columns )
df= df.merge(df_expansion, on = id_columns )
df= df.merge(df_rolling, on = id_columns )

df = df.drop( id_columns, axis = 1)
df = df.dropna()

df_kfolded = utils_exomodel.transform_KFold(df, "VentaUnidades", 2)


M= 4     
N= 4

neuronal_system, chromosome = utils_exomodel.build_neuronal_system(M, N, models_list)
NEURONAL_SOLUTIONS          = utils_exomodel.decode_neuronal_system(neuronal_system, df, SKLEARN_MODELS)
utils_exomodel.get_intramodel_chromosomes(NEURONAL_SOLUTIONS)
utils_exomodel.simulate_output(NEURONAL_SOLUTIONS, N, df_kfolded)
df_exmodel = utils_exomodel.get_models_output(NEURONAL_SOLUTIONS, df_kfolded) #runs everytime a layer is finished.



#-------------------GENETIC ALGORITM PER MODEL ----------------------------------

for n_col in NEURONAL_SOLUTIONS.keys():
    for m_row in NEURONAL_SOLUTIONS[n_col].keys():
        model_name = NEURONAL_SOLUTIONS[n_col][m_row]["model_name"]
        model = NEURONAL_SOLUTIONS[n_col][m_row]["model"]
        neuron_name = NEURONAL_SOLUTIONS[n_col][m_row]["name"]




GENLONG = 15
FIRST_ITERATION = True
N = 20
PC =.33   #Crossover probability
PM =.16  #Mutation probability
MAX_ITERATIONS = 10

GENOMA_TEST = 783
N_WORKERS = 16
round_prediction = True
max_features = 1000


ERROR_TYPES = {"MAE": mean_absolute_error, "MSE": mean_squared_error, "R2": r2_score}
POPULATION_test = utils_model_genetic.generate_model_population(df_kfolded, NEURONAL_SOLUTIONS, n_col, N, max_features)
INDIVIDUAL = POPULATION_test[0] 
model_name = "LR"
error_type = "MAE"
MODEL = SKLEARN_MODELS["linear"]
POPULATION_X, SOLUTIONS =  utils_model_genetic.solve_genetic_algorithm( N, PC, PM, N_WORKERS, 
							 MAX_ITERATIONS, MODEL,
                             NEURONAL_SOLUTIONS, n_col, df,df_kfolded, df_exmodel,
                             max_features, round_prediction)


