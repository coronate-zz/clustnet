import pandas as pd
import numpy as np
import random
from xgboost import XGBRegressor
from xgboost import plot_importance


def get_exochromoesome(df, models):
	bits_per_model = int(np.ceil(np.log2(len(models))))
	chromosome = ""
	for n_col in df.columns:
		for m_row in df.index:
			model = df.loc[m_row, n_col]
			model_position = models.index(model)
			chromosome = chromosome + "" + np.binary_repr(model_position, width= bits_per_model )
	return chromosome


def build_neuronal_system(M, N, models):
	n_columns =  list()
	for i in range(N):
		n_name = "N_" +str(i)
		n_columns.append(n_name)
	m_rows =  list()
	for j in range(M):
		m_name = "M_"+str(j)
		m_rows.append(m_name)

	neuronal_system = pd.DataFrame(columns =  n_columns, index = m_rows)
	for n_col in n_columns:
		for m_row in m_rows:
			neuronal_system.loc[m_row, n_col] = random.choice(models)

	chromosome = get_exochromoesome(neuronal_system, models)

	return neuronal_system, chromosome


def decode_neuronal_system(neuronal_system, df):
	"""
	This funtion transforms the neuronal_system into a set of models that must be trained 
	in order to score the neuronal_system.
	"""
	NEURONAL_SOLUTIONS = dict()
	for i in range(len(neuronal_system.columns)):
		n_col = neuronal_system.columns[i]
		NEURONAL_SOLUTIONS[n_col] = dict()
		for j in range(len(neuronal_system.index)):
			m_row = neuronal_system.index[j]

			model = neuronal_system.loc[m_row, n_col]
			NEURONAL_SOLUTIONS[n_col][m_row] = dict()
			NEURONAL_SOLUTIONS[n_col][m_row]["model_name"] = model
			NEURONAL_SOLUTIONS[n_col][m_row]["model"] = fit_models[model]
			NEURONAL_SOLUTIONS[n_col][m_row]["name"] = model + "_" + str(n_col) + "_" + str(m_row)

			#For each model we want to test the features that should be included in the model. 
			features_number = len(df.columns)
			baseline_fetaures = list()
			baseline_fetaures_chormosome = ""
			for f in range(features_number):
				feature = df.columns[f]
				isincluded = random.choice([True,False])
				if isincluded:
					baseline_fetaures.append(feature)
					baseline_fetaures_chormosome += "1"
				else:
					baseline_fetaures_chormosome += "0"

			NEURONAL_SOLUTIONS[n_col][m_row]["baseline_fetaures"] =  baseline_fetaures
			NEURONAL_SOLUTIONS[n_col][m_row]["baseline_fetaures_chormosome"] =  baseline_fetaures_chormosome
			NEURONAL_SOLUTIONS[n_col][m_row]["exmodel_fetaures"] =  dict()
			

			#Include exmodel features for layers>0 
			for layer in neuronal_system.columns:
				NEURONAL_SOLUTIONS[n_col][m_row]["exmodel_fetaures"][layer] = dict()
				if i <= int(layer.replace("N_", "")) or i ==0:
					NEURONAL_SOLUTIONS[n_col][m_row]["exmodel_fetaures"][layer]["features"] = []
					NEURONAL_SOLUTIONS[n_col][m_row]["exmodel_fetaures"][layer]["chromosome"] = ""
				else:

					exmodel_fetaures = list()
					exmodel_fetaures_chormosome = ""
					for j_2 in range(len(neuronal_system.index)):
						m_row_asfeature = neuronal_system.index[j_2]
						feature = NEURONAL_SOLUTIONS[layer][m_row_asfeature]["name"]
						isincluded = random.choice([True,False])

						if isincluded and NEURONAL_SOLUTIONS[layer][m_row_asfeature]["model_name"] != "":
							exmodel_fetaures.append(feature)
							exmodel_fetaures_chormosome += "1"
						else:
							exmodel_fetaures_chormosome += "0"

					NEURONAL_SOLUTIONS[n_col][m_row]["exmodel_fetaures"][layer]["features"]   = exmodel_fetaures
					NEURONAL_SOLUTIONS[n_col][m_row]["exmodel_fetaures"][layer]["chromosome"] = exmodel_fetaures_chormosome

	return NEURONAL_SOLUTIONS


def decode_exochromosome(M, N , chromosome, models):
	bits_per_model = int(np.ceil(np.log2(len(models))))

	n_columns =  list()
	for i in range(N):
		n_name = "N_" +str(i)
		n_columns.append(n_name)
	m_rows =  list()
	for j in range(M):
		m_name = "M_"+str(j)
		m_rows.append(m_name)

	neuronal_system = pd.DataFrame(columns =  n_columns, index = m_rows)
	for i in range(len(n_columns)):
		n_col =  n_columns[i]
		for j in range(len(m_rows)):
			m_row = m_rows[j]
			neuronal_system.loc[m_row, n_col] = chromosome[(i*len(m_rows)*bits_per_model)+ (j*bits_per_model) : (i*len(m_rows)*bits_per_model)+ (j*bits_per_model) +bits_per_model]

	return chromosome



df = pd.read_csv('/home/coronado/Desktop/clustnet/Data/train.csv')
Y = df.Survived
del df["Survived"]
models = ["", "XGBoost", "LR", "RandomForest", "LGBM"]
M= 4
N= 4

neuronal_system, chromosome = build_neuronal_system(M, N, models)

fit_models = {"":np.nan , 
			  "XGBoost":XGBRegressor(max_depth=14, 
                        learning_rate=0.15, #learning_rate=0.05
                        n_estimators=800, 
                        silent=True, 
                        objective='reg:linear', 
                        nthread=32, 
                        gamma=0,
                        min_child_weight=1, 
                        max_delta_step=0, 
                        subsample=0.85, 
                        colsample_bytree=0.7, 
                        colsample_bylevel=1, 
                        reg_alpha=0, 
                        reg_lambda=1, 
                        scale_pos_weight=1, 
                        seed=1440, 
                        missing=None) , 
			  "LR":1,
			  "RandomForest":2,
			  "LGBM": 3}



NEURONAL_SOLUTIONS = decode_neuronal_system(neuronal_system, df)
