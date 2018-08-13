

import pandas as pd 
import xgboost as xgb 



"""
This script test a new mrthodology to find the optimal hyperparameters for an XGBOOST model



* Tree-Specific Parameters: These affect each individual tree in the model.
min_samples_split
	Defines the minimum number of samples (or observations) which are required in a node to be considered for splitting.
	Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
	Too high values can lead to under-fitting hence, it should be tuned using CV.

min_samples_leaf
	Defines the minimum samples (or observations) required in a terminal node or leaf.
	Used to control over-fitting similar to min_samples_split.
	Generally lower values should be chosen for imbalanced class problems because the regions in which the minority class will be in majority will be very small.

min_weight_fraction_leaf
	Similar to min_samples_leaf but defined as a fraction of the total number of observations instead of an integer.
	Only one of #2 and #3 should be defined

max_depth
	The maximum depth of a tree.
	Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
	Should be tuned using CV.

max_leaf_nodes
	The maximum number of terminal nodes or leaves in a tree.
	Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves.
	If this is defined, GBM will ignore max_depth.

max_features
	The number of features to consider while searching for a best split. These will be randomly selected.
	As a thumb-rule, square root of the total number of features works great but we should check upto 30-40% of the total number of features.
	Higher values can lead to over-fitting but depends on case to case.



* Boosting Parameters: These affect the boosting operation in the model.

learning_rate
	This determines the impact of each tree on the final outcome (step 2.4). GBM works by starting with an initial estimate which is updated using the output of each tree. The learning parameter controls the magnitude of this change in the estimates.
	Lower values are generally preferred as they make the model robust to the specific characteristics of tree and thus allowing it to generalize well.
	Lower values would require higher number of trees to model all the relations and will be computationally expensive.

n_estimators
	The number of sequential trees to be modeled (step 2)
	Though GBM is fairly robust at higher number of trees but it can still overfit at a point. Hence, this should be tuned using CV for a particular learning rate.

subsample
	The fraction of observations to be selected for each tree. Selection is done by random sampling.
	Values slightly less than 1 make the model robust by reducing the variance.
	Typical values ~0.8 generally work fine but can be fine-tuned further.


* Miscellaneous Parameters: Other parameters for overall functioning.

loss 
	It refers to the loss function to be minimized in each split.
	It can have various values for classification and regression case. Generally the default values work fine. Other values should be chosen only if you understand their impact on the model.

init
	This affects initialization of the output.
	This can be used if we have made another model whose outcome is to be used as the initial estimates for GBM.

random_state ******IMPORTANT*****
The random number seed so that same random numbers are generated every time.
	This is important for parameter tuning. If we don’t fix the random number, then we’ll have different outcomes for subsequent runs on the same parameters and it becomes difficult to compare models.
	It can potentially result in overfitting to a particular random sample selected. We can try running models for different random samples, which is computationally expensive and generally not used.

verbose
	The type of output to be printed when the model fits. The different values can be:
	0: no output generated (default)
	1: output generated for trees in certain intervals
	>1: output generated for all trees

warm_start
	This parameter has an interesting application and can help a lot if used judicially.
	Using this, we can fit additional trees on previous fits of a model. It can save a lot of time and you should explore this option for advanced applications

presort 
 	Select whether to presort data for faster splits.
	It makes the selection automatically by default but it can be changed if needed.



There are two types of parameter to be tuned here – tree based and boosting parameters. 
There are no optimum values for learning rate as low values always work better,
given that we train on sufficient number of trees.



Boosting Parameters Recomendations:

	Choose a relatively high learning rate. 
	Generally the default value of 0.1 works but somewhere between 0.05 to 0.2 should work for different problems

	Determine the optimum number of trees for this learning rate. 
	This should range around 40-70. Remember to choose a value on which your system can work fairly fast. This is because it will be used for testing various scenarios and determining the tree parameters.

	Tune tree-specific parameters for decided learning rate and number of trees.
	Note that we can choose different parameters to define a tree and I’ll take up an example here.

	Lower the learning rate and increase the estimators proportionally to get more robust models.


Tree Parameters Recomendations:
	In order to decide on boosting parameters, we need to set some initial values of other parameters. Lets take the following values:

	min_samples_split = 500 : 
	This should be ~0.5-1% of total values. Since this is imbalanced class problem,
	 we’ll take a small value from the range.

	min_samples_leaf = 50 : 
	Can be selected based on intuition. This is just used for preventing overfitting and again a small value because of imbalanced classes.

	max_depth = 8 : 
	Should be chosen (5-8) based on the number of observations and predictors. This has 87K rows and 49 columns so lets take 8 here.

	max_features = ‘sqrt’ : 
	Its a general thumb-rule to start with square root.

	subsample = 0.8 : 
	This is a commonly used used start value




"""








