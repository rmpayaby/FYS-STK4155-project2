# FYS-STK4155-project2
This repository contains the codes used in project 2. 

Sample runs are to be found in the folder plots_for_sample_runs 

Data sets for the regression and classification problem are stored in the data_sets-folder. The data sets I've utilized in the project includes historical data of oil prices provided by the International Monetary Fund for regression problems, and a survey of approximately 1000 people in Siyaya county, Kenya about their living conditions and status of depression. The classification set is called ####train.csv.

All codes used are to be found in the src-folder. Since I've proposed my own data sets, OLS and Ridge regression from project 1 was reused to pinpoint what to expect during the regression analysis of the oil price data set. 

A simple SGD-algorithm prior to applying it to neural networks can be found on ####sgd_experiment.py.
To show the distribution of features in a data set, use ####distrib_data.py

The following codes are relevant for each setting:

Neural network regression: ####neural_net_regression.py, ####Regression.py, ####Regression_run.py

Neural network classification: ####Classify.py, ####Classify_run.py

Logistic regression: ####LogReg.py, ####logreg_execution.py

Sample codes of neural networks using Keras and Scikit can be found in the folder testing_code_with_scikit_keras.

