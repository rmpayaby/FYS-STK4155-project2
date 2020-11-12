# FYS-STK4155-project2
This repository contains the codes used in project 2. 

The results from sample runs are to be found in the folder plots_for_sample_runs. 

Data sets for the regression and classification problem are stored in the data_sets-folder. The data sets I've utilized in the project includes historical data of oil prices provided by the International Monetary Fund for regression problems, and a survey of approximately 1000 people in Siyaya county, Kenya about their living conditions and status of depression. The classification set is called train.csv.

All codes used are to be found in the src-folder. Since I've proposed my own data sets, OLS and Ridge regression from project 1 was reused to pinpoint what to expect during the regression analysis of the oil price data set. 

A simple SGD-algorithm prior to applying it to neural networks can be found on <strong>sgd_experiment.py</strong>.
To show the distribution of features in a data set, use <strong>distrib_data.py</strong>

The following codes are relevant for each setting:

Neural network regression: <strong>neural_net_regression.py</strong>, <strong>Regression.py</strong>, <strong>Regression_run.py</strong>

Neural network classification: <strong>Classify.py</strong>, <strong>Classify_run.py</strong>

Logistic regression: <strong>LogReg.py</strong>, <strong>logreg_execution.py</strong>

Sample codes of neural networks using Keras and Scikit can be found in the folder testing_code_with_scikit_keras inside src
