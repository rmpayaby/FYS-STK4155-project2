# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, roc_auc_score as AUC_score, r2_score
from sklearn import preprocessing

from Regression import Regression
from neural_net_regression import NeuralNetwork

import seaborn as sns

mpl.rcdefaults()
plt.style.use('seaborn-darkgrid')
mpl.rcParams['figure.figsize'] = [10.0, 4.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 18


# Reading data
df = pd.read_excel("D:/FYS-STK4155/project2/Oil_regression/oil_prices.xlsx")


# Start from 1991 where all datasets are available
X = df.iloc[132:,[1,2,5,6,9,11,12]].to_numpy()
y_prev = df.iloc[132:,[3]].to_numpy()
# Adding noise
m = np.mean(y_prev); s = np.std(y_prev)
y = y_prev + np.random.normal(m,s)


# Hyperparameters for tuning
hidden_neuron_list = [5,5,5]
epochs = 500
runs = 30
lr_rate = 0.01
lmbd = 0.001

# Calling the class function containing activaion and cost function
reg = Regression(hidden_activation='ReLU',output_activation="linear")


# Initialize storing values
r2_test_runs = np.zeros((runs,epochs))
r2_train_runs = np.zeros((runs,epochs))
r2_end_test = np.zeros(runs)
r2_end_train = np.zeros(runs)

MAPE_test_runs = np.zeros((runs,epochs))
MAPE_train_runs = np.zeros((runs,epochs))
MAPE_test_end = np.zeros(runs)
MAPE_train_end = np.zeros(runs)



for run in tqdm(range(runs)):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.2)
    Scaler = preprocessing.StandardScaler()
    X_train_scaled = Scaler.fit_transform(X_train)
    X_test_scaled = Scaler.transform(X_test)
    nn = NeuralNetwork( X_train_scaled,
                        Y_train,
                        problem=reg,
                        n_hidden_neurons_list=hidden_neuron_list,
                        n_output_neurons=1,
                        epochs=epochs,
                        batch_size=20,
                        lr_rate=lr_rate,
                        lmbd=lmbd)
    
    nn.SGD(metric=['r2'],test_data=(X_test_scaled,Y_test),train_data=(X_train_scaled,Y_train))
    #r2_test_runs[run,:] = nn.r2_test
    #r2_train_runs[run,:] = nn.r2_train
    #r2_end_test[run] = nn.r2_test[-1]
    #r2_end_train[run] = nn.r2_train[-1]
    
    
    """Used to run the MAPE-values"""
    MAPE_test_runs[run,:]= nn.MAPE_test
    MAPE_train_runs[run,:] = nn.MAPE_train
    MAPE_test_end[run] = nn.MAPE_test[-1]
    MAPE_train_end[run] = nn.MAPE_train[-1]

r2_mean_test = np.mean(r2_end_test)
r2_mean_train = np.mean(r2_end_train)

MAPE_mean_test = np.mean(MAPE_test_end); MAPE_mean_train = np.mean(MAPE_train_end)
fig,ax = plt.subplots()
for i in range(runs):
    #ax.plot(r2_train_runs[i,:],color='black',label='train, mean = {:.2f}'.format(r2_mean_train))
    #ax.plot(r2_test_runs[i,:],color='green',label='test, mean = {:.2f}'.format(r2_mean_test))
    ax.plot(MAPE_train_runs[i,:],color='black',label='train, mean = {:.2f}'.format(MAPE_mean_train))
    ax.plot(MAPE_test_runs[i,:],color='green',label='test, mean = {:.2f}'.format(MAPE_mean_test))
    if i == 0:
        ax.legend(loc=4)
ax.set_ylabel('MAPE score')
ax.set_xlabel('Epochs')
ax.set_ylim(0,20) # 0 to 1 if R2
fig.tight_layout()

plt.title("MAPE of oil data")
print('epochs',epochs,'runs',runs)
print('lr_rate',lr_rate,' lambda ',lmbd,' neuron list ',hidden_neuron_list)



"""
Grid search of hyperparameters

"""


eta_vals = np.logspace(-5, -1, 5)
lmbd_vals = np.logspace(-5, 1, 7)
# store the models for later use
DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

# grid search
for i, lr_rate in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        nn = NeuralNetwork( X_train_scaled,
                        Y_train,
                        problem=reg,
                        n_hidden_neurons_list=hidden_neuron_list,
                        n_output_neurons=1,
                        epochs=epochs,
                        batch_size=20,
                        lr_rate=lr_rate,
                        lmbd=lmbd)
        nn.SGD(test_data=(X_test_scaled,Y_test),train_data=(X_train_scaled,Y_train))
        
        DNN_numpy[i][j] = nn
        
        test_predict = nn.predict_proba(X_test_scaled)

        print("Learning rate  = ", lr_rate)
        print("Lambda = ", lmbd)
        print("R2 score on test set: ", r2_score(Y_test,test_predict))
        print()


sns.set()

test_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))

for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        nn = DNN_numpy[i][j]
        
        test_pred = nn.predict_proba(X_test_scaled)
        test_r2[i][j] = r2_score(Y_test, test_pred)

        
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_r2, annot=True, ax=ax, cmap="viridis")
ax.set_title("Grid search of R2-score, Sigmoid")
ax.set_ylabel("Learning rate")
ax.set_xlabel("Regularization parameter")
plt.show()


def MAPE(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

#print(y_predict)

#print(MAPE(Y_test,y_predict))



