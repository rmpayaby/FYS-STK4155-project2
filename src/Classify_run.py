# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:05:33 2020

@author: rmbp
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import floor
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, roc_auc_score as AUC_score, r2_score
from sklearn import preprocessing

from Classify import Classify
from neural_net_regression import NeuralNetwork

mpl.rcdefaults()
plt.style.use('seaborn-darkgrid')
mpl.rcParams['figure.figsize'] = [10.0, 4.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 18


#nn_structure = [8,7,2]

def convert_y(c_arr):
    ca = c_arr
    num_cat = np.max(ca) + 1
    num_points = len(ca)
    one_hot = np.zeros((num_points,num_cat),dtype=int)
    one_hot[range(num_points),ca] = 1
    return one_hot

df = pd.read_csv('D:/FYS-STK4155/project2/b_depressed.csv',sep=',')

input_data = df[["sex","Age","Married","Number_children","education_level",
       "living_expenses","incoming_salary","lasting_investment"]].to_numpy()
output_data = df["depressed"].to_numpy()

output_one_hot = convert_y(output_data) 

hidden_neuron_list = [7,7]
epochs = 100
runs = 10
lr_rate = 1e-3
lmbd = 0

accu_test = np.zeros((runs,epochs))
accu_train = np.zeros((runs,epochs))
clf = Classify(hidden_activation='sigmoid',output_activation='softmax')


auc = []; acc_test = []; acc_train = []


for i in tqdm(range(runs)):
    X_train, X_test, Y_train, Y_test = train_test_split(input_data,
                                                        output_one_hot,
                                                        test_size = 0.2)
    Scaler = preprocessing.StandardScaler()
    X_train_scale = Scaler.fit_transform(X_train)
    X_test_scale = Scaler.transform(X_test)
    
    nn = NeuralNetwork(X_train_scale,
                       Y_train,
                       problem=clf,
                       n_hidden_neurons_list=hidden_neuron_list,
                       n_output_neurons=2,
                       epochs=epochs,
                       batch_size=100,
                       lr_rate=lr_rate,
                       lmbd=lmbd,
                       TypeOfProblem="Classification")
    
    nn.SGD_classify(test_scale=X_test_scale,train_scale=X_train_scale,
                    test_y = Y_test, train_y=Y_train)
    
    acc_test.append(nn.accuracy_test[-1])
    acc_train.append(nn.accuracy_train[-1])
    auc.append(nn.auc_test[-1])
    accu_test[i,:] = nn.accuracy_test; accu_train[i,:] = nn.accuracy_train
    
auc_mean = np.mean(auc); mean_accuracy = np.mean(acc_test)
mean_accuracy_train = np.mean(acc_train)

print('AUC mean = ',auc_mean, ' accuracy mean = ',mean_accuracy)

fig, ax = plt.subplots()
for i in range(runs):
    ax.plot(accu_test[i,:],color='green',label='test, mean = {:.2f}'.format(mean_accuracy))
    ax.plot(accu_test[i,:],color='black',label='test, mean = {:.2f}'.format(mean_accuracy_train))
    
    if i == 0:
        ax.legend(loc=0)
        
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epochs')
ax.set_ylim(0.6,1)
fig.tight_layout()

 