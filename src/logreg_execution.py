# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:06:42 2020

@author: rmbp
"""


import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.datasets import load_wine, load_digits
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from LogReg import LogReg_SGD
import seaborn as sns


def accuracy_score(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)

model = LogReg_SGD()


#scale = StandardScaler()
df = pd.read_csv('D:/FYS-STK4155/project2/b_depressed.csv',sep=',')
X = df[["sex","Age","Married","Number_children","education_level",
       "living_expenses","incoming_salary","lasting_investment"]].to_numpy()
y = df["depressed"].to_numpy() 


print(y.size), print(X.shape[1])


epochs = np.array([100, 200, 400, 800, 1600, 3200, 6400, 12800])
lr_rate = np.array([0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4, -4.5, -5])

nr_averages = 40
acc_mat = np.zeros((len(epochs), len(lr_rate)))
area_mat = np.zeros((len(epochs), len(lr_rate)))

for k in range(len(lr_rate)):
    for j in range(len(epochs)):
        acc = 0
        area = 0
        for i in range(nr_averages):
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
            Scaler = preprocessing.StandardScaler()
            X_train_scaled = Scaler.fit_transform(X_train)
            X_test_scaled = Scaler.transform(X_test)
            logreg = LogReg_SGD()
            X_train_scaled, Y_train = logreg.create_minibatch(X_train_scaled,Y_test,epochs[j])
            logreg.fit(X_train_scaled, Y_train, epochs[j], 10**lr_rate[k])
            Y_pred = logreg.predict_proba(X_test_scaled)
            acc += accuracy_score(Y_test, Y_pred)
            area += roc_auc_score(Y_test, Y_pred)

        acc_mat[j,k] = acc/nr_averages
        area_mat[j,k] = area/nr_averages
        
fig, ax = plt.subplots(1, 1, figsize=(10,6))
plt.title("Depression detection, AUC-score")
sns.heatmap(area_mat, ax=ax, annot=True, fmt=".2f", vmin=0.48, vmax=0.52, xticklabels=lr_rate, yticklabels=epochs, cmap="Greens", square=True)
ax.set_xlabel("Log10(Learning Rate)")
ax.set_ylabel("Epochs")
plt.ylim(0, len(epochs));
ax.set_yticklabels(epochs, rotation = 45, ha="right")
plt.tight_layout()


