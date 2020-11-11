# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:27:41 2020

@author: rmbp

Inspirations and sources
https://compphysics.github.io/MachineLearning/doc/pub/week41/html/week41.html

https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
"""


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE, roc_auc_score as AUC_score, r2_score


"""
Define the parameters
"""

eta_vals = np.logspace(-5, -1, 5)
lmbd_vals = np.logspace(-5, 1, 7)
epochs = 1000
n_hidden_neurons = 5


DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

df = pd.read_excel("D:/FYS-STK4155/project2/Oil_regression/oil_prices.xlsx")


# Start from 1991 where all datasets are available
X = df.iloc[132:,[1,2,5,6,9,11,12]].to_numpy()
y_prev = df.iloc[132:,[3]].to_numpy()
# Adding noise
m = np.mean(y_prev); s = np.std(y_prev)
y = y_prev + np.random.normal(m,s)


X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2)



for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPRegressor(hidden_layer_sizes=(n_hidden_neurons), activation='relu',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs, solver = 'adam')
        dnn.fit(X_train, Y_train)
        
        DNN_scikit[i][j] = dnn
        
        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("R2-score on test set: ", dnn.score(X_test, Y_test))
        print()
        
        
        
sns.set()

train_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))
test_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))

for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        dnn = DNN_scikit[i][j]
        
        train_pred = dnn.predict(X_train) 
        test_pred = dnn.predict(X_test)

        train_r2[i][j] = r2_score(Y_train, train_pred)
        test_r2[i][j] = r2_score(Y_test, test_pred)

        
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_r2, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training R2-score")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_r2, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test R2-score")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()