# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:24:16 2020

@author: rmbp
"""

import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_boston
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error,r2_score
from numpy import random
from sklearn.model_selection import train_test_split
print("DONE")


df = pd.read_excel("D:/FYS-STK4155/project2/Oil_regression/oil_prices.xlsx")
y_prev =df.iloc[132:,[3]].to_numpy() 

# Add some noise
m = np.mean(y_prev); s = np.std(y_prev)
Y = y_prev + np.random.normal(35, s)


# Start from 1991 where all datasets are available
X =df.iloc[132:,[1,2,5,6,9,11,12]].to_numpy()
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)



# Priniting out the shapes 
print("X Shape: ",X.shape)
print("Y Shape: ",Y.shape)
print("X_Train Shape: ",x_train.shape)
print("X_Test Shape: ",x_test.shape)
print("Y_Train Shape: ",y_train.shape)
print("Y_Test Shape: ",y_test.shape)

# standardizing data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test=scaler.transform(x_test)

## Adding the target value column in the data
train_data= pd.DataFrame(x_train)
train_data['Crude Oil, WTI']= y_train
train_data.head(3)


test_data = pd.DataFrame(x_test)
test_data['Crude Oil, WTI']= y_test

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test= np.array(x_test)
y_test= np.array(y_test) 




def CustomSGD(train_data,learning_rate,n_iter,k,divide=1,C=0):
    
    # Keep  weughts and biases as 0 as per the Training Data
    w=np.zeros(shape=(1,train_data.shape[1]-1))
    b=0
    cur_iter=1
    while(cur_iter<=n_iter): 

        # Create a small training data set of size K
        temp=train_data.sample(k)
        
        # Create X and Y from the above temp dataset
        y = np.array(temp['Crude Oil, WTI'])
        x = np.array(temp.drop('Crude Oil, WTI',axis=1))
        
        
        # Keeps the initial gradients as 0
        w_gradient= np.zeros(shape=(1,train_data.shape[1]-1))
        b_gradient= 0
        
        for i in range(k): # Calculating gradients for point in our K sized dataset
            prediction=np.dot(w,x[i])+b
            w_gradient= w_gradient+ (-2)*x[i]*(y[i]-(prediction)) -2*C*w_gradient
            b_gradient=b_gradient+(-2)*(y[i]-(prediction)) # Add bias gradient
        
        #Updating the weights(W) and Bias(b) with the above calculated Gradients
        w = w-learning_rate*(w_gradient/k)
        b = b-learning_rate*(b_gradient/k)
        
        # Incrementing the iteration value
        cur_iter=cur_iter+1
        
        #Dividing the learning rate by the specified value if wanted
        learning_rate = learning_rate/divide

        
    return w,b 



def predict(x,w,b):
    y_pred=[]
    for i in range(len(x)):
        y =np.asscalar(np.dot(w,x[i])+b)
        y_pred.append(y)
    return np.array(y_pred)

#np.random.seed(293)

"""
plt.scatter(y_test,y_pred_customsgd)
plt.grid()
plt.xlabel('Actual oil price (USD per barrel)')
plt.ylabel('Predicted oil price(USD per barrel)')
plt.title('Scatter plot from actual oil price vs. predicted oil price')
plt.show()
"""

# Calculating average R2-score 
Runs = 50
r2_OLS = np.zeros(50)
for i in range(Runs):
    w,b= CustomSGD(train_data,learning_rate=0.01,n_iter=1000,k=10,C=0.1)
    y_pred_customsgd = predict(x_test,w,b)
    y_tilde = predict(x_train,w,b)
    
    
    r2_OLS[i] += r2_score(y_test, y_pred_customsgd)
    
print(np.mean(r2_OLS))



epochs = np.array([i for i in range(0,110,10)])
#epochs = np.arange(0,110,10)
penalty = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

MSEs = []; R2s = []; R2s_train = []; MSEs_train = []; MAPS = []


def MAPE(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


for i in epochs:
    w,b = CustomSGD(train_data,learning_rate=0.01,n_iter=i, k=10,C=0.1)
    y_pred_customsgd = predict(x_test,w,b)
    y_tilde_customsgd = predict(x_train,w,b)
    
    MSEs.append(MAPE(y_test, y_pred_customsgd))
    MSEs_train.append(MAPE(y_train,y_tilde_customsgd))
    R2s.append(r2_score(y_test,y_pred_customsgd))
    R2s_train.append(r2_score(y_train,y_tilde_customsgd))
    
print(min(MSEs)); print(min(MSEs_train))
print(max(R2s)); print(max(R2s_train))
    
    

plt.figure()

plt.title("Epochs vs loss function")
plt.ylabel("MAPE"); plt.xlabel(r"lambda value")
plt.plot(epochs,MSEs,label="Test set")
plt.plot(epochs, MSEs_train,label="Train set")
plt.legend()


#plt.figure()

"""
plt.title("R2 vs loss function with penalty")
plt.ylabel("R2"); plt.xlabel(r"Epochs")
plt.plot(epochs,R2s,label="Test set")
plt.plot(epochs, R2s_train,label="Train set")
plt.legend()
"""
