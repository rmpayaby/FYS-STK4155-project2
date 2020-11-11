# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 12:48:23 2020

@author: rmbp
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler


import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn import preprocessing
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from tqdm import trange

from Regress import *
from results import *
from Resampling import *

import datetime
import time
import seaborn as sns


# This is for showing the distribution oil data
"""
#df = pd.read_excel("D:/FYS-STK4155/project2/Oil_regression/oil_prices.xlsx")
#X = df.iloc[132:,[1,2,5,6,9,11,12]]
#y_prev = df.iloc[132:,[3]].to_numpy() 
#m = np.mean(y_prev); s = np.std(y_prev)
#y = y_prev + np.random.normal(35, s)
#print(X.shape)
"""



# Showing distribution of data to the depression data set 
df = pd.read_csv("D:/FYS-STK4155/train.csv")


X = df[["femaleres","age","married","children","edu","ent_wagelabor","durable_investment","fs_adskipm_often"]]
y = df["depressed"]
df.info()

plt.figure(figsize=(9, 9))
correlation = X.corr()
heatmap = sns.heatmap(correlation, annot=True,fmt='.2f')
plt.title("Correlation matrix of depression")
plt.show()


attributes = ["Crude Oil, Brent","Log-return Crude Oil, Brent",
              "Australian thermal coal","Log-return Coal, ATC",
              "Russian Gas border proce","Gas spot price, HHB",
              "Log-return Gas spot, HHB","Crude Oil, WTI"]

attributes4 = ["femaleres","age","married","children","edu","ent_wagelabor","durable_investment","fs_adskipm_often","depressed"]


for i in attributes4:
    plt.title(i)
    sns.set_style('darkgrid')
    sns.distplot(df[i])
    plt.figure()

    



