# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:06:23 2020

@author: rmbp
"""


import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from random import shuffle
import matplotlib.pyplot as plt

import list_of_func as lof

class Resampling:
    
    """
    This class contains the following resampling methods:
        
        train_test: A simple splitting of data between training and
        testing set. (Default test size: 20 %)
        
        
        kfold_cross_val: Splitting data into k-folds with shuffling
        (Default fold number: 5)
        
        sklearns_kfold: A kfold algorithm defined by scikit-learn
        Some runtime issues was experienced when running the terrain data,
        thus this was used as a reference. The default fold number
        remains the same.
        
        bootstrap: This method chooses a number of bootstraps
        to perform (default value: 100 bootstraps).
        Then, a sample size is chosen. For each bootstrap
        sample, a sample with replacement with the chosen
        size is drawn. The mean of the calculated samples is 
        then estimated at the end. 
        
    """

    def __init__(self, X, z):
        self.X = X.astype("float64")
        self.z = z.astype("float64")


    def train_test(self, model,test_size = 0.2):

        #split the data in training and test.
        # Remove intercept for scaling data
        X = self.X[:,1:]
        
        X_train, X_test, z_train, z_test = train_test_split(X, self.z, test_size = test_size)
        
        # Scaling the data and then add the intercept
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        ones1 = np.ones((X_train.shape[0],1))
        ones2 = np.ones((X_test.shape[0],1))
        
        X_train = np.hstack((ones1,X_train))
        X_test = np.hstack((ones2,X_test))

        #fit the model on the train data
        model.fit(X_train, z_train)

        #predict on the train data and test data
        z_tilde = model.predict(X_train)
        z_pred = model.predict(X_test)
        
                
        #calculate errors
        bias = np.mean((z_test - np.mean(z_pred))**2)
        variance = np.var(z_pred)        
        mse = lof.MSE(z_test,z_pred)
        r2 = lof.R_score(z_test,z_pred)
        mse_train = lof.MSE(z_train,z_tilde)

        # Calculate error for scaled data 
        return mse, bias, variance, r2, mse_train
    
    
    def kfold_cross_val(self,model,num_folds=5):
        
        X = np.array(self.X[:,1:]); z = np.array(self.z)
    
        def kfold_idx_generator(arr_len, num_folds, randomize = True):
            """
            This function generates K=num_folds (training,testing)-pairs
            from the items in the design matrix X.
            
            Each pair is a partition of X, where the test set is an iterable
            of len(X)/K. 
            
            When randomized, a copy of X is shuffled before partitioning
             
            """
            if randomize: arr_len=list(arr_len); shuffle(arr_len)
            for k in range(num_folds):
                train_idx = [x for i, x in enumerate(arr_len) if i % num_folds != k]
                test_idx = [x for i, x in enumerate(arr_len) if i % num_folds == k]
                yield train_idx, test_idx
       
        
        #Initializing values
        mse_sum = 0
        mse_train_sum = 0
        bias_sum = 0 
        variance_sum = 0 
        r2_sum = 0
        
        
        # Calling the function is similar to Scikit's kFold.split
        X_len = range(len(X))
        for train_idx, test_idx in kfold_idx_generator(X_len, num_folds=5):
            for x in X_len: assert (x in train_idx) ^ (x in test_idx), x
            X_train, X_test = X[train_idx], X[test_idx]
            z_train, z_test = z[train_idx], z[test_idx]
            
            scaler = StandardScaler(with_std=False, with_mean=True)
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
            ones1 = np.ones((X_train.shape[0],1))
            ones2 = np.ones((X_test.shape[0],1))
            
            X_train = np.hstack((ones1,X_train))
            X_test = np.hstack((ones2,X_test))
         
            
            model.fit(X_train,z_train)
            z_pred = model.predict(X_test)
            z_tilde = model.predict(X_train)
            
            bias = np.mean((z_test - np.mean(z_pred))**2)
            variance = np.var(z_pred)
            mse_test = lof.MSE(z_test,z_pred)
            
            r2 = lof.R_score(z_test, z_pred)
            mse_train = lof.MSE(z_train,z_tilde)
            
            mse_sum += mse_test
            bias_sum += bias
            variance_sum += variance
            mse_train_sum += mse_train
            r2_sum += r2
            
        # Calculating mean value 
        mse_avg = mse_sum/num_folds
        bias_avg = bias_sum/num_folds
        var_avg = variance_sum/num_folds
        mse_train_avg = mse_train_sum/num_folds
        r2_avg = r2_sum/num_folds
        
        return mse_avg, bias_avg, var_avg, r2_avg, mse_train_avg
    
    
    
    def sklearns_kfold(self, model, num_folds=5):
        """Used to check for terrain data to reduce runtime"""
        kf = KFold(n_splits=5,shuffle=True)
        
        
        X = np.array(self.X); z = np.array(self.z)
        
        mse_sum = 0
        mse_train_sum = 0
        bias_sum = 0 
        variance_sum = 0 
        r2_sum = 0
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            z_train, z_test = z[train_idx], z[test_idx]
            
            model.fit(X_train,z_train)
            z_pred = model.predict(X_test)
            z_tilde = model.predict(X_train)
            
            mse_test = mean_squared_error(z_test,z_pred)
            bias = np.mean((z_test - np.mean(z_pred))**2)
            variance = np.var(z_pred)
            r2 = r2_score(z_test, z_pred)
            mse_train = np.mean((z_train - z_tilde)**2)
            
            mse_sum += mse_test
            bias_sum += bias
            variance_sum += variance
            mse_train_sum += mse_train
            r2_sum += r2
            
        # Calculating mean value 
        mse_avg = mse_sum/num_folds
        bias_avg = bias_sum/num_folds
        var_avg = variance_sum/num_folds
        mse_train_avg = mse_train_sum/num_folds
        r2_avg = r2_sum/num_folds
        
        return mse_avg, bias_avg, var_avg, r2_avg, mse_train_avg
    
    
    
    def bootstrap(self, model, no_boots = 100, test_size=0.2):
        
        #Again, split the training and test data with scaling
        X = np.array(self.X); z = np.array(self.z)
         
        X_train, X_test, z_train, z_test = train_test_split(X, self.z,test_size=test_size)
        

        # Defining the size of sample
        size_of_sample = X_train.shape[0]
        
        # setting up arrays for storing the values
        z_pred = np.empty((z_test.shape[0], no_boots))
        z_pred_train = np.empty((z_train.shape[0], no_boots))
        z_train_bootstore = np.empty((z_train.shape[0], no_boots))
        
        r2_vals = np.empty(no_boots)
        

        
        # Now, doing the resampling
        for i in range(no_boots):
            idx = np.random.randint(0, size_of_sample, size_of_sample)
            some_X, some_z = X_train[idx], z_train[idx]
            model.fit(some_X, some_z)
            
        # Storing the values
            z_train_bootstore[:,i] = some_z

            z_pred[:,i] = model.predict(X_test)
            z_pred_train[:,i] = model.predict(some_X)
            r2_vals[i] = lof.R_score(z_pred[:,i], z_test)
        
        z_test = z_test.reshape((len(z_test), 1))
        
        #Calculating the errors
        mse = np.mean(np.mean((z_pred - z_test)**2, axis=1, keepdims=True))
        mse_train = np.mean( np.mean((z_pred_train - z_train_bootstore)**2, axis=1, keepdims=True))
        bias = np.mean((z_test - np.mean(z_pred, axis=1, keepdims=True))**2)
        variance = np.mean(np.var(z_pred, axis=1, keepdims=True))
        
        return mse, bias, variance, np.mean(r2_vals), mse_train
    
    
    

