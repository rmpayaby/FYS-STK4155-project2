# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:50:29 2020

@author: rmbp
"""

import numpy as np


class LogReg_SGD():
    
    def __init__(self, lr_rate=0.1,epochs=100,regularization="L2",batch_size=100, C = 0.1, tol = 1e-4):
        
        self.lr_rate = lr_rate
        self.epochs = epochs
        self.regularization = regularization
        self.C = C
        self.batch_size = batch_size
        self.tol = tol
        
    
    def create_minibatch(self,X,y,epochs):
        """
        Create minibatches for applying stochastic gradient descent
        """
        n = y.size
        num_batches = int(n/self.batch_size)
        
        for _ in range(epochs):
            idx = np.arange(n)
            np.random.shuffle(idx)
            j = 0
            for batch in range(num_batches):
                random_idx = idx[j*self.batch_size:(j+1)*self.batch_size]
                X_i = X[random_idx,:]
                y_i = y[random_idx]
                
                j += 1
                
        return X_i, y_i
    
        
    def fit(self,X,y,epochs, lr_rate):
        """
        Fitting the data set. Gives an option
        whether or not to apply l2-regularization 

        """
        self.beta = np.random.randn(X.shape[1]+1)
        X = np.concatenate((np.ones((X.shape[0],1)),X), axis=1)
        
        for _ in range(epochs):
            y_hat = self.sigmoid(X @ self.beta)
            errors = y_hat - y
            N = X.shape[1]
            
            if self.regularization is not None:
                delta_grad = lr_rate * ((self.C*
                        (X.T @ errors)) + np.sum(self.beta))
            else:
                delta_grad = lr_rate * (X.T @ errors)
                    
            if np.all(abs(delta_grad) >= self.tol):
                self.beta -= delta_grad/N
            else:
                break
            
        return self
    
    def predict_proba(self,X):
        return self.sigmoid((X @ self.beta[1:]) + self.beta[0])
    
    def predict(self,X):
        y_pred = np.round(self.predict_proba(X))
        
        return y_pred
        
    
    def sigmoid(self,z):
        return 1/(1+ np.exp(-z))

    
    
if __name__ == '__main__':
    pass