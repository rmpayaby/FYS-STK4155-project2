# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:10:04 2020

@author: rmbp
"""


import numpy as np
import sklearn.linear_model as skl
from list_of_func import *

class Regress:
    """
    This class contains all three methods for linear regression, and
    their respective approximation of the beta-coefficient.
    
    Note that the calculation of the pseudoinverse is defined by the function
    SVDSolver() which can be found in list_of_func.py
    
    OLS and Ridge are defined manually, while Lasso uses scikit-learn's
    package. Maximum iteration is set at 10000 in default and uses
    a tolerance of 1e-1.
    """

    def __init__(self, method = 'OLS', alpha = 0):
        """
        Method is defined as either "OLS", "Ridge" or "Lasso"
        Alpha describes the hyperparameter used for Ridge
        and Lasso
        """
        self.method = method
        self.alpha = alpha


    def ols(self):
        XT = self.X.T
        A = SVDSolver(XT.dot(self.X))
        self.beta = A.dot(XT.dot(self.z))
    
        
    def Ridge(self):
        XT = self.X.T
        L = np.identity(np.shape(self.X)[1])*self.alpha
        A = SVDSolver(XT.dot(self.X) + L)
        self.beta = A.dot(XT.dot(self.z))
        
    def Lasso(self):
        clf = skl.Lasso(alpha = self.alpha,fit_intercept=False, normalize=False,max_iter=10000, tol=1e-1).fit(self.X, self.z)
        self.beta = clf.coef_


    def fit(self, X, z):
        """
        Fits the specified model to the data. Calling in the method
        by inputting the design matrix X with dimensions
        (n,p) and the target z with dimension (n,1).
        """

        self.X = X
        self.z = z
        if self.method == 'OLS':
            self.ols()
        elif self.method == 'Ridge':
            self.Ridge()
        elif self.method == 'Lasso':
            self.Lasso()


    def predict(self, X):
        """
        Does a prediction on a data set by using the beta-coefficient values
        after fitting the data to a a specific regression method. This will be
        predicted on the design matrix.
        """
        self.z_tilde = X @ self.beta
        return self.z_tilde


    def set_alpha(self, alpha):
        """
        Change the alpha hyperparameter after initialiation. Useful when plotting
        the MSE against changing hyperparameter values in the heatmap
        """
        self.alpha = alpha
