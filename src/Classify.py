# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:35:56 2020

@author: rmbp
"""

import numpy as np


class Classify():
    
    def __init__(self, 
                 hidden_activation="ReLU",
                 output_activation="softmax",
                 cost_func="cross_entropy"):
        
        self.h_a = hidden_activation
        self.o_a = output_activation
        self.cost = cost_func
        
        
    def hidden_activation(self,x,deriv=False):

        if self.h_a == 'ReLU':
            if deriv:
                return self._ReLU_deriv(x)
            else:
                return self._ReLU(x)
        elif self.h_a == 'sigmoid':
            if deriv:
                return self._sigmoid_deriv(x)
            else:
                return self._sigmoid(x)
        elif self.h_a == "leaky_ReLU":
            if deriv:
                return self._leaky_ReLU_deriv(x)
            else:
                return self._leaky_ReLU(x)
            
    def output_activation(self,x,deriv=False):
        
        if self.o_a == 'sigmoid':
            if deriv:
                return self._sigmoid_deriv(x)
            else:
                return self._sigmoid(x)
            
        if self.o_a == 'softmax':
            return self._softmax(x)
    
    def output_error(self,a,t,x=None):
        if self.cost == 'cross_entropy':
            return (a-t)
    
    def cost_function(self,a,t):
        if self.cost == 'cross_entropy':
            return self._cross_entropy_cost(a,t)

    def _cross_entropy_cost(self,a,t):
        return -np.sum(np.nan_to_num(t*np.log(a)-(1-t)*np.log(1-a)))
        
        
        
    # Again the activation function, now including softmax
    
    
    _softmax = lambda self, x: np.exp(x)/np.sum(np.exp(x),
                                                axis=1, keepdims=True)

    _sigmoid = lambda self, x: 1/(1+np.exp(-x))
    _sigmoid_deriv = lambda self, x: self._sigmoid(x)*(1 - self._sigmoid(x))

    _leaky_ReLU = lambda self, x: np.where(x > 0, x, x * 0.01)
    _leaky_ReLU_deriv = lambda self, x: np.where(x > 0, 1, 0.01)

    _ReLU = lambda self,x: np.where(x<0,0,x)
    _ReLU_deriv = lambda self, x: np.where(x<0,0,1)
