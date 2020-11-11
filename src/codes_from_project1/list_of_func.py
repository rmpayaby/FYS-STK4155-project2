# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:09:16 2020

@author: rmbp
"""

import sys, os 

import numpy as np
import pandas as pd
from Regress import *
from Resampling import *
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from imageio import imread
from PIL import Image

import skimage.measure


"""
This python-file contains shorter functions for the purpose of
generating values for the Franke function and manipulate the size of
the terrain data. 

Additionally the homemade pseudoinverse code used for finding
the beta coefficients of each regression methods can be found here
as well.

"""

def generate_mesh(n, random_pts = 0):
    """
    Generated a mesh of n x and y values.
    """
    if random_pts == 0:
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
    if random_pts == 1:
        x = np.random.rand(n)
        y = np.random.rand(n)
    return np.meshgrid(x, y)

""" Own defined error metrics"""
def MSE(y_data,y_pred):
    n = np.size(y_pred)
    return np.sum((y_data - y_pred)**2)/n

def R_score(y_data,y_pred):
    term1 = np.sum((y_data - y_pred)**2)
    term2 = np.sum((y_data - np.mean(y_data))**2)
    return 1 - term1/term2


# Using Singular value decomposition to invert a matrix
def SVDSolver(A):
    U, S, V = np.linalg.svd(A)
    Ainv = np.dot(V.transpose(),np.dot(np.diag(S**-1),U.transpose()))
    return Ainv
    

"""The next three functions generates the Franke 
function with and without noise. The design matrix
is defined in the create_X function"""    
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    
    return term1 + term2 + term3 + term4

def FrankeWithNoise(x,y,n,sigma):
    return FrankeFunction(x,y) + np.random.normal(0,sigma,n)


def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X


"""
The last two functions are related to manipulating the data of the chosen terrain
data. get_num_pixels is self-explanatory. It gives the total number of pixels in
the image, and also the width and height.

terrain_reader on the other hand reads the image, and convert the elevation
from meters to kilometers. The pixel size can be changed to make it finer or coarser.
default pixel size = 50 x 50. 

Returns the predictor value z, and its dimension.
"""
def get_num_pixels(filepath):
    width, height = Image.open(filepath).size
    return width*height, width, height


def terrain_reader(file,pix=50):
    terrain = imread(file)/1000.0
    terrain_red = skimage.measure.block_reduce(terrain,(pix,pix),np.mean)
    dim = np.shape(terrain_red)
    z = np.ravel(terrain_red)
    return z, dim[0]


    
    
