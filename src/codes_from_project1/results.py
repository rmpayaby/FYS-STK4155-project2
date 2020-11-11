# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:07:11 2020

@author: rmbp
"""


from Regress import *
from Resampling import *
from list_of_func import *

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imageio import imread


import altair as alt
from altair import Row, Column, Chart, Text, Scale, Color

"""
This python-file contains all the plots and error metrics used in the report.
This is also the functions that are called from the main-file in order to do 
test runs. 

"""


def complexity(X,y, model_name, min_deg=1, max_deg=10, alpha = 0):
    """
    Plots the training MSE against the test MSE in an interval of
    polynomial degrees (model complexity) from 1 to 10 as a default with desired
    regression method. It also plots the bias-variance tradeoff.
    
    Inputs:
        x and y: the coordinates used to define the design matrix.
        z = the target value
        model_name: Name of regression method: "OLS", "Ridge" or "Lasso"
        min_deg: minimum degree; max_deg: maximum degree
        alpha = Hyperparameter value
        
    
    N.B to change resampling type: Edit line 77 to desired method in Resampling.py
    """

    # Create a pandas dataframe to store error metrics.
    errors = pd.DataFrame(columns=['degrees', 'mse', 'bias', 'variance', 'r2','mse_train'])
    

    #initialize regression model and arrays for saving error values
    model = Regress(model_name, alpha=alpha)
    degrees = np.linspace(min_deg, max_deg, max_deg - min_deg + 1)
    num_deg = len(degrees)
    mse = np.zeros(num_deg); mse_train = np.zeros(num_deg)
    bias = np.zeros(num_deg); variance = np.zeros(num_deg); r2 = np.zeros(num_deg)

    # Initialize the optimal error metrics
    min_mse = 1e100
    min_r2 = 0
    min_deg = 0
    i = 0

    #loop through the specified degrees to be analyzed
    for deg in degrees:
        X = X
        resample = Resampling(X, y)

        #perform bootstrap resampling and save error values
        mse[i], bias[i], variance[i], r2[i], mse_train[i] = resample.sklearns_kfold(model)

        #save to pandas dataframe
        errors = errors.append({'degrees': degrees[i],
                                            'mse': mse[i],
                                            'bias': bias[i],
                                            'variance': variance[i],
                                            'r2': r2[i],
                                           'mse_train': mse_train[i]},
                                           ignore_index=True)

        #Defines the optimal error value
        if mse[i] < min_mse:
            min_mse = mse[i]
            min_r2 = r2[i]
            min_deg = deg


        i += 1




    #plot error of test set and training set
    """
    plt.title("Bootstrap MSE, OLS (Franke function)")
    plt.plot(degrees, mse, label='test set')
    plt.plot(degrees, mse_train, label='training set')
    plt.legend()
    plt.xlabel('Model complexity [degree]')
    plt.ylabel('Mean Squared Error')
    plt.show()
    

    #plot bias^2 variance decomposition of the test error
    plt.title("Bias-variance tradeoff, Bootstrap")
    plt.plot(degrees, mse, label='mse')
    plt.plot(degrees, bias,'--', label='bias')
    plt.plot(degrees, variance, label='variance')
    plt.xlabel('Model complexity [degree]')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()
    """
    print('min mse:', min_mse)
    print('r2:', min_r2)
    print('deg:', min_deg)
    print(r2)
    
    
def alpha_tests(X,y, model_name,min_alpha=-10,max_alpha=2,num_alpha = 13):
    """
    Plots the MSE for a fixed polynomial degree against the hyperparameters.
    
    Input values remain mostly the same as for the function above

    """
    
    #Store the errors in pandas dataframe 
    errors = pd.DataFrame(columns=['log lambda', 'mse', 'bias', 'variance', 'r2','mse_train'])
    
    model = Regress(model_name)
    alpha_vals = np.linspace(min_alpha,max_alpha,num_alpha)
    
    # Again, initialize values
    mse = np.zeros(num_alpha); mse_train = np.zeros(num_alpha)
    bias = np.zeros(num_alpha); variance = np.zeros(num_alpha); r2 = np.zeros(num_alpha)

    
    # Making a loop to perform regression analysis with different 
    # hyperparameter
    
    i = 0
    for alpha in alpha_vals:
        X = X
        resample = Resampling(X, y)
        model.set_alpha(10**alpha)
        
        mse[i], bias[i], variance[i], r2[i], mse_train[i] = resample.bootstrap(model)
        
        errors = errors.append({'log lambda': alpha_vals[i],
                                            'mse': mse[i],
                                            'bias': bias[i],
                                            'variance': variance[i],
                                            'r2': r2[i],
                                           'mse_train': mse_train[i]},
                                           ignore_index=True)
        
        i += 1
    
    
    # Plotting with MSE and R2-score together.
    fig, ax1 = plt.subplots()
    color = "tab:green"
    
    #plt.hlines(0.955,-10,0,linestyles="dashed",label="OLS R2-score")
    #plt.hlines(0.004,-10,0,linestyles="dashed",label="OLS MSE",colors="b")
    #plt.legend()
    
    
    ax1.set_xlabel(r'$log_{10}\lambda$')
    ax1.set_ylabel("MSE",color=color)
    ax1.plot(alpha_vals,mse,color=color,label="MSE")
    ax1.tick_params(axis="y",labelcolor=color)
    
    ax2 = ax1.twinx()
    
    color = "tab:red"
    ax2.set_ylabel("R2-score",color=color)
    ax2.plot(alpha_vals,r2,color=color,label="R2")
    ax2.tick_params(axis="y",labelcolor=color)
    
    fig.tight_layout()
    plt.title("Error metrics, Ridge")
    
    plt.show()

    
def learning_curve(model_name,X,y):
    """
    Setting up the learning curve for a fixed hyperparameter value.
    
    Takes the design matrix (X) and target value (y) as input.
    
    Default test size is defined as 20% of the data set

    """
    
    
    X = X[:,1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_error, test_error = [], []  # For MSE
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
        
    ones1 = np.ones((X_train.shape[0],1))
    ones2 = np.ones((X_test.shape[0],1))
     
    X_train = np.hstack((ones1,X_train))
    X_test = np.hstack((ones2,X_test))
    
    model = Regress(model_name)
    
    if model_name == "Ridge" or model_name == "Lasso":
        model.set_alpha(10**-5)
        
    for m in range(1,len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_pred = model.predict(X_train[:m])
        y_test_pred = model.predict(X_test)
        train_error.append(mean_squared_error(y_train[:m],y_train_pred))
        test_error.append(mean_squared_error(y_test, y_test_pred))
        
    plt.xlim(0,150); plt.ylim(0,0.8)
    
    plt.plot(np.sqrt(train_error), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(test_error), "b-", linewidth=3, label="test")
    plt.legend()
    plt.title("Learning curve, OLS (Franke function)")
    plt.ylabel("RMSE")
    plt.xlabel("Training set size")
    
        
def confidence_intervals(X,z):
    
    """
    Calculates the confidence intervals and plots it's values for OLS-regression. A histogram of 
    the beta-variance is also plotted here. Takes the design matrix (X) and target value (z)
    as input
    """
    
    # Fitting the model and calls in the beta-values
    model = Regress("OLS")
    model.fit(X,z)
    beta_vals = model.beta
    
    length = np.arange(0,len(beta_vals))
    
    # Defining sigma
    XTX_diag = np.diagonal(SVDSolver(X.T@X))
    sigma2 = 1/(X.shape[0] - X.shape[1] - 1)*np.sum((model.predict(X) - model.z)**2)
    sigma_bets = np.sqrt(XTX_diag*sigma2)
    
    # Calculates the confidence intervals 
    CI = 1.96*sigma_bets
    
    plt.errorbar(range(len(beta_vals)), beta_vals, CI, fmt="b.", capsize=3, label=r'$\beta_j \pm 1.96 \sigma$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend()
    plt.xlabel(r'index $j$')
    plt.ylabel(r'$\beta_j$')
    plt.grid()
    
    plt.figure()
    
    plt.title("Distribution of variance of beta")
    plt.bar(length, sigma_bets, width=0.3, color="green", label=r"Var$\left[\, \hat{\beta}\, \right]$");
    plt.legend()
    plt.show()
    
    
def generate_heatmaps(X,y,model_name,min_deg=1,max_deg=10,num_alpha=13,min_alpha=-10,max_alpha=2):
    
    """
    In general, this function uses the same concept as complexity and alpha_tests, but
    now, we study the three variables: model complexity, hyperparameters, and MSE
    altogether by making a heatmap.
    
    N.B: Vizualization of the heatmap is performed by using the 
    Python-library Altair. 
    """
    
    # Calling in the model and the variables used
    model = Regress(model_name)
    alpha_vals = np.linspace(min_alpha,max_alpha,num_alpha)
    degrees = np.linspace(min_deg, max_deg, max_deg - min_deg + 1)
    variables = pd.DataFrame(columns=['degrees', 'mse', 'log lambda','r2'])
    
    min_mse = 1e100
    min_lambda = 0
    min_degree = 0
    min_r2 = 0

    
    i = 0
    
    # Running the loop for degrees and hyperparameter. 
    # Due to runtime issues, tqdm was used to measure the progress in the 
    # loops the verify that the program is functioning!
    for deg in degrees:
        j = 0
        X = X
        resample = Resampling(X,y)
        
        for alpha in tqdm(alpha_vals):
            model.set_alpha(10**alpha)
            
            mse, bias, variance, r2, mse_train = resample.sklearns_kfold(model)
            variables = variables.append({'degrees': deg,
                                          'log lambda': alpha,
                                          'mse': mse,
                                          'r2':r2},ignore_index=True)
            
        if mse < min_mse:
            min_mse = mse
            min_r2 = r2
            min_deg = deg
            min_alpha = alpha
            
            j+=1
        i+=1
            
    # Save values in a csv-file, and pivot the varialvles
    variables.to_csv("test.csv")
    
    doc_raw = pd.read_csv('test.csv')
    doc_matrix = doc_raw.pivot("degrees","log lambda","r2")

    # Creating heat maps using Altair
    alt.renderers.enable('altair_viewer')
    
    chart = alt.Chart(variables).mark_rect().encode(
    x='degrees:O',
    y='log lambda:O',
    color='r2:Q'
)
    
    text = chart.mark_text(baseline='middle').encode(
    text=alt.Text('r2:Q', format=',.2r'))
    
    tot = chart + text
   
    # Directs you into Altair's local host
    # and gives you oportunity to save figure
    tot.show()
    
    print("R2:", min_r2)
    print("Optimal deg:",min_deg)
    print("Optimal alpha:",min_alpha)
    print(r2)
    
def terrainplotter(path):
    """Plots the original terrain data in 3D"""
    terrain = imread(path)/1000
    ny, nx = terrain.shape
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    
    xv, yv = np.meshgrid(x,y)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dem3d=ax.plot_surface(xv,yv,terrain,cmap='viridis', edgecolor='none')
    
    ax.set_title('Orriginal terrain, Hokkaido, Japan')
    ax.set_zlabel('Elevation (km)')
    plt.show()

def predicted_terrain(x,y,z,model_name,poly_deg):
    """Plots the predicted terrain in 3D"""
    model = Regress(model_name)
    X = create_X(x, y, poly_deg)
    model.fit(X,z)
    z_pred = model.predict(X).reshape(len(x),len(y))
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z_pred,cmap='viridis', edgecolor='none')
    ax.set_title('Predicted terrain, OLS, 50 degree polynomial')
    plt.show()
