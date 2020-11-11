# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


from tqdm import tqdm

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error as MSE, roc_auc_score as AUC_score
from sklearn import preprocessing

from classification_problem import Classification
from Classify import Classify

mpl.rcdefaults()
plt.style.use('seaborn-darkgrid')
mpl.rcParams['figure.figsize'] = [10.0, 4.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 18

"""
The structure is similar to neural_net_regression.py,
but the hidden layer and output activation and functions are defined differently.

"""

class NeuralNetwork:

    def __init__(
            self,
            X_data,
            Y_data,
            problem,    
            n_hidden_neurons_list =[2],    
            n_output_neurons=2,    
            epochs=10,
            batch_size=100,
            lr_rate=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_layers = len(n_hidden_neurons_list)
        self.n_hidden_neurons_list = n_hidden_neurons_list
        self.n_output_neurons = n_output_neurons

        self.Problem = problem
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.lr_rate = lr_rate
        self.lmbd = lmbd

        self.accuracy_train = np.zeros(epochs)
        self.accuracy_test = np.zeros(epochs)
        self.auc_train = np.zeros(epochs)
        self.auc_test = np.zeros(epochs)

        self.initialize_layers()

    def initialize_layers(self):
        n_hidden = self.n_hidden_neurons_list
        
        self.bias_list = [np.zeros(n)+0.01 for n in n_hidden]
        self.bias_list.append(np.zeros(self.n_output_neurons)+0.01)
        
        self.weights_list = [np.random.randn(self.n_features,n_hidden[0])]    
        for i in range(1,self.n_layers):
            self.weights_list.append(np.random.randn(n_hidden[i-1],n_hidden[i]))
        self.weights_list.append(np.random.randn(n_hidden[-1], self.n_output_neurons))
    
            
    def FeedForward(self):

        problem = self.Problem
        self.a_list = [self.X_data]
        self.z_list = []
        
        for w,b in zip(self.weights_list,self.bias_list):
            
            self.z_list.append(np.matmul(self.a_list[-1],w)+b)
            self.a_list.append(problem.hidden_activation(self.z_list[-1]))
            
        self.a_list[-1] = problem.output_activation(self.z_list[-1])

    def FeedForward_out(self, X):
        problem = self.Problem
        a_list = [X]
        z_list = []

        for w,b in zip(self.weights_list,self.bias_list):
            z_list.append(np.matmul(a_list[-1],w)+b)
            a_list.append(problem.hidden_activation(z_list[-1]))

        a_list[-1] = problem.output_activation(z_list[-1])
        return a_list[-1]

    def Backpropagation(self):

        problem = self.Problem
        
        error_list = []; grad_w_list = []; grad_b_list = []
        
        output_error = problem.output_error(self.a_list[-1],self.Y_data)
        error_list.append(output_error)
        
        L = self.n_layers   
        
        for l in range(2,L+2): 
            prev_error = error_list[-1]
            prev_w = self.weights_list[-l+1]
            current_z = self.z_list[-l]
            error_hidden = np.matmul(prev_error,prev_w.T)*problem.hidden_activation(current_z,deriv=True)  
            error_list.append(error_hidden)
        error_list.reverse()

        for l in range(L+1):
            grad_b_list.append(np.sum(error_list[l],axis=0))
            grad_w_list.append(np.matmul(self.a_list[l].T,error_list[l]))

            if self.lmbd > 0.0: 
                grad_w_list[l] += self.lmbd * self.weights_list[l]
            
            self.weights_list[l] -= self.lr_rate*grad_w_list[l]
            self.bias_list[l] -= self.lr_rate*grad_b_list[l]
        
    def predict(self, X):
        probabilities = self.FeedForward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X):
        probabilities = self.FeedForward_out(X)
        return probabilities

    def SGD(self,auc=True):
        data_idx = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                chosen_datapoints = np.random.choice(
                    data_idx, size=self.batch_size, replace=False
                )

                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.FeedForward()
                self.Backpropagation()
            
            pred_test = self.predict_proba(X_test_scaled)
            pred_train = self.predict_proba(X_train_scaled)
            self.accuracy_test[i] = accuracy_score(from_one_hot(Y_test),np.argmax(pred_test,axis=1))
            self.accuracy_train[i] = accuracy_score(from_one_hot(self.Y_data_full),np.argmax(pred_train,axis=1))
            
            if auc==True:
                self.auc_test[i] = AUC_score(Y_test,pred_test)
                self.auc_train[i] = AUC_score(Y_train,pred_train)

def accuracy_score(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)


"""
Defining one-hot encoder
"""
def to_one_hot(category_array):
    ca = category_array # 1D array with values of the categories
    nr_categories = np.max(ca)+1
    nr_points = len(ca)
    one_hot = np.zeros((nr_points,nr_categories),dtype=int)
    one_hot[range(nr_points),ca] = 1
    return one_hot

def from_one_hot(one_hot_array):
    category_arr = np.nonzero(one_hot_array)[1]
    return category_arr


# Running the data

# Defining the data sets
df = pd.read_csv("D:/FYS-STK4155/train.csv")
input_data = df[["femaleres","age","married","children","edu","ent_wagelabor","durable_investment","fs_adskipm_often"]]
output = df["depressed"].to_numpy()
output_one_hot = to_one_hot(output)



# Huperparameters for tuning
hidden_neuron_list = [6]
epochs = 100
runs = 30
lr_rate = 0.001
lmbd = 0

AUC = []
accuracy = []



# Defining the parameters run for grid search
acc_test = np.zeros((runs,epochs))
acc_train = np.zeros((runs,epochs))
clf = Classify(hidden_activation="leaky_ReLU",output_activation="softmax")

for i in tqdm(range(runs)):
    X_train, X_test, Y_train, Y_test = train_test_split(input_data, output_one_hot,test_size=0.2)
    Scaler = preprocessing.StandardScaler()
    X_train_scaled = Scaler.fit_transform(X_train)
    X_test_scaled = Scaler.transform(X_test)
    nn = NeuralNetwork( X_train_scaled,
                        Y_train,
                        problem = clf,
                        n_hidden_neurons_list=hidden_neuron_list,
                        n_output_neurons=2,
                        epochs=epochs,
                        batch_size=100,
                        lr_rate=lr_rate,
                        lmbd=lmbd)
    nn.SGD(auc=True)
    AUC.append(nn.auc_test[-1])
    accuracy.append(nn.accuracy_test[-1])
    acc_test[i,:] = nn.accuracy_test
    acc_train[i,:] = nn.accuracy_train

AUC_mean = np.mean(AUC)
accuracy_mean = np.mean(accuracy)
print('AUC mean = ',AUC_mean, ' accuracy mean = ',accuracy_mean)


fig,ax = plt.subplots()
for i in range(len(acc_test)):
    ax.plot(acc_test[i],color='green',label='test')
    ax.plot(acc_train[i],color='black',label='train')
    if i == 0:
        ax.legend(loc=4)
        ax.set_ylim(0.5,1)
        plt.title("Depression detection, sigmoid")
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epochs')
        plt.tight_layout()
        
        
        
"""
Grid search

"""

eta_vals = np.logspace(-5, -1, 5)
lmbd_vals = np.logspace(-5, 1, 7)
# store the models for later use
DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

# grid search
for i, lr_rate in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        nn = NeuralNetwork( X_train_scaled,
                        Y_train,
                        problem=clf,
                        n_hidden_neurons_list=hidden_neuron_list,
                        n_output_neurons=2,
                        epochs=epochs,
                        batch_size=100,
                        lr_rate=lr_rate,
                        lmbd=lmbd)
        nn.SGD(auc=False)
        
        DNN_numpy[i][j] = nn
        
        test_predict = nn.predict_proba(X_test_scaled)

        print("Learning rate  = ", lr_rate)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", accuracy_score(from_one_hot(Y_test),np.argmax(test_predict,axis=1)))
        print()


sns.set()

test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        nn = DNN_numpy[i][j]
        
        test_predict = nn.predict_proba(X_test_scaled)
        test_accuracy[i][j] = accuracy_score(from_one_hot(Y_test),np.argmax(test_predict,axis=1))
        
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Grid search of Accuracy-score, Softmax")
ax.set_ylabel("Learning rate")
ax.set_xlabel("Regularization parameter")
plt.show()

