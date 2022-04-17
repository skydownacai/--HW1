import numpy as np
from typing import List, Union
import pickle

def softmax(X : np.array):
    
    '''
    X : array : (batch, input_dim)
    '''
    for i in range(len(X)):
        X[i,:] = np.exp(X[i,:])
        sum_weight = np.sum(X[i,:])
        X[i,:] = X[i,:]/ sum_weight
    return X


class NN_Classfier(object):
    '''
    An Implementation of Full Connected Nerual Network with Relu activation and CrossEntropy Loss Function
    '''
    def __init__(self, hidden_size : int, input_dim : int, output_dim : int) -> None:
        # In 
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        #initialize Parameters with Normal Distribution
        self.initial_parameters()
        self.zero_grad()
        self.train_mode = False

    def set_train_mode(self):
        self.train_mode = True
    
    def set_eval_mode(self):
        self.train_mode = False

    def initial_parameters(self):
        self.parameters = []
        self.parameters.append(np.random.rand(self.input_dim,self.hidden_size)/self.input_dim)
        self.parameters.append(np.random.rand(self.hidden_size,self.output_dim)/self.hidden_size)

    def loss_fn(self,tgt : np.array, prob : np.array):
        loss = np.zeros((len(tgt),1))
        for i in range(len(tgt)):   
            tgt_i = tgt[i,:].reshape(-1,1)
            prob_i = prob[i,:].reshape(-1,1)
            loss[i][0] = -1 * tgt_i.T @ np.log(prob_i)
        return loss



    def forward(self, X : np.array):
        '''
        The input shape of X is (batch,input_dim)
        '''


        input_X = X.copy()
        if self.train_mode:
            self.input_history["X"] = X.copy()
        hidden = input_X @ self.parameters[0]
        hidden[hidden < 0] = 0
        if self.train_mode:
            self.input_history["hidden"] = hidden.copy()
        softmax_input = hidden @ self.parameters[1]
        prob = softmax(softmax_input)
        if self.train_mode:
            self.input_history["prob"] = prob.copy()
        return prob


    def backward(self,tgt : np.array, Lambda : float = 0 ):
        '''
        backward the loss and calculate the gradient
        '''
        hidden_values = self.input_history["hidden"]
        probs  = self.input_history["prob"]
        input_X = self.input_history["X"]

        batch_size = len(hidden_values)
        W0_gradient = np.zeros(self.parameters[0].shape)
        W1_gradient = np.zeros(self.parameters[1].shape)
        for i in range(batch_size):
            
            
            X_i = input_X[i,:].reshape(-1,1)
            tgt_i = tgt[i,:].reshape(-1,1)
            prob_i = probs[i,:].reshape(-1,1)
            hidden_i = hidden_values[i,:].reshape(-1,1)

            o = -1 * tgt_i + prob_i
            g_W1_i = hidden_i @ o.T
            sign = np.zeros(hidden_i . shape)
            sign[hidden_i >= 0] = 1
            g_W0_i = X_i @ ((self.parameters[1] @ o ) * sign).T

            W0_gradient += g_W0_i / batch_size
            W1_gradient += g_W1_i / batch_size

        W0_gradient += Lambda*self.parameters[0]
        W1_gradient += Lambda*self.parameters[1]

        self.gradients = {"W1" : W1_gradient, "W0" : W0_gradient}


    def zero_grad(self):

        self.input_history = {}
        self.gradients = {"W1":None,"W2":None}

    def step(self, lr_rate : float):
        '''
        update the paramter
        '''
        self.parameters[0] -= lr_rate * self.gradients["W0"]
        self.parameters[1] -= lr_rate * self.gradients["W1"]

    def save_model(self, filepath : str):
        with open(filepath,"wb") as f:
            pickle.dump(self.parameters,f)
    
    def load_model(self, filepath: str):
        with open(filepath,"rb") as f:
            self.parameters = pickle.load(f)
        
