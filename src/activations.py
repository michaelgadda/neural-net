import numpy as np

def sigmoid(z: np.float):
    s = 1 / (1 + np.exp(-z))
    return s
     
def tanh(z: np.float):
    (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)) 

def relu(z: np.float): 
    return np.max(0, z) 

def linear(z: np.float):
    return z 