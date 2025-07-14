import numpy as np
from src.utils import one_hot_encode


def sigmoid(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return_arr = np.zeros(z.shape)
    
    pos_indices = np.where(z >= 0)
    neg_indices = np.where(z < 0)
    return_arr[pos_indices] = 1 / (1 + np.exp(-z[pos_indices]))
    return_arr[neg_indices] = np.exp(z[neg_indices]) / (1 + np.exp(z[neg_indices]))
    deriv = (1 - return_arr)* (return_arr)
    return return_arr, deriv
     
def tanh(z: np.array) -> tuple[np.ndarray, np.ndarray]:
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)) 

def relu(z: np.array)  -> tuple[np.ndarray, np.ndarray]:
    deriv_arr = np.ndarray(z.shape)
    one_indices = np.where(z > 0)
    zero_indcies = np.where(z <= 0)
    deriv_arr[one_indices] = 1
    deriv_arr[zero_indcies] = .01
    return np.maximum(z * .01, z), deriv_arr 

def linear(z: np.array)  -> tuple[np.ndarray, np.ndarray]:
    return z, 1

def softmax(z: np.array, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true_ohe = one_hot_encode(y_true)
    max_x = np.max(z, axis=1).reshape(-1,1)
    new_z = z - max_x
    denom = np.sum(np.exp(new_z), axis=1).reshape(-1,1)
    vals = np.exp(new_z) / denom
    correct_val = np.sum(y_true_ohe * vals, axis=1).reshape(-1,1)
    derivative = correct_val * (vals - y_true_ohe)  
    return vals, derivative