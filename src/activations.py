import numpy as np

def sigmoid(z: np.float) -> tuple[np.ndarray, np.ndarray]:
    return_arr = np.ndarray(z.shape)
    pos_indices = np.where(z >= 0)
    neg_indices = np.where(z < 0)
    return_arr[pos_indices] = 1 / (1 + np.exp(-z))
    return_arr[neg_indices] = np.exp(z) / (1 + np.exp(z))
    deriv = (1 - return_arr) (return_arr)
    return return_arr, deriv
     
def tanh(z: np.float) -> tuple[np.ndarray, np.ndarray]:
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)) 

def relu(z: np.float)  -> tuple[np.ndarray, np.ndarray]:
    deriv_arr = np.ndarray(z.shape)
    one_indices = np.where(z > 0)
    zero_indcies = np.where(z <= 0)
    deriv_arr[one_indices] = 1
    deriv_arr[zero_indcies] = 0
    return np.max(0, z), deriv_arr 

def linear(z: np.float)  -> tuple[np.ndarray, np.ndarray]:
    return z 