import numpy as np
from typing import Any

def one_hot_encode(Y: np.array, classes: list[Any] = None):
    """
    Y: Array that needs to be one hot encoded
    classes: If you expect more classes to be seen than what is in Y than you can manually input them. classes must be > # of unique classes in Y. 
    """
    if classes and len(classes) < len(np.unique(Y)):
        raise ValueError("Classess must be set to a number higher than # of classes in Y *( len(np.unique(Y)) )")
    else:
        classes = np.unique(Y)
    zero_mtx = np.zeros((Y.shape[0], len(classes)))
    for index, p_class in enumerate(classes): 
        idxs = np.where(Y == p_class)
        zero_mtx[idxs, index] = 1

    return zero_mtx
    
