import numpy as np
from src.utils import one_hot_encode

def negative_log_loss(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[int, np.array]:
    """
    Equation is {y_true_class = y_class} * y_pred 
    """
    y_true_ohe = one_hot_encode(y_true)
    ohe_class_probs = np.multiply(y_true_ohe, np.log(y_pred))
    loss = np.sum(ohe_class_probs, axis=1)
    loss = -1 * np.sum(loss)
    derivative = np.sum(y_true_ohe / y_pred, axis = 1).reshape(-1,1)
    print("Loss: ", loss)
    return loss, derivative



