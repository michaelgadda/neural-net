from dataclasses import dataclass
from enums import activations
import numpy as np


@dataclass
class ActivatedLayer: 
    features_in: int
    features_out: int
    weights: np.ndarray
    biases: np.array
    neurons: np.array = None
    activation_derivatives: np.ndarray = None
    activation: activations = "Relu"
    next: 'ActivatedLayer' = None
    prev: 'ActivatedLayer' = None