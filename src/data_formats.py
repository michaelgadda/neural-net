from dataclasses import dataclass
from src.enums import activations
from src.enums import losses
import numpy as np


@dataclass
class ActivatedLayer: 
    features_in: int
    features_out: int
    weights: np.ndarray = None
    biases: np.array = None
    activation: activations = "relu"
    neurons: np.array = None
    derivative: np.ndarray = None
    next: 'ActivatedLayer' = None
    prev: 'ActivatedLayer' = None

@dataclass
class OutputLayer:
    features_in: int
    features_out: int
    y_true: np.ndarray
    weights: np.ndarray = None
    biases: np.array = None
    activation: activations  = "relu"
    neurons: np.array = None
    derivative: np.ndarray = None
    next: 'LossLayer' = None
    prev: 'ActivatedLayer' = None

@dataclass
class LossLayer:
    features_in: int
    y_true: np.ndarray
    loss_type: losses = "negative_log_loss"
    features_out: int = 1
    neurons: np.array = None
    derivative: np.ndarray = None
    prev: 'ActivatedLayer' = None
    next = None