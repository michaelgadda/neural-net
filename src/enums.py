from enum import Enum

class activations(str, Enum):
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LINEAR = "linear"