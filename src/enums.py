from enum import Enum

class activations(str, Enum):
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LINEAR = "linear"

class losses(str, Enum):
    NEGATIVE_LOG_LOSS = "negative_log_loss"
