from src.activations import tanh
from src.activations import linear
from src.activations import relu
from src.activations import sigmoid
from src.activations import softmax
from src.metrics import negative_log_loss

LAYER_MAPPING = {"linear": linear, "tanh": tanh, "relu": relu, "sigmoid": sigmoid, "softmax": softmax}

OUTPUT_LAYER_MAPPING = {"softmax": softmax}

LOSS_MAPPING = {"negative_log_loss": negative_log_loss}