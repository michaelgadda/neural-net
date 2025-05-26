from enums import activations
from data_formats import ActivatedLayer
import numpy as np
from activation_mapping import ACTIVATION_MAPPING

class Network:
    def __init__(self):
        self.next_layer = None
        self.layer_head = None

    def add_input_layer(self, inputs: np.ndarray):
        if self.layer_head is None and self.next_layer is None:
            self.next_layer = ActivatedLayer(neurons=inputs, features_in=None, features_out=inputs.shape[0], weights=None, biases=None)
            self.layer_head = ActivatedLayer(None, None, None, None, None, None, self.next_layer)
            self.layer_head.next = self.next_layer
            self.next_layer.prev = self.layer_head
        else:
            raise ValueError("You must add the input layer before adding any hidden layers.")

    def add_layer(self, features_in: int, features_out: int, activations: np.ndarray = None, activation: activations = "relu"):
        init_weights = np.random.normal(0, (1/features_in), (features_in, features_out))
        init_biases = np.zero(features_out)
        next_layer = ActivatedLayer(features_in, features_out, init_weights, init_biases, activation)
        if self.layer_head is None and self.next_layer is None:
            raise ValueError("You must add the input layer using .add_input_layer() before adding any hidden layers.")
        else:
            self.next_layer.next = next_layer
            self.next_layer.next.prev = self.next_layer
            self.next_layer = self.next_layer.next

    def forward(self) -> np.ndarray:
        curr_neurons = None
        layer = None
        while True:
            if layer.next == None:
                return layer.neurons
            if layer is None:   
                layer = self.layer_head.next
            if layer.neurons.shape[1] != layer.next.weights.shape[0]:
                raise ValueError("")
            z = layer.neurons @ layer.next.weights
            activations = ACTIVATION_MAPPING[layer.next.activation](z)
            layer.next.neurons = activations
            layer = layer.next

    


            
        