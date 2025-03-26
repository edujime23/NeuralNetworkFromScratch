from typing import Optional, Callable
import numpy as np

class Neuron:
    def __init__(self, num_inputs, activation_function: Optional[Callable[[float], float]] = None):
        self.activation_function = activation_function
        self.weights = np.random.randn(num_inputs) * np.sqrt(2.0 / num_inputs)  # This should be a 1D array of size num_inputs
        self.bias = np.random.randn() * np.sqrt(2.0 / num_inputs)  # Bias should be scalar
        self.signal = 0
        self.delta = 0
        self.gradients = np.zeros(num_inputs)
        self.inputs = np.zeros(num_inputs)