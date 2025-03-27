from typing import Optional, Callable
import numpy as np

class Neuron:
    def __init__(self, num_inputs: int, activation_function: Optional[Callable[[float], float]] = None):
        # Use more precise initialization with He initialization
        scale = np.sqrt(2.0 / num_inputs)
        self.weights = np.random.randn(num_inputs) * scale
        self.bias = np.random.randn() * scale
        
        # Precompute zero arrays to avoid repeated allocation
        self.signal: float = 0.0
        self.delta: float = 0.0
        self.gradients = np.zeros_like(self.weights)
        self.inputs = np.zeros_like(self.weights)
        
        self.activation_function = activation_function