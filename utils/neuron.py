from . import ActivationFunctions
from typing import Optional, Callable
import math
import random

class Neuron:
    def __init__(self, num_inputs, activation_function: Optional[Callable[[float], float]] = None):
        self.activation_function = activation_function
        self.delta = 0
        self.weights = []
        self.bias = 0
        self.signal = 0
        self.error_signal = 0
        self.gradients = []  # New attribute for weight gradients

        # Adam optimizer values
        self.first_moment = []
        self.second_moment = []
        self.first_moment_bias = 0
        self.second_moment_bias = 0
        self.time_step = 0

        if num_inputs >= 1:
            # Improved weight initialization
            if activation_function in [
                ActivationFunctions.relu,
                ActivationFunctions.leaky_relu,
            ]:
                std = math.sqrt(2.0 / num_inputs)  # He initialization for ReLU
            else:
                std = math.sqrt(1.0 / num_inputs)  # Xavier initialization for sigmoid/tanh

            self.weights = [random.gauss(0, std) for _ in range(num_inputs)]
            self.bias = random.gauss(0, std)
            self.first_moment = [0.0 for _ in range(num_inputs)]
            self.second_moment = [0.0 for _ in range(num_inputs)]
            self.gradients = [0.0 for _ in range(num_inputs)]  # Initialize gradients for weights

    def activate(self, inputs, threshold):
        signal_sum = self.bias + sum(w * i for w, i in zip(self.weights, inputs))
        self.signal = self.activation_function(signal_sum / threshold)
        if math.isnan(self.signal): # Added check to prevent NaN and None values.
            self.signal = 0.0