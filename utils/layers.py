import numpy as np
from typing import Optional, Callable

from . import Neuron, ActivationFunctions, Optimizer

class Layer:
    def __init__(self, num_neurons: Optional[int] = None, num_inputs: Optional[int] = None, activation_function: Optional[Callable[[np.ndarray], np.ndarray]] = None, threshold: float = 1.0):
        """Base class for all layers."""
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.activation_function = activation_function
        self.threshold = threshold
        self.epsilon = 1e-5
        self.signals = None  # Initialize as None, as the shape depends on the layer
        self.inputs = None   # Initialize as None
        self.optimizer: Optimizer = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        raise NotImplementedError("Subclasses must implement forward method")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass through the layer."""
        raise NotImplementedError("Subclasses must implement backward method")

    def update(self) -> None:
        """Update the layer's parameters."""
        raise NotImplementedError("Subclasses must implement update method")

    def _init_optimizer(self, optimizer):
        """Initialize the optimizer for the layer's neurons (if applicable)."""
        if hasattr(self, 'neurons') and self.neurons:
            self.optimizer = optimizer
            for neuron in self.neurons:
                self.optimizer.initialize(neuron)
        else:
            self.optimizer = optimizer # For layers without neurons

class DenseLayer(Layer):
    def __init__(self, num_neurons: int, num_inputs: int, activation_function, dropout_rate: float = 0.0, batch_norm: bool = False, threshold: float = 1.0):
        """Initialize a dense layer with the given parameters."""
        super().__init__(num_neurons, num_inputs, activation_function, threshold)
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.gamma = np.ones(num_neurons) if batch_norm else None
        self.beta = np.zeros(num_neurons) if batch_norm else None
        self.mean = np.zeros(num_neurons) if batch_norm else None
        self.variance = np.ones(num_neurons) if batch_norm else None
        self.signals = np.zeros(num_neurons)
        self.inputs = np.zeros(num_inputs)
        self.gradients = None
        self.backward_delta = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs

        # Extract weights and biases for each neuron
        weights = np.array([neuron.weights for neuron in self.neurons])  # Shape: (num_neurons, num_inputs)
        biases = np.array([neuron.bias for neuron in self.neurons])      # Shape: (num_neurons,)

        # Ensure proper shape for dot product
        z = np.dot(weights, inputs) + biases  # Shape: (num_neurons,)

        # Apply activation function and threshold
        self.signals = self.activation_function(z) * self.threshold
        return self.signals


    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass through the layer."""

        # Calculate derivative of the activation function for the backward pass
        derivative = ActivationFunctions.derivative(self.activation_function, self.signals)
        delta = grad * derivative  # Shape: (num_neurons,)
        self.backward_delta = delta  # Store delta for bias update

        # Store gradients for weight update (weights: (num_neurons, num_inputs))
        self.gradients = np.dot(delta[:, np.newaxis], self.inputs[np.newaxis, :])  # Shape: (num_neurons, num_inputs)

        # Propagate the delta backward to the previous layer
        return np.dot(np.array([neuron.weights for neuron in self.neurons]).T, delta)  # Shape: (num_inputs,)

    def update(self) -> None:
        """Update the layer's parameters."""
        if self.optimizer:
            for i, neuron in enumerate(self.neurons):
                grads = (self.gradients[i], self.backward_delta[i])
                self.optimizer.update(neuron, grads)