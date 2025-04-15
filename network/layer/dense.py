import numpy as np
from typing import Callable, Tuple, List, Optional
from .base import Layer
from ..optimizer import Optimizer
from ..functions import derivative

class DenseLayer(Layer):
    """
    A fully connected layer in a neural network.
    """
    def __init__(self, num_neurons: int, num_inputs: Optional[int] = None, activation_function: Optional[Callable] = None, threshold: float = 1.0):
        """
        Initializes the DenseLayer.

        Args:
            num_neurons (int): The number of neurons in the layer.
            num_inputs (Optional[int], optional): The number of input features. Defaults to None.
            activation_function (Optional[Callable], optional): The activation function to apply. Defaults to None.
            threshold (float, optional): A scaling factor for the activation. Defaults to 1.0.
        """
        super().__init__(num_neurons=num_neurons, num_inputs=num_inputs, activation_function=activation_function, threshold=threshold)
        self.weights: Optional[np.ndarray] = None
        self.biases: Optional[np.ndarray] = None
        self.gradients: Optional[np.ndarray] = None
        self.d_biases: Optional[np.ndarray] = None
        self.inputs: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None
        self.signals: Optional[np.ndarray] = None
        self._initialized = bool(num_inputs)
        if self._initialized and self.num_inputs is not None:
            self._init_weights_and_biases()

    def _init_weights_and_biases(self):
        """
        Optimized initialization of weights and biases using He initialization.
        """
        if self.num_inputs is None:
            raise ValueError("Cannot initialize weights without num_inputs specified.")
        scale = np.sqrt(2.0 / self.num_inputs)
        self.weights = np.random.randn(self.num_neurons, self.num_inputs) * scale
        self.biases = np.zeros(self.num_neurons)
        
        if self.use_complex:
            self.weights = self.weights.astype(np.complex64)
            self.biases = self.biases.astype(np.complex64)
            
            self.weights += 1j * np.random.randn(self.num_neurons, self.num_inputs) * scale
            self.biases += 1j * np.zeros(self.num_neurons)
        
        self._initialized = True

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Optimized forward pass of the layer.
        """
        if inputs.ndim != 2:
            raise ValueError(f"Input must be 2D (batch_size, num_inputs). Got shape {inputs.shape}")

        self.inputs = inputs
        if not self._initialized:
            self.num_inputs = inputs.shape[1]
            self._init_weights_and_biases()

        if self.weights is None or self.biases is None:
            raise RuntimeError("Layer not initialized.")
        if inputs.shape[1] != self.num_inputs:
            raise ValueError(f"Input shape mismatch. Layer expected {self.num_inputs} features, got {inputs.shape[1]}")

        # Compute output and apply activation in one step
        self.output = np.matmul(inputs, self.weights.T) + self.biases
        self.signals = self.activation_function(self.output) * self.threshold
        return self.signals

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Optimized backward pass of the layer.
        """
        batch_size = self.inputs.shape[0]

        # Precompute activation derivative and delta
        activation_deriv = derivative(self.activation_function, self.output)
        delta = grad * activation_deriv * self.threshold

        # Efficiently compute gradients
        self.gradients = np.matmul(delta.T, self.inputs) / batch_size
        self.d_biases = np.mean(delta, axis=0)

        # Return gradient for the previous layer
        return np.matmul(delta, self.weights)

    def _init_optimizer(self, optimizer: Optimizer):
        """
        Initializes the optimizer for the layer's parameters.

        Args:
            optimizer (Optimizer): The optimizer instance.
        """
        # Register weights and biases as parameters to be optimized
        if self.weights is not None and self.biases is not None:
            if hasattr(self.optimizer, 'register_parameter') and callable(self.optimizer.register_parameter):
                self.optimizer.register_parameter(self.weights, 'weights')
                self.optimizer.register_parameter(self.biases, 'biases')
            else:
                print("Warning: Optimizer missing 'register_parameter' method.")
                
        super()._init_optimizer(optimizer)

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Optimized retrieval of parameters and gradients.
        """
        if self.gradients is not None and self.d_biases is not None:
            return [(self.weights, self.gradients), (self.biases, self.d_biases)]
        return []