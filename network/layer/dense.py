import numpy as np
from typing import Callable, Tuple, List, Optional
from .base import Layer
from ..optimizer import Optimizer
from ..functions import derivative as deriv

class DenseLayer(Layer):
    """
    Blazingly fast fully connected (dense) layer using NumPy optimizations and Numba.

    Attributes:
        weights (np.ndarray): Weight matrix of shape (num_neurons, num_inputs).
        biases (np.ndarray): Bias vector of shape (num_neurons,).
        gradients (Optional[Tuple[np.ndarray, np.ndarray]]): Gradients of weights and biases. Defaults to None.
        inputs (Optional[np.ndarray]): Input to the layer during the forward pass. Defaults to None.
        signals (Optional[np.ndarray]): Output of the layer after activation. Defaults to None.
        _is_initialized (bool): Flag to track if the layer has been initialized with the input size.
    """
    def __init__(self, num_neurons: int, num_inputs: Optional[int] = None, activation_function: Optional[Callable] = None, threshold: float = 1.0):
        """Initializes the optimized dense layer.

        Args:
            num_neurons (int): Number of neurons in the layer.
            num_inputs (Optional[int]): Number of input features to each neuron. Defaults to None.
            activation_function (Callable[[np.ndarray], np.ndarray]): Activation function for the neurons.
            threshold (float): Threshold value for the layer's output. Defaults to 1.0.
        """
        super().__init__(num_neurons=num_neurons, num_inputs=num_inputs, activation_function=activation_function, threshold=threshold)
        self.weights = None
        self.biases = None
        self.gradients = None
        self.inputs = None
        self.signals = None
        self.d_biases = None
        self._initialized = bool(num_inputs)
        if self._initialized:
            self._init_weights_and_biases()

    def _init_weights_and_biases(self):
        self.weights: np.ndarray = np.random.randn(self.num_neurons, self.num_inputs) / np.sqrt(self.num_inputs)
        self.biases = np.zeros(self.num_neurons)
        self._initialized = True

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Performs the forward pass through the dense layer.

        Args:
            inputs (np.ndarray): The input to the layer.

        Returns:
            np.ndarray: The output signals of the layer after activation.
        """
        self.inputs = inputs
        if not self._initialized:
            self.num_inputs = inputs.shape[1]
            self._init_weights_and_biases()

        output = np.matmul(inputs, self.weights.T) + self.biases

        self.signals = self.activation_function(output) * self.threshold

        return self.signals

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Performs the backward pass through the dense layer.

        Args:
            grad (np.ndarray): The gradient from the next layer.

        Returns:
            np.ndarray: The gradient passed to the previous layer.
        """
        batch_size = self.inputs.shape[0]

        # Calculate derivative
        derivative = deriv(self.activation_function, 'all', self.signals)

        # Use element-wise multiplication for delta calculation (faster than einsum for this case)
        delta = grad * derivative

        # Fast gradient computation using transpose dot product
        # This is often faster than einsum for this operation
        self.gradients = np.matmul(delta.T, self.inputs) / batch_size

        # Use faster axis sum instead of mean for bias gradients
        # divide by batch size only once
        self.d_biases = np.sum(delta, axis=0) / batch_size

        return np.matmul(delta, self.weights)

    def _init_optimizer(self, optimizer: Optimizer):
        """Initializes the optimizer for the weights and biases of the layer.

        Args:
            optimizer (Optimizer): The optimizer object to use.
        """
        super()._init_optimizer(optimizer)
        # Register weights and biases as parameters to be optimized
        if self.weights is not None and self.biases is not None:
            self.optimizer.register_parameter(self.weights, 'weights')
            self.optimizer.register_parameter(self.biases, 'biases')

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Returns a list of (parameter, gradient) tuples for the weights and biases.

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: A list where each tuple contains a parameter (weight or bias) and its corresponding gradient.
        """
        params_and_grads = []
        if self.gradients is not None and self.d_biases is not None and self.weights is not None and self.biases is not None:
            params_and_grads.extend([
                (self.weights, self.gradients),
                (self.biases, self.d_biases)
            ])
        return params_and_grads