import numpy as np
from typing import Callable, Tuple, List, Optional
from layer.base import Layer
from utils.optimizer import Optimizer
from utils.functions import derivative as deriv

class DenseLayer(Layer):
    """
    Optimized fully connected (dense) layer using NumPy.

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
        super().__init__(num_neurons=num_neurons, num_inputs=num_inputs, activation_function=activation_function)
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.activation_function = activation_function
        self.threshold = threshold
        self.weights = None
        self.biases = None
        self.gradients = None
        self.inputs = None
        self.signals = None
        self.d_biases = None
        self._cache = None
        self._grad_cache = None
        self._is_initialized = False
        if num_inputs is not None:
            self._initialize_weights_and_biases(num_inputs)
            self._is_initialized = True

    def _initialize_weights_and_biases(self, num_inputs: int):
        """Initializes weights using He initialization."""
        self.weights = np.random.randn(self.num_neurons, num_inputs) / np.sqrt(num_inputs)
        self.biases = np.zeros(self.num_neurons, dtype=np.float64)
        self.num_inputs = num_inputs

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Performs the forward pass through the dense layer.

        Args:
            inputs (np.ndarray): The input to the layer.

        Returns:
            np.ndarray: The output signals of the layer after activation.
        """
        if not self._is_initialized:
            self._initialize_weights_and_biases(inputs.shape[1])
            self._is_initialized = True
    
        
        # Ensure inputs are contiguous and float64
        self.inputs = np.ascontiguousarray(inputs, dtype=np.float64)
        
        # Pre-allocate cache if needed
        if self._cache is None or self._cache.shape != (inputs.shape[0], self.num_neurons):
            self._cache = np.empty((inputs.shape[0], self.num_neurons), dtype=np.float64)
        
        # Use einsum for batch matrix multiplication
        np.einsum('ij,kj->ik', self.inputs, self.weights, out=self._cache)
        self._cache += self.biases
        
        # Apply activation in-place
        self.signals = self.activation_function(self._cache) * self.threshold
        return self.signals

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Performs the backward pass through the dense layer.

        Args:
            grad (np.ndarray): The gradient from the next layer.

        Returns:
            np.ndarray: The gradient passed to the previous layer.
        """
        derivative = deriv(self.activation_function, 'all', self.signals)
        
        # In-place multiplication for gradients
        delta = np.multiply(grad, derivative, out=derivative)
        
        # Pre-allocate grad_cache if needed
        if self._grad_cache is None or self._grad_cache.shape != (self.num_neurons, self.num_inputs):
            self._grad_cache = np.empty((self.num_neurons, self.num_inputs), dtype=grad.dtype)
        
        # Use einsum for efficient gradient computation
        batch_size = max(1, self.inputs.shape[0])
        self.gradients = np.einsum('ij,ik->jk', delta, self.inputs, out=self._grad_cache) / batch_size
        self.d_biases = np.mean(delta, axis=0)
        
        # Use matmul (@) operator for backward pass
        return delta @ self.weights

    def _init_optimizer(self, optimizer: 'Optimizer'):
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