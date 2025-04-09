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
        Initializes the weights and biases of the layer.
        Uses He initialization for weights.
        """
        if self.num_inputs is None:
            raise ValueError("Cannot initialize weights without num_inputs specified.")
        # He initialization for weights
        scale = np.sqrt(2.0 / self.num_inputs)
        self.weights = np.random.randn(self.num_neurons, self.num_inputs) * scale
        # Initialize biases to zeros
        self.biases = np.zeros(self.num_neurons)
        self._initialized = True

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the layer.

        Args:
            inputs (np.ndarray): The input to the layer (batch_size, num_inputs).

        Returns:
            np.ndarray: The output of the layer after activation (batch_size, num_neurons).
        """
        # Ensure the input has the correct dimensions
        if inputs.ndim != 2:
            raise ValueError(f"Input must be 2D (batch_size, num_inputs). Got shape {inputs.shape}")

        # Store the input for the backward pass
        self.inputs = inputs
        # Initialize the layer if it hasn't been initialized yet
        if not self._initialized:
            self.num_inputs = inputs.shape[1]
            self._init_weights_and_biases()

        # Check if weights and biases have been initialized
        if self.weights is None or self.biases is None:
            raise RuntimeError("Layer not initialized.")
        # Check for input shape mismatch
        if inputs.shape[1] != self.num_inputs:
            raise ValueError(f"Input shape mismatch. Layer expected {self.num_inputs} features, got {inputs.shape[1]}")

        # Calculate the output before activation: dot product of inputs and weights, plus biases
        self.output = np.matmul(inputs, self.weights.T) + self.biases

        # Apply the activation function and the threshold
        self.signals = self.activation_function(self.output) * self.threshold

        return self.signals

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of the layer.

        Args:
            grad (np.ndarray): The incoming gradient from the next layer (batch_size, num_neurons).

        Returns:
            np.ndarray: The gradient to be passed to the previous layer (batch_size, num_inputs).
        """
        # Check if forward pass has been called
        if self.inputs is None or self.output is None:
            raise RuntimeError("Backward pass called before forward pass.")
        # Check if weights have been initialized
        if self.weights is None:
            raise RuntimeError("Layer weights not initialized.")
        # Check for batch size mismatch
        if grad.shape[0] != self.inputs.shape[0]:
            raise ValueError(f"Incoming gradient batch size {grad.shape[0]} != input batch size {self.inputs.shape[0]}.")
        # Check for gradient shape mismatch
        if grad.shape[1] != self.num_neurons:
            raise ValueError(f"Incoming gradient shape mismatch. Expected {self.num_neurons} neurons, got {grad.shape[1]}")

        batch_size = self.inputs.shape[0]

        # Calculate the derivative of the activation function
        activation_deriv = derivative(
            func=self.activation_function,
            args=self.output,
            mode='derivative'
        )

        # Calculate the derivative of the signals with respect to the output
        ds_dz = activation_deriv * self.threshold

        # Calculate the delta (error term) for this layer
        delta = grad * ds_dz

        # Calculate the gradient of the weights
        self.gradients = np.matmul(delta.T, self.inputs) / batch_size

        # Calculate the gradient of the biases
        self.d_biases = np.sum(delta, axis=0) / batch_size

        # Calculate the gradient to be passed to the previous layer
        return np.matmul(delta, self.weights)

    def _init_optimizer(self, optimizer: Optimizer):
        """
        Initializes the optimizer for the layer's parameters.

        Args:
            optimizer (Optimizer): The optimizer instance.
        """
        super()._init_optimizer(optimizer)
        # Register weights and biases as parameters to be optimized
        if self.weights is not None and self.biases is not None:
            if hasattr(self.optimizer, 'register_parameter') and callable(self.optimizer.register_parameter):
                self.optimizer.register_parameter(self.weights, 'weights')
                self.optimizer.register_parameter(self.biases, 'biases')
            else:
                print("Warning: Optimizer missing 'register_parameter' method.")


    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns a list of tuples containing the layer's parameters and their gradients.

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: A list of (parameter, gradient) tuples.
        """
        params_and_grads = []
        # Add weights and their gradients if they are available
        if self.gradients is not None and self.d_biases is not None and self.weights is not None and self.biases is not None:
            params_and_grads.extend([
                (self.weights, self.gradients),
                (self.biases, self.d_biases)
            ])
        return params_and_grads