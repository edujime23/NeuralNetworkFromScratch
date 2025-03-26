import itertools
import numpy as np
from typing import Optional, Callable, Tuple

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
                self.optimizer.initialize(neuron, 'weights')
                self.optimizer.initialize(neuron, 'bias')
        elif isinstance(self, Conv2DLayer):
            self.optimizer = optimizer
            self.optimizer.initialize(self, 'filters')
            self.optimizer.initialize(self, 'biases')
        else:
            self.optimizer = optimizer # For layers without neurons
            
class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.original_shape = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.original_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self.original_shape)

    def update(self) -> None:
        pass # No parameters to update

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
        batch_size = inputs.shape[0] if inputs.ndim > 1 else 1

        # Extract weights and biases for each neuron
        weights = np.array([neuron.weights for neuron in self.neurons])  # Shape: (num_neurons, num_inputs)
        biases = np.array([neuron.bias for neuron in self.neurons])      # Shape: (num_neurons,)

        # Perform batched dot product
        # inputs shape: (batch_size, num_inputs)
        # weights shape: (num_neurons, num_inputs)
        # Transpose weights to (num_inputs, num_neurons) for dot product
        z = np.dot(inputs, weights.T) + biases  # Shape: (batch_size, num_neurons)

        # Apply activation function and threshold
        self.signals = self.activation_function(z) * self.threshold
        return self.signals

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass through the layer."""
        # Calculate derivative of the activation function for the backward pass
        derivative = ActivationFunctions.derivative(self.activation_function, self.signals)
        delta = grad * derivative  # Shape: (batch_size, num_neurons)
        self.backward_delta = delta  # Store delta for bias update

        # Store gradients for weight update
        # inputs shape: (batch_size, num_inputs)
        # delta shape: (batch_size, num_neurons)
        self.gradients = np.dot(self.inputs.T, delta) / self.inputs.shape[0] # Shape: (num_inputs, num_neurons), normalized by batch size

        # Propagate the delta backward to the previous layer
        weights = np.array([neuron.weights for neuron in self.neurons]) # Shape: (num_neurons, num_inputs)
        return np.dot(delta, weights)  # Shape: (batch_size, num_inputs)

    def update(self) -> None:
        """Update the layer's parameters."""
        if self.optimizer:
            for i, neuron in enumerate(self.neurons):
                self.optimizer.update(neuron, 'weights', self.gradients[:, i])
                self.optimizer.update(neuron, 'bias', self.backward_delta[:, i].mean(axis=0) if self.backward_delta.ndim > 1 else self.backward_delta[i])

class Conv2DLayer(Layer):
    def __init__(self, num_filters: int, kernel_size: Tuple[int, int], stride: int = 1, padding: str = 'valid', activation_function=None, input_shape: Optional[Tuple[int, int, int]] = None):
        super().__init__(activation_function=activation_function)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape
        self.filters = np.random.randn(num_filters, kernel_size[0], kernel_size[1], input_shape[2] if input_shape else 1) * np.sqrt(2.0 / (kernel_size[0] * kernel_size[1] * (input_shape[2] if input_shape else 1)))
        self.biases = np.zeros(num_filters)
        self.d_filters = np.zeros_like(self.filters)
        self.d_biases = np.zeros_like(self.biases)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Assuming inputs is a single image with shape (height, width, channels) or a batch of images (batch_size, height, width, channels)
        self.inputs = inputs
        if inputs.ndim == 3:
            inputs = np.expand_dims(inputs, axis=0) # Add batch dimension

        batch_size, input_height, input_width, input_channels = inputs.shape
        kernel_height, kernel_width = self.kernel_size

        if self.padding == 'same':
            pad_height = int(((input_height - 1) * self.stride + kernel_height - input_height) / 2)
            pad_width = int(((input_width - 1) * self.stride + kernel_width - input_width) / 2)
            padded_inputs = np.pad(inputs, ((0, 0), (pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
        else:
            padded_inputs = inputs

        input_height_padded = padded_inputs.shape[1]
        input_width_padded = padded_inputs.shape[2]

        output_height = int((input_height_padded - kernel_height) / self.stride + 1)
        output_width = int((input_width_padded - kernel_width) / self.stride + 1)
        output = np.zeros((batch_size, output_height, output_width, self.num_filters))

        for b, f, h, w in itertools.product(range(batch_size), range(self.num_filters), range(output_height), range(output_width)):
            input_slice = padded_inputs[b,
                                        h * self.stride:h * self.stride + kernel_height,
                                        w * self.stride:w * self.stride + kernel_width, :]
            output[b, h, w, f] = np.sum(input_slice * self.filters[f]) + self.biases[f]

        if self.activation_function:
            self.signals = self.activation_function(output)
            return self.signals
        else:
            self.signals = output
            return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # grad shape: (batch_size, output_height, output_width, num_filters)
        inputs = self.inputs
        if inputs.ndim == 3:
            inputs = np.expand_dims(inputs, axis=0)

        batch_size, input_height, input_width, input_channels = inputs.shape
        num_filters, kernel_height, kernel_width, _ = self.filters.shape
        _, output_height, output_width, _ = grad.shape

        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input = np.zeros_like(inputs, dtype=np.float64)

        for b, f in itertools.product(range(batch_size), range(num_filters)):
            # Gradient with respect to biases
            d_biases[f] += np.sum(grad[b, :, :, f])

            for h, w in itertools.product(range(output_height), range(output_width)):
                # Gradient with respect to filters
                input_slice = inputs[b,
                                    h * self.stride:h * self.stride + kernel_height,
                                    w * self.stride:w * self.stride + kernel_width, :]
                d_filters[f] += input_slice * grad[b, h, w, f]

                # Gradient with respect to input (for backpropagation to previous layer)
                for c in range(input_channels):
                    d_input[b,
                            h * self.stride:h * self.stride + kernel_height,
                            w * self.stride:w * self.stride + kernel_width, c] += grad[b, h, w, f] * self.filters[f, :, :, c]

        self.d_filters = d_filters / batch_size
        self.d_biases = d_biases / batch_size

        # Remove the added batch dimension if the original input was 3D
        if self.inputs.ndim == 3:
            d_input = np.squeeze(d_input, axis=0)

        return d_input

    def update(self) -> None:
        if self.optimizer:
            self.optimizer.update(self, 'filters', self.d_filters)
            self.optimizer.update(self, 'biases', self.d_biases)

    def _init_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.initialize(self, 'filters')
        self.optimizer.initialize(self, 'biases')