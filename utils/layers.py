import numpy as np
from typing import Optional, Callable, Tuple
from . import Neuron, ActivationFunctions, Optimizer

class Layer:
    def __init__(self, num_neurons: Optional[int] = None, num_inputs: Optional[int] = None, 
                 activation_function: Optional[Callable[[np.ndarray], np.ndarray]] = None, 
                 threshold: float = 1.0):
        """Streamlined base layer initialization."""
        self.num_neurons, self.num_inputs = num_neurons, num_inputs
        self.activation_function = activation_function
        self.threshold = threshold
        self.signals = self.inputs = None
        self.optimizer = None

class Flatten(Layer):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.original_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self.original_shape)

    def update(self) -> None:
        pass

class DenseLayer(Layer):
    def __init__(self, num_neurons: int, num_inputs: int, activation_function, threshold: float = 1.0):
        super().__init__(num_neurons, num_inputs, activation_function, threshold)
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]
        self.signals = np.zeros(num_neurons)
        self.inputs = np.zeros(num_inputs)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        weights = np.array([neuron.weights for neuron in self.neurons])
        biases = np.array([neuron.bias for neuron in self.neurons])
        z = np.dot(inputs, weights.T) + biases
        self.signals = self.activation_function(z) * self.threshold
        return self.signals

    def backward(self, grad: np.ndarray) -> np.ndarray:
        derivative = ActivationFunctions.derivative(self.activation_function, self.signals)
        delta = grad * derivative
        self.gradients = np.dot(self.inputs.T, delta) / self.inputs.shape[0]
        weights = np.array([neuron.weights for neuron in self.neurons])
        return np.dot(delta, weights)

    def update(self) -> None:
        if self.optimizer:
            for i, neuron in enumerate(self.neurons):
                self.optimizer.update(neuron, 'weights', self.gradients[:, i])
                bias_update = delta_mean = (
                    self.backward_delta[:, i].mean() 
                    if self.backward_delta.ndim > 1 
                    else self.backward_delta[i]
                )
                self.optimizer.update(neuron, 'bias', bias_update)

import numpy as np
from typing import Union, Optional, Tuple, Callable

class Conv2DLayer:
    def __init__(self, 
                 num_filters: int, 
                 kernel_size: Union[int, Tuple[int, int]], 
                 stride: int = 1, 
                 padding: str = 'valid', 
                 activation_function: Optional[Callable] = None, 
                 input_shape: Optional[Tuple[int, int, int]] = None):
        """
        Initialize a 2D Convolutional Layer
        
        Args:
            num_filters: Number of convolutional filters/kernels
            kernel_size: Size of the convolution window (int or tuple)
            stride: Stride of the convolution
            padding: Padding method ('valid' or 'same')
            activation_function: Optional activation function to apply after convolution
            input_shape: Optional input shape for pre-initialization
        """
        # Normalize kernel size to tuple if single int is provided
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        
        # Store layer parameters
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding.lower()
        self.activation_function = activation_function
        
        # Placeholder for input and layer parameters
        self.inputs = None
        self.signals = None
        self.input_shape = input_shape
        self.optimizer = None
        
        # Initialization of filters and biases
        if input_shape:
            input_channels = input_shape[2]
            # He initialization for better gradient flow
            scale = np.sqrt(2.0 / (np.prod(self.kernel_size) * input_channels))
            self.filters = np.random.randn(num_filters, *self.kernel_size, input_channels) * scale
            self.biases = np.zeros(num_filters)
        else:
            self.filters = None
            self.biases = None
        
        # Pre-allocate gradient arrays
        self.d_filters = None
        self.d_biases = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform forward convolution
        
        Args:
            inputs: Input feature map
        
        Returns:
            Convolved and activated feature map
        """
        # Lazy initialization of filters if not done during __init__
        if self.filters is None:
            input_channels = inputs.shape[-1]
            scale = np.sqrt(2.0 / (np.prod(self.kernel_size) * input_channels))
            self.filters = np.random.randn(self.num_filters, *self.kernel_size, input_channels) * scale
            self.biases = np.zeros(self.num_filters)
        
        self.inputs = inputs
        batch_size, input_height, input_width, input_channels = inputs.shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride, self.stride

        # Apply padding
        if self.padding == 'same':
            pad_h = int(((input_height - 1) * stride_h + kernel_height - input_height) / 2)
            pad_w = int(((input_width - 1) * stride_w + kernel_width - input_width) / 2)
            padded_inputs = np.pad(inputs, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:
            padded_inputs = inputs

        # Compute output dimensions
        output_height = (padded_inputs.shape[1] - kernel_height) // stride_h + 1
        output_width = (padded_inputs.shape[2] - kernel_width) // stride_w + 1

        # Use numpy's stride tricks for efficient convolution
        shape = (batch_size, output_height, output_width, kernel_height, kernel_width, input_channels)
        strides = (
            padded_inputs.strides[0], 
            stride_h * padded_inputs.strides[1], 
            stride_w * padded_inputs.strides[2],
            padded_inputs.strides[1], 
            padded_inputs.strides[2], 
            padded_inputs.strides[3]
        )
        patches = np.lib.stride_tricks.as_strided(padded_inputs, shape=shape, strides=strides)

        # Perform convolution using einsum for efficiency
        output = np.einsum('bhwkyc,fkyc->bhwf', patches, self.filters) + self.biases.reshape(1, 1, 1, -1)

        # Apply activation function if specified
        if self.activation_function:
            self.signals = self.activation_function(output)
            return self.signals
        else:
            self.signals = output
            return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Compute gradients for backpropagation
        
        Args:
            grad: Gradient from the subsequent layer
        
        Returns:
            Gradient with respect to input
        """
        inputs = self.inputs
        batch_size, input_height, input_width, input_channels = inputs.shape
        num_filters, kernel_height, kernel_width, num_input_channels_filters = self.filters.shape
        _, output_height, output_width, _ = grad.shape
        stride_h, stride_w = self.stride, self.stride

        # Apply padding similar to forward pass
        if self.padding == 'same':
            pad_h = int(((input_height - 1) * stride_h + kernel_height - input_height) / 2)
            pad_w = int(((input_width - 1) * stride_w + kernel_width - input_width) / 2)
            padded_inputs = np.pad(inputs, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:
            padded_inputs = inputs

        # Apply activation derivative if activation function exists
        if self.activation_function:
            derivative = ActivationFunctions.derivative(self.activation_function, self.signals)
            grad = grad * derivative

        # Compute gradient for filters
        shape_patches = (batch_size, output_height, output_width, kernel_height, kernel_width, input_channels)
        strides_patches = (
            padded_inputs.strides[0], 
            stride_h * padded_inputs.strides[1], 
            stride_w * padded_inputs.strides[2],
            padded_inputs.strides[1], 
            padded_inputs.strides[2], 
            padded_inputs.strides[3]
        )
        input_patches = np.lib.stride_tricks.as_strided(padded_inputs, shape=shape_patches, strides=strides_patches)

        self.d_filters = np.einsum('bhwkyc,bhwf->fkyc', input_patches, grad) / inputs.shape[0]

        # Compute gradient for input
        flipped_filters = np.flip(self.filters, axis=(1, 2))
        padded_grad = np.pad(grad, ((0, 0), (kernel_height - 1, kernel_height - 1), (kernel_width - 1, kernel_width - 1), (0, 0)), mode='constant')

        shape_grad_patches = (batch_size, input_height, input_width, kernel_height, kernel_width, num_filters)
        strides_grad_patches = (
            padded_grad.strides[0], 
            stride_h * padded_grad.strides[1], 
            stride_w * padded_grad.strides[2],
            padded_grad.strides[1], 
            padded_grad.strides[2], 
            padded_grad.strides[3]
        )
        grad_patches = np.lib.stride_tricks.as_strided(padded_grad, shape=shape_grad_patches, strides=strides_grad_patches)

        d_input_padded = np.einsum('bhwkyf,fkyc->bhwc', grad_patches, flipped_filters)

        # Handle padding for input gradient
        if self.padding == 'same' and (kernel_height > 1 or kernel_width > 1):
            d_input = d_input_padded[:, pad_h:-pad_h, pad_w:-pad_w, :]
        elif self.padding == 'valid':
            d_input = d_input_padded

        # Compute biases gradient
        self.d_filters /= batch_size
        self.d_biases = np.mean(grad, axis=(0, 1, 2))

        # Handle different input dimensions
        if self.inputs.ndim == 3:
            d_input = np.squeeze(d_input, axis=0)

        return d_input

    def update(self) -> None:
        """
        Update layer parameters using the optimizer
        """
        if self.optimizer:
            self.optimizer.update(self, 'filters', self.d_filters)
            self.optimizer.update(self, 'biases', self.d_biases)

    def _init_optimizer(self, optimizer):
        """
        Initialize the optimizer for the layer
        
        Args:
            optimizer: Optimizer to be used for parameter updates
        """
        self.optimizer = optimizer
        self.optimizer.initialize(self, 'filters')
        self.optimizer.initialize(self, 'biases')