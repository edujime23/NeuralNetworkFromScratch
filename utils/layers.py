import numpy as np
from typing import Optional, Callable, Tuple, Union, List
from utils import Neuron, ActivationFunctions, Optimizer
import scipy.signal

class Layer:
    def __init__(self, num_neurons: Optional[int] = None, num_inputs: Optional[int] = None,
                 activation_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 threshold: float = 1.0):
        """Streamlined base layer initialization."""
        self.num_neurons, self.num_inputs = num_neurons, num_inputs
        self.activation_function = activation_function
        self.threshold = threshold
        self.signals = self.inputs = None
        self.optimizer: Optional[Optimizer] = None

    def _init_optimizer(self, optimizer: Optimizer):
        """Initializes the optimizer for the layer."""
        self.optimizer = optimizer

    def update(self) -> None:
        """Updates the parameters of the layer."""
        if self.optimizer:
            params_and_grads = self._get_params_and_grads()
            self.optimizer.update(params_and_grads)

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Returns a list of (parameter, gradient) tuples for the layer."""
        raise NotImplementedError

class Flatten(Layer):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.original_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self.original_shape)

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return []

class DenseLayer(Layer):
    def __init__(self, num_neurons: int, num_inputs: int, activation_function, threshold: float = 1.0):
        super().__init__(num_neurons, num_inputs, activation_function, threshold)
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]
        self.signals = np.zeros((1, num_neurons)) # Initialize as 2D array
        self.inputs = np.zeros((1, num_inputs)) # Initialize as 2D array
        self.gradients = None
        self.backward_delta = None

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
        self.gradients = np.dot(self.inputs.T, delta) / self.inputs.shape[0] if self.inputs.shape[0] > 0 else np.dot(self.inputs.T, delta)
        weights = np.array([neuron.weights for neuron in self.neurons])
        self.backward_delta = delta
        return np.dot(delta, weights)

    def _init_optimizer(self, optimizer: Optimizer):
        super()._init_optimizer(optimizer)
        for neuron in self.neurons:
            self.optimizer.register_parameter(neuron.weights, 'weights')
            self.optimizer.register_parameter(neuron.bias, 'bias')

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        params_and_grads = []
        if self.gradients is not None and self.backward_delta is not None:
            for i, neuron in enumerate(self.neurons):
                weight_grad = self.gradients[:, i] if self.gradients.ndim > 1 else self.gradients[i]
                bias_grad = self.backward_delta[:, i].mean() if self.backward_delta.ndim > 1 else self.backward_delta[i]
                params_and_grads.extend(
                    ((neuron.weights, weight_grad), (neuron.bias, bias_grad))
                )
        return params_and_grads

class Conv2DLayer(Layer):
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
        self.d_filters = None
        self.d_biases = None

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

    def _initialize_filters_biases(self, input_shape):
        if self.filters is None:
            input_channels = input_shape[-1]
            scale = np.sqrt(2.0 / (np.prod(self.kernel_size) * input_channels))
            self.filters = np.random.randn(self.num_filters, *self.kernel_size, input_channels) * scale
            self.biases = np.zeros(self.num_filters)
            if self.optimizer:
                self.optimizer.register_parameter(self.filters, 'filters')
                self.optimizer.register_parameter(self.biases, 'biases')

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform forward convolution

        Args:
            inputs: Input feature map

        Returns:
            Convolved and activated feature map
        """
        self._initialize_filters_biases(inputs.shape)
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
        output = np.einsum('bhwkyc,fkyc->bhwf', patches, self.filters, optimize="greedy") + self.biases.reshape(1, 1, 1, -1)

        # Apply activation function if specified
        if self.activation_function:
            self.signals = self.activation_function(output)
            return self.signals
        else:
            self.signals = output
            return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        inputs = self.inputs
        batch_size, input_height, input_width, input_channels = inputs.shape
        num_filters, kernel_height, kernel_width, num_input_channels_filters = self.filters.shape
        _, output_height, output_width, _ = grad.shape
        stride_h, stride_w = self.stride, self.stride

        # Apply activation derivative if activation function exists
        if self.activation_function:
            derivative = ActivationFunctions.derivative(self.activation_function, self.signals)
            grad = grad * derivative

        # Compute gradient for filters
        shape_patches = (batch_size, output_height, output_width, kernel_height, kernel_width, input_channels)
        strides_patches = (
            inputs.strides[0],
            stride_h * inputs.strides[1],
            stride_w * inputs.strides[2],
            inputs.strides[1],
            inputs.strides[2],
            inputs.strides[3]
        )
        input_patches = np.lib.stride_tricks.as_strided(inputs, shape=shape_patches, strides=strides_patches)
        self.d_filters = np.einsum('bhwkyc,bhwf->fkyc', input_patches, grad, optimize="greedy") / batch_size

        # Compute gradient for input (Transposed Convolution)
        flipped_filters = np.flip(self.filters, axis=(1, 2))
        d_input = np.zeros_like(self.inputs)

        for b in range(batch_size):
            for f in range(num_filters):
                for c in range(input_channels):
                    # Perform convolution of gradient with flipped filter in 'full' mode
                    conv_result = scipy.signal.convolve2d(
                        grad[b, :, :, f],
                        flipped_filters[f, :, :, c],
                        mode='full'
                    )
                    # Calculate padding needed to match input size
                    pad_h_total = conv_result.shape[0] - input_height
                    pad_w_total = conv_result.shape[1] - input_width
                    pad_top = pad_h_total // 2
                    pad_left = pad_w_total // 2
                    # Crop the result to the original input size
                    d_input[b, :, :, c] += conv_result[pad_top:pad_top + input_height, pad_left:pad_left + input_width]

        # Compute biases gradient
        self.d_biases = np.mean(grad, axis=(0, 1, 2))

        return d_input

    def _init_optimizer(self, optimizer: Optimizer):
        super()._init_optimizer(optimizer)
        if self.filters is not None and self.biases is not None and self.optimizer is not None:
            self.optimizer.register_parameter(self.filters, 'filters')
            self.optimizer.register_parameter(self.biases, 'biases')

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        params_and_grads = []
        if self.d_filters is not None and self.d_biases is not None and self.filters is not None and self.biases is not None:
            params_and_grads.extend(
                ((self.filters, self.d_filters), (self.biases, self.d_biases))
            )
        return params_and_grads