import numpy as np
from typing import Optional, Callable, Tuple, Union, List
from utils import ActivationFunctions, Optimizer
from scipy.signal import convolve2d

class Layer:
    """
    Base class for neural network layers.

    Attributes:
        num_neurons (Optional[int]): Number of neurons in the layer. Defaults to None.
        num_inputs (Optional[int]): Number of input features to the layer. Defaults to None.
        activation_function (Optional[Callable[[np.ndarray], np.ndarray]]): Activation function to apply to the layer's output. Defaults to None.
        threshold (float): Threshold value for the layer's output scaling. Defaults to 1.0.
        signals (Optional[np.ndarray]): The output signals of the layer after the forward pass. Defaults to None.
        inputs (Optional[np.ndarray]): The inputs to the layer during the forward pass. Defaults to None.
        optimizer (Optional[Optimizer]): Optimizer object for updating the layer's parameters. Defaults to None.
    """
    def __init__(self, num_neurons: Optional[int] = None, num_inputs: Optional[int] = None,
                 activation_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 threshold: float = 1.0):
        """Initializes the base layer.

        Args:
            num_neurons (Optional[int]): Number of neurons in the layer.
            num_inputs (Optional[int]): Number of input features to the layer.
            activation_function (Optional[Callable[[np.ndarray], np.ndarray]]): Activation function for the layer's output.
            threshold (float): Threshold value for scaling the layer's output. Defaults to 1.0.
        """
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.activation_function = activation_function
        self.threshold = threshold
        self.signals = None
        self.inputs = None
        self.optimizer: Optional[Optimizer] = None

    def _init_optimizer(self, optimizer: Optimizer):
        """Initializes the optimizer for the layer.

        Args:
            optimizer (Optimizer): The optimizer object to use.
        """
        self.optimizer = optimizer

    def update(self) -> None:
        """Updates the parameters of the layer using the optimizer."""
        if self.optimizer:
            params_and_grads = self._get_params_and_grads()
            self.optimizer.update(params_and_grads)

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Returns a list of (parameter, gradient) tuples for the layer.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

class Flatten(Layer):
    """
    Layer that flattens the input tensor.
    """
    def __init__(self):
        """Initializes the flatten layer."""
        super().__init__(num_neurons=None, num_inputs=None, activation_function=None)
        self.original_shape = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Performs the forward pass by flattening the input.

        Args:
            inputs (np.ndarray): The input tensor.

        Returns:
            np.ndarray: The flattened output tensor.
        """
        self.original_shape = inputs.shape
        # Flatten the input, keeping the first dimension (batch size)
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Performs the backward pass by reshaping the gradient to the original shape.

        Args:
            grad (np.ndarray): The gradient from the next layer.

        Returns:
            np.ndarray: The reshaped gradient.
        """
        return grad.reshape(self.original_shape)

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Returns an empty list as the Flatten layer has no trainable parameters."""
        return []

class DenseLayer(Layer):
    """
    Optimized fully connected (dense) layer using NumPy.

    Attributes:
        weights (np.ndarray): Weight matrix of shape (num_neurons, num_inputs).
        biases (np.ndarray): Bias vector of shape (num_neurons,).
        gradients (Optional[Tuple[np.ndarray, np.ndarray]]): Gradients of weights and biases. Defaults to None.
        inputs (Optional[np.ndarray]): Input to the layer during the forward pass. Defaults to None.
        signals (Optional[np.ndarray]): Output of the layer after activation. Defaults to None.
    """
    def __init__(self, num_neurons: int, num_inputs: int, activation_function, threshold: float = 1.0):
        """Initializes the optimized dense layer.

        Args:
            num_neurons (int): Number of neurons in the layer.
            num_inputs (int): Number of input features to each neuron.
            activation_function (Callable[[np.ndarray], np.ndarray]): Activation function for the neurons.
            threshold (float): Threshold value for the layer's output. Defaults to 1.0.
        """
        super().__init__(num_neurons, num_inputs, activation_function, threshold)
        # Initialize weights with He initialization
        scale = np.sqrt(2.0 / num_inputs)
        self.weights = np.random.randn(num_neurons, num_inputs) * scale
        # Initialize biases to zeros
        self.biases = np.zeros(num_neurons)
        self.gradients = None
        self.inputs = None
        self.signals = None
        self.d_biases = None # Initialize d_biases here

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Performs the forward pass through the dense layer.

        Args:
            inputs (np.ndarray): The input to the layer.

        Returns:
            np.ndarray: The output signals of the layer after activation.
        """
        self.inputs = inputs
        # Calculate the weighted sum of inputs and add biases
        z = np.dot(inputs, self.weights.T) + self.biases
        # Apply the activation function and the threshold
        self.signals = self.activation_function(z) * self.threshold
        return self.signals

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Performs the backward pass through the dense layer.

        Args:
            grad (np.ndarray): The gradient from the next layer.

        Returns:
            np.ndarray: The gradient passed to the previous layer.
        """
        # Assuming you have an ActivationFunctions class with a derivative method
        derivative = derivative(self.activation_function, 0, self.signals)
        # Calculate the delta for this layer
        delta = grad * derivative
        # Calculate the gradients of the weights
        self.gradients = np.dot(self.inputs.T, delta).T / self.inputs.shape[0] if self.inputs.shape[0] > 0 else np.dot(self.inputs.T, delta).T
        # Calculate the gradient for the biases (mean over the batch)
        self.d_biases = np.mean(delta, axis=0)
        # Calculate the gradient for the previous layer
        return np.dot(delta, self.weights)

    def _init_optimizer(self, optimizer: 'Optimizer'):
        """Initializes the optimizer for the weights and biases of the layer.

        Args:
            optimizer (Optimizer): The optimizer object to use.
        """
        super()._init_optimizer(optimizer)
        # Register weights and biases as parameters to be optimized
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

class Conv1DLayer(Layer):
    """
    Performs 1D convolution, useful for sequence data like text, audio, or time series.
    """
    def __init__(self, num_filters: int, kernel_size: int, stride: int = 1, padding: str = 'valid', activation_function: Optional[Callable] = None, input_shape: Optional[Tuple[int, int, int]] = None):
        super().__init__(num_neurons=None, num_inputs=None, activation_function=activation_function)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding.lower()
        if self.padding not in ['valid', 'same']:
            raise ValueError(f"Invalid padding type: '{padding}'. Must be 'valid' or 'same'.")
        self.input_shape = input_shape
        self.filters = None
        self.biases = None
        self.d_filters = None
        self.d_biases = None

    def _initialize_filters_biases(self, input_shape):
        if self.filters is None:
            input_channels = input_shape[-1]
            scale = np.sqrt(2.0 / (self.kernel_size * input_channels))
            self.filters = np.random.randn(self.num_filters, self.kernel_size, input_channels) * scale
            self.biases = np.zeros(self.num_filters)
            if self.optimizer:
                self.optimizer.register_parameter(self.filters, 'filters')
                self.optimizer.register_parameter(self.biases, 'biases')

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._initialize_filters_biases(inputs.shape)
        # Placeholder for 1D convolution implementation
        raise NotImplementedError("Conv1DLayer forward pass not yet implemented.")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Placeholder for Conv1D backward pass implementation
        raise NotImplementedError("Conv1DLayer backward pass not yet implemented.")

    def _init_optimizer(self, optimizer):
        super()._init_optimizer(optimizer)
        if self.filters is not None and self.biases is not None and self.optimizer is not None:
            self.optimizer.register_parameter(self.filters, 'filters')
            self.optimizer.register_parameter(self.biases, 'biases')

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        params_and_grads = []
        if self.d_filters is not None and self.d_biases is not None and self.filters is not None and self.biases is not None:
            params_and_grads.extend([
                (self.filters, self.d_filters),
                (self.biases, self.d_biases)
            ])
        return params_and_grads

class Conv2DLayer(Layer):
    """
    Performs 2D convolution, the fundamental building block for processing image data.
    It learns spatial hierarchies of features.
    """
    def __init__(self,
                 num_filters: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int = 1,
                 padding: str = 'valid',
                 activation_function: Optional[Callable] = None,
                 input_shape: Optional[Tuple[int, int, int]] = None):
        super().__init__(num_neurons=None, num_inputs=None, activation_function=activation_function)
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding.lower()
        if self.padding not in ['valid', 'same']:
            raise ValueError(f"Invalid padding type: '{padding}'. Must be 'valid' or 'same'.")

        self.inputs = None
        self.signals = None
        self.input_shape = input_shape
        self.d_filters = None
        self.d_biases = None
        self.filters = None
        self.biases = np.zeros(self.num_filters)

    def _initialize_filters_biases(self, input_shape):
        if self.filters is None:
            input_channels = input_shape[-1]
            scale = np.sqrt(2.0 / (np.prod(self.kernel_size) * input_channels))
            self.filters = np.random.randn(self.num_filters, *self.kernel_size, input_channels) * scale
            if self.optimizer:
                self.optimizer.register_parameter(self.filters, 'filters')
                self.optimizer.register_parameter(self.biases, 'biases')

    def _im2col(self, inputs: np.ndarray) -> np.ndarray:
        batch_size, input_height, input_width, input_channels = inputs.shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride, self.stride

        if self.padding == 'same':
            pad_h = int(((input_height - 1) * stride_h + kernel_height - input_height) / 2)
            pad_w = int(((input_width - 1) * stride_w + kernel_width - input_width) / 2)
            padded_inputs = np.pad(inputs, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:
            padded_inputs = inputs

        padded_height, padded_width = padded_inputs.shape[1], padded_inputs.shape[2]
        output_height = (padded_height - kernel_height) // stride_h + 1
        output_width = (padded_width - kernel_width) // stride_w + 1

        col = np.zeros((batch_size, kernel_height * kernel_width * input_channels, output_height * output_width))

        for b in range(batch_size):
            for y in range(output_height):
                for x in range(output_width):
                    start_y = y * stride_h
                    end_y = start_y + kernel_height
                    start_x = x * stride_w
                    end_x = start_x + kernel_width
                    patch = padded_inputs[b, start_y:end_y, start_x:end_x, :].reshape(-1)
                    col[b, :, y * output_width + x] = patch
        return col

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._initialize_filters_biases(inputs.shape)
        self.inputs = inputs
        batch_size, input_height, input_width, input_channels = inputs.shape
        num_filters, kernel_height, kernel_width, _ = self.filters.shape
        stride_h, stride_w = self.stride, self.stride

        if self.padding == 'same':
            pad_h = int(((input_height - 1) * stride_h + kernel_height - input_height) / 2)
            pad_w = int(((input_width - 1) * stride_w + kernel_width - input_width) / 2)
            padded_inputs = np.pad(inputs, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:
            padded_inputs = inputs

        padded_height, padded_width = padded_inputs.shape[1], padded_inputs.shape[2]
        output_height = (padded_height - kernel_height) // stride_h + 1
        output_width = (padded_width - kernel_width) // stride_w + 1

        # im2col transformation
        input_cols = self._im2col(inputs)  # Shape: (batch_size, kernel_h*kernel_w*in_c, output_h*output_w)

        # Reshape filters for matrix multiplication
        filter_cols = self.filters.reshape(num_filters, -1).T  # Shape: (kernel_h*kernel_w*in_c, num_filters)

        # Perform convolution as matrix multiplication
        output_cols = np.einsum('bij,ik->bjk', input_cols, filter_cols) # Shape: (batch_size, output_h*output_w, num_filters)

        # Add biases
        output_cols += self.biases[None, None, :]

        # Reshape the output back to the original format
        output = output_cols.reshape(batch_size, output_height, output_width, num_filters)

        if self.activation_function:
            self.signals = self.activation_function(output)
            return self.signals
        else:
            self.signals = output
            return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        inputs = self.inputs
        batch_size, input_height, input_width, input_channels = inputs.shape
        num_filters, kernel_height, kernel_width, _ = self.filters.shape
        _, output_height, output_width, _ = grad.shape
        stride_h, stride_w = self.stride, self.stride

        # Apply activation derivative if activation function exists
        if self.activation_function:
            derivative = derivative(self.activation_function, 0, self.signals)
            grad = grad * derivative

        # im2col of the input
        input_cols = self._im2col(inputs) # Shape: (batch_size, kernel_h*kernel_w*in_c, output_h*output_w)

        # Gradient for biases
        self.d_biases = np.sum(grad, axis=(0, 1, 2)) / batch_size

        # Gradient for filters
        grad_reshaped = grad.transpose(0, 1, 2, 3).reshape(batch_size, output_height * output_width, num_filters) # Shape: (batch_size, output_h * output_w, num_filters)

        d_filters_unbatched = np.einsum('bpo,bof->fp', input_cols, grad_reshaped)
        self.d_filters = d_filters_unbatched.reshape(num_filters, kernel_height, kernel_width, input_channels) / batch_size


        # Gradient for the input (Transposed Convolution)
        d_input = np.zeros_like(inputs)

        for b in range(batch_size):
            for c in range(input_channels):
                for f in range(num_filters):
                    flipped_filter = np.flip(self.filters[f, :, :, c], axis=(0, 1))
                    conv_result = convolve2d(grad[b, :, :, f], flipped_filter, mode='full')
                    sh, sw = conv_result.shape
                    ih, iw = input_height, input_width
                    h_start = (sh - ih) // 2
                    w_start = (sw - iw) // 2
                    d_input[b, :, :, c] += conv_result[h_start:h_start + ih, w_start:w_start + iw]


        return d_input

    def _col2im(self, col: np.ndarray, input_shape: Tuple[int, int, int, int]) -> np.ndarray:
        batch_size, input_height, input_width, input_channels = input_shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride, self.stride

        if self.padding == 'same':
            pad_h = int(((input_height - 1) * stride_h + kernel_height - input_height) / 2)
            pad_w = int(((input_width - 1) * stride_w + kernel_width - input_width) / 2)
            padded_height, padded_width = input_height + 2 * pad_h, input_width + 2 * pad_w
        else:
            padded_height, padded_width = input_height, input_width

        output_height = (padded_height - kernel_height) // stride_h + 1
        output_width = (padded_width - kernel_width) // stride_w + 1

        d_input_padded = np.zeros((batch_size, padded_height, padded_width, input_channels))
        col_reshaped = col.reshape(batch_size, output_height, output_width, kernel_height, kernel_width, input_channels).transpose(0, 3, 4, 5, 1, 2)

        for b in range(batch_size):
            for y in range(output_height):
                for x in range(output_width):
                    start_y = y * stride_h
                    end_y = start_y + kernel_height
                    start_x = x * stride_w
                    end_x = start_x + kernel_width
                    d_input_padded[b, start_y:end_y, start_x:end_x, :] += col_reshaped[b, :, :, :, y, x, :]

        if self.padding == 'same':
            return d_input_padded[:, pad_h:-pad_h, pad_w:-pad_w, :]
        else:
            return d_input_padded

    def _init_optimizer(self, optimizer):
        super()._init_optimizer(optimizer)
        if self.filters is not None and self.biases is not None and self.optimizer is not None:
            self.optimizer.register_parameter(self.filters, 'filters')
            self.optimizer.register_parameter(self.biases, 'biases')

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        params_and_grads = []
        if self.d_filters is not None and self.d_biases is not None and self.filters is not None and self.biases is not None:
            params_and_grads.extend([
                (self.filters, self.d_filters),
                (self.biases, self.d_biases)
            ])
        return params_and_grads

class Conv3DLayer(Layer):
    """
    Performs 3D convolution, used for processing volumetric data like 3D medical scans or video.
    """
    def __init__(self, num_filters: int, kernel_size: Union[int, Tuple[int, int, int]], stride: int = 1, padding: str = 'valid', activation_function: Optional[Callable] = None, input_shape: Optional[Tuple[int, int, int, int, int]] = None):
        super().__init__(num_neurons=None, num_inputs=None, activation_function=activation_function)
        self.num_filters = num_filters
        self.kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding.lower()
        if self.padding not in ['valid', 'same']:
            raise ValueError(f"Invalid padding type: '{padding}'. Must be 'valid' or 'same'.")
        self.input_shape = input_shape
        self.filters = None
        self.biases = None
        self.d_filters = None
        self.d_biases = None

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
        self._initialize_filters_biases(inputs.shape)
        raise NotImplementedError("Conv3DLayer forward pass not yet implemented.")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Conv3DLayer backward pass not yet implemented.")

    def _init_optimizer(self, optimizer):
        super()._init_optimizer(optimizer)
        if self.filters is not None and self.biases is not None and self.optimizer is not None:
            self.optimizer.register_parameter(self.filters, 'filters')
            self.optimizer.register_parameter(self.biases, 'biases')

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        params_and_grads = []
        if self.d_filters is not None and self.d_biases is not None and self.filters is not None and self.biases is not None:
            params_and_grads.extend([
                (self.filters, self.d_filters),
                (self.biases, self.d_biases)
            ])
        return params_and_grads

class SeparableConv2DLayer(Layer):
    """
    Performs a depthwise separable convolution, which can be more efficient than standard Conv2D layers
    with a similar number of parameters.
    """
    def __init__(self, num_filters: int, kernel_size: Union[int, Tuple[int, int]], stride: int = 1, padding: str = 'valid', activation_function: Optional[Callable] = None, input_shape: Optional[Tuple[int, int, int]] = None):
        super().__init__(num_neurons=None, num_inputs=None, activation_function=activation_function)
        self.num_filters = num_filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding.lower()
        if self.padding not in ['valid', 'same']:
            raise ValueError(f"Invalid padding type: '{padding}'. Must be 'valid' or 'same'.")
        self.input_shape = input_shape
        self.depthwise_filters = None
        self.pointwise_filters = None
        self.biases = None
        self.d_depthwise_filters = None
        self.d_pointwise_filters = None
        self.d_biases = None

    def _initialize_filters_biases(self, input_shape):
        if self.depthwise_filters is None:
            input_channels = input_shape[-1]
            scale_depthwise = np.sqrt(2.0 / (np.prod(self.kernel_size) * 1)) # 1 because each input channel is processed independently
            self.depthwise_filters = np.random.randn(input_channels, *self.kernel_size, 1) * scale_depthwise
            scale_pointwise = np.sqrt(2.0 / (input_channels * 1 * 1)) # 1x1 convolution
            self.pointwise_filters = np.random.randn(self.num_filters, 1, 1, input_channels) * scale_pointwise
            self.biases = np.zeros(self.num_filters)
            if self.optimizer:
                self.optimizer.register_parameter(self.depthwise_filters, 'depthwise_filters')
                self.optimizer.register_parameter(self.pointwise_filters, 'pointwise_filters')
                self.optimizer.register_parameter(self.biases, 'biases')

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._initialize_filters_biases(inputs.shape)
        raise NotImplementedError("SeparableConv2DLayer forward pass not yet implemented.")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError("SeparableConv2DLayer backward pass not yet implemented.")

    def _init_optimizer(self, optimizer):
        super()._init_optimizer(optimizer)
        if self.depthwise_filters is not None and self.pointwise_filters is not None and self.biases is not None and self.optimizer is not None:
            self.optimizer.register_parameter(self.depthwise_filters, 'depthwise_filters')
            self.optimizer.register_parameter(self.pointwise_filters, 'pointwise_filters')
            self.optimizer.register_parameter(self.biases, 'biases')

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        params_and_grads = []
        if self.d_depthwise_filters is not None and self.d_pointwise_filters is not None and self.d_biases is not None and self.depthwise_filters is not None and self.pointwise_filters is not None and self.biases is not None:
            params_and_grads.extend([
                (self.depthwise_filters, self.d_depthwise_filters),
                (self.pointwise_filters, self.d_pointwise_filters),
                (self.biases, self.d_biases)
            ])
        return params_and_grads

class DepthwiseConv2DLayer(Layer):
    """
    A type of separable convolution where each input channel is convolved independently.
    """
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], stride: int = 1, padding: str = 'valid', activation_function: Optional[Callable] = None, input_shape: Optional[Tuple[int, int, int]] = None):
        super().__init__(num_neurons=None, num_inputs=None, activation_function=activation_function)
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding.lower()
        if self.padding not in ['valid', 'same']:
            raise ValueError(f"Invalid padding type: '{padding}'. Must be 'valid' or 'same'.")
        self.input_shape = input_shape
        self.filters = None
        self.biases = None
        self.d_filters = None
        self.d_biases = None

    def _initialize_filters_biases(self, input_shape):
        if self.filters is None:
            input_channels = input_shape[-1]
            scale = np.sqrt(2.0 / (np.prod(self.kernel_size) * 1)) # 1 because each input channel has its own filter
            self.filters = np.random.randn(input_channels, *self.kernel_size, 1) * scale
            self.biases = np.zeros(input_channels) # One bias per input channel
            if self.optimizer:
                self.optimizer.register_parameter(self.filters, 'filters')
                self.optimizer.register_parameter(self.biases, 'biases')

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._initialize_filters_biases(inputs.shape)
        raise NotImplementedError("DepthwiseConv2DLayer forward pass not yet implemented.")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError("DepthwiseConv2DLayer backward pass not yet implemented.")

    def _init_optimizer(self, optimizer):
        super()._init_optimizer(optimizer)
        if self.filters is not None and self.biases is not None and self.optimizer is not None:
            self.optimizer.register_parameter(self.filters, 'filters')
            self.optimizer.register_parameter(self.biases, 'biases')

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        params_and_grads = []
        if self.d_filters is not None and self.d_biases is not None and self.filters is not None and self.biases is not None:
            params_and_grads.extend([
                (self.filters, self.d_filters),
                (self.biases, self.d_biases)
            ])
        return params_and_grads

class ConvTranspose2DLayer(Layer):
    """
    Performs the transpose of a convolution operation, often used for upsampling in tasks like image segmentation or generative models.
    """
    def __init__(self, num_filters: int, kernel_size: Union[int, Tuple[int, int]], stride: int = 1, padding: str = 'valid', activation_function: Optional[Callable] = None, input_shape: Optional[Tuple[int, int, int]] = None):
        super().__init__(num_neurons=None, num_inputs=None, activation_function=activation_function)
        self.num_filters = num_filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding.lower()
        if self.padding not in ['valid', 'same']:
            raise ValueError(f"Invalid padding type: '{padding}'. Must be 'valid' or 'same'.")
        self.input_shape = input_shape
        self.filters = None
        self.biases = None
        self.d_filters = None
        self.d_biases = None

    def _initialize_filters_biases(self, input_shape):
        if self.filters is None:
            input_channels = input_shape[-1]
            scale = np.sqrt(2.0 / (np.prod(self.kernel_size) * input_channels))
            self.filters = np.random.randn(input_channels, *self.kernel_size, self.num_filters) * scale # Note the order of dimensions
            self.biases = np.zeros(self.num_filters)
            if self.optimizer:
                self.optimizer.register_parameter(self.filters, 'filters')
                self.optimizer.register_parameter(self.biases, 'biases')

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._initialize_filters_biases(inputs.shape)
        raise NotImplementedError("ConvTranspose2DLayer forward pass not yet implemented.")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ConvTranspose2DLayer backward pass not yet implemented.")

    def _init_optimizer(self, optimizer):
        super()._init_optimizer(optimizer)
        if self.filters is not None and self.biases is not None and self.optimizer is not None:
            self.optimizer.register_parameter(self.filters, 'filters')
            self.optimizer.register_parameter(self.biases, 'biases')

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        params_and_grads = []
        if self.d_filters is not None and self.d_biases is not None and self.filters is not None and self.biases is not None:
            params_and_grads.extend([
                (self.filters, self.d_filters),
                (self.biases, self.d_biases)
            ])
        return params_and_grads