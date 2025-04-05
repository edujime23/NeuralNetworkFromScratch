import numpy as np
from typing import Optional, Callable, List, Tuple, Union
from utils import Optimizer, ActivationFunctions

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
        self.activation_function = activation_function or ActivationFunctions.linear
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

class ConvolutionalLayer(Layer):
    """
    Base class for convolutional layers.
    """
    def __init__(self, num_filters: int, kernel_size: Union[int, Tuple[int, ...]], stride: int = 1, padding: str = 'valid', activation_function: Optional[Callable] = None, input_shape: Optional[Tuple[int, ...]] = None):
        super().__init__(num_neurons=None, num_inputs=None, activation_function=activation_function)
        self.num_filters = num_filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size  # Store original kernel_size
        self.stride = stride  # Store original stride
        self.padding = padding.lower()
        if self.padding not in ['valid', 'same']:
            raise ValueError(f"Invalid padding type: '{padding}'. Must be 'valid' or 'same'.")
        self.filters = None
        self.biases = None
        self.d_filters = None
        self.d_biases = None
        self._normalized_kernel_size = None  # Will be set during forward pass
        self._normalized_stride = None  # Will be set during forward pass

    def _normalize_kernel_size(self, kernel_size: Union[int, Tuple[int, ...]], num_dims: int) -> Tuple[int, ...]:
        if isinstance(kernel_size, int):
            return (kernel_size,) * num_dims
        return tuple(kernel_size)

    def _get_num_spatial_dims(self, input_shape: Optional[Tuple[int, ...]] = None) -> int:
        shape = input_shape if input_shape is not None else self.input_shape
        if shape is None:
            raise ValueError("Input shape must be provided to determine the number of spatial dimensions.")
        return len(shape) - 2  # Subtract batch size and number of channels

    def _normalize_shapes(self, input_shape: Tuple[int, ...]):
        num_dims = self._get_num_spatial_dims(input_shape)
        if self._normalized_kernel_size is None:
            self._normalized_kernel_size = self._normalize_kernel_size(self.kernel_size, num_dims)
        if self._normalized_stride is None:
            if isinstance(self.stride, int):
                self._normalized_stride = (self.stride,) * num_dims
            else:
                self._normalized_stride = tuple(self.stride)

    def _initialize_filters_biases(self, input_shape):
        if self.filters is None:
            self._normalize_shapes(input_shape)
            input_channels = input_shape[-1]
            scale = np.sqrt(2.0 / (np.prod(self._normalized_kernel_size) * input_channels))
            self.filters = np.random.randn(self.num_filters, *self._normalized_kernel_size, input_channels) * scale
            self.biases = np.zeros(self.num_filters)
            if self.optimizer:
                self.optimizer.register_parameter(self.filters, 'filters')
                self.optimizer.register_parameter(self.biases, 'biases')

    def _init_optimizer(self, optimizer: Optimizer):
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
    
class PoolingLayer(Layer):
    """
    Base class for pooling layers.
    """
    def __init__(self, pool_size: Union[int, Tuple[int, int]], stride: Optional[Union[int, Tuple[int, int]]] = None, input_shape: Optional[Tuple[int, ...]] = None):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.input_shape = input_shape
        self._normalized_pool_size = None
        self._normalized_stride = None

    def _normalize_shapes(self, input_shape: Tuple[int, ...]):
        num_dims = len(input_shape) - 2  # Subtract batch size and channels
        if self._normalized_pool_size is None:
            if isinstance(self.pool_size, int):
                self._normalized_pool_size = (self.pool_size,) * num_dims
            else:
                self._normalized_pool_size = tuple(self.pool_size)
        
        if self._normalized_stride is None:
            if self.stride is None:
                self._normalized_stride = self._normalized_pool_size
            elif isinstance(self.stride, int):
                self._normalized_stride = (self.stride,) * num_dims
            else:
                self._normalized_stride = tuple(self.stride)

    def _get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        raise NotImplementedError("Subclasses must implement _get_output_shape")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return self._pool_forward(inputs)

    def _pool_forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement _pool_forward")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement backward")
    
class RecurrentLayer(Layer):
    """
    Base class for recurrent layers.
    """
    def __init__(self, units: int, activation: Optional[Callable[[np.ndarray], np.ndarray]] = None, return_sequences: bool = False, trainable: bool = True):
        super().__init__(num_neurons=units, activation_function=activation)
        self.units = units  # Number of hidden units
        self.return_sequences = return_sequences  # Whether to return the full sequence of outputs or just the last one
        self.hidden_state = None
        self.inputs = None
        self.outputs = None
        self.trainable = trainable

    def _initialize_weights(self, input_shape):
        raise NotImplementedError(f"{self.__class__.__name__} must implement the '_initialize_weights' method to set up layer parameters.")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        batch_size, time_steps, input_dim = inputs.shape
        self.hidden_state = np.zeros((batch_size, self.units))  # Initialize hidden state
        self.outputs = np.zeros((batch_size, time_steps, self.units)) if self.return_sequences else np.zeros((batch_size, self.units))
        self.signals = self.outputs # Assign outputs to signals as per your Layer class structure

        for t in range(time_steps):
            # Process each time step
            output_at_t = self._step(inputs[:, t, :])
            if self.return_sequences:
                self.outputs[:, t, :] = output_at_t
            else:
                self.outputs = output_at_t
            self.signals = self.outputs # Update signals at each step (or at the end if not return_sequences)

        return self.outputs

    def _step(self, input_at_t: np.ndarray) -> np.ndarray:
        """
        Performs the recurrent computation for a single time step.
        To be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the '_step' method to define the recurrent computation for each time step.")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'backward' method to handle backpropagation through the recurrent layer.")

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement the '_get_params_and_grads' method to return trainable parameters and their gradients.")

    def _init_optimizer(self, optimizer: Optimizer):
        """Initializes the optimizer for the layer and its parameters.

        Args:
            optimizer (Optimizer): The optimizer object to use.
        """
        super()._init_optimizer(optimizer)