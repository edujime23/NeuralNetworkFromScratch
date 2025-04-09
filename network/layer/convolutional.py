import itertools
import numpy as np
from typing import Optional, Callable, Tuple, Union, List
from .base import ConvolutionalLayer
from ..functions import derivative
from scipy.signal import convolve2d, convolve

class ConvNDLayer(ConvolutionalLayer):
    """
    A convolutional layer that can handle an arbitrary number of spatial dimensions.
    """
    def __init__(self,
                 num_filters: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: str = 'valid',
                 activation_function: Optional[Callable] = None,
                 input_shape: Optional[Tuple[int, ...]] = None):
        """
        Initializes the ConvNDLayer.

        Args:
            num_filters (int): The number of output filters (channels).
            kernel_size (Union[int, Tuple[int, ...]]): The size of the convolutional kernel.
            stride (Union[int, Tuple[int, ...]], optional): The stride of the convolution. Defaults to 1.
            padding (str, optional): The padding strategy ('valid' or 'same'). Defaults to 'valid'.
            activation_function (Optional[Callable], optional): The activation function to apply. Defaults to None.
            input_shape (Optional[Tuple[int, ...]], optional): The shape of the input tensor (excluding batch size). Defaults to None.
        """
        super().__init__(num_filters, kernel_size, stride, padding, activation_function, input_shape)
        # Store the input tensor during the forward pass
        self.inputs = None
        # Store the output tensor after the activation function
        self.signals = None
        # Flag to track if parameters have been normalized
        self._normalized = False
        # Store the original kernel size for normalization
        self._original_kernel_size = kernel_size
        # Store the original stride for normalization
        self._original_stride = stride
        # Temporary variable to store gradients (may be unused)
        self._tmp_grad = None

    def _normalize_params(self, num_spatial_dims: int):
        """
        Normalizes kernel size and stride to tuples based on the number of spatial dimensions.

        Args:
            num_spatial_dims (int): The number of spatial dimensions in the input.
        """
        if self._normalized:
            return
        # Handle integer kernel size by converting it to a tuple
        if isinstance(self._original_kernel_size, int):
            self.kernel_size = (self._original_kernel_size,) * num_spatial_dims
        # Pad kernel size tuple with ones if it's shorter than the number of spatial dimensions
        elif len(self._original_kernel_size) < num_spatial_dims:
            self.kernel_size = self._original_kernel_size + (1,) * (num_spatial_dims - len(self._original_kernel_size))
        # Raise an error if the kernel size has more dimensions than the input
        elif len(self._original_kernel_size) > num_spatial_dims:
            raise ValueError(f"Kernel size {self._original_kernel_size} has more dimensions than input {num_spatial_dims}")
        else:
            self.kernel_size = self._original_kernel_size

        # Handle integer stride by converting it to a tuple
        if isinstance(self._original_stride, int):
            self.stride = (self._original_stride,) * num_spatial_dims
        # Pad stride tuple with ones if it's shorter than the number of spatial dimensions
        elif len(self._original_stride) < num_spatial_dims:
            self.stride = self._original_stride + (1,) * (num_spatial_dims - len(self._original_stride))
        # Raise an error if the stride has more dimensions than the input
        elif len(self._original_stride) > num_spatial_dims:
            raise ValueError(f"Stride {self._original_stride} has more dimensions than input {num_spatial_dims}")
        else:
            self.stride = self._original_stride
        self._normalized = True

    def _get_num_spatial_dims(self) -> int:
        """
        Determines the number of spatial dimensions from the input shape.

        Returns:
            int: The number of spatial dimensions.

        Raises:
            ValueError: If input shape is not provided and input is None.
        """
        if self.inputs is not None:
            return len(self.inputs.shape) - 2
        if self.input_shape is None:
            raise ValueError("Input shape must be provided to determine the number of spatial dimensions.")
        return len(self.input_shape) - 2

    def _initialize_filters_biases(self, input_shape: Tuple[int, ...]):
        """
        Initializes the convolutional filters and biases.

        Args:
            input_shape (Tuple[int, ...]): The shape of the input tensor (including batch size).
        """
        if self.filters is None:
            num_spatial_dims = self._get_num_spatial_dims()
            in_channels = input_shape[-1]
            kernel_shape = (self.num_filters, *self.kernel_size, in_channels)
            # He initialization for filters
            scale = np.sqrt(2.0 / np.prod(self.kernel_size) * in_channels)
            self.filters = np.random.randn(*kernel_shape) * scale
            # Initialize biases to zeros
            self.biases = np.zeros(self.num_filters)
            # Register parameters with the optimizer if it exists
            if self.optimizer:
                self._reg_params()

    def _reg_params(self):
        """
        Registers the filters and biases with the optimizer.
        """
        if self.optimizer:
            self.optimizer.register_parameter(self.filters, 'filters')
            self.optimizer.register_parameter(self.biases, 'biases')

    def _init_optimizer(self, optimizer):
        """
        Initializes the optimizer for the layer's parameters.

        Args:
            optimizer: The optimizer instance.
        """
        super()._init_optimizer(optimizer)
        # Register filters and biases if they are initialized
        if self.filters is not None and self.biases is not None:
            self._reg_params()

    def _pad_input_nd(self, inputs: np.ndarray, kernel_size: Tuple[int, ...], stride: Tuple[int, ...], padding: str) -> np.ndarray:
        """
        Pads the input tensor based on the specified padding type.

        Args:
            inputs (np.ndarray): The input tensor.
            kernel_size (Tuple[int, ...]): The size of the convolutional kernel.
            stride (Tuple[int, ...]): The stride of the convolution.
            padding (str): The padding strategy ('valid' or 'same').

        Returns:
            np.ndarray: The padded input tensor.

        Raises:
            ValueError: If an invalid padding type is specified.
        """
        num_spatial_dims = self._get_num_spatial_dims()
        padding_amounts = [(0, 0)]  # Batch dimension
        if padding == 'same':
            input_spatial_shape = inputs.shape[1:-1]
            for i in range(num_spatial_dims):
                input_len = input_spatial_shape[i]
                kernel_len = kernel_size[i]
                stride_len = stride[i]
                output_len = int(np.ceil(input_len / stride_len))
                padding_needed = max(0, (output_len - 1) * stride_len + kernel_len - input_len)
                padding_before = padding_needed // 2
                padding_after = padding_needed - padding_before
                padding_amounts.append((padding_before, padding_after))
        elif padding == 'valid':
            for _ in range(num_spatial_dims):
                padding_amounts.append((0, 0))
        else:
            raise ValueError(f"Invalid padding type: {padding}")
        padding_amounts.append((0, 0))  # Channel dimension
        return np.pad(inputs, padding_amounts, mode='constant')

    def _get_output_shape(self, input_shape: Tuple[int, ...], kernel_size: Tuple[int, ...], stride: Tuple[int, ...], padding: str) -> Tuple[int, ...]:
        """
        Calculates the output shape of the convolutional layer.

        Args:
            input_shape (Tuple[int, ...]): The shape of the input tensor (including batch size).
            kernel_size (Tuple[int, ...]): The size of the convolutional kernel.
            stride (Tuple[int, ...]): The stride of the convolution.
            padding (str): The padding strategy ('valid' or 'same').

        Returns:
            Tuple[int, ...]: The shape of the output tensor.

        Raises:
            ValueError: If an invalid padding type is specified.
        """
        batch_size = input_shape[0]
        in_channels = input_shape[-1]
        num_spatial_dims = self._get_num_spatial_dims()
        input_spatial_shape = input_shape[1:-1]
        output_spatial_shape = []

        if padding == 'same':
            output_spatial_shape.extend(
                int(np.ceil(input_spatial_shape[i] / stride[i]))
                for i in range(num_spatial_dims)
            )
        elif padding == 'valid':
            output_spatial_shape.extend(
                int((input_spatial_shape[i] - kernel_size[i]) / stride[i] + 1)
                for i in range(num_spatial_dims)
            )
        else:
            raise ValueError(f"Invalid padding type: {padding}")

        return (batch_size, *output_spatial_shape, self.num_filters)

    def _extract_patches(self, padded_inputs):
        """
        Extracts patches from the padded input tensor using stride tricks.

        Args:
            padded_inputs (np.ndarray): The padded input tensor.

        Returns:
            np.ndarray: The extracted patches.
        """
        batch_size = padded_inputs.shape[0]
        in_channels = padded_inputs.shape[-1]
        num_spatial_dims = self._get_num_spatial_dims()

        output_shape = self._get_output_shape(self.inputs.shape, self.kernel_size, self.stride, self.padding)
        output_spatial_shape = output_shape[1:-1]

        patches_shape = (batch_size,) + output_spatial_shape + self.kernel_size + (in_channels,)

        patches_strides = (padded_inputs.strides[0],)
        for i in range(num_spatial_dims):
            patches_strides += (padded_inputs.strides[i+1] * self.stride[i],)
        for i in range(num_spatial_dims):
            patches_strides += (padded_inputs.strides[i+1],)
        patches_strides += (padded_inputs.strides[-1],)

        return np.lib.stride_tricks.as_strided(padded_inputs, shape=patches_shape, strides=patches_strides)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the convolutional layer.

        Args:
            inputs (np.ndarray): The input tensor.

        Returns:
            np.ndarray: The output tensor after convolution and activation.
        """
        self.inputs = inputs
        # Infer input shape if not provided during initialization
        if self.input_shape is None:
            self.input_shape = inputs.shape

        num_spatial_dims = self._get_num_spatial_dims()
        self._normalize_params(num_spatial_dims)
        self._initialize_filters_biases(inputs.shape)

        batch_size = inputs.shape[0]
        # Pad the input tensor
        padded_inputs = self._pad_input_nd(inputs, self.kernel_size, self.stride, self.padding)

        # Extract patches from the padded input
        patches = self._extract_patches(padded_inputs)

        num_patches = np.prod(patches.shape[1:1+num_spatial_dims])
        patches_reshaped = patches.reshape(batch_size * num_patches, -1)
        filters_reshaped = self.filters.reshape(self.num_filters, -1)

        # Perform the convolution by matrix multiplication
        output_flat = np.matmul(patches_reshaped, filters_reshaped.T) + self.biases

        # Reshape the output to the correct dimensions
        output_shape = self._get_output_shape(inputs.shape, self.kernel_size, self.stride, self.padding)
        output = output_flat.reshape(output_shape)

        # Apply the activation function if provided
        if self.activation_function:
            self.signals = self.activation_function(output)
            return self.signals
        else:
            self.signals = output
            return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of the convolutional layer.

        Args:
            grad (np.ndarray): The incoming gradient from the next layer.

        Returns:
            np.ndarray: The gradient to be passed to the previous layer.
        """
        inputs = self.inputs
        batch_size = inputs.shape[0]
        in_channels = inputs.shape[-1]
        num_filters = self.filters.shape[0]
        num_spatial_dims = self._get_num_spatial_dims()

        # Apply the derivative of the activation function if it exists
        if self.activation_function:
            derivative_values = derivative(self.activation_function, self.signals, 'derivative', arg_index=0)
            grad = grad * derivative_values

        # Pad the input tensor
        padded_input = self._pad_input_nd(inputs, self.kernel_size, self.stride, self.padding)
        # Extract patches from the padded input
        patches = self._extract_patches(padded_input)

        num_patches = np.prod(patches.shape[1:1+num_spatial_dims])
        patches_reshaped = patches.reshape(batch_size, num_patches, -1)
        grad_reshaped = grad.reshape(batch_size, num_patches, num_filters)

        # Calculate the gradient of the filters
        self.d_filters = np.zeros_like(self.filters)
        for b in range(batch_size):
            d_filters_b = np.matmul(grad_reshaped[b].transpose(1, 0), patches_reshaped[b])
            self.d_filters += d_filters_b.reshape(self.filters.shape) / batch_size

        # Calculate the gradient of the biases
        self.d_biases = np.mean(np.sum(grad, axis=tuple(range(1, grad.ndim-1))), axis=0)

        # Compute the gradient with respect to the input
        return self._compute_input_gradient(grad)

    def _compute_input_gradient(self, grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient with respect to the input tensor.

        Args:
            grad (np.ndarray): The incoming gradient.

        Returns:
            np.ndarray: The gradient with respect to the input.
        """
        batch_size = self.inputs.shape[0]
        input_shape = self.inputs.shape
        num_spatial_dims = self._get_num_spatial_dims()

        # Dilate the gradient if the stride is greater than 1
        if any(s > 1 for s in self.stride):
            grad = self._dilate_gradient(grad)

        padded_grad = grad
        # Pad the gradient if the padding was 'valid' during the forward pass
        if self.padding == 'valid':
            pad_width = [(0, 0)]
            pad_width.extend(
                (self.kernel_size[i] - 1, self.kernel_size[i] - 1)
                for i in range(num_spatial_dims)
            )
            pad_width.append((0, 0))
            padded_grad = np.pad(padded_grad, pad_width, mode='constant')

        # Flip the filters for convolution with the gradient
        flipped_filters = np.flip(self.filters, axis=tuple(range(1, num_spatial_dims + 1)))

        d_input = np.zeros_like(self.inputs)

        # Perform convolution of the padded gradient with the flipped filters
        for b, i, f in itertools.product(range(batch_size), range(input_shape[-1]), range(self.filters.shape[0])):
            current_grad = padded_grad[b, ..., f]
            current_filter = flipped_filters[f, ..., i]

            if num_spatial_dims == 1:
                result = np.convolve(current_grad, current_filter, mode='valid')
            elif num_spatial_dims == 2:
                result = convolve2d(current_grad, current_filter, mode='valid')
            else:
                result = convolve(current_grad, current_filter, mode='valid')
            target_shape = input_shape[1:-1]
            result_shape = result.shape
            adjusted_result = np.zeros(target_shape)
            slices_result = []
            slices_target = []
            for dim in range(num_spatial_dims):
                if result_shape[dim] > target_shape[dim]:
                    diff = result_shape[dim] - target_shape[dim]
                    start = diff // 2
                    end = start + target_shape[dim]
                    slices_result.append(slice(start, end))
                    slices_target.append(slice(None))
                elif result_shape[dim] < target_shape[dim]:
                    diff = target_shape[dim] - result_shape[dim]
                    start = diff // 2
                    end = start + result_shape[dim]
                    slices_result.append(slice(None))
                    slices_target.append(slice(start, end))
                else:
                    slices_result.append(slice(None))
                    slices_target.append(slice(None))
            adjusted_result[tuple(slices_target)] = result[tuple(slices_result)]
            d_input[b, ..., i] += adjusted_result

        return d_input

    def _dilate_gradient(self, grad: np.ndarray) -> np.ndarray:
        """
        Dilates the gradient based on the stride used in the forward pass.

        Args:
            grad (np.ndarray): The incoming gradient.

        Returns:
            np.ndarray: The dilated gradient.
        """
        batch_size = grad.shape[0]
        num_filters = grad.shape[-1]
        num_spatial_dims = self._get_num_spatial_dims()

        dilated_spatial_shape = tuple(
            (grad.shape[d+1] - 1) * self.stride[d] + 1
            for d in range(num_spatial_dims)
        )

        dilated_shape = (batch_size,) + dilated_spatial_shape + (num_filters,)
        dilated_grad = np.zeros(dilated_shape)

        slices = tuple(
            slice(0, dilated_spatial_shape[d], self.stride[d])
            for d in range(num_spatial_dims)
        )

        for b in range(batch_size):
            for f in range(num_filters):
                dilated_grad[(b,) + slices + (f,)] = grad[b, ..., f]

        return dilated_grad