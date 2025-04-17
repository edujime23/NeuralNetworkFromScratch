import itertools
import numpy as np
from typing import Optional, Callable, Tuple, Union, List
from .base import ConvolutionalLayer
from ..functions import derivative
from scipy.signal import convolve2d, convolve, fftconvolve
import collections.abc

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
        self.inputs = None
        self.signals = None
        self._normalized = False
        self._original_kernel_size = kernel_size
        self._original_stride = stride
        self._tmp_grad = None

    def _normalize_single_param(self,
                                 value: Union[int, collections.abc.Sequence[int]],
                                 num_spatial_dims: int,
                                 param_name: str) -> tuple[int, ...]:
        """
        Normalizes a single parameter (e.g., kernel size, stride) to a tuple.

        Args:
            value: The input value (int or sequence of ints).
            num_spatial_dims: The target number of spatial dimensions.
            param_name: The name of the parameter for error messages.

        Returns:
            A tuple of integers of length num_spatial_dims.

        Raises:
            ValueError: If the value is invalid (e.g., non-positive, wrong length).
            TypeError: If the value is not an int or sequence of ints.
        """
        if isinstance(value, int):
            if value <= 0:
                raise ValueError(f"{param_name} must be a positive integer, got {value}")
            return (value,) * num_spatial_dims
        # Check if it's a sequence (like tuple or list)
        elif isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
            if not all(isinstance(x, int) and x > 0 for x in value):
                raise ValueError(f"{param_name} sequence must contain only positive integers, got {value}")

            len_value = len(value)
            if len_value == num_spatial_dims:
                return tuple(value) # Ensure it's a tuple
            elif len_value < num_spatial_dims:
                # Pad with 1s
                return tuple(value) + (1,) * (num_spatial_dims - len_value)
            else: # len_value > num_spatial_dims
                raise ValueError(
                    f"{param_name} {value} has more dimensions ({len_value}) "
                    f"than input ({num_spatial_dims})"
                )
        else:
            raise TypeError(
                f"{param_name} must be an int or a sequence of ints, "
                f"got {type(value).__name__}"
            )


    def _normalize_params(self, num_spatial_dims: int):
        """
        Normalizes kernel size and stride to tuples based on the number of
        spatial dimensions.

        Args:
            num_spatial_dims (int): The number of spatial dimensions in the input.
                                     Must be positive.
        """
        if self._normalized:
            return

        if not isinstance(num_spatial_dims, int) or num_spatial_dims <= 0:
            raise ValueError(f"num_spatial_dims must be a positive integer, got {num_spatial_dims}")

        try:
            self.kernel_size = self._normalize_single_param(
                self._original_kernel_size, num_spatial_dims, "Kernel size"
            )
            self.stride = self._normalize_single_param(
                self._original_stride, num_spatial_dims, "Stride"
            )
        except (ValueError, TypeError) as e:
            # Optionally re-raise or handle normalization errors
            print(f"Error normalizing parameters: {e}")
            raise # Re-raise the specific error

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
        else:
            raise ValueError("Input shape must be provided to determine the number of spatial dimensions.")

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
        # Register filters and biases if they are initialized
        if self.filters is not None and self.biases is not None:
            self._reg_params()

        super()._init_optimizer(optimizer)

    def _pad_input_nd(self, inputs: np.ndarray, kernel_size: Tuple[int, ...], stride: Tuple[int, ...], padding: str) -> np.ndarray:
        """
        Pads the input tensor based on the specified padding type (Refined).

        Args:
            inputs (np.ndarray): The input tensor (e.g., shape N, D1, ..., Dn, C).
            kernel_size (Tuple[int, ...]): The size of the convolutional kernel (length n).
            stride (Tuple[int, ...]): The stride of the convolution (length n).
            padding (str): The padding strategy ('valid' or 'same').

        Returns:
            np.ndarray: The padded input tensor.

        Raises:
            ValueError: If an invalid padding type is specified.
            NotImplementedError: If _get_num_spatial_dims is not implemented.
        """
        num_spatial_dims = self._get_num_spatial_dims()
        if len(kernel_size) != num_spatial_dims or len(stride) != num_spatial_dims:
            raise ValueError("kernel_size and stride must have length equal to num_spatial_dims")

        # Initialize padding for Batch (0) and Channel (-1) dimensions
        padding_amounts = [(0, 0)] * (num_spatial_dims + 2) # Pre-allocate list size

        if padding == 'same':
            input_spatial_shape = inputs.shape[1:-1]
            if len(input_spatial_shape) != num_spatial_dims:
                raise ValueError("Input tensor spatial dimensions mismatch")

            for i in range(num_spatial_dims):
                input_len = input_spatial_shape[i]
                kernel_len = kernel_size[i]
                stride_len = stride[i]

                # Calculate output size for 'same' padding
                output_len = int(np.ceil(input_len / stride_len))

                # Calculate total padding needed for this dimension
                padding_needed = max(0, (output_len - 1) * stride_len + kernel_len - input_len)

                padding_before = padding_needed // 2
                padding_after = padding_needed - padding_before

                # Assign padding for the i-th spatial dimension (index i+1 in padding_amounts)
                padding_amounts[i+1] = (padding_before, padding_after)

        elif padding != 'valid':
            raise ValueError(f"Invalid padding type: {padding}. Choose 'valid' or 'same'.")

        return np.pad(inputs, padding_amounts, mode='constant')

    def _get_output_shape(self, input_shape: Tuple[int, ...], kernel_size: Tuple[int, ...], stride: Tuple[int, ...], padding: str) -> Tuple[int, ...]:
        """
        Calculates the output shape of the convolutional layer using optimized integer arithmetic.

        Args:
            input_shape (Tuple[int, ...]): The shape of the input tensor (including batch size).
                                            Expected format: (batch_size, spatial_dim_1, ..., spatial_dim_N, in_channels)
            kernel_size (Tuple[int, ...]): The size of the convolutional kernel across spatial dimensions.
            stride (Tuple[int, ...]): The stride of the convolution across spatial dimensions.
            padding (str): The padding strategy ('valid' or 'same').

        Returns:
            Tuple[int, ...]: The shape of the output tensor.
                             Format: (batch_size, out_spatial_dim_1, ..., out_spatial_dim_N, num_filters)

        Raises:
            ValueError: If an invalid padding type is specified.
            ValueError: If input_shape dimensions don't match expectations.
            ValueError: If kernel_size or stride dimensions don't match input spatial dimensions.
        """
        if len(input_shape) < 3:
            raise ValueError(f"Input shape must have at least 3 dimensions (batch, spatial..., channels), got {input_shape}")

        batch_size = input_shape[0]
        in_channels = input_shape[-1]
        input_spatial_shape = input_shape[1:-1]
        num_spatial_dims = len(input_spatial_shape)

        # --- Input Validation ---
        if num_spatial_dims != self._get_num_spatial_dims():
            raise ValueError(f"Input spatial dimensions ({num_spatial_dims}) does not match layer's expected spatial dimensions ({self._get_num_spatial_dims()})")
        if len(kernel_size) != num_spatial_dims:
            raise ValueError(f"Kernel size dimensions ({len(kernel_size)}) must match input spatial dimensions ({num_spatial_dims})")
        if len(stride) != num_spatial_dims:
            raise ValueError(f"Stride dimensions ({len(stride)}) must match input spatial dimensions ({num_spatial_dims})")

        output_spatial_shape = []

        if padding == 'same':
            output_spatial_shape = [
                (input_spatial_shape[i] + stride[i] - 1) // stride[i]
                for i in range(num_spatial_dims)
            ]
        elif padding == 'valid':
            output_spatial_shape = [
                (input_spatial_shape[i] - kernel_size[i]) // stride[i] + 1
                for i in range(num_spatial_dims)
            ]
        else:
            raise ValueError(f"Invalid padding type: {padding}. Choose 'valid' or 'same'.")

        # Check for invalid output dimensions
        for i, dim in enumerate(output_spatial_shape):
            if dim <= 0:
                raise ValueError(
                    f"Calculated output spatial dimension {i} is non-positive ({dim}). "
                    f"Input: {input_spatial_shape[i]}, Kernel: {kernel_size[i]}, Stride: {stride[i]}, Padding: {padding}. "
                    "Check kernel size and stride relative to input size."
                )

        return (batch_size, *output_spatial_shape, self.num_filters)

    def _extract_patches(self, padded_inputs: np.ndarray) -> np.ndarray:
        """
        Extracts patches from the padded input tensor using stride tricks.

        This method creates a view of the input data, avoiding costly data copying.

        Args:
            padded_inputs (np.ndarray): The padded input tensor.
                                        Expected shape: (batch_size, padded_dim_1, ..., padded_dim_N, in_channels)

        Returns:
            np.ndarray: A view of the input tensor representing the extracted patches.
                        Shape: (batch_size, out_dim_1, ..., out_dim_N, kernel_dim_1, ..., kernel_dim_N, in_channels)
        """
        # Get input dimensions
        batch_size = padded_inputs.shape[0]
        in_channels = padded_inputs.shape[-1]
        num_spatial_dims = self._get_num_spatial_dims()

        output_shape = self._get_output_shape(self.inputs.shape, self.kernel_size, self.stride, self.padding)
        output_spatial_shape = output_shape[1:-1] # Extract spatial dimensions from (batch, spatial..., filters)

        # Define the shape of the resulting patches array
        patches_shape = (batch_size,) + output_spatial_shape + self.kernel_size + (in_channels,)

        # Define the strides for the patches array view
        input_strides = padded_inputs.strides

        # Stride for batch dimension
        patches_strides = (input_strides[0],)

        # Strides for output spatial dimensions (stepping by stride * original spatial stride)
        for i in range(num_spatial_dims):
            patches_strides += (input_strides[i+1] * self.stride[i],) # Step by stride[i] elements in the padded input

        # Strides for kernel dimensions (stepping by 1 * original spatial stride)
        for i in range(num_spatial_dims):
            patches_strides += (input_strides[i+1],) # Step by 1 element in the padded input

        # Stride for channel dimension
        patches_strides += (input_strides[-1],)

        return np.lib.stride_tricks.as_strided(
            padded_inputs, shape=patches_shape, strides=patches_strides
        )

    def _should_use_fftconvolve(self, input_shape: Tuple[int, ...], kernel_size: Tuple[int, ...]) -> bool:
        """
        Determines if fftconvolve should be used based on input and kernel size.
        """
        num_spatial_dims = self._get_num_spatial_dims()
        if num_spatial_dims > 2:  # fftconvolve in scipy might not be optimal for very high dimensions
            return False
        input_spatial_dims = input_shape[1:-1]
        kernel_volume = np.prod(kernel_size)
        input_volume = np.prod(input_spatial_dims)
        # Heuristic: use fftconvolve if kernel size is reasonably large compared to input
        # This threshold might need tuning based on experimentation
        return kernel_volume > 50 # Example threshold

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the convolutional layer.

        Args:
            inputs (np.ndarray): The input tensor.

        Returns:
            np.ndarray: The output tensor after convolution and activation.
        """
        self.inputs = inputs
        if self.input_shape is None:
            self.input_shape = inputs.shape

        num_spatial_dims = self._get_num_spatial_dims()
        self._normalize_params(num_spatial_dims)
        self._initialize_filters_biases(inputs.shape)

        use_fft = self._should_use_fftconvolve(inputs.shape, self.kernel_size)

        if use_fft:
            output_shape = self._get_output_shape(inputs.shape, self.kernel_size, self.stride, self.padding)
            output = np.zeros(output_shape, dtype=inputs.dtype)
            batch_size = inputs.shape[0]
            in_channels = inputs.shape[-1]

            for b in range(batch_size):
                for out_f in range(self.num_filters):
                    # Perform convolution for each output filter by summing over input channels
                    conv_result = None
                    for in_c in range(in_channels):
                        input_slice = inputs[b, ..., in_c]
                        filter_slice = self.filters[out_f, ..., in_c]

                        if num_spatial_dims == 1:
                            mode = 'same' if self.padding == 'same' else 'valid'
                            conv_single = fftconvolve(input_slice, filter_slice, mode=mode)
                        elif num_spatial_dims == 2:
                            mode = 'same' if self.padding == 'same' else 'valid'
                            conv_single = fftconvolve(input_slice, filter_slice, mode=mode)
                        else:
                            # Handle higher dimensions - scipy's fftconvolve might not be ideal
                            # Consider using numpy.fft.fftn and inverse fftn for more control
                            raise NotImplementedError(f"fftconvolve for {num_spatial_dims} dimensions is not yet fully supported.")

                        if conv_result is None:
                            conv_result = conv_single
                        else:
                            conv_result += conv_single

                    # Add bias and store the result
                    output[b, ..., out_f] = conv_result + self.biases[out_f]

            if self.activation_function:
                self.signals = self.activation_function(output)
            else:
                self.signals = output

            return self.signals

        else:
            # Pad the input tensor
            padded_inputs = self._pad_input_nd(inputs, self.kernel_size, self.stride, self.padding)

            # Extract patches and reshape for efficient matrix multiplication
            patches = self._extract_patches(padded_inputs)
            patches_reshaped = patches.reshape(-1, np.prod(self.kernel_size) * inputs.shape[-1])
            filters_reshaped = self.filters.reshape(self.num_filters, -1).T

            # Perform convolution using matrix multiplication
            output_flat = patches_reshaped @ filters_reshaped + self.biases

            # Reshape the output to the correct dimensions
            output_shape = self._get_output_shape(inputs.shape, self.kernel_size, self.stride, self.padding)
            output = output_flat.reshape(output_shape)

            # Apply activation function if provided
            if self.activation_function:
                self.signals = self.activation_function(output)
            else:
                self.signals = output

            return self.signals

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of the convolutional layer using vectorized operations.

        Args:
            grad (np.ndarray): The incoming gradient from the next layer.

        Returns:
            np.ndarray: The gradient to be passed to the previous layer.
        """
        # Apply the derivative of the activation function if it exists
        if self.activation_function:
            derivative_values = derivative(self.activation_function, self.signals, arg_index=0, complex_diff=False)
            grad = grad * derivative_values

        batch_size = self.inputs.shape[0]
        num_spatial_dims = self._get_num_spatial_dims()

        # Pad the input tensor and extract patches (reusing existing methods)
        padded_input = self._pad_input_nd(self.inputs, self.kernel_size, self.stride, self.padding)
        patches = self._extract_patches(padded_input)

        # Vectorized reshape operations
        patches_shape = patches.shape
        num_patches = np.prod(patches_shape[1:1+num_spatial_dims])
        patches_reshaped = patches.reshape(batch_size, num_patches, -1)
        grad_reshaped = grad.reshape(batch_size, num_patches, self.num_filters)

        # Vectorized filter gradient computation
        # Reshape for efficient batch matrix multiplication
        patches_reshaped_t = np.transpose(patches_reshaped, (0, 2, 1))
        grad_reshaped_t = np.transpose(grad_reshaped, (0, 2, 1))

        # Compute d_filters using batch matrix multiplication
        d_filters_batch = np.matmul(grad_reshaped_t, patches_reshaped)
        self.d_filters = np.mean(d_filters_batch, axis=0).reshape(self.filters.shape)

        # Vectorized bias gradient computation
        self.d_biases = np.mean(grad.reshape(batch_size, -1, self.num_filters).sum(axis=1), axis=0)

        # Compute input gradient
        return self._compute_input_gradient(grad)

    def _compute_input_gradient(self, grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient with respect to the input tensor using optimized convolution.
        """
        num_spatial_dims = self._get_num_spatial_dims()
        batch_size, *spatial_dims, out_channels = grad.shape
        in_channels = self.inputs.shape[-1]

        # For 2D convolutions, use optimized implementation
        if num_spatial_dims == 2:
            return self._compute_input_gradient_2d(grad)

        # For other dimensions, use existing implementation
        # Flip filters for convolution transposition
        flipped_filters = np.flip(self.filters, axis=tuple(range(1, num_spatial_dims + 1)))
        transposed_filters = np.transpose(flipped_filters,
                                            ((1,) + tuple(range(2, num_spatial_dims + 1)) +
                                             (num_spatial_dims + 1, 0)))

        # Pad gradient if necessary
        if self.padding == 'valid':
            pad_width = [(0, 0)] + [(k - 1, k - 1) for k in self.kernel_size] + [(0, 0)]
            grad = np.pad(grad, pad_width, mode='constant')

        # Dilate gradient if stride > 1
        if any(s > 1 for s in self.stride):
            grad = self._dilate_gradient(grad)

        # Vectorized computation for other dimensions
        d_input = np.zeros_like(self.inputs)
        for b in range(batch_size):
            d_input[b] = self._compute_single_batch_gradient(
                grad[b], transposed_filters, in_channels, out_channels
            )

        return d_input

    def _compute_input_gradient_2d(self, grad: np.ndarray) -> np.ndarray:
        """
        Optimized input gradient computation for 2D convolutions.
        """
        batch_size, h, w, out_channels = grad.shape
        input_h, input_w = self.inputs.shape[1:3]
        in_channels = self.inputs.shape[-1]

        # Prepare filters
        flipped_filters = np.flip(np.flip(self.filters, axis=1), axis=2)
        filters_reshaped = flipped_filters.transpose(1, 2, 3, 0)  # [h, w, in_c, out_c]

        # Handle stride first
        if any(s > 1 for s in self.stride):
            grad = self._dilate_gradient(grad)

        # Calculate necessary padding
        if self.padding == 'valid':
            h_pad, w_pad = self.kernel_size
            grad = np.pad(grad, [(0,0), (h_pad-1, h_pad-1),
                                    (w_pad-1, w_pad-1), (0,0)],
                                mode='constant')

        # Pre-allocate output array
        d_input = np.zeros_like(self.inputs)

        # Compute convolution for each batch and channel
        for b in range(batch_size):
            grad_b = grad[b]
            for i in range(in_channels):
                # Process each output channel and accumulate results
                conv_results = []
                for o in range(out_channels):
                    curr_grad = grad_b[..., o]
                    curr_filter = filters_reshaped[..., i, o]

                    # Compute convolution with full output
                    result = convolve2d(curr_grad, curr_filter, mode='valid')
                    conv_results.append(result)

                # Sum all results
                total_result = np.sum(np.array(conv_results), axis=0)

                # Calculate centering offsets
                h_offset = (total_result.shape[0] - input_h) // 2
                w_offset = (total_result.shape[1] - input_w) // 2

                # Copy centered result to output
                d_input[b, :input_h, :input_w, i] = total_result[
                    h_offset:h_offset + input_h,
                    w_offset:w_offset + input_w
                ]

        return d_input

    def _compute_single_batch_gradient(self, grad_item, filters, in_channels, out_channels):
        """
        Helper method to compute gradient for a single batch item.
        """
        result = np.zeros(self.inputs.shape[1:])
        for i in range(in_channels):
            for o in range(out_channels):
                if self._get_num_spatial_dims() == 2:
                    result[..., i] += convolve2d(
                        grad_item[..., o],
                        filters[..., i, o],
                        mode='valid'
                    )
                else:
                    temp_conv = convolve(grad_item[..., o], filters[..., i, o], mode='valid')
                    slices = tuple(slice(0, result.shape[j]) for j in range(self._get_num_spatial_dims()))
                    result[..., i] += temp_conv[slices]
        return result

    def _dilate_gradient(self, grad: np.ndarray) -> np.ndarray:
        """
        Dilates the gradient based on the stride used in the forward pass.

        Args:
            grad (np.ndarray): The incoming gradient.

        Returns:
            np.ndarray: The dilated gradient.
        """
        num_spatial_dims = self._get_num_spatial_dims()
        dilated_shape = [
            (grad.shape[i + 1] - 1) * self.stride[i] + 1 for i in range(num_spatial_dims)
        ]
        dilated_grad = np.zeros((grad.shape[0], *dilated_shape, grad.shape[-1]))

        slices = tuple(
            slice(None, None, self.stride[i]) for i in range(num_spatial_dims)
        )
        dilated_grad[(slice(None), *slices, slice(None))] = grad

        return dilated_grad