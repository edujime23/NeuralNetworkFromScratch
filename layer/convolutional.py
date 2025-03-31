import itertools
import numpy as np
from typing import Optional, Callable, Tuple, Union, List
from layer.base import ConvolutionalLayer
from utils.functions import derivative as deriv
from scipy.signal import convolve2d, convolve

class ConvNDLayer(ConvolutionalLayer):
    """
    Performs N-dimensional convolution with optimized implementation.
    """
    def __init__(self,
                 num_filters: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: str = 'valid',
                 activation_function: Optional[Callable] = None,
                 input_shape: Optional[Tuple[int, ...]] = None):
        super().__init__(num_filters, kernel_size, stride, padding, activation_function, input_shape)
        self.inputs = None
        self.signals = None
        self._normalized = False
        self._original_kernel_size = kernel_size
        self._original_stride = stride
        self._tmp_grad = None  # For caching intermediate results

    def _normalize_params(self, num_spatial_dims: int):
        if self._normalized:
            return
        if isinstance(self._original_kernel_size, int):
            self.kernel_size = (self._original_kernel_size,) * num_spatial_dims
        elif len(self._original_kernel_size) < num_spatial_dims:
            # Pad with 1s for remaining dimensions
            self.kernel_size = self._original_kernel_size + (1,) * (num_spatial_dims - len(self._original_kernel_size))
        elif len(self._original_kernel_size) > num_spatial_dims:
            raise ValueError(f"Kernel size {self._original_kernel_size} has more dimensions than input {num_spatial_dims}")
        else:
            self.kernel_size = self._original_kernel_size

        if isinstance(self._original_stride, int):
            self.stride = (self._original_stride,) * num_spatial_dims
        elif len(self._original_stride) < num_spatial_dims:
            # Pad with 1s for remaining dimensions
            self.stride = self._original_stride + (1,) * (num_spatial_dims - len(self._original_stride))
        elif len(self._original_stride) > num_spatial_dims:
            raise ValueError(f"Stride {self._original_stride} has more dimensions than input {num_spatial_dims}")
        else:
            self.stride = self._original_stride
        self._normalized = True

    def _get_num_spatial_dims(self) -> int:
        if self.inputs is not None:
            return len(self.inputs.shape) - 2
        if self.input_shape is None:
            raise ValueError("Input shape must be provided to determine the number of spatial dimensions.")
        return len(self.input_shape) - 2

    def _initialize_filters_biases(self, input_shape: Tuple[int, ...]):
        if self.filters is None:
            num_spatial_dims = self._get_num_spatial_dims()
            in_channels = input_shape[-1]
            kernel_shape = (self.num_filters, *self.kernel_size, in_channels)
            scale = np.sqrt(2.0 / np.prod(self.kernel_size) * in_channels)
            self.filters = np.random.randn(*kernel_shape) * scale
            self.biases = np.zeros(self.num_filters)
            if self.optimizer:
                # Ensure parameters are registered with the optimizer
                self._reg_params()

    def _reg_params(self):
        """Register parameters with the optimizer"""
        if self.optimizer:
            self.optimizer.register_parameter(self.filters, 'filters')
            self.optimizer.register_parameter(self.biases, 'biases')

    def _init_optimizer(self, optimizer):
        """Initialize optimizer and register parameters"""
        super()._init_optimizer(optimizer)
        if self.filters is not None and self.biases is not None:
            self._reg_params()

    def _pad_input_nd(self, inputs: np.ndarray, kernel_size: Tuple[int, ...], stride: Tuple[int, ...], padding: str) -> np.ndarray:
        num_spatial_dims = self._get_num_spatial_dims()
        padding_amounts = [(0, 0)]  # For batch size
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

        padding_amounts.append((0, 0))  # For channels
        return np.pad(inputs, padding_amounts, mode='constant')

    def _get_output_shape(self, input_shape: Tuple[int, ...], kernel_size: Tuple[int, ...], stride: Tuple[int, ...], padding: str) -> Tuple[int, ...]:
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
        """Extract patches using stride tricks for efficiency"""
        batch_size = padded_inputs.shape[0]
        in_channels = padded_inputs.shape[-1]
        num_spatial_dims = self._get_num_spatial_dims()
        
        # Calculate output spatial dimensions
        output_shape = self._get_output_shape(self.inputs.shape, self.kernel_size, self.stride, self.padding)
        output_spatial_shape = output_shape[1:-1]
        
        # Create proper shape and strides for as_strided
        patches_shape = (batch_size,) + output_spatial_shape + self.kernel_size + (in_channels,)
        
        # Calculate strides for each dimension
        patches_strides = (padded_inputs.strides[0],)
        for i in range(num_spatial_dims):
            patches_strides += (padded_inputs.strides[i+1] * self.stride[i],)
        for i in range(num_spatial_dims):
            patches_strides += (padded_inputs.strides[i+1],)
        patches_strides += (padded_inputs.strides[-1],)
        
        # Use as_strided for efficient patch extraction
        return np.lib.stride_tricks.as_strided(padded_inputs, shape=patches_shape, strides=patches_strides)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        if self.input_shape is None:
            self.input_shape = inputs.shape

        num_spatial_dims = self._get_num_spatial_dims()
        self._normalize_params(num_spatial_dims)
        self._initialize_filters_biases(inputs.shape)

        # Use im2col approach for efficient convolution
        batch_size = inputs.shape[0]
        padded_inputs = self._pad_input_nd(inputs, self.kernel_size, self.stride, self.padding)
        
        # Extract patches efficiently
        patches = self._extract_patches(padded_inputs)
        
        # Reshape for efficient matrix multiplication
        num_patches = np.prod(patches.shape[1:1+num_spatial_dims])
        patches_reshaped = patches.reshape(batch_size * num_patches, -1)
        filters_reshaped = self.filters.reshape(self.num_filters, -1)
        
        # Perform matrix multiplication
        output_flat = np.matmul(patches_reshaped, filters_reshaped.T) + self.biases
        
        # Reshape to correct output dimensions
        output_shape = self._get_output_shape(inputs.shape, self.kernel_size, self.stride, self.padding)
        output = output_flat.reshape(output_shape)
        
        if self.activation_function:
            self.signals = self.activation_function(output)
            return self.signals
        else:
            self.signals = output
            return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        inputs = self.inputs
        batch_size = inputs.shape[0]
        in_channels = inputs.shape[-1]
        num_filters = self.filters.shape[0]
        num_spatial_dims = self._get_num_spatial_dims()

        # Apply activation derivative if needed
        if self.activation_function:
            derivative_values = deriv(self.activation_function, 'all', self.signals)
            grad = grad * derivative_values

        # Calculate filter gradients using im2col approach
        padded_input = self._pad_input_nd(inputs, self.kernel_size, self.stride, self.padding)
        patches = self._extract_patches(padded_input)

        # Reshape patches for matrix multiplication
        num_patches = np.prod(patches.shape[1:1+num_spatial_dims])
        patches_reshaped = patches.reshape(batch_size, num_patches, -1)
        grad_reshaped = grad.reshape(batch_size, num_patches, num_filters)

        # Compute filter gradients
        self.d_filters = np.zeros_like(self.filters)
        for b in range(batch_size):
            d_filters_b = np.matmul(grad_reshaped[b].transpose(1, 0), patches_reshaped[b])
            self.d_filters += d_filters_b.reshape(self.filters.shape) / batch_size

        # Compute bias gradients (more efficiently)
        self.d_biases = np.mean(np.sum(grad, axis=tuple(range(1, grad.ndim-1))), axis=0)

        return self._compute_input_gradient(grad)

    def _compute_input_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Compute gradient with respect to inputs using transposed convolution"""
        batch_size = self.inputs.shape[0]
        input_shape = self.inputs.shape
        num_spatial_dims = self._get_num_spatial_dims()

        # Handle strides greater than 1 by inserting zeros
        if any(s > 1 for s in self.stride):
            grad = self._dilate_gradient(grad)

        # Pad the gradient for full convolution if needed
        padded_grad = grad
        if self.padding == 'valid':
            # For valid padding, we need to pad the gradient
            pad_width = [(0, 0)]  # No padding for batch dimension
            pad_width.extend(
                (self.kernel_size[i] - 1, self.kernel_size[i] - 1)
                for i in range(num_spatial_dims)
            )
            pad_width.append((0, 0))  # No padding for channel dimension
            padded_grad = np.pad(padded_grad, pad_width, mode='constant')

        # Flip filters for transposed convolution
        flipped_filters = np.flip(self.filters, axis=tuple(range(1, num_spatial_dims + 1)))

        # Initialize input gradients
        d_input = np.zeros_like(self.inputs)

        # Compute gradients for each batch and input channel
        for b, i, f in itertools.product(range(batch_size), range(input_shape[-1]), range(self.filters.shape[0])):
            # Get the gradient for this filter
            current_grad = padded_grad[b, ..., f]

            # Get the corresponding filter weights
            current_filter = flipped_filters[f, ..., i]

                # Compute convolution contribution
            if num_spatial_dims == 1:
                result = np.convolve(current_grad, current_filter, mode='valid')
            elif num_spatial_dims == 2:
                result = convolve2d(current_grad, current_filter, mode='valid')
            else:
                result = convolve(current_grad, current_filter, mode='valid')
            # Handle shape differences between result and target
            target_shape = input_shape[1:-1]
            result_shape = result.shape

            # Create a properly sized array for the result
            adjusted_result = np.zeros(target_shape)

            slices_result = []

            slices_target = []
            for dim in range(num_spatial_dims):
                if result_shape[dim] > target_shape[dim]:
                    # Need to crop the result
                    diff = result_shape[dim] - target_shape[dim]
                    start = diff // 2
                    end = start + target_shape[dim]
                    slices_result.append(slice(start, end))
                    slices_target.append(slice(None))
                elif result_shape[dim] < target_shape[dim]:
                    # Need to place the result in a larger array
                    diff = target_shape[dim] - result_shape[dim]
                    start = diff // 2
                    end = start + result_shape[dim]
                    slices_result.append(slice(None))
                    slices_target.append(slice(start, end))
                else:
                    # Dimensions match
                    slices_result.append(slice(None))
                    slices_target.append(slice(None))

            # Place the result in the properly sized array
            adjusted_result[tuple(slices_target)] = result[tuple(slices_result)]

            # Add to input gradients
            d_input[b, ..., i] += adjusted_result

        return d_input

    def _dilate_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Insert zeros between gradient values for stride > 1"""
        batch_size = grad.shape[0]
        num_filters = grad.shape[-1]
        num_spatial_dims = self._get_num_spatial_dims()
        
        # Calculate dilated gradient shape
        dilated_spatial_shape = tuple(
            (grad.shape[d+1] - 1) * self.stride[d] + 1
            for d in range(num_spatial_dims)
        )
        
        dilated_shape = (batch_size,) + dilated_spatial_shape + (num_filters,)
        dilated_grad = np.zeros(dilated_shape)
        
        # Create slices to insert gradient values at proper positions
        slices = tuple(
            slice(0, dilated_spatial_shape[d], self.stride[d])
            for d in range(num_spatial_dims)
        )
        
        # Insert values
        for b in range(batch_size):
            for f in range(num_filters):
                # Create dynamic indexing
                dilated_grad[(b,) + slices + (f,)] = grad[b, ..., f]
        
        return dilated_grad

class SeparableConv2DLayer(ConvolutionalLayer):
    """
    Performs a depthwise separable convolution, which can be more efficient than standard Conv2D layers
    with a similar number of parameters.
    """
    def __init__(self, num_filters: int, kernel_size: Union[int, Tuple[int, int]], stride: int = 1, padding: str = 'valid', activation_function: Optional[Callable] = None, input_shape: Optional[Tuple[int, int, int]] = None):
        super().__init__(num_filters, kernel_size, stride, padding, activation_function, input_shape)
        self.depthwise_filters = None
        self.pointwise_filters = None
        self.inputs = None
        self.depthwise_output = None

    def _get_num_spatial_dims(self) -> int:
        return 2

    def _initialize_filters_biases(self, input_shape):
        if self.depthwise_filters is None:
            input_channels = input_shape[-1]
            scale_depthwise = np.sqrt(2.0 / (np.prod(self.kernel_size) * 1)) # 1 because each input channel is processed independently
            self.depthwise_filters = np.random.randn(input_channels, *self.kernel_size, 1) * scale_depthwise
            scale_pointwise = np.sqrt(2.0 / (input_channels * 1 * 1)) # 1x1 convolution
            self.pointwise_filters = np.random.randn(self.num_filters, 1, 1, input_channels) * scale_pointwise
            self.biases = np.zeros(self.num_filters)
            if self.optimizer:
                self._reg_params()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._initialize_filters_biases(inputs.shape)
        self.inputs = inputs
        batch_size, input_height, input_width, input_channels = inputs.shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride, self.stride

        # Depthwise convolution
        depthwise_output = np.zeros_like(inputs)
        for i, c in itertools.product(range(batch_size), range(input_channels)):
            padded_input = self._pad_input_2d(inputs[i, :, :, c][..., np.newaxis], kernel_height, kernel_width, stride_h, stride_w, self.padding)
            convolved = convolve2d(padded_input[:, :, 0], self.depthwise_filters[c, :, :, 0], mode='valid')[::stride_h, ::stride_w]
            depthwise_output[i, :convolved.shape[0], :convolved.shape[1], c] = convolved

        self.depthwise_output = depthwise_output

        # Pointwise convolution (1x1 convolution)
        pointwise_output = np.zeros((batch_size, depthwise_output.shape[1], depthwise_output.shape[2], self.num_filters))
        for i in range(batch_size):
            for f in range(self.num_filters):
                for c in range(input_channels):
                    pointwise_output[i, :, :, f] += convolve2d(depthwise_output[i, :, :, c], self.pointwise_filters[f, 0, 0, c], mode='same')

        pointwise_output += self.biases

        if self.activation_function:
            self.signals = self.activation_function(pointwise_output)
            return self.signals
        else:
            self.signals = pointwise_output
            return pointwise_output

    def _pad_input_2d(self, input_2d: np.ndarray, kernel_height: int, kernel_width: int, stride_h: int, stride_w: int, padding: str) -> np.ndarray:
        input_height, input_width, _ = input_2d.shape
        if padding == 'same':
            output_height = int(np.ceil(input_height / stride_h))
            output_width = int(np.ceil(input_width / stride_w))
            pad_h_needed = max(0, (output_height - 1) * stride_h + kernel_height - input_height)
            pad_w_needed = max(0, (output_width - 1) * stride_w + kernel_width - input_width)
            pad_top = pad_h_needed // 2
            pad_bottom = pad_h_needed - pad_top
            pad_left = pad_w_needed // 2
            pad_right = pad_w_needed - pad_left
            return np.pad(input_2d, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
        elif padding == 'valid':
            return input_2d
        else:
            raise ValueError(f"Invalid padding type: {padding}")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        inputs = self.inputs
        batch_size, input_height, input_width, input_channels = inputs.shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride, self.stride

        # Apply activation derivative
        if self.activation_function:
            derivative = deriv(self.activation_function, 'all', self.signals)
            grad = grad * derivative

        # Gradient for biases
        self.d_biases = np.sum(grad, axis=(0, 1, 2)) / batch_size

        # Gradient for pointwise filters
        self.d_pointwise_filters = np.zeros_like(self.pointwise_filters)
        for i in range(batch_size):
            for f in range(self.num_filters):
                for c in range(input_channels):
                    self.d_pointwise_filters[f, 0, 0, c] += np.sum(self.depthwise_output[i, :, :, c] * grad[i, :, :, f])
        self.d_pointwise_filters /= batch_size

        # Gradient for depthwise output
        d_depthwise_output = np.zeros_like(self.depthwise_output)
        for i in range(batch_size):
            for c in range(input_channels):
                for f in range(self.num_filters):
                    d_depthwise_output[i, :, :, c] += convolve2d(grad[i, :, :, f], self.pointwise_filters[f, 0, 0, c], mode='same')

        # Gradient for depthwise filters
        self.d_depthwise_filters = np.zeros_like(self.depthwise_filters)
        for i in range(batch_size):
            for c in range(input_channels):
                padded_input = self._pad_input_2d(inputs[i, :, :, c][..., np.newaxis], kernel_height, kernel_width, stride_h, stride_w, self.padding)
                self.d_depthwise_filters[c, :, :, 0] += convolve2d(padded_input[:, :, 0], d_depthwise_output[i, :, :, c], mode='valid')[::stride_h, ::stride_w]
        self.d_depthwise_filters /= batch_size

        # Gradient for input
        d_input = np.zeros_like(inputs)
        for i in range(batch_size):
            for c in range(input_channels):
                padded_d_depthwise = self._pad_input_2d(d_depthwise_output[i, :, :, c][..., np.newaxis], kernel_height, kernel_width, stride_h, stride_w, self.padding)
                d_input[i, :, :, c] += convolve2d(padded_d_depthwise[:, :, 0], np.flip(np.flip(self.depthwise_filters[c, :, :, 0], axis=0), axis=1), mode='full')

        return d_input

    def _init_optimizer(self, optimizer):
        super()._init_optimizer(optimizer)
        if self.depthwise_filters is not None and self.pointwise_filters is not None and self.biases is not None and self.optimizer is not None:
            self._reg_params()

    def _reg_params(self):
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

class DepthwiseConv2DLayer(ConvolutionalLayer):
    """
    A type of separable convolution where each input channel is convolved independently.
    """
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], stride: int = 1, padding: str = 'valid', activation_function: Optional[Callable] = None, input_shape: Optional[Tuple[int, int, int]] = None):
        super().__init__(num_filters=None, kernel_size=kernel_size, stride=stride, padding=padding, activation_function=activation_function, input_shape=input_shape)
        self.inputs = None
        self.signals = None

    def _get_num_spatial_dims(self) -> int:
        return 2

    def _initialize_filters_biases(self, input_shape):
        if self.filters is None:
            input_channels = input_shape[-1]
            scale = np.sqrt(2.0 / (np.prod(self.kernel_size) * 1)) # 1 because each input channel has its own filter
            self.filters = np.random.randn(input_channels, *self.kernel_size, 1) * scale
            self.biases = np.zeros(input_channels) # One bias per input channel
            if self.optimizer:
                self.optimizer.register_parameter(self.filters, 'filters')
                self.optimizer.register_parameter(self.biases, 'biases')
        self.num_filters = self.filters.shape[0] # Set num_filters based on input channels

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._initialize_filters_biases(inputs.shape)
        self.inputs = inputs
        batch_size, input_height, input_width, input_channels = inputs.shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride, self.stride
        output_height = int((input_height - kernel_height + 2 * self._get_padding_amount_2d(input_height, kernel_height, stride_h, self.padding)) / stride_h + 1)
        output_width = int((input_width - kernel_width + 2 * self._get_padding_amount_2d(input_width, kernel_width, stride_w, self.padding)) / stride_w + 1)
        output = np.zeros((batch_size, output_height, output_width, input_channels))

        for i in range(batch_size):
            for c in range(input_channels):
                padded_input = self._pad_input_2d(inputs[i, :, :, c][..., np.newaxis], kernel_height, kernel_width, stride_h, stride_w, self.padding)
                convolved = convolve2d(padded_input[:, :, 0], self.filters[c, :, :, 0], mode='valid')[::stride_h, ::stride_w]
                output[i, :convolved.shape[0], :convolved.shape[1], c] = convolved + self.biases[c]

        if self.activation_function:
            self.signals = self.activation_function(output)
            return self.signals
        else:
            self.signals = output
            return output

    def _get_padding_amount_2d(self, input_dim: int, kernel_dim: int, stride: int, padding: str) -> int:
        if padding == 'same':
            output_dim = int(np.ceil(input_dim / stride))
            return max(0, (output_dim - 1) * stride + kernel_dim - input_dim)
        elif padding == 'valid':
            return 0
        else:
            raise ValueError(f"Invalid padding type: {padding}")

    def _pad_input_2d(self, input_2d: np.ndarray, kernel_height: int, kernel_width: int, stride_h: int, stride_w: int, padding: str) -> np.ndarray:
        input_height, input_width, _ = input_2d.shape
        if padding == 'same':
            output_height = int(np.ceil(input_height / stride_h))
            output_width = int(np.ceil(input_width / stride_w))
            pad_h_needed = max(0, (output_height - 1) * stride_h + kernel_height - input_height)
            pad_w_needed = max(0, (output_width - 1) * stride_w + kernel_width - input_width)
            pad_top = pad_h_needed // 2
            pad_bottom = pad_h_needed - pad_top
            pad_left = pad_w_needed // 2
            pad_right = pad_w_needed - pad_left
            return np.pad(input_2d, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
        elif padding == 'valid':
            return input_2d
        else:
            raise ValueError(f"Invalid padding type: {padding}")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        inputs = self.inputs
        batch_size, input_height, input_width, input_channels = inputs.shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride, self.stride

        # Apply activation derivative
        if self.activation_function:
            derivative = deriv(self.activation_function, 'all', self.signals)
            grad = grad * derivative

        # Initialize gradients
        self.d_filters = np.zeros_like(self.filters)
        self.d_biases = np.sum(grad, axis=(0, 1, 2)) / batch_size
        d_input = np.zeros_like(inputs)

        for i in range(batch_size):
            for c in range(input_channels):
                padded_input = self._pad_input_2d(inputs[i, :, :, c][..., np.newaxis], kernel_height, kernel_width, stride_h, stride_w, self.padding)
                grad_slice = grad[i, :, :, c]

                # Gradient for filters
                self.d_filters[c, :, :, 0] += convolve2d(padded_input[:, :, 0], grad_slice, mode='valid')[::stride_h, ::stride_w]

                # Gradient for input
                padded_grad_output = self._pad_input_2d(grad_slice[..., np.newaxis], kernel_height, kernel_width, stride_h, stride_w, self.padding)
                d_input[i, :, :, c] += convolve2d(padded_grad_output[:, :, 0], np.flip(np.flip(self.filters[c, :, :, 0], axis=0), axis=1), mode='valid')

        self.d_filters /= batch_size
        return d_input

class ConvTranspose2DLayer(ConvolutionalLayer):
    """
    Performs the transpose of a convolution operation, often used for upsampling in tasks like image segmentation or generative models.
    """
    def __init__(self, num_filters: int, kernel_size: Union[int, Tuple[int, int]], stride: int = 1, padding: str = 'valid', activation_function: Optional[Callable] = None, input_shape: Optional[Tuple[int, int, int]] = None):
        super().__init__(num_filters, kernel_size, stride, padding, activation_function, input_shape)
        self.inputs = None
        self.signals = None

    def _get_num_spatial_dims(self) -> int:
        return 2

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
        self.inputs = inputs
        batch_size, input_height, input_width, input_channels = inputs.shape
        num_filters, kernel_height, kernel_width, _ = self.filters.shape
        stride_h, stride_w = self.stride, self.stride

        output_height = (input_height - 1) * stride_h + kernel_height
        output_width = (input_width - 1) * stride_w + kernel_width
        output = np.zeros((batch_size, output_height, output_width, num_filters))

        for i in range(batch_size):
            for f in range(num_filters):
                for c in range(input_channels):
                    padded_input = np.zeros((input_height * stride_h, input_width * stride_w))
                    for h in range(input_height):
                        for w in range(input_width):
                            padded_input[h * stride_h, w * stride_w] = inputs[i, h, w, c]
                    convolved = convolve2d(padded_input, np.flip(np.flip(self.filters[c, :, :, f], axis=0), axis=1), mode='full')
                    output[i, :, :, f] += convolved

        if self.padding == 'valid':
            output = output[:, kernel_height - 1:output_height - (kernel_height - 1), kernel_width - 1:output_width - (kernel_width - 1), :]
        elif self.padding == 'same':
            pad_h = (kernel_height - 1)
            pad_w = (kernel_width - 1)
            output = output[:, pad_h // 2:output_height - pad_h // 2, pad_w // 2:output_width - pad_w // 2, :]

        output += self.biases

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
        stride_h, stride_w = self.stride, self.stride

        # Apply activation derivative
        if self.activation_function:
            derivative = deriv(self.activation_function, 'all', self.signals)
            grad = grad * derivative

        # Initialize gradients
        self.d_filters = np.zeros_like(self.filters)
        self.d_biases = np.sum(grad, axis=(0, 1, 2)) / batch_size
        d_input = np.zeros_like(inputs)

        for i, f in itertools.product(range(batch_size), range(num_filters)):
            for c in range(input_channels):
                # Gradient for filters
                self.d_filters[c, :, :, f] += convolve2d(inputs[i, :, :, c], grad[i, :, :, f], mode='valid')

                # Gradient for input
                padded_grad = np.zeros((grad.shape[1] * stride_h, grad.shape[2] * stride_w))
                for h in range(grad.shape[1]):
                    for w in range(grad.shape[2]):
                        padded_grad[h * stride_h, w * stride_w] = grad[i, h, w, f]

                d_input[i, :, :, c] += convolve2d(padded_grad, self.filters[c, :, :, f], mode='same')

        self.d_filters /= batch_size
        return d_input