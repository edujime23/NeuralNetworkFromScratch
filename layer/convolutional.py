import itertools
import numpy as np
from typing import Optional, Callable, Tuple, Union, List
from layer.base import ConvolutionalLayer
from utils.functions import derivative as deriv
from scipy.signal import convolve2d

class Conv1DLayer(ConvolutionalLayer):
    """
    Performs 1D convolution, useful for sequence data like text, audio, or time series.
    """
    def __init__(self, num_filters: int, kernel_size: int, stride: int = 1, padding: str = 'valid', activation_function: Optional[Callable] = None, input_shape: Optional[Tuple[int, int, int]] = None):
        super().__init__(num_filters, kernel_size, stride, padding, activation_function, input_shape)
        self.inputs = None
        self.signals = None

    def _get_num_spatial_dims(self) -> int:
        return 1

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._initialize_filters_biases(inputs.shape)
        self.inputs = inputs
        batch_size, seq_len, in_channels = inputs.shape
        num_filters, kernel_size, _ = self.filters.shape
        output_len = int((seq_len - kernel_size + 2 * self._get_padding_amount(seq_len, kernel_size, self.stride, self.padding)) / self.stride + 1)
        output = np.zeros((batch_size, output_len, num_filters))

        for i in range(batch_size):
            for f in range(num_filters):
                for c in range(in_channels):
                    padded_input = self._pad_input_1d(inputs[i, :, c], kernel_size, self.stride, self.padding)
                    convolved = np.convolve(padded_input, self.filters[f, :, c], mode='valid')[::self.stride]
                    output[i, :, f] += convolved

        output += self.biases

        if self.activation_function:
            self.signals = self.activation_function(output)
            return self.signals
        else:
            self.signals = output
            return output

    def _pad_input_1d(self, input_1d: np.ndarray, kernel_size: int, stride: int, padding: str) -> np.ndarray:
        if padding == 'same':
            output_len = int(np.ceil(input_1d.shape[0] / stride))
            padding_needed = max(0, (output_len - 1) * stride + kernel_size - input_1d.shape[0])
            padding_before = padding_needed // 2
            padding_after = padding_needed - padding_before
            return np.pad(input_1d, (padding_before, padding_after), mode='constant')
        elif padding == 'valid':
            return input_1d
        else:
            raise ValueError(f"Invalid padding type: {padding}")

    def _get_padding_amount(self, input_len: int, kernel_size: int, stride: int, padding: str) -> int:
        if padding == 'same':
            output_len = int(np.ceil(input_len / stride))
            return max(0, (output_len - 1) * stride + kernel_size - input_len)
        elif padding == 'valid':
            return 0
        else:
            raise ValueError(f"Invalid padding type: {padding}")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        inputs = self.inputs
        batch_size, seq_len, in_channels = inputs.shape
        num_filters, kernel_size, _ = self.filters.shape
        output_len = grad.shape[1]
        stride = self.stride

        # Apply activation derivative
        if self.activation_function:
            derivative = deriv(self.activation_function, 0, self.signals)
            grad = grad * derivative

        # Initialize gradients
        self.d_filters = np.zeros_like(self.filters)
        self.d_biases = np.sum(grad, axis=(0, 1)) / batch_size
        d_input = np.zeros_like(inputs)

        for i in range(batch_size):
            for f in range(num_filters):
                for c in range(in_channels):
                    input_slice = self.inputs[i, :, c]
                    padded_input = self._pad_input_1d(input_slice, kernel_size, stride, self.padding)

                    # Gradient for filters
                    grad_slice = grad[i, :, f]
                    upsampled_grad = np.zeros(padded_input.shape[0])
                    for j in range(output_len):
                        index = j * stride
                        if index < upsampled_grad.shape[0]:
                            upsampled_grad[index] = grad_slice[j]
                    self.d_filters[f, :, c] += np.convolve(padded_input, upsampled_grad, mode='valid')

                    # Gradient for input
                    padded_grad_output = np.pad(grad_slice, (kernel_size - 1, kernel_size - 1), mode='constant')
                    d_input[i, :, c] += np.convolve(padded_grad_output, np.flip(self.filters[f, :, c]), mode='valid')

        self.d_filters /= batch_size
        return d_input

class Conv2DLayer(ConvolutionalLayer):
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
        super().__init__(num_filters, kernel_size, stride, padding, activation_function, input_shape)
        self.inputs = None
        self.signals = None

    def _get_num_spatial_dims(self) -> int:
        return 2

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
            derivative = deriv(self.activation_function, 0, self.signals)
            grad = grad * derivative

        # im2col of the input
        input_cols = self._im2col(inputs) # Shape: (batch_size, kernel_h*kernel_w*in_c, output_h*output_w)

        # Gradient for biases
        self.d_biases = np.sum(grad, axis=(0, 1, 2)) / batch_size

        # Gradient for filters
        grad_reshaped = grad.transpose(0, 1, 2, 3).reshape(batch_size, output_height * output_width, num_filters) # Shape: (batch_size, output_h * output_w, num_filters)

        d_filters_unbatched = np.einsum('bpo,bof->fp', input_cols, grad_reshaped)
        self.d_filters = d_filters_unbatched.reshape(num_filters, kernel_height, kernel_width, input_channels) / batch_size

        # Gradient for the input (Transposed Convolution using im2col)
        filter_cols_rotated = np.flip(self.filters, axis=(1, 2)).reshape(num_filters, -1) # Shape: (num_filters, kernel_h*kernel_w*in_c)
        grad_cols = grad.transpose(0, 1, 2, 3).reshape(batch_size, output_height * output_width, num_filters) # Shape: (batch_size, output_h*output_w, num_filters)

        d_input_cols = np.einsum('bop,pf->bof', grad_cols, filter_cols_rotated) # Shape: (batch_size, output_h*output_w, kernel_h*kernel_w*in_c)
        d_input_cols_reshaped = d_input_cols.reshape(batch_size, output_height, output_width, kernel_height, kernel_width, input_channels)

        d_input_padded = np.zeros_like(inputs)
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride, self.stride
        if self.padding == 'same':
            pad_h = int(((input_height - 1) * stride_h + kernel_height - input_height) / 2)
            pad_w = int(((input_width - 1) * stride_w + kernel_width - input_width) / 2)
            d_input_padded = np.pad(d_input_padded, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

        for b in range(batch_size):
            for y in range(output_height):
                for x in range(output_width):
                    start_y = y * stride_h
                    end_y = start_y + kernel_height
                    start_x = x * stride_w
                    end_x = start_x + kernel_width
                    d_input_padded[b, start_y:end_y, start_x:end_x, :] += d_input_cols_reshaped[b, y, x, :, :, :]

        if self.padding == 'same':
            return d_input_padded[:, pad_h:-pad_h, pad_w:-pad_w, :]
        else:
            return d_input_padded

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

class Conv3DLayer(ConvolutionalLayer):
    """
    Performs 3D convolution, used for processing volumetric data like 3D medical scans or video.
    """
    def __init__(self, num_filters: int, kernel_size: Union[int, Tuple[int, int, int]], stride: int = 1, padding: str = 'valid', activation_function: Optional[Callable] = None, input_shape: Optional[Tuple[int, int, int, int, int]] = None):
        super().__init__(num_filters, kernel_size, stride, padding, activation_function, input_shape)
        self.inputs = None
        self.signals = None

    def _get_num_spatial_dims(self) -> int:
        return 3

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._initialize_filters_biases(inputs.shape)
        self.inputs = inputs
        batch_size, input_depth, input_height, input_width, input_channels = inputs.shape
        num_filters, kernel_depth, kernel_height, kernel_width, _ = self.filters.shape
        stride_d, stride_h, stride_w = self.stride, self.stride, self.stride

        if self.padding == 'same':
            pad_d = int(((input_depth - 1) * stride_d + kernel_depth - input_depth) / 2)
            pad_h = int(((input_height - 1) * stride_h + kernel_height - input_height) / 2)
            pad_w = int(((input_width - 1) * stride_w + kernel_width - input_width) / 2)
            padded_inputs = np.pad(inputs, ((0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:
            padded_inputs = inputs

        output_depth = (padded_inputs.shape[1] - kernel_depth) // stride_d + 1
        output_height = (padded_inputs.shape[2] - kernel_height) // stride_h + 1
        output_width = (padded_inputs.shape[3] - kernel_width) // stride_w + 1
        output = np.zeros((batch_size, output_depth, output_height, output_width, num_filters))

        for i in range(batch_size):
            for f in range(num_filters):
                for id in range(output_depth):
                    for ih in range(output_height):
                        for iw in range(output_width):
                            start_d = id * stride_d
                            end_d = start_d + kernel_depth
                            start_h = ih * stride_h
                            end_h = start_h + kernel_height
                            start_w = iw * stride_w
                            end_w = start_w + kernel_width
                            input_slice = padded_inputs[i, start_d:end_d, start_h:end_h, start_w:end_w, :]
                            output[i, id, ih, iw, f] = np.sum(input_slice * self.filters[f]) + self.biases[f]

        if self.activation_function:
            self.signals = self.activation_function(output)
            return self.signals
        else:
            self.signals = output
            return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        inputs = self.inputs
        batch_size, input_depth, input_height, input_width, input_channels = inputs.shape
        num_filters, kernel_depth, kernel_height, kernel_width, _ = self.filters.shape
        stride_d, stride_h, stride_w = self.stride, self.stride, self.stride
        output_depth, output_height, output_width = grad.shape[1:4]

        # Apply activation derivative
        if self.activation_function:
            derivative = deriv(self.activation_function, 0, self.signals)
            grad = grad * derivative

        # Initialize gradients
        self.d_filters = np.zeros_like(self.filters)
        self.d_biases = np.sum(grad, axis=(0, 1, 2, 3)) / batch_size
        d_input = np.zeros_like(inputs)

        for i, f, id in itertools.product(range(batch_size), range(num_filters), range(output_depth)):
            for ih, iw in itertools.product(range(output_height), range(output_width)):
                start_d = id * stride_d
                end_d = start_d + kernel_depth
                start_h = ih * stride_h
                end_h = start_h + kernel_height
                start_w = iw * stride_w
                end_w = start_w + kernel_width
                input_slice = inputs[i, start_d:end_d, start_h:end_h, start_w:end_w, :]
                self.d_filters[f] += input_slice * grad[i, id, ih, iw, f]

                # Gradient for input
                grad_slice = grad[i, id, ih, iw, f]
                for c in range(input_channels):
                    d_input[i, start_d:end_d, start_h:end_h, start_w:end_w, c] += self.filters[f, :, :, :, c] * grad_slice

        self.d_filters /= batch_size
        return d_input

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
                self.optimizer.register_parameter(self.depthwise_filters, 'depthwise_filters')
                self.optimizer.register_parameter(self.pointwise_filters, 'pointwise_filters')
                self.optimizer.register_parameter(self.biases, 'biases')

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
            derivative = deriv(self.activation_function, 0, self.signals)
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
            derivative = deriv(self.activation_function, 0, self.signals)
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
            derivative = deriv(self.activation_function, 0, self.signals)
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