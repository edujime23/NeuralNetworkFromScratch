import itertools
import numpy as np
from typing import Optional, Tuple, Union, List
from layer.base import Layer, PoolingLayer

class MaxPoolingLayer(PoolingLayer):
    """
    Performs max pooling over the input.
    """
    def __init__(self, pool_size: Union[int, Tuple[int, int]] = (2, 2), stride: Optional[Union[int, Tuple[int, int]]] = None, input_shape: Optional[Tuple[int, int, int]] = None):
        super().__init__(pool_size, stride, input_shape)
        self.max_indices = None

    def _get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        input_height, input_width, num_channels = input_shape
        output_height = (input_height - self.pool_size[0]) // self.stride[0] + 1
        output_width = (input_width - self.pool_size[1]) // self.stride[1] + 1
        return (output_height, output_width, num_channels)

    def _pool_forward(self, inputs: np.ndarray) -> np.ndarray:
        batch_size, input_height, input_width, num_channels = inputs.shape
        kh, kw = self.pool_size
        sh, sw = self.stride
        output_height, output_width, _ = self._get_output_shape((input_height, input_width, num_channels))
        output = np.zeros((batch_size, output_height, output_width, num_channels), dtype=np.float64)
        self.max_indices = np.zeros_like(inputs, dtype=bool)

        for b in range(batch_size):
            for c in range(num_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_h = i * sh
                        end_h = start_h + kh
                        start_w = j * sw
                        end_w = start_w + kw
                        pool_region = inputs[b, start_h:end_h, start_w:end_w, c]
                        max_val = np.max(pool_region)
                        output[b, i, j, c] = max_val
                        max_index_in_region = np.argmax(pool_region)
                        index_h = start_h + max_index_in_region // kw
                        index_w = start_w + max_index_in_region % kw
                        self.max_indices[b, index_h, index_w, c] = True
        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        d_input = np.zeros_like(self.inputs)
        batch_size, input_height, input_width, num_channels = self.inputs.shape
        output_height, output_width, _ = grad.shape[1:]

        for b, c, i in itertools.product(range(batch_size), range(num_channels), range(output_height)):
            for j in range(output_width):
                start_h = i * self.stride[0]
                end_h = start_h + self.pool_size[0]
                start_w = j * self.stride[1]
                end_w = start_w + self.pool_size[1]
                grad_val = grad[b, i, j, c]

                input_slice = self.inputs[b, start_h:end_h, start_w:end_w, c]
                max_index = np.argmax(input_slice)
                index_h = start_h + max_index // self.pool_size[1]
                index_w = start_w + max_index % self.pool_size[1]

                d_input[b, index_h, index_w, c] += grad_val
        return d_input

class AveragePoolingLayer(PoolingLayer):
    """
    Performs average pooling over the input.
    """
    def __init__(self, pool_size: Union[int, Tuple[int, int]] = (2, 2), stride: Optional[Union[int, Tuple[int, int]]] = None, input_shape: Optional[Tuple[int, int, int]] = None):
        super().__init__(pool_size, stride, input_shape)

    def _get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        input_height, input_width, num_channels = input_shape
        output_height = (input_height - self.pool_size[0]) // self.stride[0] + 1
        output_width = (input_width - self.pool_size[1]) // self.stride[1] + 1
        return (output_height, output_width, num_channels)

    def _pool_forward(self, inputs: np.ndarray) -> np.ndarray:
        batch_size, input_height, input_width, num_channels = inputs.shape
        kh, kw = self.pool_size
        sh, sw = self.stride
        output_height, output_width, _ = self._get_output_shape((input_height, input_width, num_channels))

        # Optimized forward pass using reshaping and mean
        input_reshaped = np.lib.stride_tricks.as_strided(
            inputs,
            shape=(batch_size, output_height, output_width, num_channels, kh, kw),
            strides=(inputs.strides[0], sh * inputs.strides[1], sw * inputs.strides[2],
                     inputs.strides[3], inputs.strides[1], inputs.strides[2])
        )
        return np.mean(input_reshaped, axis=(4, 5))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        batch_size, output_height, output_width, num_channels = grad.shape
        kh, kw = self.pool_size
        sh, sw = self.stride
        input_height, input_width, _ = self.inputs.shape[1:]
        d_input = np.zeros_like(self.inputs)

        # Optimized backward pass using broadcasting
        for b, c, i, j in itertools.product(range(batch_size), range(num_channels), range(output_height), range(output_width)):
            start_h = i * sh
            end_h = start_h + kh
            start_w = j * sw
            end_w = start_w + kw
            d_input[b, start_h:end_h, start_w:end_w, c] += grad[b, i, j, c] / (kh * kw)
        return d_input

class ReshapeLayer(Layer):
    """
    Reshapes the input tensor to a new shape.
    """
    def __init__(self, output_shape: Tuple[int, ...], input_shape: Optional[Tuple[int, ...]] = None):
        super().__init__(input_shape)
        self.output_shape = output_shape
        self.input_shape_forward = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input_shape_forward = inputs.shape
        batch_size = inputs.shape[0]
        return inputs.reshape((batch_size,) + self.output_shape)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.input_shape_forward is None:
            raise RuntimeError("Must call forward before backward for ReshapeLayer.")
        return grad.reshape(self.input_shape_forward)
    
class FlattenLayer(Layer):
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

class PermuteLayer(Layer):
    """
    Permutes the dimensions of the input tensor.
    """
    def __init__(self, dims: Tuple[int, ...], input_shape: Optional[Tuple[int, ...]] = None):
        super().__init__(input_shape)
        self.dims = dims
        self.inverse_dims = None

    def build(self, input_shape: Tuple[int, ...]):
        super().build(input_shape)
        # Create inverse permutation for the backward pass
        self.inverse_dims = np.argsort(np.array(self.dims)).tolist()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.transpose(inputs, axes=self.dims)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.inverse_dims is None:
            raise RuntimeError("Permute layer needs to be built to determine inverse dims.")
        return np.transpose(grad, axes=self.inverse_dims)