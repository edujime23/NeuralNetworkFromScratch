import numpy as np
from numba import njit

@njit(cache=True)
def linear(x: np.ndarray):
    return x

@njit(cache=True)
def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function, works with complex numbers.
    """
    return 1 / (1 + np.exp(-x))

@njit(cache=True)
def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation function for complex numbers.
    """
    return x + (np.sqrt(np.power(x, 2)) - x) / 2

@njit(cache=True)
def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return x + ((np.sqrt(np.power(x, 2)) - x) / 2) * (1-alpha)

@njit(cache=True)
def tanh(x: np.ndarray) -> np.ndarray:
    """
    Tanh activation function, works with complex numbers.
    """
    return np.tanh(x)

@njit(cache=True)
def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation function for complex numbers (attempt 8).
    Handling 1D case separately in the original approach.
    """
    if x.ndim == 0:
        return np.array([1.0])
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    else:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

@njit(cache=True)
def softplus(x: np.ndarray) -> np.ndarray:
    """
    Softplus activation function, works with complex numbers.
    """
    return np.log(1 + np.exp(x))

@njit(cache=True)
def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Exponential Linear Unit (ELU) activation function for complex numbers.
    """
    epsilon = 1e-12
    return (np.sqrt(np.power(x, 2)) + x) / 2 + alpha * (np.exp(x) - 1) * (((np.sqrt(np.power(x, 2)) - x) / 2) / ((np.sqrt(np.power(x, 2)) - x) / 2 + epsilon))

@njit(cache=True)
def swish(x: np.ndarray) -> np.ndarray:
    """
    Swish activation function, works with complex numbers.
    """
    return x * sigmoid(x)