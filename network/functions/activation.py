import numpy as np
from typing import Union

def linear(x: np.ndarray):
    return x

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function, works with complex numbers.
    """
    return 1 / (1 + np.exp(-x))

def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation function for complex numbers (applied element-wise to real and imaginary parts).
    """
    real_part = np.where(x.real > 0, x.real, 0)
    imag_part = np.where(x.imag > 0, x.imag, 0)
    return real_part + 1j * imag_part

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Leaky ReLU activation function for complex numbers (applied element-wise to real and imaginary parts).
    """
    real_part = np.where(x.real > 0, x.real, alpha * x.real)
    imag_part = np.where(x.imag > 0, x.imag, alpha * x.imag)
    return real_part + 1j * imag_part

def tanh(x: np.ndarray) -> np.ndarray:
    """
    Tanh activation function, works with complex numbers.
    """
    return np.tanh(x)

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation function, works with complex numbers.
    Note: The interpretation of softmax on complex numbers might differ from the real case.
    This implementation applies the standard formula.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def softplus(x: np.ndarray) -> np.ndarray:
    """
    Softplus activation function, works with complex numbers.
    """
    return np.log(1 + np.exp(x))

def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Exponential Linear Unit (ELU) activation function for complex numbers (applied element-wise to real and imaginary parts).
    """
    real_part = np.where(x.real > 0, x.real, alpha * (np.exp(x.real) - 1))
    imag_part = np.where(x.imag > 0, x.imag, alpha * (np.exp(x.imag) - 1))
    return real_part + 1j * imag_part

def swish(x: np.ndarray) -> np.ndarray:
    """
    Swish activation function, works with complex numbers.
    """
    return x * sigmoid(x)