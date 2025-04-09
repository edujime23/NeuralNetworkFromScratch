import numpy as np
from typing import Union

def linear(x: np.ndarray):
    return x

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -700, 700)))

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def softplus(x: np.ndarray) -> np.ndarray:
    """
    Softplus activation function.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Softplus output.
    """
    return np.log(1 + np.exp(np.clip(x, -700, 700)))

def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Exponential Linear Unit (ELU) activation function.

    Args:
        x (np.ndarray): Input array.
        alpha (float): The alpha parameter for ELU.

    Returns:
        np.ndarray: ELU output.
    """
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -700, 700)) - 1))

def swish(x: np.ndarray) -> np.ndarray:
    """
    Swish activation function.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Swish output.
    """
    return x * sigmoid(x)
