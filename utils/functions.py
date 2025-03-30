import numpy as np
from typing import Callable

class CostFunctions:
    @staticmethod
    def mean_squared_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        return (predicted - target) ** 2

    @staticmethod
    def binary_cross_entropy(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        epsilon = 1e-15  # To avoid division by zero
        predicted = predicted + epsilon
        return -(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))

class ActivationFunctions:
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -700, 700)))

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def leaky_relu(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, 0.01 * x)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

def derivative(func: Callable, arg_index: int, *args, dx: float = np.float64(1) / np.uint64(2**32)) -> np.ndarray:
    args_list = list(args)
    original_arg = args_list[arg_index].copy()
    perturbed_arg = original_arg + dx
    args_list[arg_index] = perturbed_arg
    return (func(*args_list) - func(*args)) / dx