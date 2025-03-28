import numpy as np
from typing import Callable

import numpy as np

class CostFunctions:
    @staticmethod
    def mean_squared_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        return (predicted - target) ** 2

    @staticmethod
    def mean_squared_error_derivative(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        return 2 * (predicted - target)

    @staticmethod
    def binary_cross_entropy(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        epsilon = 1e-15  # To avoid division by zero
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))

    @staticmethod
    def binary_cross_entropy_derivative(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return - (target / predicted - (1 - target) / (1 - predicted))

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

    @staticmethod
    def derivative(func: Callable[[np.ndarray], np.ndarray], *args, dx: float = 1e-6) -> np.ndarray:
        return (func(*[arg + dx for arg in args]) - func(*args)) / dx