import numpy as np
from typing import Callable

class CostFunctions:
    @staticmethod
    def mean_squared_error(predicted: np.ndarray, actual: np.ndarray) -> float:
        return np.mean((predicted - actual) ** 2)

    @staticmethod
    def cross_entropy(predicted: np.ndarray, actual: np.ndarray) -> float:
        epsilon = 1e-7
        return -np.mean(actual * np.log(predicted + epsilon) + (1 - actual) * np.log(1 - predicted + epsilon))

    @staticmethod
    def binary_cross_entropy(predicted: np.ndarray, actual: np.ndarray) -> float:
        return CostFunctions.cross_entropy(predicted, actual)

class Loss:
    def __init__(self, loss_function: Callable):
        self.loss_function = loss_function

    def __call__(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        return self.loss_function(predicted, actual)

    def backward(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        epsilon = 1e-7
        if self.loss_function == CostFunctions.mean_squared_error:
            return 2 * (predicted - actual) / predicted.size
        elif self.loss_function in (CostFunctions.cross_entropy, CostFunctions.binary_cross_entropy):
            return -(actual / (predicted + epsilon)) + ((1 - actual) / (1 - predicted + epsilon))
        raise NotImplementedError("Backward method not implemented for this loss function")

class Metrics:
    @staticmethod
    def accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
        return np.mean(np.round(predicted) == actual)

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
    def derivative(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, dx: float = 1e-6) -> np.ndarray:
        return (func(x + dx) - func(x)) / dx