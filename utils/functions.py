import math
from typing import List, Callable
import numpy as np

# Cost Functions
class CostFunctions:
    @staticmethod
    def mean_squared_error(predicted: List[float], actual: List[float]) -> float:
        """Calculate the mean squared error between predicted and actual values."""
        return sum((p - a) ** 2 for p, a in zip(predicted, actual)) / len(predicted)

    @staticmethod
    def cross_entropy(predicted: List[float], actual: List[float]) -> float:
        """Calculate the cross-entropy loss between predicted and actual values."""
        epsilon = 1e-6  # Prevents log(0) issues
        return -sum(a * math.log(p + epsilon) + (1 - a + epsilon) * math.log(1 - p + epsilon) for a, p in zip(actual, predicted)) / len(predicted)

# Loss Class
class Loss:
    def __init__(self, loss_function):
        self.loss_function = loss_function

    def __call__(self, predicted: List[float], actual: List[float]) -> float:
        return self.loss_function(predicted, actual)

    def backward(self, predicted: List[float], actual: List[float]) -> List[float]:
        if self.loss_function == CostFunctions.mean_squared_error:
            return [2 * (p - a) / len(predicted) for p, a in zip(predicted, actual)]
        elif self.loss_function == CostFunctions.cross_entropy:
            epsilon = 1e-6
            return [-(a / (p + epsilon)) + ((1 - a) / (1 - p + epsilon)) for p, a in zip(predicted, actual)]
        else:
            raise NotImplementedError("Backward method not implemented for this loss function")

# Metrics Class
class Metrics:
    @staticmethod
    def accuracy(predicted: List[float], actual: List[float]) -> float:
        """Calculate the accuracy between predicted and actual values."""
        correct = sum(round(p) == a for p, a in zip(predicted, actual))
        return correct / len(predicted)

# Activation Functions and Their Derivatives
class ActivationFunctions:
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Calculate the sigmoid activation function."""
        x = np.clip(x, -700, 700)  # Prevent overflow
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Calculate the ReLU activation function."""
        return np.maximum(0, x)

    @staticmethod
    def leaky_relu(x: np.ndarray) -> np.ndarray:
        """Calculate the leaky ReLU activation function."""
        return np.where(x > 0, x, 0.01 * x)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Calculate the tanh activation function."""
        return np.tanh(x)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Calculate the softmax activation function."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def derivative(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, dx: float = 1e-6) -> np.ndarray:
        """Calculate the derivative of a function using the limit definition."""
        return (func(x + dx) - func(x)) / dx
    
