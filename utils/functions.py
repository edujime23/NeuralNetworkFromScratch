import math
from typing import List

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


# Activation Functions and Their Derivatives
class ActivationFunctions:
    @staticmethod
    def sigmoid(x: float) -> float:
        """Calculate the sigmoid activation function."""
        x = max(min(x, 700), -700)  # Prevent overflow
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def relu(x: float) -> float:
        """Calculate the ReLU activation function."""
        return max(0, x)

    @staticmethod
    def leaky_relu(x: float) -> float:
        """Calculate the leaky ReLU activation function."""
        return x if x > 0 else 0.01 * x