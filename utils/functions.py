import numpy as np
from typing import Callable, Union

class CostFunctions:
    @staticmethod
    def mean_squared_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        return (predicted - target) ** 2

    @staticmethod
    def cross_entropy(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        epsilon = 1e-15  # To avoid division by zero
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))

    @staticmethod
    def binary_cross_entropy(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Binary Cross-Entropy loss function.

        Args:
            predicted (np.ndarray): Predicted probabilities (between 0 and 1).
            target (np.ndarray): True binary labels (0 or 1).

        Returns:
            np.ndarray: Binary Cross-Entropy loss.
        """
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))

    @staticmethod
    def categorical_cross_entropy(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Categorical Cross-Entropy loss function.

        Args:
            predicted (np.ndarray): Predicted probabilities for each class (summing to 1 along the last axis).
            target (np.ndarray): True labels as one-hot encoded vectors.

        Returns:
            np.ndarray: Categorical Cross-Entropy loss.
        """
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -np.sum(target * np.log(predicted), axis=-1, keepdims=True)

    @staticmethod
    def mean_absolute_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Mean Absolute Error (MAE) loss function.

        Args:
            predicted (np.ndarray): Predicted values.
            target (np.ndarray): True values.

        Returns:
            np.ndarray: Mean Absolute Error.
        """
        return np.abs(predicted - target)

    @staticmethod
    def huber_loss(predicted: np.ndarray, target: np.ndarray, delta: float = 1.0) -> np.ndarray:
        """
        Huber loss function.

        Args:
            predicted (np.ndarray): Predicted values.
            target (np.ndarray): True values.
            delta (float): Threshold for the Huber loss.

        Returns:
            np.ndarray: Huber loss.
        """
        abs_diff = np.abs(predicted - target)
        quadratic_loss = 0.5 * abs_diff**2
        linear_loss = delta * abs_diff - 0.5 * delta**2
        return np.where(abs_diff <= delta, quadratic_loss, linear_loss)

class ActivationFunctions:
    @staticmethod
    def linear(x: np.ndarray):
        return x

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
    def softplus(x: np.ndarray) -> np.ndarray:
        """
        Softplus activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Softplus output.
        """
        return np.log(1 + np.exp(np.clip(x, -700, 700)))

    @staticmethod
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

    @staticmethod
    def swish(x: np.ndarray) -> np.ndarray:
        """
        Swish activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Swish output.
        """
        return x * ActivationFunctions.sigmoid(x)

def derivative(func: Callable, arg_index: Union[int, 'all'], *args, dx: float = 1e-12) -> np.ndarray:
    """Calculate numerical derivative of a function with respect to specified argument.

    Args:
        func: Function to differentiate
        arg_index: Index of argument to differentiate with respect to, or 'all'
        *args: Arguments to pass to the function
        dx: Small change for numerical differentiation

    Returns:
        Numerical derivative array
    """
    # Convert args to numpy array for vectorized operations
    args = [np.asarray(arg) for arg in args]

    if not args:
        raise ValueError("At least one argument required")

    if isinstance(arg_index, int):
        if not 0 <= arg_index < len(args):
            raise ValueError(f"arg_index {arg_index} out of range")

        # Create copy of arguments
        perturbed_args = [arg.copy() for arg in args]
        perturbed_args[arg_index] += dx

    elif arg_index == "all":
        perturbed_args = [arg + dx for arg in args]

    else:
        raise ValueError("arg_index must be an integer or 'all'")

    # Central difference for better accuracy
    forward = func(*perturbed_args)
    backward = func(*args)

    return (forward - backward) / dx