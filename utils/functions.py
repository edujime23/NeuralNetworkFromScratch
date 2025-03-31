import numpy as np
from typing import Callable, Union

class CostFunctions:
    @staticmethod
    def mean_squared_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        return (predicted - target) ** 2

    @staticmethod
    def cross_entropy(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
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

def derivative(func: Callable, arg_index: Union[int, str], *args, dx: float = 1e-12) -> np.ndarray:
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