import numpy as np

def mean_squared_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (predicted - target) ** 2

def cross_entropy(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    epsilon = 1e-15  # To avoid division by zero
    predicted = np.clip(predicted, epsilon, 1 - epsilon)
    return -(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))

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