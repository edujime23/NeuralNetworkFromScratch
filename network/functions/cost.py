import numpy as np
from typing import Union
from numba import jit

@jit(nopython=True, cache=True)
def mean_squared_error(predicted: np.ndarray, target: np.ndarray) -> Union[np.complex128, np.float64]:
    """
    Mean Squared Error loss function for complex numbers.
    Returns a complex number where the real part is the mean of the squared real difference
    and the imaginary part is the mean of the squared imaginary difference.
    """
    diff = predicted - target
    real_error = np.mean(np.square(diff.real))
    if np.iscomplexobj(target):
        imag_error = np.mean(np.square(diff.imag))
        return real_error + 1j * imag_error
    else:
        return real_error

@jit(nopython=True, cache=True)
def cross_entropy(predicted: np.ndarray, target: np.ndarray) -> Union[np.complex128, np.float64]:
    """
    Cross-Entropy loss function for complex numbers.
    Returns a complex number where the real part is the mean of the real component of the loss
    and the imaginary part is the mean of the imaginary component of the loss.
    """
    epsilon = 1e-12  # More robust way to get epsilon
    loss = -(target * np.log(predicted + epsilon) + (1 - target) * np.log(1 - predicted + epsilon))
    if hasattr(predicted, 'size'):
        return 0.0 if loss.size == 0 else loss.sum() / loss.size
    else:
        return loss

@jit(nopython=True, cache=True)
def binary_cross_entropy(predicted: np.ndarray, target: np.ndarray) -> Union[np.complex128, np.float64]:
    """
    Binary Cross-Entropy loss function for complex numbers.
    Returns a complex number where the real part is the mean of the real component of the loss
    and the imaginary part is the mean of the imaginary component of the loss.
    """
    epsilon = 1e-12
    loss = -(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))
    if hasattr(predicted, 'size'):
        return 0.0 if loss.size == 0 else loss.sum() / loss.size
    else:
        return loss

@jit(nopython=True, cache=True)
def categorical_cross_entropy(predicted: np.ndarray, target: np.ndarray) -> Union[np.complex128, np.float64]:
    """
    Categorical Cross-Entropy loss function for complex numbers.
    Returns a complex number where the real part is the mean of the real component of the loss
    and the imaginary part is the mean of the imaginary component of the loss.
    """
    epsilon = 1e-12
    loss = -np.sum(target * np.log(predicted + epsilon), axis=-1)
    if hasattr(loss, 'size'):
        return 0.0 if loss.size == 0 else loss.sum() / loss.size
    else:
        return loss  # Handle case where loss might be a scalar

@jit(nopython=True, cache=True)
def mean_absolute_error(predicted: np.ndarray, target: np.ndarray) -> Union[np.complex128, np.float64]:
    """
    Mean Absolute Error (MAE) loss function for complex numbers.
    Returns a complex number where the real part is the mean of the real component of the absolute difference
    and the imaginary part is the mean of the imaginary component of the absolute difference.
    """
    diff = predicted - target
    real_error = np.mean(np.abs(diff.real))
    if np.iscomplexobj(target):
        imag_error = np.mean(np.abs(diff.imag))
        return real_error + 1j * imag_error
    else:
        return real_error

@jit(nopython=True, cache=True)
def huber_loss(predicted: np.ndarray, target: np.ndarray, delta: float = 1.0) -> Union[np.complex128, np.float64]:
    """
    Huber loss function for complex numbers.
    Returns a complex number where the real part is the mean of the real component of the Huber loss
    and the imaginary part is the mean of the imaginary component of the Huber loss.
    """
    diff = predicted - target
    abs_diff = np.abs(diff)
    quadratic_loss = 0.5 * np.square(abs_diff)
    linear_loss = delta * abs_diff - 0.5 * delta**2
    loss = np.where(abs_diff <= delta, quadratic_loss, linear_loss)
    if hasattr(predicted, 'size'):
        return 0.0 if loss.size == 0 else loss.sum() / loss.size
    else:
        return loss

if __name__ == '__main__':
    # Example usage with complex numbers
    predicted_complex = np.array([1 + 2j, 3 - 1j, 0.5 + 0.5j])
    target_complex = np.array([1.5 + 1.8j, 2.5 - 1.2j, 0.6 + 0.4j])

    print("--- Complex Number Example ---")
    print("Predicted:", predicted_complex)
    print("Target:", target_complex)
    print("Mean Squared Error:", mean_squared_error(predicted_complex, target_complex))
    print("Cross Entropy:", cross_entropy(predicted_complex, target_complex))
    print("Binary Cross Entropy:", binary_cross_entropy(predicted_complex, target_complex))
    print("Categorical Cross Entropy:", categorical_cross_entropy(predicted_complex, target_complex))
    print("Mean Absolute Error:", mean_absolute_error(predicted_complex, target_complex))
    print("Huber Loss:", huber_loss(predicted_complex, target_complex))

    # Example usage with real numbers
    predicted_real = np.array([1.0, 3.0, 0.5])
    target_real = np.array([1.5, 2.5, 0.6])

    print("\n--- Real Number Example ---")
    print("Predicted:", predicted_real)
    print("Target:", target_real)
    print("Mean Squared Error:", mean_squared_error(predicted_real, target_real))
    print("Cross Entropy:", cross_entropy(predicted_real, target_real))
    print("Binary Cross Entropy:", binary_cross_entropy(predicted_real, target_real))
    print("Categorical Cross Entropy:", categorical_cross_entropy(predicted_real, target_real))
    print("Mean Absolute Error:", mean_absolute_error(predicted_real, target_real))
    print("Huber Loss:", huber_loss(predicted_real, target_real))