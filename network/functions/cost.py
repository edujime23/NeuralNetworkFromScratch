import numpy as np

def mean_squared_error(predicted: np.ndarray, target: np.ndarray) -> np.complex128:
    """
    Mean Squared Error loss function for complex numbers.
    Returns a complex number where the real part is the mean of the squared real difference
    and the imaginary part is the mean of the squared imaginary difference.
    """
    diff = predicted - target
    real_error = np.mean(diff.real**2)
    imag_error = np.mean(diff.imag**2)
    return np.complex128(real_error, imag_error) if np.any(target.imag != 0) else real_error

def cross_entropy(predicted: np.ndarray, target: np.ndarray) -> np.complex128:
    """
    Cross-Entropy loss function for complex numbers.
    Returns a complex number where the real part is the mean of the real component of the loss
    and the imaginary part is the mean of the imaginary component of the loss.
    """
    epsilon = 1e-15  # To avoid division by zero
    loss = -(target * np.log(predicted + epsilon) + (1 - target) * np.log(1 - predicted + epsilon))
    return np.mean(loss, dtype=np.complex128)

def binary_cross_entropy(predicted: np.ndarray, target: np.ndarray) -> np.complex128:
    """
    Binary Cross-Entropy loss function for complex numbers.
    Returns a complex number where the real part is the mean of the real component of the loss
    and the imaginary part is the mean of the imaginary component of the loss.
    """
    epsilon = 1e-15
    loss = -(target * np.log(predicted + epsilon) + (1 - target) * np.log(1 - predicted + epsilon))
    return np.mean(loss, dtype=target.dtype)

def categorical_cross_entropy(predicted: np.ndarray, target: np.ndarray) -> np.complex128:
    """
    Categorical Cross-Entropy loss function for complex numbers.
    Returns a complex number where the real part is the mean of the real component of the loss
    and the imaginary part is the mean of the imaginary component of the loss.
    """
    epsilon = 1e-15
    loss = -np.sum(target * np.log(predicted + epsilon), axis=-1, keepdims=True)
    return np.mean(loss, dtype=target.dtype)

def mean_absolute_error(predicted: np.ndarray, target: np.ndarray) -> np.complex128:
    """
    Mean Absolute Error (MAE) loss function for complex numbers.
    Returns a complex number where the real part is the mean of the real component of the absolute difference
    and the imaginary part is the mean of the imaginary component of the absolute difference.
    """
    diff = predicted - target
    real_error = np.mean(np.abs(diff.real))
    imag_error = np.mean(np.abs(diff.imag))
    return np.complex128(real_error, imag_error) if np.any(target.imag != 0) else real_error

def huber_loss(predicted: np.ndarray, target: np.ndarray, delta: float = 1.0) -> np.complex128:
    """
    Huber loss function for complex numbers.
    Returns a complex number where the real part is the mean of the real component of the Huber loss
    and the imaginary part is the mean of the imaginary component of the Huber loss.
    """
    diff = predicted - target
    abs_diff = np.abs(diff)
    quadratic_loss = 0.5 * abs_diff**2
    linear_loss = delta * abs_diff - 0.5 * delta**2
    loss = np.where(abs_diff <= delta, quadratic_loss, linear_loss)
    return np.mean(loss, dtype=target.dtype)

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