import numpy as np
from typing import Callable, Tuple, Iterable, Union

def derivative(
    func: Callable,
    args: Tuple,
    arg_index: int = 0,
    dx: float = 1e-7,
    complex_diff: bool = False,
) -> Union[np.ndarray, float]:
    if not isinstance(args, tuple):
        if isinstance(args, Iterable) and not isinstance(args, np.ndarray):
            args = tuple(args)
        else:
            args = (args,)
    if not callable(func):
        raise ValueError("func must be a callable function")
    if not isinstance(arg_index, int):
        raise ValueError("arg_index must be an integer")
    if not 0 <= arg_index < len(args):
        raise ValueError("arg_index is out of bounds for the provided args")

    args_list = list(args)
    arg_to_diff = args_list[arg_index]

    if isinstance(arg_to_diff, np.ndarray):
        original_shape = arg_to_diff.shape
        perturbation = np.zeros_like(arg_to_diff, dtype=np.complex128 if complex_diff else np.float64)
        perturbation.flat[:] = (dx * (1j if complex_diff else 1))

        args_plus = list(args_list)
        args_plus[arg_index] = arg_to_diff + perturbation
        f_plus = func(*args_plus)

        args_minus = list(args_list)
        args_minus[arg_index] = arg_to_diff - perturbation
        f_minus = func(*args_minus)

        if complex_diff:
            derivative_result = np.imag(f_plus) / dx
        else:
            derivative_result = (f_plus - f_minus) / (2 * dx)

        if np.isscalar(derivative_result):
            return np.full(original_shape, derivative_result)
        else:
            return derivative_result.reshape(original_shape)
    else:
        h = dx * (1j if complex_diff else 1)
        f_plus = func(*(list(args[:arg_index]) + [arg_to_diff + h] + list(args[arg_index + 1:])))
        f_minus = func(*(list(args[:arg_index]) + [arg_to_diff - h] + list(args[arg_index + 1:])))

        return np.imag(f_plus) / dx if complex_diff else (f_plus - f_minus) / (2 * dx)

if __name__ == '__main__':
    def linear_func(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 2 * x

    def linear_derivative(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.full_like(x, 2)

    def constant_func(x: Union[float, np.ndarray]) -> int:
        return 2

    def constant_derivative(x: Union[float, np.ndarray]) -> np.ndarray:
        return np.zeros_like(x)

    def absolute_func(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.abs(2 * x)

    def absolute_derivative(x: Union[float, np.ndarray]) -> np.ndarray:
        return np.where(x >= 0, 2, -2)

    def quadratic_func(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return x**2

    def quadratic_derivative(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 2 * x

    test_array = np.arange(-5, 5)
    args_tuple = (test_array,)

    print("--- Linear Function ---")
    print("Args:", args_tuple)
    print("Derivative (normal diff):", derivative(linear_func, args_tuple, dx=1e-6, complex_diff=False))
    print("Expected derivative:", linear_derivative(test_array))
    print("Derivative (complex diff):", derivative(linear_func, args_tuple, dx=1e-6, complex_diff=True))
    print("Expected derivative:", linear_derivative(test_array))

    print("\n--- Constant Function ---")
    print("Args:", args_tuple)
    print("Derivative (normal diff):", derivative(constant_func, args_tuple, dx=1e-6, complex_diff=False))
    print("Expected derivative:", constant_derivative(test_array))
    print("Derivative (complex diff):", derivative(constant_func, args_tuple, dx=1e-6, complex_diff=True))
    print("Expected derivative:", constant_derivative(test_array))

    print("\n--- Absolute Value Function ---")
    print("Args:", args_tuple)
    print("Derivative (normal diff):", derivative(absolute_func, args_tuple, dx=1e-6, complex_diff=False))
    print("Expected derivative:", absolute_derivative(test_array))
    print("Derivative (complex diff):", derivative(absolute_func, args_tuple, dx=1e-6, complex_diff=True))
    print("Expected derivative:", absolute_derivative(test_array))
    print("Note: The complex difference method yields zeros because the absolute value function is not analytic.")

    epsilon = 1e-8
    def smooth_absolute_func(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.sqrt((2 * x)**2 + epsilon**2)

    def smooth_absolute_derivative(x: Union[float, np.ndarray]) -> np.ndarray:
        return 4 * x / np.sqrt((2 * x)**2 + epsilon**2)

    print("\n--- Smoothed Absolute Value Function (Approximation) ---")
    print("Args:", args_tuple)
    print("Derivative (normal diff):", derivative(smooth_absolute_func, args_tuple, dx=1e-6, complex_diff=False))
    print("Expected derivative:", smooth_absolute_derivative(test_array))
    print("Derivative (complex diff):", derivative(smooth_absolute_func, args_tuple, dx=1e-6, complex_diff=True))
    print("Expected derivative:", smooth_absolute_derivative(test_array))
    print(f"Note: This uses the approximation sqrt((2x)^2 + {epsilon}^2), which is analytic.")

    print("\n--- Quadratic Function ---")
    print("Args:", args_tuple)
    print("Derivative (normal diff):", derivative(quadratic_func, args_tuple, dx=1e-6, complex_diff=False))
    print("Expected derivative:", quadratic_derivative(test_array))
    print("Derivative (complex diff):", derivative(quadratic_func, args_tuple, dx=1e-6, complex_diff=True))
    print("Expected derivative:", quadratic_derivative(test_array))