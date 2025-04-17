import numpy as np
from typing import Callable, Tuple, Iterable, Union
from numba import njit, prange

@njit(cache=True, parallel=True)
def _compute_complex_derivative(f_plus, dx):
    """Optimized complex derivative computation with parallel processing"""
    result = np.empty(f_plus.shape[0], dtype=np.float64)
    for i in prange(f_plus.shape[0]):
        result[i] = f_plus[i].imag / dx
    return result

@njit(cache=True)
def _compute_real_derivative(f_plus, f_minus, dx):
    """Vectorized real derivative computation"""
    return (f_plus - f_minus) / (2 * dx)

def _compute_diff(f_plus, f_minus, dx: float, complex_diff: bool, original_shape):
    """Vectorized wrapper for derivative computation"""
    if complex_diff:
        f_plus_array = np.asarray(f_plus, dtype=np.complex128)
        if f_plus_array.ndim == 0:
            return np.full(original_shape, np.imag(f_plus_array) / dx)
        derivative_result = _compute_complex_derivative(f_plus_array.ravel(), dx)
    else:
        f_plus_array = np.asarray(f_plus, dtype=np.float64)
        f_minus_array = np.asarray(f_minus, dtype=np.float64)
        if f_plus_array.ndim == 0:
            return np.full(original_shape, (f_plus_array - f_minus_array) / (2 * dx))
        derivative_result = _compute_real_derivative(f_plus_array.ravel(), f_minus_array.ravel(), dx)
    
    return derivative_result.reshape(original_shape)

def _compute_array_derivative(func, arg_to_diff, dx: float, complex_diff: bool, original_shape, args, arg_index):
    """Memory-efficient array derivative computation"""
    args_list = list(args)
    if complex_diff:
        h = dx * 1j
        args_list[arg_index] = arg_to_diff + h
        f_plus = func(*args_list)
        return _compute_diff(f_plus, None, dx, True, original_shape)
    
    h = dx
    args_list[arg_index] = arg_to_diff + h
    f_plus = func(*args_list)
    args_list[arg_index] = arg_to_diff - h
    f_minus = func(*args_list)
    return _compute_diff(f_plus, f_minus, dx, False, original_shape)

def _compute_scalar_derivative(func, args_before, arg_to_diff, args_after, h, dx: float, complex_diff: bool):
    """Auxiliary function to compute derivative for scalar inputs"""
    f_plus = func(*(args_before + (arg_to_diff + h,) + args_after))
    f_minus = func(*(args_before + (arg_to_diff - h,) + args_after))
    
    return np.imag(f_plus) / dx if complex_diff else (f_plus - f_minus) / (2 * dx)

def derivative(
    func,
    args: Tuple,
    arg_index: int = 0,
    dx: float = 1e-7,
    complex_diff: bool = False,
) -> Union[np.ndarray, float]:
    """Main derivative function, handles input validation and dispatches to appropriate compute function"""
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

        return _compute_array_derivative(func, arg_to_diff, dx, complex_diff, original_shape, args, arg_index)
    else:
        h = dx * (1j if complex_diff else 1)
        args_before = args[:arg_index]
        args_after = args[arg_index + 1:]
        
        return _compute_scalar_derivative(func, args_before, arg_to_diff, args_after, h, dx, complex_diff)

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