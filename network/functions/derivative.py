import numpy as np
from typing import Callable, Tuple, Iterable, Union

def derivative(
    func: Callable,
    args: Tuple,
    arg_index: int = 0,
    dx: float = 1e-6,
    complex_diff: bool = False,
) -> Union[np.ndarray, float]:
    """
    Calculates the numerical derivative of a function with respect to one of its arguments.

    Uses either the finite difference method or the complex step derivative approximation.

    Args:
        func: The callable function to differentiate.
        args: A tuple containing the arguments to the function.
        arg_index: The index of the argument with respect to which the derivative is calculated (default: 0).
        dx: The step size for the finite difference or complex step (default: 1e-6).
        complex_diff: If True, uses the complex step derivative approximation; otherwise, uses the central finite difference method (default: False).

    Returns:
        The numerical derivative as a NumPy array or a float, depending on the input argument.
    """
    if not isinstance(args, Tuple):
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
    
    dtype = np.complex64 if complex_diff or any(isinstance(arg, np.ndarray) and np.iscomplexobj(arg) for arg in args) else np.float64

    args_list = list(args)
    arg_to_diff = args_list[arg_index]

    if isinstance(arg_to_diff, np.ndarray):
        result_array = np.zeros_like(arg_to_diff, dtype=dtype)
        iterator = np.nditer(arg_to_diff, flags=['multi_index'])
        while not iterator.finished:
            multi_index = iterator.multi_index
            original_value = arg_to_diff[multi_index].copy()

            f_plus_args = list(args_list)
            if complex_diff:
                f_plus_args[arg_index] = arg_to_diff.astype(np.complex64).copy()
                f_plus_args[arg_index][multi_index] += dx * 1j
            else:
                f_plus_args[arg_index] = arg_to_diff.copy()
                f_plus_args[arg_index][multi_index] += dx
            f_plus = func(*f_plus_args)

            f_minus_args = list(args_list)
            if complex_diff:
                f_minus_args[arg_index] = arg_to_diff.astype(np.complex64).copy()
                f_minus_args[arg_index][multi_index] -= dx * 1j
            else:
                f_minus_args[arg_index] = arg_to_diff.copy()
                f_minus_args[arg_index][multi_index] -= dx
            f_minus = func(*f_minus_args)

            if complex_diff:
                deriv = (
                    np.complex64(
                        f_plus[multi_index] - f_minus[multi_index]
                    ).imag
                    / (2 * dx)
                    if isinstance(f_plus, np.ndarray)
                    and isinstance(f_minus, np.ndarray)
                    else np.complex64(f_plus - f_minus).imag / (2 * dx)
                )
            elif isinstance(f_plus, np.ndarray) and isinstance(f_minus, np.ndarray):
                deriv = (f_plus[multi_index] - f_minus[multi_index]) / (2 * dx)
            else:
                deriv = (f_plus - f_minus) / (2 * dx)

            result_array[multi_index] = deriv

            args_list[arg_index][multi_index] = original_value
            iterator.iternext()
        return result_array if result_array.imag.any() else result_array.real
    else:
        # Case where the argument is a single number
        original_value = arg_to_diff
        h = dx * 1j if complex_diff else dx
        f_plus = func(*(list(args[:arg_index]) + [original_value + h] + list(args[arg_index + 1:])))
        f_minus = func(*(list(args[:arg_index]) + [original_value - h] + list(args[arg_index + 1:])))

        if complex_diff:
            return np.complex64(f_plus - f_minus).imag / (2 * dx)
        else:
            return (f_plus - f_minus) / (2 * dx)

if __name__ == '__main__':
    def linear_func(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """A simple linear function: f(x) = 2 * x."""
        return 2 * x

    def linear_derivative(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """The derivative of linear_func: f'(x) = 2."""
        return np.full_like(x, 2)

    def constant_func(x: Union[float, np.ndarray]) -> int:
        """A constant function: f(x) = 2."""
        return 2

    def constant_derivative(x: Union[float, np.ndarray]) -> np.ndarray:
        """The derivative of constant_func: f'(x) = 0."""
        return np.zeros_like(x)

    def absolute_func(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """The absolute value function: f(x) = |2 * x|."""
        return np.abs(2 * x)

    def absolute_derivative(x: Union[float, np.ndarray]) -> np.ndarray:
        """The derivative of absolute_func: f'(x) = 2 if x >= 0 else -2."""
        return np.where(x >= 0, 2, -2)

    def quadratic_func(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """A quadratic function: f(x) = x**2."""
        return x**2

    def quadratic_derivative(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """The derivative of quadratic_func: f'(x) = 2 * x."""
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

    print("\n--- Quadratic Function ---")
    print("Args:", args_tuple)
    print("Derivative (normal diff):", derivative(quadratic_func, args_tuple, dx=1e-6, complex_diff=False))
    print("Expected derivative:", quadratic_derivative(test_array))
    print("Derivative (complex diff):", derivative(quadratic_func, args_tuple, dx=1e-6, complex_diff=True))
    print("Expected derivative:", quadratic_derivative(test_array))