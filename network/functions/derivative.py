import numpy as np
from typing import Callable, Union

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