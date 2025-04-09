import numpy as np
import warnings
from typing import Callable, Optional, Union, Tuple, Literal, List

class NumericalDerivation:
    """
    A class for calculating numerical derivatives, gradients, and Jacobians.

    This class encapsulates the core logic for numerical differentiation using
    complex-step and finite difference methods.  The 'derivative' function,
    which serves as the user-facing entry point, is kept outside the class
    for organizational purposes, as requested.
    """

    # --- Constants ---
    _DEFAULT_COMPLEX_H_LD = np.longdouble(np.sqrt(np.finfo(np.float64).eps))
    _FD_ORDER = 8
    _FD_COEFFS_LD = np.array([
        1/280., -4/105., 1/5., -4/5., 0., 4/5., -1/5., 4/105., -1/280.
    ], dtype=np.longdouble)
    _FD_STEPS_LD = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.longdouble)
    _DEFAULT_FD_BASE_DX_LD = np.longdouble(1e-3)
    _RICHARDSON_RATIO_LD = np.longdouble(2.0)

    def __init__(self):
        """
        Initializes the NumericalDerivation class.
        """
        pass  # No specific initialization needed

    @staticmethod
    def _ensure_float64_tuple(args_in: Tuple) -> Tuple[np.ndarray, ...]:
        """Converts input args to a tuple of float64 numpy arrays/scalars."""
        if not args_in:
            raise ValueError("At least one argument required for the function.")
        try:
            return tuple(np.asarray(arg, dtype=np.float64) for arg in args_in)
        except Exception as e:
            raise TypeError(f"Cannot convert all arguments to numpy float64: {e}") from e

    @staticmethod
    def _complex_diff(
            func: Callable,
            args_f64: Tuple[np.ndarray, ...],
            idx: int,
            complex_h_ld: np.longdouble
    ) -> np.ndarray:
        """Attempt complex-step differentiation."""
        if complex_h_ld <= 0:
            raise ValueError("Complex step h must be positive.")

        args_complex = list(args_f64)
        original_arg = args_f64[idx]

        # Use float64 for the imaginary step to match expected input type often
        complex_h_f64 = np.float64(complex_h_ld)
        complex_perturbation = np.complex128(1j * complex_h_f64)

        # Create complex argument
        complex_arg = original_arg.astype(np.complex128, copy=True)
        complex_arg += complex_perturbation
        args_complex[idx] = complex_arg

        # Evaluate function with complex input
        try:
            result_complex = func(*args_complex)
        except Exception as e:
            raise RuntimeError(f"User function failed during complex step evaluation: {e}") from e

        # Check and convert result type
        if not np.iscomplexobj(result_complex) and not np.isrealobj(result_complex):
            try:
                result_complex = np.asarray(result_complex) # Try converting list/tuple output
                if not np.iscomplexobj(result_complex) and not np.isrealobj(result_complex):
                    raise TypeError # Re-raise if still not numeric
            except (TypeError, ValueError) as e:
                raise TypeError(
                    "Function did not return a complex, real, or convertible "
                    f"result for complex input. Got type: {type(result_complex)}"
                ) from e

        # Calculate derivative
        deriv = np.imag(result_complex) / complex_h_f64

        if np.isnan(deriv).any() or np.isinf(deriv).any():
            raise ValueError("Complex step resulted in NaN or Inf.")

        return np.asarray(deriv, dtype=np.float64)

    @staticmethod
    def _central_diff_single_h(
            func: Callable,
            args_f64: Tuple[np.ndarray, ...],
            idx: int,
            h_ld: np.longdouble,
            func_accepts_ld: bool
    ) -> np.ndarray:
        """Calculate Nth-order central FD for a single step size h."""
        if h_ld <= 0:
            raise ValueError("Finite difference step h must be positive.")

        args_list_ld = [np.asarray(arg, dtype=np.longdouble) for arg in args_f64]
        original_arg_ld = args_list_ld[idx]

        # Adaptive Step Size based on argument magnitude
        magnitude_ld = np.mean(np.abs(original_arg_ld)) if isinstance(original_arg_ld, np.ndarray) and original_arg_ld.size > 0 else np.abs(original_arg_ld)
        if magnitude_ld < np.finfo(np.longdouble).tiny:
            magnitude_ld = np.longdouble(1.0)

        # Scale h, ensuring it doesn't become too small
        scale_factor = max(np.longdouble(1e-6), min(np.longdouble(1e6), magnitude_ld))
        adj_h_ld = h_ld * scale_factor
        min_allowable_h = np.finfo(np.longdouble).eps * np.longdouble(100.0) * (np.longdouble(1.0) + magnitude_ld)
        adj_h_ld = max(adj_h_ld, min_allowable_h)

        # Prepare coefficients and steps for the adjusted h
        coeffs_ld = NumericalDerivation._FD_COEFFS_LD / adj_h_ld
        steps_ld = NumericalDerivation._FD_STEPS_LD * adj_h_ld
        deriv_sum_ld = np.longdouble(0.0)
        result_dtype = np.longdouble # Accumulate in long double

        for i, step_ld in enumerate(steps_ld):
            if NumericalDerivation._FD_COEFFS_LD[i] == 0: continue # Skip zero coefficient (center point)

            temp_args_ld = list(args_list_ld)
            temp_args_ld[idx] = original_arg_ld + step_ld

            # Determine dtype for function call based on user hint
            call_dtype = np.longdouble if func_accepts_ld else np.float64
            try:
                temp_args_call = tuple(np.asarray(arg, dtype=call_dtype) for arg in temp_args_ld)
                func_eval = func(*temp_args_call)
            except Exception as e:
                raise RuntimeError(f"User function failed during FD evaluation with args {temp_args_call} (dtype {call_dtype}): {e}") from e

            # Convert result to long double for accumulation
            try:
                func_eval_ld = np.asarray(func_eval, dtype=result_dtype)
                if i == 0: # Initialize accumulator shape on first non-zero coefficient evaluation
                    deriv_sum_ld = np.zeros_like(func_eval_ld, dtype=result_dtype)
            except Exception as e:
                raise TypeError(
                    f"Function result could not be converted to {result_dtype.__name__} numpy array. "
                    f"Got type: {type(func_eval)}. Error: {e}"
                ) from e

            deriv_sum_ld += coeffs_ld[i] * func_eval_ld

        return deriv_sum_ld # Return as long double for Richardson

    @staticmethod
    def _finite_diff_richardson(
            func: Callable,
            args_f64: Tuple[np.ndarray, ...],
            idx: int,
            base_h_ld: np.longdouble,
            func_accepts_ld: bool,
            return_error: bool
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Combine central FD with Richardson extrapolation."""
        num_extrap = 3 # Use 3 steps for 2 levels of extrapolation
        h_steps_ld = [base_h_ld / (NumericalDerivation._RICHARDSON_RATIO_LD**k) for k in range(num_extrap)]

        d_estimates_ld = [
            NumericalDerivation._central_diff_single_h(func, args_f64, idx, h_ld, func_accepts_ld)
            for h_ld in h_steps_ld
        ]

        # Perform Richardson Extrapolation (Neville's algorithm idea)
        # T[i, k] = estimate using h_i and order k
        T = list(d_estimates_ld) # T[k] corresponds to h_k = base_h / r^k
        p = NumericalDerivation._FD_ORDER # Order of the underlying FD method

        for k in range(1, num_extrap): # k is extrapolation level
            power_r_p_k = NumericalDerivation._RICHARDSON_RATIO_LD**(p + 2*(k-1)) # Adjusted power for higher order error terms
            denom = power_r_p_k - np.longdouble(1.0)
            if abs(denom) < np.finfo(np.longdouble).tiny:
                warnings.warn(f"Richardson extrapolation denominator near zero at level {k}.", RuntimeWarning)
                denom = np.sign(denom) * np.finfo(np.longdouble).tiny if denom != 0 else np.finfo(np.longdouble).tiny

            for i in range(num_extrap - k): # i is row index
                T[i] = (power_r_p_k * T[i+1] - T[i]) / denom

        final_extrap_ld = T[0] # Most refined estimate

        # Error Estimation (difference between last two extrapolation levels)
        error_est_ld = None
        if return_error:
            # Estimate error as the magnitude of the difference between the final
            # and the second-to-last Richardson estimate.
            # The second to last estimate was T[0] before the final loop iteration (k=num_extrap-1)
            # which used T[0] and T[1] from the previous level (k=num_extrap-2).
            # Let's recalculate the second best estimate for comparison.
            if num_extrap >= 2:
                second_best_ld = d_estimates_ld[0] # Start from base estimates again for clarity
                T_prev = list(d_estimates_ld)
                for k_prev in range(1, num_extrap - 1): # Extrapolate up to level num_extrap-2
                    power_r_p_k_prev = NumericalDerivation._RICHARDSON_RATIO_LD**(p + 2*(k_prev-1))
                    denom_prev = power_r_p_k_prev - np.longdouble(1.0)
                    if abs(denom_prev) < np.finfo(np.longdouble).tiny: denom_prev = np.finfo(np.longdouble).tiny # Avoid division by zero
                    for i_prev in range(num_extrap - k_prev):
                        T_prev[i_prev] = (power_r_p_k_prev * T_prev[i_prev+1] - T_prev[i_prev]) / denom_prev
                second_best_ld = T_prev[0]
                error_est_ld = np.abs(final_extrap_ld - second_best_ld)
            else: # Not enough steps for error estimate this way
                error_est_ld = np.full_like(final_extrap_ld, np.nan, dtype=np.longdouble)


        # Final Conversion to float64
        final_result_f64 = np.asarray(final_extrap_ld, dtype=np.float64)

        if not return_error:
            return final_result_f64
        error_est_f64 = np.asarray(error_est_ld, dtype=np.float64)
        return final_result_f64, error_est_f64

    @staticmethod
    def _get_single_partial(
            func: Callable,
            args_f64: Tuple[np.ndarray, ...],
            idx: int,
            complex_h_ld: np.longdouble,
            base_fd_h_ld: np.longdouble,
            func_accepts_ld: bool,
            return_error: bool
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Tries complex step, falls back to FD Richardson."""
        try:
            result_f64 = NumericalDerivation._complex_diff(func, args_f64, idx, complex_h_ld)
            # Complex step is successful, error is theoretically near machine epsilon, return None for error estimate
            return (result_f64, None) if return_error else result_f64
        except Exception as complex_err:
            warnings.warn(
                f"Complex step failed for arg {idx} ({complex_err}). "
                f"Falling back to finite difference.",
                RuntimeWarning
            )
            try:
                # Finite difference returns result or (result, error_estimate)
                return NumericalDerivation._finite_diff_richardson(
                    func, args_f64, idx,
                    base_fd_h_ld, func_accepts_ld, return_error
                )
            except Exception as fd_err:
                raise RuntimeError(
                    f"Finite difference fallback also failed for arg {idx}: {fd_err}"
                ) from fd_err

def derivative(
    func: Callable,
    args: Union[list, tuple, float, int, np.ndarray],
    mode: Literal['derivative', 'gradient', 'jacobian', 'sum_partials'] = 'derivative',
    arg_index: Optional[int] = 0,
    complex_step_h: Optional[float] = None,
    fd_base_dx: Optional[float] = None,
    func_accepts_longdouble: bool = False,
    return_error_estimate: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculates numerical derivatives, gradients, or Jacobians using complex-step
    or high-order finite difference with Richardson extrapolation.

    Args:
        func: Callable function to differentiate.
        args: Arguments for `func` (single value or list/tuple).
        mode: Type of derivative ('derivative', 'gradient', 'jacobian', 'sum_partials').
        arg_index: Index for 'derivative' mode (default: 0). Ignored otherwise.
        complex_step_h: Step size 'h' for complex step (default: sqrt(eps)). Must be > 0.
        fd_base_dx: Base step size 'dx' for finite difference before scaling (default: 1e-3). Must be > 0.
        func_accepts_longdouble: Hint if `func` handles np.longdouble (improves FD).
        return_error_estimate: If True, return (result, error_estimate) tuple. Error is None if complex step succeeds.

    Returns:
        Calculated derivative/gradient/Jacobian (np.ndarray, float64).
        If return_error_estimate is True, returns (result, error_estimate).

    Raises:
        TypeError: Invalid input types.
        ValueError: Invalid mode, index, step size, or function output incompatible with complex step.
        RuntimeError: User function evaluation fails, or both methods fail.
    """
    # --- Input Validation and Setup ---
    if not callable(func):
        raise TypeError("`func` must be a callable function.")

    # Ensure args is always a tuple for internal consistency
    args_tuple = args if isinstance(args, (list, tuple)) else (args,)
    if not args_tuple:
        raise ValueError("At least one argument must be provided.")

    args_f64 = NumericalDerivation._ensure_float64_tuple(args_tuple)
    num_args = len(args_f64)

    valid_modes = ['derivative', 'gradient', 'jacobian', 'sum_partials']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")

    if mode == 'derivative':
        if not isinstance(arg_index, int):
            raise ValueError("`arg_index` must be an integer for mode='derivative'.")
        if not 0 <= arg_index < num_args:
            raise ValueError(f"arg_index {arg_index} out of range for {num_args} arguments.")
    elif arg_index != 0: # Don't error, just warn if unused
        warnings.warn(f"arg_index={arg_index} is ignored when mode='{mode}'.", UserWarning)
        if mode == 'derivative': arg_index = 0 # Reset to default if mode was corrected later

    current_complex_h_ld = np.longdouble(complex_step_h) if complex_step_h is not None else NumericalDerivation._DEFAULT_COMPLEX_H_LD
    current_fd_base_dx_ld = np.longdouble(fd_base_dx) if fd_base_dx is not None else NumericalDerivation._DEFAULT_FD_BASE_DX_LD

    if current_complex_h_ld <= 0: raise ValueError("complex_step_h must be positive.")
    if current_fd_base_dx_ld <= 0: raise ValueError("fd_base_dx must be positive.")

    # --- Perform Calculation Based on Mode ---
    results_list: List[np.ndarray] = []
    errors_list: List[Optional[np.ndarray]] = [] if return_error_estimate else None # type: ignore[assignment]
    final_result: Optional[np.ndarray] = None
    final_error: Optional[np.ndarray] = None

    deriv_instance = NumericalDerivation() # Create an instance of the class.  This is necessary to call the methods.

    if mode == 'derivative':
        # Directly calculate the single partial derivative
        result_or_tuple = deriv_instance._get_single_partial(
            func, args_f64, arg_index,
            current_complex_h_ld, current_fd_base_dx_ld,
            func_accepts_longdouble, return_error_estimate
        )
        if return_error_estimate:
            final_result, final_error = result_or_tuple # type: ignore[misc]
        else:
            final_result = result_or_tuple # type: ignore[assignment]

    elif mode == 'sum_partials':
        # Calculate all partials and sum them
        total_deriv_ld = np.longdouble(0.0)
        total_error_ld = np.longdouble(0.0) if return_error_estimate else None
        any_fd_used = False
        first_partial_shape = None

        for i in range(num_args):
            res_or_tup = deriv_instance._get_single_partial(
                func, args_f64, i,
                current_complex_h_ld, current_fd_base_dx_ld,
                func_accepts_longdouble, return_error_estimate
            )
            if return_error_estimate:
                partial_f64, error_f64 = res_or_tup # type: ignore[misc]
                partial_ld = np.asarray(partial_f64, dtype=np.longdouble)
                if i == 0: # Initialize sum shape
                    total_deriv_ld = np.zeros_like(partial_ld, dtype=np.longdouble)
                    if total_error_ld is not None:
                        total_error_ld = np.zeros_like(partial_ld, dtype=np.longdouble)
                total_deriv_ld += partial_ld
                if error_f64 is not None:
                    total_error_ld += np.asarray(error_f64, dtype=np.longdouble) # type: ignore[union-attr]
                    any_fd_used = True
            else:
                partial_f64 = res_or_tup # type: ignore[assignment]
                partial_ld = np.asarray(partial_f64, dtype=np.longdouble)
                if i == 0: # Initialize sum shape
                    total_deriv_ld = np.zeros_like(partial_ld, dtype=np.longdouble)
                total_deriv_ld += partial_ld

        final_result = np.asarray(total_deriv_ld, dtype=np.float64)
        if return_error_estimate:
            final_error = np.asarray(total_error_ld, dtype=np.float64) if any_fd_used else None

    elif mode in ['gradient', 'jacobian']:
        # Calculate all partials and stack them
        any_fd_used_for_error = False
        for i in range(num_args):
            res_or_tup = deriv_instance._get_single_partial(
                func, args_f64, i,
                current_complex_h_ld, current_fd_base_dx_ld,
                func_accepts_longdouble, return_error_estimate
            )
            if return_error_estimate:
                partial_res, partial_err = res_or_tup # type: ignore[misc]
                results_list.append(partial_res)
                errors_list.append(partial_err)
                if partial_err is not None:
                    any_fd_used_for_error = True
            else:
                results_list.append(res_or_tup) # type: ignore[arg-type]

        # --- Post-process Gradient/Jacobian ---
        if not results_list:
            raise RuntimeError("Internal error: No partial derivatives were computed for gradient/jacobian.")

        stack_axis = 0 if mode == 'gradient' else -1 # Stack along new first axis for grad, last axis for jacobian
        try:
            final_result = np.stack(results_list, axis=stack_axis)
        except ValueError as e:
            shapes = [np.shape(r) for r in results_list]
            raise ValueError(
                f"Could not stack partial derivatives into {mode}. Inconsistent shapes: {shapes}. Error: {e}"
            ) from e

        # Check expected dimensions (optional warning)
        # if mode == 'gradient' and final_result.ndim > 1 and final_result.shape[0] != num_args:
        #     warnings.warn(f"Gradient shape {final_result.shape} might be unexpected for {num_args} inputs.", RuntimeWarning)
        # elif mode == 'jacobian' and (final_result.ndim < 2 or final_result.shape[-1] != num_args):
        #     warnings.warn(f"Jacobian shape {final_result.shape} might be unexpected for {num_args} inputs.", RuntimeWarning)


        if return_error_estimate:
            if any_fd_used_for_error:
                # Replace None errors with NaN arrays of the correct shape before stacking
                processed_errors = []
                for res, err in zip(results_list, errors_list):
                    if err is None:
                        processed_errors.append(np.full_like(res, np.nan, dtype=np.float64))
                    else:
                        processed_errors.append(np.asarray(err, dtype=np.float64)) # Ensure float64

                try:
                    final_error = np.stack(processed_errors, axis=stack_axis)
                except ValueError:
                    # Fallback if shapes are inconsistent even after NaN padding (shouldn't happen if results stacked)
                    warnings.warn(f"Could not stack error estimates for {mode} due to inconsistent shapes after NaN padding.", RuntimeWarning)
                    final_error = np.full_like(final_result, np.nan, dtype=np.float64) # Match result shape
            else:
                # All complex steps succeeded, error is None (or NaN array)
                final_error = None # Will be handled in final return logic


    # --- Return Final Result ---
    if final_result is None:
        raise RuntimeError("Internal error: final_result was not computed.") # Should not happen

    if not return_error_estimate:
        return final_result
    # Ensure error array exists and matches result shape if FD was used anywhere
    if final_error is None:
        # Either complex step was used everywhere, or mode was 'derivative' with complex step
        final_error = np.full_like(final_result, np.nan, dtype=np.float64)
    elif final_error.shape != final_result.shape:
        warnings.warn(f"Result shape {final_result.shape} and error shape {final_error.shape} mismatch. Broadcasting error.", RuntimeWarning)
        try:
            # Try broadcasting error to result shape, useful if error was scalar but result array
            final_error = np.broadcast_to(final_error, final_result.shape).copy()
        except ValueError:
            final_error = np.full_like(final_result, np.nan, dtype=np.float64) # Fallback if broadcast fails

    return final_result, final_error
