import numpy as np
import warnings
from typing import Callable, Optional, Union, Tuple, Literal, List

class NumericalDerivation:
    _DEFAULT_COMPLEX_H_LD = np.longdouble(np.finfo(np.longdouble).eps**0.5)
    _FD_ORDER = 10
    _FD_COEFFS_LD = np.array([
        1/34650., -6/1155., 1/77., -4/21., 1/5., 0., -1/5., 4/21., -1/77., 6/1155., -1/34650.
    ], dtype=np.longdouble)
    _FD_STEPS_LD = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.longdouble)
    _DEFAULT_FD_BASE_DX_LD = np.longdouble(1e-4)
    _RICHARDSON_RATIO_LD = np.longdouble(2.0)

    def __init__(self):
        pass

    @staticmethod
    def _ensure_float64_tuple(args_in: Tuple) -> Tuple[np.ndarray, ...]:
        if not args_in:
            raise ValueError("At least one argument required for the function.")
        try:
            return tuple(np.asarray(arg, dtype=np.longdouble) for arg in args_in)
        except Exception as e:
            raise TypeError(f"Cannot convert all arguments to numpy longdouble: {e}") from e

    @staticmethod
    def _complex_diff(
            func: Callable,
            args_f128: Tuple[np.ndarray, ...],
            idx: int,
            complex_h_ld: np.longdouble
    ) -> np.ndarray:
        if complex_h_ld <= 0:
            raise ValueError("Complex step h must be positive.")

        args_complex = list(args_f128)
        original_arg = args_f128[idx]

        complex_h_f128 = np.longdouble(complex_h_ld)
        complex_perturbation = np.complex128(1j * complex_h_f128)

        complex_arg = original_arg.astype(np.complex128, copy=True)
        complex_arg += complex_perturbation
        args_complex[idx] = complex_arg

        try:
            result_complex = func(*args_complex)
        except Exception as e:
            raise RuntimeError(f"User function failed during complex step evaluation: {e}") from e

        if not np.iscomplexobj(result_complex) and not np.isrealobj(result_complex):
            try:
                result_complex = np.asarray(result_complex)
                if not np.iscomplexobj(result_complex) and not np.isrealobj(result_complex):
                    raise TypeError
            except (TypeError, ValueError) as e:
                raise TypeError(
                    "Function did not return a complex, real, or convertible "
                    f"result for complex input. Got type: {type(result_complex)}"
                ) from e

        deriv = np.imag(result_complex) / complex_h_f128

        if np.isnan(deriv).any() or np.isinf(deriv).any():
            raise ValueError("Complex step resulted in NaN or Inf.")

        return np.asarray(deriv, dtype=np.longdouble)

    @staticmethod
    def _central_diff_single_h(
            func: Callable,
            args_f128: Tuple[np.ndarray, ...],
            idx: int,
            h_ld: np.longdouble,
            func_accepts_ld: bool
    ) -> np.ndarray:
        if h_ld <= 0:
            raise ValueError("Finite difference step h must be positive.")

        args_list_ld = [np.asarray(arg, dtype=np.longdouble) for arg in args_f128]
        original_arg_ld = args_list_ld[idx]

        magnitude_ld = np.mean(np.abs(original_arg_ld)) if isinstance(original_arg_ld, np.ndarray) and original_arg_ld.size > 0 else np.abs(original_arg_ld)
        if magnitude_ld < np.finfo(np.longdouble).tiny:
            magnitude_ld = np.longdouble(1.0)

        scale_factor = max(np.longdouble(1e-8), min(np.longdouble(1e8), magnitude_ld))
        adj_h_ld = h_ld * scale_factor
        min_allowable_h = np.finfo(np.longdouble).eps * np.longdouble(1000.0) * (np.longdouble(1.0) + magnitude_ld)
        adj_h_ld = max(adj_h_ld, min_allowable_h)

        coeffs_ld = NumericalDerivation._FD_COEFFS_LD / adj_h_ld
        steps_ld = NumericalDerivation._FD_STEPS_LD * adj_h_ld
        deriv_sum_ld = np.longdouble(0.0)
        result_dtype = np.longdouble

        for i, step_ld in enumerate(steps_ld):
            if NumericalDerivation._FD_COEFFS_LD[i] == 0: continue

            temp_args_ld = list(args_list_ld)
            temp_args_ld[idx] = original_arg_ld + step_ld

            call_dtype = np.longdouble
            try:
                temp_args_call = tuple(np.asarray(arg, dtype=call_dtype) for arg in temp_args_ld)
                func_eval = func(*temp_args_call)
            except Exception as e:
                raise RuntimeError(f"User function failed during FD evaluation with args {temp_args_call} (dtype {call_dtype}): {e}") from e

            try:
                func_eval_ld = np.asarray(func_eval, dtype=result_dtype)
                if i == 0:
                    deriv_sum_ld = np.zeros_like(func_eval_ld, dtype=result_dtype)
            except Exception as e:
                raise TypeError(
                    f"Function result could not be converted to {result_dtype.__name__} numpy array. "
                    f"Got type: {type(func_eval)}. Error: {e}"
                ) from e

            deriv_sum_ld += coeffs_ld[i] * func_eval_ld

        return deriv_sum_ld

    @staticmethod
    def _finite_diff_richardson(
            func: Callable,
            args_f128: Tuple[np.ndarray, ...],
            idx: int,
            base_h_ld: np.longdouble,
            func_accepts_ld: bool,
            return_error: bool
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        num_extrap = 4
        h_steps_ld = [base_h_ld / (NumericalDerivation._RICHARDSON_RATIO_LD**k) for k in range(num_extrap)]

        d_estimates_ld = [
            NumericalDerivation._central_diff_single_h(func, args_f128, idx, h_ld, func_accepts_ld)
            for h_ld in h_steps_ld
        ]

        T = list(d_estimates_ld)
        p = NumericalDerivation._FD_ORDER

        for k in range(1, num_extrap):
            power_r_p_k = NumericalDerivation._RICHARDSON_RATIO_LD**(p + 2*(k-1))
            denom = power_r_p_k - np.longdouble(1.0)
            if abs(denom) < np.finfo(np.longdouble).tiny:
                warnings.warn(f"Richardson extrapolation denominator near zero at level {k}.", RuntimeWarning)
                denom = np.sign(denom) * np.finfo(np.longdouble).tiny if denom != 0 else np.finfo(np.longdouble).tiny

            for i in range(num_extrap - k):
                T[i] = (power_r_p_k * T[i+1] - T[i]) / denom

        final_extrap_ld = T[0]

        error_est_ld = None
        if return_error:
            if num_extrap >= 2:
                second_best_ld = d_estimates_ld[0]
                T_prev = list(d_estimates_ld)
                for k_prev in range(1, num_extrap - 1):
                    power_r_p_k_prev = NumericalDerivation._RICHARDSON_RATIO_LD**(p + 2*(k_prev-1))
                    denom_prev = power_r_p_k_prev - np.longdouble(1.0)
                    if abs(denom_prev) < np.finfo(np.longdouble).tiny: denom_prev = np.finfo(np.longdouble).tiny
                    for i_prev in range(num_extrap - k_prev):
                        T_prev[i_prev] = (power_r_p_k_prev * T_prev[i_prev+1] - T_prev[i_prev]) / denom_prev
                second_best_ld = T_prev[0]
                error_est_ld = np.abs(final_extrap_ld - second_best_ld)
            else:
                error_est_ld = np.full_like(final_extrap_ld, np.nan, dtype=np.longdouble)

        final_result_f128 = np.asarray(final_extrap_ld, dtype=np.longdouble)

        if not return_error:
            return final_result_f128
        error_est_f128 = np.asarray(error_est_ld, dtype=np.longdouble)
        return final_result_f128, error_est_f128

    @staticmethod
    def _get_single_partial(
            func: Callable,
            args_f128: Tuple[np.ndarray, ...],
            idx: int,
            complex_h_ld: np.longdouble,
            base_fd_h_ld: np.longdouble,
            func_accepts_ld: bool,
            return_error: bool
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        try:
            result_f128 = NumericalDerivation._complex_diff(func, args_f128, idx, complex_h_ld)
            return (result_f128, None) if return_error else result_f128
        except Exception as complex_err:
            warnings.warn(
                f"Complex step failed for arg {idx} ({complex_err}). "
                f"Falling back to finite difference.",
                RuntimeWarning
            )
            try:
                return NumericalDerivation._finite_diff_richardson(
                    func, args_f128, idx,
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
    """Computes numerical derivatives using complex step differentiation with finite difference fallback.

    Args:
        func (Callable): Function to differentiate
        args (Union[list, tuple, float, int, np.ndarray]): Input arguments for function
        mode (Literal['derivative', 'gradient', 'jacobian', 'sum_partials'], optional): 
            'derivative': Compute partial derivative with respect to single argument
            'gradient': Compute gradient with respect to all arguments 
            'jacobian': Compute Jacobian matrix
            'sum_partials': Sum partial derivatives with respect to all arguments
            Defaults to 'derivative'.
        arg_index (Optional[int], optional): Index of argument to differentiate for 'derivative' mode. Defaults to 0.
        complex_step_h (Optional[float], optional): Step size for complex step differentiation. Defaults to optimal value.
        fd_base_dx (Optional[float], optional): Base step size for finite difference fallback. Defaults to 1e-4.
        func_accepts_longdouble (bool, optional): Whether function accepts longdouble inputs. Defaults to False.
        return_error_estimate (bool, optional): Whether to return error estimates. Defaults to False.

    Raises:
        TypeError: If func is not callable
        ValueError: If no arguments provided
        ValueError: If mode is invalid
        ValueError: If arg_index invalid for derivative mode
        ValueError: If complex_step_h is not positive
        ValueError: If fd_base_dx is not positive
        RuntimeError: If all derivative computation methods fail
        ValueError: If partial derivatives have inconsistent shapes
        RuntimeError: If internal computation error occurs

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
            Single return: Computed derivative/gradient/jacobian
            Tuple return: (Derivative result, Error estimate) if return_error_estimate=True
    """
    if not callable(func):
        raise TypeError("`func` must be a callable function.")

    args_tuple = args if isinstance(args, (list, tuple)) else (args,)
    if not args_tuple:
        raise ValueError("At least one argument must be provided.")

    args_f128 = NumericalDerivation._ensure_float64_tuple(args_tuple)
    num_args = len(args_f128)

    valid_modes = ['derivative', 'gradient', 'jacobian', 'sum_partials']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")

    if mode == 'derivative':
        if not isinstance(arg_index, int):
            raise ValueError("`arg_index` must be an integer for mode='derivative'.")
        if not 0 <= arg_index < num_args:
            raise ValueError(f"arg_index {arg_index} out of range for {num_args} arguments.")
    elif arg_index != 0:
        warnings.warn(f"arg_index={arg_index} is ignored when mode='{mode}'.", UserWarning)
        if mode == 'derivative': arg_index = 0

    current_complex_h_ld = np.longdouble(complex_step_h) if complex_step_h is not None else NumericalDerivation._DEFAULT_COMPLEX_H_LD
    current_fd_base_dx_ld = np.longdouble(fd_base_dx) if fd_base_dx is not None else NumericalDerivation._DEFAULT_FD_BASE_DX_LD

    if current_complex_h_ld <= 0: raise ValueError("complex_step_h must be positive.")
    if current_fd_base_dx_ld <= 0: raise ValueError("fd_base_dx must be positive.")

    results_list: List[np.ndarray] = []
    errors_list: List[Optional[np.ndarray]] = [] if return_error_estimate else None
    final_result: Optional[np.ndarray] = None
    final_error: Optional[np.ndarray] = None

    deriv_instance = NumericalDerivation()

    if mode == 'derivative':
        result_or_tuple = deriv_instance._get_single_partial(
            func, args_f128, arg_index,
            current_complex_h_ld, current_fd_base_dx_ld,
            func_accepts_longdouble, return_error_estimate
        )
        if return_error_estimate:
            final_result, final_error = result_or_tuple
        else:
            final_result = result_or_tuple

    elif mode == 'sum_partials':
        total_deriv_ld = np.longdouble(0.0)
        total_error_ld = np.longdouble(0.0) if return_error_estimate else None
        any_fd_used = False
        first_partial_shape = None

        for i in range(num_args):
            res_or_tup = deriv_instance._get_single_partial(
                func, args_f128, i,
                current_complex_h_ld, current_fd_base_dx_ld,
                func_accepts_longdouble, return_error_estimate
            )
            if return_error_estimate:
                partial_f128, error_f128 = res_or_tup
                partial_ld = np.asarray(partial_f128, dtype=np.longdouble)
                if i == 0:
                    total_deriv_ld = np.zeros_like(partial_ld, dtype=np.longdouble)
                    if total_error_ld is not None:
                        total_error_ld = np.zeros_like(partial_ld, dtype=np.longdouble)
                total_deriv_ld += partial_ld
                if error_f128 is not None:
                    total_error_ld += np.asarray(error_f128, dtype=np.longdouble)
                    any_fd_used = True
            else:
                partial_f128 = res_or_tup
                partial_ld = np.asarray(partial_f128, dtype=np.longdouble)
                if i == 0:
                    total_deriv_ld = np.zeros_like(partial_ld, dtype=np.longdouble)
                total_deriv_ld += partial_ld

        final_result = np.asarray(total_deriv_ld, dtype=np.longdouble)
        if return_error_estimate:
            final_error = np.asarray(total_error_ld, dtype=np.longdouble) if any_fd_used else None

    elif mode in ['gradient', 'jacobian']:
        any_fd_used_for_error = False
        for i in range(num_args):
            res_or_tup = deriv_instance._get_single_partial(
                func, args_f128, i,
                current_complex_h_ld, current_fd_base_dx_ld,
                func_accepts_longdouble, return_error_estimate
            )
            if return_error_estimate:
                partial_res, partial_err = res_or_tup
                results_list.append(partial_res)
                errors_list.append(partial_err)
                if partial_err is not None:
                    any_fd_used_for_error = True
            else:
                results_list.append(res_or_tup)

        if not results_list:
            raise RuntimeError("Internal error: No partial derivatives were computed for gradient/jacobian.")

        stack_axis = 0 if mode == 'gradient' else -1
        try:
            final_result = np.stack(results_list, axis=stack_axis)
        except ValueError as e:
            shapes = [np.shape(r) for r in results_list]
            raise ValueError(
                f"Could not stack partial derivatives into {mode}. Inconsistent shapes: {shapes}. Error: {e}"
            ) from e

        if return_error_estimate:
            if any_fd_used_for_error:
                processed_errors = []
                for res, err in zip(results_list, errors_list):
                    if err is None:
                        processed_errors.append(np.full_like(res, np.nan, dtype=np.longdouble))
                    else:
                        processed_errors.append(np.asarray(err, dtype=np.longdouble))

                try:
                    final_error = np.stack(processed_errors, axis=stack_axis)
                except ValueError:
                    warnings.warn(f"Could not stack error estimates for {mode} due to inconsistent shapes after NaN padding.", RuntimeWarning)
                    final_error = np.full_like(final_result, np.nan, dtype=np.longdouble)
            else:
                final_error = None

    if final_result is None:
        raise RuntimeError("Internal error: final_result was not computed.")

    if not return_error_estimate:
        return final_result
    if final_error is None:
        final_error = np.full_like(final_result, np.nan, dtype=np.longdouble)
    elif final_error.shape != final_result.shape:
        warnings.warn(f"Result shape {final_result.shape} and error shape {final_error.shape} mismatch. Broadcasting error.", RuntimeWarning)
        try:
            final_error = np.broadcast_to(final_error, final_result.shape).copy()
        except ValueError:
            final_error = np.full_like(final_result, np.nan, dtype=np.longdouble)

    return final_result, final_error