#   Copyright 2023 neffint
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
from enum import Enum
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from .utils import complex_pchip

MAX_TAYLOR_ITERATIONS = 1000

class InterpolationMode(Enum):
    PCHIP = "pchip"
    LINEAR = "linear"

def _lambda(x: ArrayLike) -> np.ndarray:
    """Calculates the Lambda function defined in equations E.136 in [N.Mounet-PhD]_, which is given by:

    Lambda(x) = -j*exp(j*x)/x + (exp(j*x)-1)/x^2,

    where j = sqrt(-1) is the imaginary unit.

    For \|x\| < 1, a Taylor series approximation is summed until convergence is reached.

    :param x: The input variable x, given as a single float or an array of floats.
    :type x: ArrayLike
    :raises RuntimeError: If the Taylor series summation did not converge after 1000 iterations.
    :return: Lambda, an array with the same shape as x
    :rtype: np.ndarray
    """
    # Make input into array if it is not already, and prepare output arrays
    x = np.asarray(x)
    result = np.zeros_like(1j*x)

    # Set relative tolerance to machine precision
    rel_tolerance = np.finfo(x.dtype).eps

    # Define a mask of where to use a Taylor series approximation
    taylor_mask = np.abs(x) < 1.0

    # Calculate phi and psi for x >= 1
    xx = x[~taylor_mask]
    exp_jx = np.exp(1j*xx)
    result[~taylor_mask] = -1j*exp_jx/xx + (exp_jx-1)/xx**2

    # Set up for convergence loop for x < 1
    xx = x[taylor_mask]
    inv_factorial = 1.
    jx_to_n = np.ones_like(1j*xx) # Starts as x**0
    abs_x_to_n_plus_1 = np.abs(xx) # Starts as x**(0+1)
    exp_abs_x = np.exp(np.abs(xx))

    # Run convergence loop until phi and psi converged
    for n in range(MAX_TAYLOR_ITERATIONS+1):
        
        # Add next term
        result[taylor_mask] += 1/(n+2) * jx_to_n * inv_factorial
        
        # Check convergence
        min_component = np.min([np.abs(np.real(result[taylor_mask])), np.abs(np.imag(result[taylor_mask]))], axis=0)
        remaining_term_bound = abs_x_to_n_plus_1*exp_abs_x * inv_factorial * 1/((n+3)*(n+1))
        if np.all(remaining_term_bound < rel_tolerance * min_component):
            break
        
        # Update variables
        jx_to_n *= 1j*xx
        abs_x_to_n_plus_1 *= np.abs(xx)
        inv_factorial /= n + 1
        
    # If loop finished without converging, raise error
    if n == MAX_TAYLOR_ITERATIONS:
        raise RuntimeError(f"Lambda calculation failed to converge after {MAX_TAYLOR_ITERATIONS} iterations")
    
    return result

def _phi_and_psi(x: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the Phi and Psi functions defined in equations E.142 and E.143 in [N.Mounet-PhD]_, which are given by:

    Phi(x) = -j*exp(j*x)/x - 6j*(exp(j*x)+1)/x^3 + 12(exp(j*x)-1)/x^4,

    Psi(x) = exp(j*x)/x^2 + 2j*(2*exp(j*x)+1)/x^3 - 6(exp(j*x)-1)/x^4,

    Where j = sqrt(-1) is the imaginary unit.

    For \|x\| < 1, a Taylor series approximation is summed until convergence is reached.

    :param x: The input variable x, given as a single float or an array of floats.
    :type x: ArrayLike
    :raises RuntimeError: If the Taylor series summation did not converge after 1000 iterations.
    :return: Tuple of two arrays: Phi and Psi, with the same shape as x
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    # Make input into array if it is not already, and prepare output arrays
    x = np.asarray(x)
    phi = np.zeros_like(1j*x)
    psi = np.zeros_like(phi)

    # Set relative tolerance to machine precision
    rel_tolerance = np.finfo(x.dtype).eps

    # Define a mask of where to use a Taylor series approximation
    taylor_mask = np.abs(x) < 1.0

    # Calculate phi and psi for x >= 1
    xx = x[~taylor_mask]
    exp_jx = np.exp(1j*xx)
    phi[~taylor_mask] = (-1j*xx**3 * exp_jx - 6j*xx*(exp_jx+1) + 12*(exp_jx-1))/xx**4
    psi[~taylor_mask] = ( xx**2 * exp_jx + 2j*xx*(2*exp_jx+1) - 6*(exp_jx-1))/xx**4

    # Set up for convergence loop for x < 1
    xx = x[taylor_mask]
    inv_factorial = 1.
    jx_to_n = np.ones_like(1j*xx) # Starts as x**0
    abs_x_to_n_plus_1 = np.abs(xx) # Starts as x**(0+1)
    exp_abs_x = np.exp(np.abs(xx))

    # Run convergence loop until phi and psi converged
    phi_converged = False
    psi_converged = False
    for n in range(MAX_TAYLOR_ITERATIONS+1):

        # Stop loop if converged
        if phi_converged and psi_converged:
            break

        # Only calculate phi if not yet converged (to save computation time)
        if not phi_converged:
            phi[taylor_mask] += (n+6)/((n+3)*(n+4)) * jx_to_n * inv_factorial

            # Check for convergence
            min_phi_component = np.min([np.abs(np.real(phi[taylor_mask])), np.abs(np.imag(phi[taylor_mask]))], axis=0)
            phi_remaining_terms_bound = 2 * abs_x_to_n_plus_1 * exp_abs_x * inv_factorial / ((n+1)*(n+5))
            phi_converged = np.all(phi_remaining_terms_bound < rel_tolerance * min_phi_component)
        
        # Only calculate psi if not yet converged
        if not psi_converged:
            psi[taylor_mask] += - 1/((n+3)*(n+4)) * jx_to_n * inv_factorial

            # Check for convergence
            min_psi_component = np.min([np.abs(np.real(psi[taylor_mask])), np.abs(np.imag(psi[taylor_mask]))], axis=0)
            psi_remaining_terms_bound = abs_x_to_n_plus_1 * exp_abs_x * inv_factorial / ((n+1)*(n+4)*(n+5))
            psi_converged = np.all(psi_remaining_terms_bound < rel_tolerance*min_psi_component)
        
        # Update variables
        jx_to_n *= 1j*xx
        abs_x_to_n_plus_1 *= np.abs(xx)
        inv_factorial /= n+1

    # If loop finished without converging, raise error
    if not (phi_converged and psi_converged):
        raise RuntimeError(f"Phi and Psi calculation failed to converge after {MAX_TAYLOR_ITERATIONS} iterations")
    
    return phi, psi

def _fourier_integral_inf_correction(
    times: ArrayLike,
    omega_end: float,
    func_value_end: ArrayLike,
    func_derivative_end: ArrayLike=0.,
    positive_inf: bool=True
) -> np.ndarray:
    """Calculate the asymptotic correction term as omega -> +-inf of a Fourier integral for the given times.
    Uses a first or second (if func_derivative_end is given) order Taylor expansion around the angular frequency omega_end.

    :param times: Float or 1D array of floats of length M, the time(s) [s]* to compute the fourier integral for
    :type times: ArrayLike
    :param omega_end: An angular frequency [rad/s]* to expand the Taylor series from. If ``positive_inf == True``, this should be the highest angular frequency used. Otherwise, it should be the lowest (closest to -inf).
    :type omega_end: float
    :param func_value_end: An array of shape (X1, X2, ...) containing the values of the function to be transformed evaluated at omega_end
    :type func_value_end: ArrayLike
    :param func_derivative_end: An array of shape (X1, X2, ...) containing the derivative of the function to be transformed, evaluated at omega_end, defaults to 0
    :type func_derivative_end: ArrayLike, optional
    :param positive_inf: Flag set to True if the asymptotic correction towards +inf should be calculated, and False if the correction towards -inf should be calculated
    :type positive_inf: bool, optional
    :return: The asymptotic terms of the Fourier Integral for all times, given as an array of shape (M, X1, X2, ...)
    :rtype: np.ndarray
    
    \*Though the units s and rad/s are used here, any coherent set of time and angular frequency units will work
    """
    
    sign = (1 if positive_inf else -1)
    
    times = np.asarray(times)
    func_value_end = np.asarray(func_value_end)
    func_derivative_end = np.asarray(func_derivative_end)
    
    # Add axes from function output
    for _ in range(func_value_end.ndim):
        times = times[..., np.newaxis]
    
    # Add time axis
    func_value_end = func_value_end[np.newaxis, ...]
    func_derivative_end = func_derivative_end[np.newaxis, ...]
    
    return sign * np.exp(1j*times*omega_end) * (1j*func_value_end/times - func_derivative_end/times**2)

def fourier_integral_fixed_sampling(
    times: ArrayLike,
    frequencies: ArrayLike,
    func_values: ArrayLike,
    pos_inf_correction_term: bool,
    neg_inf_correction_term: bool,
    interpolation: str
) -> np.ndarray:
    """Calculates the fourier integral of a function for the time values given as input:
    
    integral from fmin to fmax of : exp(2*pi*j*f*t)*func(f)*df,
    
    where t is the time, fmin is the first frequency in the input ``frequencies``, fmax is either the highest frequency or (positive) infinity, depending on ``inf_correction_term``,
    and func(f) is the input function of frequency. The function is given as an array of outputs corresponding to the array of frequencies given. 
    
    A Filon type algorithm using a either a piecewise cubic Hermite interpolating polynomial ([PCHIP]_) or a piecewise linear polynomial, depending on the interpolation argument.
    Optionally, an asymptotic correction term can also be computed at each end.
    For details on implementation, see [N.Mounet-PhD]_.

    :param times: Float or 1D array of floats of length M, the time(s) [s]* to compute the fourier integral for
    :type times: ArrayLike
    :param frequencies: Float or 1D array of floats of length N, the frequencies [Hz]* the function to be transformed have been evaluated at
    :type frequencies: ArrayLike
    :param func_values: Complex or ND array of complex of shape (N, X1, X2, ...), the outputs of the function to be transformed at the frequencies given as input
    :type func_values: ArrayLike
    :param pos_inf_correction_term: True if an asymptotic correction term towards +infinity should be added, otherwise the integral is effectively truncated at the highest frequency
    :type pos_inf_correction_term: bool
    :param neg_inf_correction_term: True if an asymptotic correction term towards -infinity should be added, otherwise the integral is effectively truncated at the lowest (closest to -inf) frequency
    :type neg_inf_correction_term: bool
    :param interpolation: String either equal to "pchip" or "linear", to select the integration methods using PCHIP or piecewise linear interpolation, respectively.
    :type interpolation: str
    :return: The fourier integral of the input function at the input times, given as an array of shape (M, X1, X2, ...)
    :rtype: np.ndarray
    
    \*Though the units s and Hz are used here, any coherent set of time and frequency units will work
    """
    
    assert interpolation in [mode.value for mode in InterpolationMode]
    interpolation_mode = InterpolationMode(interpolation)
    
    # Turn inputs into arrays if they are not already, switch to angular frequencies
    # Filter out infinities and nan's
    isfinite = np.isfinite(frequencies) & np.all(np.isfinite(func_values), axis=tuple(range(1, len(func_values.shape))))
    if not np.all(isfinite):
        logging.info("Removing infinite frequencies and values from arrays.")
        
    omegas = 2 * np.pi * np.asarray(frequencies[isfinite])
    func_values = np.asarray(func_values[isfinite])
    times = np.asarray(times)
    
    # Set up result array, shape (M, X1, X2, ...)
    result = np.zeros((len(times), *[axis_size for axis_size in func_values.shape[1:]]), dtype=complex)

    # Do mode specific setup
    if interpolation_mode == InterpolationMode.PCHIP:
        # Calculate derivatives
        func_derivatives = complex_pchip(omegas, func_values, omegas, derivative_order=1)
        func_derivative_high_end = func_derivatives[-1]
        func_derivative_low_end = func_derivatives[0]
        
    elif interpolation_mode == InterpolationMode.LINEAR:
        # Set derivative at end to 0
        func_derivative_high_end = 0
        func_derivative_low_end = 0

    else:
        raise NotImplementedError(f"This state should be unreachable. Need to implement derivative calculation for {interpolation_mode}")
    
    # Add asymptotic correction terms
    if pos_inf_correction_term:
        result += _fourier_integral_inf_correction(
            times=times,
            omega_end=omegas[-1],
            func_value_end=func_values[-1],
            func_derivative_end=func_derivative_high_end,
            positive_inf=True
        )
        
    if neg_inf_correction_term:
        result += _fourier_integral_inf_correction(
            times=times,
            omega_end=omegas[0],
            func_value_end=func_values[0],
            func_derivative_end=func_derivative_low_end,
            positive_inf=False
        )

    # Reshape arrays for correct broadcasting
    # Now in all arrays except result:
    # - Axis 0  : times
    # - Axis 1  : frequencies
    # - Axis 2+ : func output dims
    # In result, the frequency axis is absent, and axes 1+ correspond to the func output dims

    # Add frequency axis
    times = times[..., np.newaxis]

    # Add time axis
    omegas = omegas[np.newaxis, ...]
    func_values = func_values[np.newaxis, ...]

    # Add new axes for each axis of func_values except the frequency and time axes
    for _ in range(func_values.ndim - 2):
        omegas = omegas[..., np.newaxis]
        times = times[..., np.newaxis]
    
    
    # Do the fourier integration
    if interpolation_mode == InterpolationMode.PCHIP:
        func_derivatives = func_derivatives[np.newaxis, ...] # Add time axis
        result += _fourier_integral_fixed_sampling_pchip(times, omegas, func_values, func_derivatives)
    
    elif interpolation_mode == InterpolationMode.LINEAR:
        result += _fourier_integral_fixed_sampling_linear(times, omegas, func_values)
    
    else:
        raise NotImplementedError(f"This state should be unreachable. Need to implement fourier integral calculation for {interpolation_mode}")
    
    return result
        

def _fourier_integral_fixed_sampling_pchip(
    times: np.ndarray,
    omegas: np.ndarray,
    func_values: np.ndarray,
    func_derivatives: np.ndarray,
) -> np.ndarray:
    
    # Calculate intermediate values
    delta_omegas = np.diff(omegas, axis=1)
    exp_omegas = np.exp(1j * omegas[:, :-1] * times)
    x = delta_omegas*times
    exp_x = np.exp(1j * x)
    phi_x, psi_x = _phi_and_psi(x)
    phi_minus_x, psi_minus_x = _phi_and_psi(-x)

    # Calculate integral by summing over frequency axis
    result = np.sum(delta_omegas*exp_omegas*(
        func_values[:, :-1] * phi_minus_x * exp_x + 
        func_values[:, 1: ] * phi_x - 
        delta_omegas*func_derivatives[:, :-1] * psi_minus_x * exp_x + 
        delta_omegas*func_derivatives[:, 1: ] * psi_x
    ), axis=1) # Sum over frequency axis

    return result

def _fourier_integral_fixed_sampling_linear(
    times: np.ndarray,
    omegas: np.ndarray,
    func_values: np.ndarray
) -> np.ndarray:
    
    delta_omegas = np.diff(omegas, axis=1)

    # Calculate integral by summing over frequency axis
    result = np.sum(delta_omegas*(
        func_values[:, :-1]  * np.exp(1j*omegas[:, 1: ]*times) * _lambda(-delta_omegas*times)
        + func_values[:, 1:] * np.exp(1j*omegas[:, :-1]*times) * _lambda( delta_omegas*times) 
    ), axis=1) # Sum over frequency axis

    return result
