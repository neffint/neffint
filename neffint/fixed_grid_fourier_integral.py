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

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import pchip_interpolate

MAX_PHI_PSI_ITERATIONS = 1000

def complex_pchip(xi: ArrayLike, zi: ArrayLike, x: ArrayLike, derivative_order: int = 0, axis: int = 0) -> np.ndarray:
    """Compute the piecewise cubic hermite interpolating polynomial (PCHIP) characterized by the points (xi, zi), where xi are real and zi are complex,
    and evaluate this polynomial or its derivatives at the points x.
    The real and imaginary components are treated separately, using the implementation in scipy.interpolate for
    each component, as that implementation is only designed to take in real-valued functions.
    The PCHIP interpolation is a cubic interpolation that guarantees to preserve monotonicity on each subinterval of the input data. In this case,
    monotonicity is therefore preserved for both real and imaginary components.
    
    For more information about the PCHIP algorithm, see [1].
    
    [1] F. N. Fritsch and J. Butland, A method for constructing local monotone piecewise cubic interpolants, SIAM J. Sci. Comput., 5(2), 300-304 (1984). DOI:10.1137/0905021.

    :param xi: A 1D array of N real input points
    :type xi: ArrayLike
    :param func_values: An array of of shape (N, X1, X2, ...) containing the output of the function to compute the interpolation for at each xi.
    :type func_values: ArrayLike
    :param x: A 1D array of M real inputs to evaluate the interpolation at
    :type x: ArrayLike
    :param derivative_order: The order of derivatives to compute, defaults to 0
    :type derivative_order: int, optional
    :param axis: The axis in func_values that corresponds to xi, defaults to 0. By setting this argument, N does not need to be the first axis of func_values
    :type axis: int, optional
    :return: The computed PCHIP evaluated at the points x
    :rtype: np.ndarray
    """
    return (
        (pchip_interpolate(xi, np.real(zi), x, der=derivative_order, axis=axis)
        + 1j*pchip_interpolate(xi, np.imag(zi), x, der=derivative_order, axis=axis))
    )

def _phi_and_psi(x: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the Phi and Psi functions defined in equations E.142 and E.143 in [1], which are given by:

    Phi(x) = -j*exp(j*x)/x - 6j*(exp(j*x)+1)/x^3 + 12(exp(j*x)-1)/x^4,

    Psi(x) = exp(j*x)/x^2 + 2j*(2*exp(j*x)+1)/x^3 - 6(exp(j*x)-1)/x^4,

    Where j = sqrt(-1) is the imaginary unit.

    For |x| < 1, an Taylor series approximation is summed until convergence is reached.

    :param x: The input variable x, given as a single float or an array of floats.
    :type x: ArrayLike
    :raises RuntimeError: If the Taylor series summation did not converge after 1000 iterations.
    :return: Tuple of two arrays: Phi and Psi, with the same shape as x
    :rtype: Tuple[np.ndarray, np.ndarray]

    [1] N. Mounet. The LHC Transverse Coupled-Bunch Instability, PhD thesis 5305 (EPFL, 2012)
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
    for n in range(MAX_PHI_PSI_ITERATIONS+1):

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
        
        # Only calculate pis if not yet converged
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
        raise RuntimeError(f"Phi and Psi calculation failed to converge after {MAX_PHI_PSI_ITERATIONS} iterations")
    
    return phi, psi

def fourier_integral_inf_correction(times: np.ndarray, omega_end: float, func_value_end: ArrayLike, func_derivative_end: ArrayLike=0.):
    """Calculate the asymptotic correction term as omega->inf of a Fourier integral for the given times.
    Uses a first or second (if func_derivative_end is given) order Taylor expansion around the angular frequency omega_end.

    :param times: A 1D array of length M with the times to compute the Fourier integral for
    :type times: np.ndarray
    :param omega_end: A frequency to expand the Taylor series from
    :type omega_end: float
    :param func_value_end: An array of shape (X1, X2, ...) containing the values of the function to be transformed evaluated at omega_end
    :type func_value_end: ArrayLike
    :param func_derivative_end: An array of shape (X1, X2, ...) containing the derivative of the function to be transformed, evaluated at omega_end, defaults to 0.
    :type func_derivative_end: ArrayLike, optional
    :return: The asymptotic terms of the Fourier Integral for all times, given as an array of shape (M, X1, X2, ...)
    :rtype: np.ndarray
    """

    # TODO: Reconsider func_value_end and func_derivative_end types, shapes and type hints
    
    func_value_end = np.asarray(func_value_end)
    func_derivative_end = np.asarray(func_derivative_end)
    
    # Add axes from function output
    for _ in range(func_value_end.ndim):
        times = times[..., np.newaxis]
    
    # Add time axis
    func_value_end = func_value_end[np.newaxis, ...]
    func_derivative_end = func_derivative_end[np.newaxis, ...]
    
    return np.exp(1j*times*omega_end) * (1j*func_value_end/times - func_derivative_end/times**2)

def fourier_integral_fixed_sampling_pchip(
    times: ArrayLike,
    frequencies: ArrayLike,
    func_values: ArrayLike,
    inf_correction_term: bool
) -> np.ndarray:
    """Calculates the fourier integral of the function, given as an array of values corresponding to an array of frequencies, for the time values given as input.
    A Filon type algorithm using a piecewise cubic Hermite interpolating polynomial (pchip) and optionally an asymptotic correction term.
    For details on implementation, see [1].

    :param times: Float or 1D array of floats of length M, the time(s) to compute the fourier integral for.
    :type times: ArrayLike
    :param frequencies: Float or 1D array of floats of length N, the frequencies the function to be transformed have been evaluated at.
    :type frequencies: ArrayLike
    :param func_values: Complex or ND array of complex of shape (N, X1, X2, ...), the outputs of the function to be transformed at the frequencies given as input.
    :type func_values: ArrayLike
    :param inf_correction_term: True if an asymptotic correction term should be added to the output, otherwise the integral is effectively truncated at the highest frequency.
    :type inf_correction_term: bool
    :return: The fourier integral of the input function at the input times, given as an array of shape (M, X1, X2, ...)
    :rtype: np.ndarray
    
    [1] N. Mounet. The LHC Transverse Coupled-Bunch Instability, PhD thesis 5305 (EPFL, 2012)
    """

    # Turn inputs into arrays if they are not already, switch to angular frequencies
    omegas = 2 * np.pi * np.asarray(frequencies)
    func_values = np.asarray(func_values)
    times = np.asarray(times)
    
    # Set up result array
    result = np.zeros((len(times), *[axis_size for axis_size in func_values.shape[1:]]), dtype=complex)

    # Calculate derivatives
    func_derivatives = complex_pchip(omegas, func_values, omegas, derivative_order=1)
    
    # Add asymptotic correction term
    if inf_correction_term:
        result += fourier_integral_inf_correction(
            times=times,
            omega_end=omegas[-1],
            func_value_end=func_values[-1],
            func_derivative_end=func_derivatives[-1]
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
    func_derivatives = func_derivatives[np.newaxis, ...]

    # Add new axes for each axis of func_values except the frequency and time axes
    for _ in range(func_values.ndim - 2):
        omegas = omegas[..., np.newaxis]
        times = times[..., np.newaxis]
    
    # Find deltas
    delta_omegas = np.diff(omegas, axis=1)
    exp_omegas = np.exp(1j * omegas[:, :-1] * times)

    x = delta_omegas*times

    exp_x = np.exp(1j * x)
    phi_x, psi_x = _phi_and_psi(x)
    phi_minus_x, psi_minus_x = _phi_and_psi(-x)

    result += np.sum(delta_omegas*exp_omegas*(
        func_values[:, :-1] * phi_minus_x * exp_x + 
        func_values[:, 1: ] * phi_x - 
        delta_omegas*func_derivatives[:, :-1] * psi_minus_x * exp_x + 
        delta_omegas*func_derivatives[:, 1: ] * psi_x
    ), axis=1) # Sum over frequency axis

    return result
