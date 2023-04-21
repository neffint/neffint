import itertools

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import pchip_interpolate

MAX_PHI_PSI_ITERATIONS = 1000

def phi_and_psi(x: ArrayLike) -> np.ndarray:

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

def fourier_integral_fixed_sampling_pchip(
    times: ArrayLike,
    frequencies: ArrayLike,
    func_values: ArrayLike,
    inf_correction_term: bool) -> np.ndarray:

    # Turn inputs into arrays if they are not already, switch to angular frequencies
    omegas = 2 * np.pi * np.asarray(frequencies)
    func_values = np.asarray(func_values)
    times = np.asarray(times)

    # Calculate derivatives
    func_derivatives = pchip_interpolate(omegas, func_values, omegas, der=1, axis=0)

    # Reshape arrays for correct broadcasting
    # Now in all arrays:
    # - Axis 0  : times
    # - Axis 1  : frequencies
    # - Axis 2+ : func output dims

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
    phi_x, psi_x = phi_and_psi(x)
    phi_minus_x, psi_minus_x = phi_and_psi(-x)

    result = np.sum(delta_omegas*exp_omegas*(
        func_values[:, :-1] * phi_minus_x * exp_x + 
        func_values[:, 1: ] * phi_x - 
        delta_omegas*func_derivatives[:, :-1] * psi_minus_x * exp_x + 
        delta_omegas*func_derivatives[:, 1: ] * psi_x
    ), axis=1, keepdims=True) # Sum over frequency axis

    if inf_correction_term:
        result += np.exp(1j*times*omegas[:, -1]) * (1j*func_values[:, -1]/times - func_derivatives[:, -1]/times**2)

    # Remove frequency axis
    result = np.squeeze(result, axis=1)

    return result
