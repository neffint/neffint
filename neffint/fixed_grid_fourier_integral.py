import itertools

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import pchip_interpolate

MAX_PHI_PSI_ITERATIONS = 1000


def phi_and_psi(x: ArrayLike) -> np.ndarray:

    x = np.asarray(x)
    phi = np.zeros_like(1j*x)
    psi = np.zeros_like(phi)

    # Set relative tolerance to machine precision
    rel_tolerance = np.finfo(x.dtype).eps

    big_x_mask = np.abs(x) >= 1

    x_high = x[big_x_mask]
    exp_jx = np.exp(1j*x_high)
    phi[big_x_mask] = (-1j*x_high**3 * exp_jx - 6j*x_high*(exp_jx+1) + 12*(exp_jx-1))/x_high**4
    psi[big_x_mask] = ( x_high**2 * exp_jx + 2j*x_high*(2*exp_jx+1) - 6*(exp_jx-1))/x_high**4

    phi_converged = False
    psi_converged = False
    x_low = x[~big_x_mask]
    inv_factorial = np.ones_like(x_low)
    power = np.ones_like(x_low, dtype=np.complex256)
    exp_absx = np.exp(np.abs(x_low))

    for n in itertools.count():
        if phi_converged and psi_converged:
            break

        if not phi_converged:
            phi[~big_x_mask] += (n+6)/((n+3)*(n+4)) * power * inv_factorial
            phi_converged = np.all(
                2* np.abs(x_low) * exp_absx * inv_factorial / ((n+1)*(n+5)) <
                rel_tolerance*np.min([np.abs(np.real(phi[~big_x_mask])), np.abs(np.imag(phi[~big_x_mask]))], axis=0)
            )
        
        if not psi_converged:
            psi[~big_x_mask] -= 1/((n+3)*(n+4)) * power * inv_factorial
            psi_converged = np.all(
                np.abs(x_low) * exp_absx * inv_factorial / ((n+1)*(n+4)*(n+5)) <
                rel_tolerance*np.min([np.abs(np.real(psi[~big_x_mask])), np.abs(np.imag(psi[~big_x_mask]))], axis=0)
            )
        
        power *= 1j*x_low
        inv_factorial /= n+1


        
        if n >= MAX_PHI_PSI_ITERATIONS:
            raise RuntimeError(f"Phi and Psi calculation failed to converge after {n} iterations")
    
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
