import logging
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import sortednp
from numpy.typing import ArrayLike
from scipy.interpolate import pchip_interpolate

from .fixed_grid_fourier_integral import fourier_integral_fixed_sampling_pchip

# TODO: go over types
# TODO: Write something about units in documentation

FuncOutputType = Any # TODO: reconsider func output type
FuncType = Callable[[float], FuncOutputType]

class CachedFunc:
    def __init__(self, function: FuncType):
        self.function = function
        self.cache: Dict[float, FuncOutputType] = dict()
    
    def __call__(self, x: float) -> FuncOutputType: 
        if x not in self.cache.keys():
            self.cache[x] = self.function(x)
        return self.cache[x]


def bisect_intervals(interval_endpoints: np.ndarray, linear_bisection_mask: np.ndarray) -> np.ndarray:
    midpoints = np.zeros(len(interval_endpoints)-1)

    left_ends = interval_endpoints[:-1]
    right_ends = interval_endpoints[1:]
    
    midpoints[~linear_bisection_mask] = np.sqrt(left_ends[~linear_bisection_mask] * right_ends[~linear_bisection_mask]) # geometric/logarithmic mean
    midpoints[ linear_bisection_mask] =  1/2 * (left_ends[ linear_bisection_mask] + right_ends[ linear_bisection_mask]) # arithmetic mean

    return midpoints


def integrate_interpolation_error(
    frequencies: np.ndarray,
    linear_bisection_mask: np.ndarray,
    interpolation_error_at_midpoints: np.ndarray
) -> Tuple[float, int]:

    left_ends = frequencies[:-1]
    right_ends = frequencies[1:]

    interval_error = np.zeros_like(interpolation_error_at_midpoints)

    # Logarithic Simpson's rule
    # Endpoints of intervals not included in formula, as they are interpolated exactly
    a =  left_ends[~linear_bisection_mask]
    b = right_ends[~linear_bisection_mask]
    interval_error[~linear_bisection_mask] = 2/3 * np.sqrt(a*b) * np.log(b/a) * interpolation_error_at_midpoints[~linear_bisection_mask]

    # Normal (linear) Simpson's rule
    # Endpoints of intervals not included in formula, as they are interpolated exactly
    a =  left_ends[ linear_bisection_mask]
    b = right_ends[ linear_bisection_mask]
    interval_error[ linear_bisection_mask] = 2/3 * (b - a) * interpolation_error_at_midpoints[linear_bisection_mask]

    total_error = np.sum(interval_error)
    index_max_error = np.argmax(interval_error)

    return total_error, index_max_error


def find_interval_with_largest_error(
    frequencies: np.ndarray,
    func_values: np.ndarray,
    bisection_mode_condition: Callable[[np.ndarray], np.ndarray],
    func: FuncType,
    interpolation_error_metric: Callable[[FuncOutputType, np.ndarray], float]
) -> Tuple[np.ndarray, int, float]:
    
    # Make mask array where True indicates that the bisection of the interval of the same index should be done linearly
    linear_bisection_mask = bisection_mode_condition(frequencies[:-1])

    # Bisect all intervals either arithmetically or geometrically depending on frequency
    freq_midpoints = bisect_intervals(frequencies, linear_bisection_mask)
    
    # Compute func at bisections
    values_at_midpoints = np.array([func(freq) for freq in freq_midpoints])

    # Interpolate func at bisections using pchip
    # TODO: same for linear interp?
    # Note: From 2nd iteration and onwards, it is slightly faster to only interpolate around the new point, but not much.
    # This approach then allows for much simpler code.
    interpolated_values_at_midpoints = pchip_interpolate(xi=frequencies, yi=func_values, x=freq_midpoints, axis=0)

    # Find interpolation error at midpoints using a user defined function
    interpolation_error_at_midpoints = np.array([interpolation_error_metric(values_at_midpoints[i], interpolated_values_at_midpoints[i]) for i in range(len(freq_midpoints))])

    # Integrate the interpolation error over the frequency range, find midpoint frequency of the interval with largest error
    total_interpolation_error, index_max_error_interval = integrate_interpolation_error(frequencies, linear_bisection_mask, interpolation_error_at_midpoints)

    return freq_midpoints, index_max_error_interval, total_interpolation_error


def add_points_until_interpolation_converged(
        starting_frequencies: np.ndarray,
        starting_func_values: np.ndarray,
        func: FuncType,
        bisection_mode_condition: Optional[Callable[[np.ndarray], np.ndarray]],
        interpolation_error_metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
        bisection_tolerance: float
    ):

    logging.info(f"Starting adaptive bisection algorithm of frequency range: {starting_frequencies[0]} to {starting_frequencies[-1]}, "
                 f"starting with {len(starting_frequencies)} frequencies.")
    
    frequencies = starting_frequencies
    func_values = starting_func_values

    if bisection_mode_condition is None:
        bisection_mode_condition = lambda x: np.zeros_like(x, dtype=bool)
    
    total_interpolation_error = np.inf

    while total_interpolation_error > bisection_tolerance:

        midpoint_freqs, index_freq_largest_error, total_interpolation_error = find_interval_with_largest_error(frequencies, func_values, bisection_mode_condition, func, interpolation_error_metric)
        midpoint_freq_interval_largest_error = midpoint_freqs[index_freq_largest_error]

        logging.info(f"Largest error in interval {index_freq_largest_error} between frequencies"
                     f"{frequencies[index_freq_largest_error]} and {frequencies[index_freq_largest_error+1]}.\n"
                     f"Integrated interpolation error: {total_interpolation_error}\n"
                     f"Number of frequencies used to interpolate: {len(frequencies)}")

        # Insert midpoint from interval with largest error into frequencies and impedances
        frequencies = np.insert(frequencies, index_freq_largest_error+1, midpoint_freq_interval_largest_error)
        func_values = np.insert(func_values, index_freq_largest_error+1, func(midpoint_freq_interval_largest_error), axis=0)

    logging.info(f"Bisection algorithm converged. Final number of frequencies: {len(frequencies)}.\n"
                 f"Adding {len(midpoint_freqs)-1} already evaluated midpoint frequencies to frequency array.")

    # After bisection algorithm, return all frequencies with cached func results
    frequencies = sortednp.merge(frequencies, midpoint_freqs, duplicates=sortednp.DROP)
    func_values = np.array([func(freq) for freq in frequencies])

    return frequencies, func_values


def adaptive_fourier_integral(
    times: ArrayLike,
    initial_frequency_range: Sequence,
    func: FuncType, 
    interpolation_error_metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
    bisection_mode_condition: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> np.ndarray:
    # TODO: Consider changing all insert and append to work in-place on a larger array

    # TODO: rename and move
    # TODO: Multiply by pi to denormalize
    bisection_tolerance = 1e-3
    relative_wake_tolerance = 1e-8

    # Starting frequencies
    frequencies = np.array(initial_frequency_range)
    assert len(frequencies) >= 2
    assert np.all(frequencies > 0) # TODO: Implement negative frequency support

    func = CachedFunc(func)

    # Set up sorted array of func outputs
    func_values = np.array([func(freq) for freq in frequencies])

    frequencies, func_values = add_points_until_interpolation_converged(
        starting_frequencies=frequencies,
        starting_func_values=func_values,
        func=func,
        bisection_mode_condition=bisection_mode_condition,
        interpolation_error_metric=interpolation_error_metric,
        bisection_tolerance=bisection_tolerance
    )

    func_derivative_values = pchip_interpolate(frequencies, func_values, frequencies, der=1, axis=0)
    integral_values = fourier_integral_fixed_sampling_pchip(times, frequencies, func_values, inf_correction_term=True)

    logging.info("Iteratively adding lower frequencies until convergence.")

    # TODO: Consider moving while loop to its own function
    integral_relative_error = np.inf
    while integral_relative_error > relative_wake_tolerance:

        # Make a new frequency and calculate func and derivative
        frequencies = np.insert(frequencies, 0, frequencies[0]/2**0.2) # TODO: reconsider magic number 2 here
        func_values = np.insert(func_values, 0, func(frequencies[0]), axis=0)
        func_derivative_values = np.insert(func_derivative_values, 0, pchip_interpolate(frequencies[:3], func_values[:3], frequencies[0], der=1, axis=0), axis=0) # 3 values needed to compute pchip derivative

        # Calculate wake contribution from the new interval
        new_integral_contributions = fourier_integral_fixed_sampling_pchip(times, frequencies[:2], func_values[:2], inf_correction_term=False)

        # TODO: Allow for absolute error in addition to relative?
        # TODO: Remove magic mode
        integral_relative_error = max_relative_error(integral_values + new_integral_contributions, integral_values, mode="abs")

        integral_values += new_integral_contributions

        logging.info(f"New lowest frequency: {frequencies[0]}\nFourier integral relative error between last two iterations: {integral_relative_error}")
    
    logging.info("Iteratively adding higher frequencies until convergence.")

    # TODO: As above, consider moving while loop to its own function
    integral_relative_error = np.inf
    while integral_relative_error > relative_wake_tolerance:
        
        # Make a new frequency and calculate func and derivative
        frequencies = np.append(frequencies, 2*frequencies[-1]) # TODO reconsider magic 2 here
        func_values = np.append(func_values, [func(frequencies[-1])], axis=0)
        func_derivative_values = np.append(func_derivative_values, [pchip_interpolate(frequencies[-3:], func_values[-3:], frequencies[-1], der=1, axis=0)], axis=0)

        new_integral_contributions = (
            fourier_integral_fixed_sampling_pchip(times, frequencies[-2:], func_values[-2:], inf_correction_term=True)
            - fourier_integral_fixed_sampling_pchip(times, [frequencies[-2]], [func_values[-2]], inf_correction_term=True) # TODO: clean up this ugly indexing
        )

        # TODO: Allow for absolute error in addition to relative?
        # TODO: Remove magic mode
        integral_relative_error = max_relative_error(integral_values + new_integral_contributions, integral_values, mode="abs")

        integral_values += new_integral_contributions

        logging.info(f"New highest frequency: {frequencies[-1]}\nFourier integral relative error between last two iterations: {integral_relative_error}")

    return integral_values, frequencies


def _max_relative_error_kernel(x: np.ndarray, reference: np.ndarray):
    return np.max( np.abs((x - reference) / reference)  )


def max_relative_error(x: np.ndarray, reference: np.ndarray, mode: str) -> float:
    # TODO: add index return if necessary
    if mode == "abs":
        return _max_relative_error_kernel(np.abs(x), np.abs(reference))
    elif mode == "real":
        return _max_relative_error_kernel(np.real(x), np.real(reference))
    elif mode == "imag":
        return _max_relative_error_kernel(np.imag(x), np.imag(reference))
    elif mode == "realimag":
        max_real_error = _max_relative_error_kernel(np.real(x), np.real(reference))
        max_imag_error = _max_relative_error_kernel(np.imag(x), np.imag(reference))
        return max(max_real_error, max_imag_error)
    else:
        raise ValueError(f"{mode} is not a valid mode. Use one of: 'abs', 'real', 'imag' or 'realimag'.")
