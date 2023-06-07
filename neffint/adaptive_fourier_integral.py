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
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import sortednp
from numpy.typing import ArrayLike

from .fixed_grid_fourier_integral import (_fourier_integral_inf_correction,
                                          fourier_integral_fixed_sampling)
from .utils import complex_pchip

# TODO: go over types
# TODO: Write something about units in documentation

FuncOutputType = Any # TODO: reconsider func output type
FuncType = Callable[[float], FuncOutputType]

class CachedFunc:
    """Wrapper class around a function to cache function calls for faster evaluation of the function with the same arguments.
    """
    
    def __init__(self, function: FuncType):
        """Create a wrapper around the input function that saves function calls in a dictionary for faster reevaluation with the same argument.

        :param function: The function to be wrapped. Must be a function of one parameter, which must be hashable.
        :type function: Callable
        """ # TODO: Look over types in docstring
        
        self.function = function
        self.cache: Dict[float, FuncOutputType] = dict()
    
    def __call__(self, x: float) -> FuncOutputType:
        """Evaluate func, defaulting to looking up the input variable x in the dictionary of cached calls,
        then evaluates the function itself if no cached output was stored for that input.

        :param x: The input variable of the wrapped function
        :type x: float
        :return: The outputs of the wrapped function for the given input
        :rtype: FuncOutputType
        """
        if x not in self.cache.keys():
            self.cache[x] = self.function(x)
        return self.cache[x]


def bisect_intervals(interval_endpoints: np.ndarray, linear_bisection_mask: np.ndarray) -> np.ndarray:
    """Takes in an array and finds the midpoint between all all neighboring elements in the input array.
    The midpoints are computed either as an arithmetic mean or a geometric mean of the two elements, depending on linear_bisection_mask.

    :param interval_endpoints: A 1D array of length N (N>1) of floats
    :type interval_endpoints: np.ndarray
    :param linear_bisection_mask: A 1D array of length N-1 of booleans. A value of True indicates that the corresponding midpoint
    should be calculated with a arithmetic (linear) mean. A value of False indicates that a geometric (logarithmic) mean should be used
    for the corresponding interval.
    :type linear_bisection_mask: np.ndarray
    :return: A 1D array of length N-1 containing the midpoints
    :rtype: np.ndarray
    """
    
    # NOTE: With normal (static) numpy arrays, testing shows it is faster to recalculate all midpoints
    # than to calculate one midpoint and insert into an array of existing midpoints
    midpoints = np.zeros(len(interval_endpoints)-1)

    left_ends = interval_endpoints[:-1]
    right_ends = interval_endpoints[1:]
    
    midpoints[~linear_bisection_mask] = np.sqrt(left_ends[~linear_bisection_mask] * right_ends[~linear_bisection_mask]) # geometric/logarithmic mean
    midpoints[ linear_bisection_mask] =  1/2 * (left_ends[ linear_bisection_mask] + right_ends[ linear_bisection_mask]) # arithmetic mean

    return midpoints


def integrate_interpolation_error(
    interval_endpoints: np.ndarray,
    linear_bisection_mask: np.ndarray,
    interpolation_error_at_midpoints: np.ndarray
) -> Tuple[float, int]:
    """Integrate the input interpolation error array over all intervals, using Simpson's method.
    For logarithmically bisected intervals, the Simpson formula is modified accordingly.
    Since the interpolation equals zero at the interval end points, only the contribution from the midpoint is used in the Simpson formulae.
    
    The formulae used are:
    
    For linearly bisected intervals: 2/3 * (b-a) * err((a+b)/2).
    
    For logarithmically bisected intervals: 2/3 * sqrt(a*b) * log(b/a) * err(sqrt(a*b)),
    
    where a and b are the interval endpoints, and err is the interpolation error.

    :param interval_endpoints: A 1D array of size N containing the endpoints (float) of all intervals to be integrated over
    :type interval_endpoints: np.ndarray
    :param linear_bisection_mask: A 1D array of booleans of size N-1 where True indicates that the interval with the same index is bisected linearly,
    and False indicates that the corresponding interval was bisected logarithmically
    :type linear_bisection_mask: np.ndarray
    :param interpolation_error_at_midpoints: A 1D array of size N-1 containing the interpolation error 
    at the logarithmic or linear midpoints (depending on linear_bisection_mask) of the intervals
    :type interpolation_error_at_midpoints: np.ndarray
    :return: The integral approximation over all intervals given by Simpson's method, and the index of the interval with the largest error.
    :rtype: Tuple[float, int]
    """


    left_ends = interval_endpoints[:-1]
    right_ends = interval_endpoints[1:]

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
    linear_bisection_condition: Callable[[np.ndarray], np.ndarray],
    func: FuncType,
    interpolation_error_metric: Callable[[FuncOutputType, np.ndarray], float]
) -> Tuple[np.ndarray, int, float]:
    """Make a pchip interpolation of func over the input frequencies, bisect all intervals, evaluate func and its interpolation on all midpoints,
    and integrate the interpolation error on the midpoints over the entire frequency range to find the total interpolation error and the interval with the highest error.

    :param frequencies: A 1D array of size N containing frequencies (floats)
    :type frequencies: np.ndarray
    :param func_values: A 1D array of size N containing the values of func at the input frequencies
    :type func_values: np.ndarray
    :param linear_bisection_condition: A callable that takes in a frequency array and returns an array of the same size, where the value of
    element i is True if the interval following frequency i should be bisected linearly, and False if it should be bisected logarithmically
    :type linear_bisection_condition: Callable[[np.ndarray], np.ndarray]
    :param func: The function to be interpolated
    :type func: FuncType
    :param interpolation_error_metric: A callable that takes in the the output of func for a given frequency,
    and an array of interpolated values for the same frequency (same shape as the func output), and reduces it down to a single float metric of the error for that frequency
    :type interpolation_error_metric: Callable[[FuncOutputType, np.ndarray], float]
    :return: An array of size N-1 containing the calculated frequency interval midpoints,
    the index where the interpolation had the largest error (integrated over the interval),
    and the total integrated interpolation error over the entire frequency range
    :rtype: Tuple[np.ndarray, int, float]
    """
    
    # Make mask array where True indicates that the bisection of the interval of the same index should be done linearly
    linear_bisection_mask = linear_bisection_condition(frequencies[:-1])

    # Bisect all intervals either arithmetically or geometrically depending on frequency
    freq_midpoints = bisect_intervals(frequencies, linear_bisection_mask)
    
    # Compute func at bisections
    values_at_midpoints = np.array([func(freq) for freq in freq_midpoints])

    # Interpolate func at bisections using pchip
    # TODO: same for linear interp?
    # Note: From 2nd iteration and onwards, it would be slightly faster to only interpolate around the new point, but not much.
    # This approach then allows for much simpler code (i.e not having to pass arrays of midpoints and interpolations in and out of functions).
    # Benchmarking shows that this approach costs < 1 ms per iteration, which typically adds up to ~ 1 s for the entire algorithm
    interpolated_values_at_midpoints = complex_pchip(xi=frequencies, zi=func_values, x=freq_midpoints, axis=0)

    # Find interpolation error at midpoints using a user defined function
    interpolation_error_at_midpoints = interpolation_error_metric(values_at_midpoints, interpolated_values_at_midpoints)

    # Integrate the interpolation error over the frequency range, find midpoint frequency of the interval with largest error
    total_interpolation_error, index_max_error_interval = integrate_interpolation_error(frequencies, linear_bisection_mask, interpolation_error_at_midpoints)

    return freq_midpoints, index_max_error_interval, total_interpolation_error


def add_points_until_interpolation_converged(
        starting_frequencies: np.ndarray,
        starting_func_values: np.ndarray,
        func: FuncType,
        bisection_mode_condition: Optional[Callable[[np.ndarray], np.ndarray]],
        interpolation_error_metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
        absolute_error_tolerance: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Take in an array of starting frequencies, interpolate func using those frequencies, and iteratively add
    new frequencies in the intervals where the interpolation error is the greatest until the interpolation error for the
    entire frequency range falls below the input tolerance.
    
    Since func must be evaluated at the frequency interval midpoints to approximate the error, and the outputs of func are cached,
    after the convergence loop terminates with N_converged frequencies, there are N_converged - 1 more func evaluations saved in the cache.
    The midpoint frequencies and the func evaluations at these frequencies are therefore inserted into the output arrays before they are returned,
    so that the interpolation error afterwards might be quite a bit lower than the tolerance would suggest.

    :param starting_frequencies: A 1D array of size N containing an initial range of frequencies. The algorithm should perform better if this is already a good guess
    of the final frequencies, i.e. if interpolating func on these frequencies already approximates func well.
    :type starting_frequencies: np.ndarray
    :param starting_func_values: An array of of shape (N, X1, X2, ...) containing the func outputs for each frequency in starting_frequencies
    :type starting_func_values: np.ndarray
    :param func: The function to be interpolated
    :type func: FuncType
    :param bisection_mode_condition: A callable that takes in a frequency array and returns an array of the same size, where the value of
    element i is True if the interval following frequency i should be bisected linearly, and False if it should be bisected logarithmically
    :type bisection_mode_condition: Optional[Callable[[np.ndarray], np.ndarray]]
    :param interpolation_error_metric: A callable that takes in the the output of func for a given frequency,
    and an array of interpolated values for the same frequency (same shape as the func output), and reduces it down to a single float metric of the error for that frequency
    :type interpolation_error_metric: Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param absolute_error_tolerance: The tolerance for the convergence loop. When the total integrated interpolation error falls below this value, the loop terminates.
    :type absolute_error_tolerance: float
    :return: Arrays of frequencies and func outputs, respectively, with all new values needed for convergence inserted.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    
    logging.info(f"Starting adaptive bisection algorithm of frequency range: {starting_frequencies[0]} to {starting_frequencies[-1]}, "
                 f"starting with {len(starting_frequencies)} frequencies.")
    
    frequencies = starting_frequencies
    func_values = starting_func_values

    # Default bisection mode: Only logarithmic bisection
    if bisection_mode_condition is None:
        bisection_mode_condition = lambda x: np.zeros_like(x, dtype=bool)
    
    # Initialize with dummy value before while loop
    total_interpolation_error = np.inf

    while total_interpolation_error > absolute_error_tolerance:

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
    initial_frequencies: Sequence,
    func: FuncType, 
    interpolation_error_metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
    absolute_integral_tolerance: Union[float, Sequence[float]] = 1e-3,
    frequency_bound_scan_logstep: Union[float, Sequence[float]] = 2**(1/5),
    bisection_mode_condition: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> np.ndarray:
    """An adaptive algorithm to compute the fourier integral for a function of frequency func for the input times.
    
    The algorithm consists of three convergence loops:
    
    1) Starting from the frequencies in initial_frequencies, iteratively add more points to the interior of this frequency range until
    interpolating func over the frequencies using the pchip* algrithm gives a sufficiently accurate interpolation. The interpolation is considered accurate when
    the integral of the interpolation error over the entire frequency range (using Simpson's method) falls below a tolerance.
    
    2) Iteratively add points below the lowest frequency until the contribution to the fourier integral from the added points falls below a tolerance.
    
    3) Iteratively add points above the highest frequency until the contribution from the added points falls below a tolerance.
    
    The fourier integral itself is evaluated using a Filon type method where the interpolating polynomial is a pchip interpolation.
    
    The algorithm is developed with smooth functions spanning with long evaluation times or spanning several orders of magnitude in mind,
    as it often evaluating the function at much fewer point than e.g. the IFFT, but at a cost of more overhead from the algorithm itself.
    
    * pchip = piecewise cubic hermite interpolating polynomial

    :param times: A 1D array of floats of size M (or a single float) with time values to compute the fourier integral for
    :type times: ArrayLike
    :param initial_frequencies: A 1D array of size N (N >= 2) with an initial guess of frequencies which already serve as good interpolation points for func.
    Only accepts positive frequencies
    :type initial_frequencies: Sequence
    :param func: The function to be transformed. Takes in a single frequency (i.e. does not need to be vectorized) and outputs a numpy array of arbitrary shape (X1, X2, ...).
    :type func: FuncType
    :param interpolation_error_metric: A callable that takes in the the output of func for a given frequency,
    and an array of interpolated values for the same frequency (same shape as the func output), and reduces it down to a single float metric of the error for that frequency
    :type interpolation_error_metric: Callable[[np.ndarray, np.ndarray], np.ndarray]
    :param absolute_integral_tolerance: The absolute tolerance for the convergence loops. Either given as a single float, which is then used for all three loops,
    or given as a sequence of three floats, for the three loops. Defaults to 1e-3
    :type absolute_integral_tolerance: Union[float, Sequence[float]], optional
    :param frequency_bound_scan_logstep: The multiplicative step size used for convergence loops 2) and 3). For convergence loop 2) new lowest frequencies
    are generated by dividing the old lowest frequency by this number. For convergence loop 3), new highest frequencies are generated with multiplication with this number.
    Can be given either as a single float, which will then be used for both loops, or as a sequence of two floats, one for each loop. Defaults to 2**(1/5)
    :type frequency_bound_scan_logstep: Union[float, Sequence[float]], optional
    :param bisection_mode_condition: A callable that takes in the the output of func for a given frequency,
    and an array of interpolated values for the same frequency (same shape as the func output), and reduces it down to a single float metric of the error for that frequency.
    Can be given as None, in which case only logarithmic bisection is used. Defaults to None
    :type bisection_mode_condition: Optional[Callable[[np.ndarray], np.ndarray]], optional
    :raises ValueError: If absolute_integral_tolerance or frequency_bound_scan_logstep are not the correct type or size
    :raises AssertionError: If fewer than 2 frequencies are given, or any of the frequencies are negative
    :return: An array of size (M, X1, X2, ... ) containing the fourier integral of func at the input times
    :rtype: np.ndarray
    """
    
    # TODO: Consider changing all insert and append to work in-place on a larger array. I.e. a dynamic array approach

    # Starting frequencies
    frequencies = np.asarray(initial_frequencies)
    assert len(frequencies) >= 2, "Need more than 1 frequency to integrate over"
    assert np.all(frequencies >= 0), "All frequencies must be positive" # TODO: Implement negative frequency support
    
    times = np.asarray(times)

    func = CachedFunc(func)

    # Set up sorted array of func outputs
    func_values = np.array([func(freq) for freq in frequencies])
    
    if isinstance(absolute_integral_tolerance, (int, float)):
        abs_bisection_tolerance = abs_lowscan_tolerance = abs_highscan_tolerance = absolute_integral_tolerance
    elif len(absolute_integral_tolerance) == 1:
        abs_bisection_tolerance = abs_lowscan_tolerance = abs_highscan_tolerance = absolute_integral_tolerance[0]
    elif len(absolute_integral_tolerance) == 3:
        abs_bisection_tolerance, abs_lowscan_tolerance, abs_highscan_tolerance = absolute_integral_tolerance
    else:
        raise ValueError("Need either 1 or 3 values for absolute_integral_tolerance.")
    
    if isinstance(frequency_bound_scan_logstep, (int, float)):
        low_frequency_scan_logstep = high_frequency_scan_logstep = frequency_bound_scan_logstep
    elif len(frequency_bound_scan_logstep) == 1:
        low_frequency_scan_logstep = high_frequency_scan_logstep = frequency_bound_scan_logstep[0]
    elif len(frequency_bound_scan_logstep) == 2:
        low_frequency_scan_logstep, high_frequency_scan_logstep = frequency_bound_scan_logstep
    else:
        raise ValueError("Need either 1 or 3 values for absolute_integral_tolerance.")
    

    frequencies, func_values = add_points_until_interpolation_converged(
        starting_frequencies=frequencies,
        starting_func_values=func_values,
        func=func,
        bisection_mode_condition=bisection_mode_condition,
        interpolation_error_metric=interpolation_error_metric,
        absolute_error_tolerance=abs_bisection_tolerance/(2*np.pi) # Convert from angular units
    )

    func_derivative_values = complex_pchip(frequencies, func_values, frequencies, derivative_order=1, axis=0)
    integral_values = fourier_integral_fixed_sampling(times, frequencies, func_values, inf_correction_term=True, interpolation="pchip")

    logging.info("Iteratively adding lower frequencies until convergence.")

    # Loop to converge on low enough first frequency. Skip if first frequency is 0
    # TODO: Consider moving while loop to its own function
    integral_absolute_error = np.inf
    while integral_absolute_error > abs_lowscan_tolerance and frequencies[0] > 0:

        # Make a new frequency and calculate func and derivative
        frequencies = np.insert(frequencies, 0, frequencies[0]/low_frequency_scan_logstep)
        func_values = np.insert(func_values, 0, func(frequencies[0]), axis=0)
        func_derivative_values = np.insert(func_derivative_values, 0, complex_pchip(frequencies[:3], func_values[:3], frequencies[0], derivative_order=1, axis=0), axis=0) # 3 values needed to compute pchip derivative

        # Calculate wake contribution from the new interval
        new_integral_contributions = fourier_integral_fixed_sampling(times, frequencies[:2], func_values[:2], inf_correction_term=False, interpolation="pchip")

        # TODO: Allow for relative error
        integral_absolute_error = np.max(np.abs(new_integral_contributions))

        integral_values += new_integral_contributions

        logging.info(f"New lowest frequency: {frequencies[0]}\nFourier integral relative error between last two iterations: {integral_absolute_error}")
    
    logging.info("Iteratively adding higher frequencies until convergence.")

    # Loop to converge on low enough first frequency
    # TODO: As above, consider moving while loop to its own function
    integral_absolute_error = np.inf
    while integral_absolute_error > abs_highscan_tolerance:
        
        # Make a new frequency and calculate func and derivative
        frequencies = np.append(frequencies, high_frequency_scan_logstep*frequencies[-1])
        func_values = np.append(func_values, [func(frequencies[-1])], axis=0)
        func_derivative_values = np.append(func_derivative_values, [complex_pchip(frequencies[-3:], func_values[-3:], frequencies[-1], derivative_order=1, axis=0)], axis=0)

        new_integral_contributions = (
            fourier_integral_fixed_sampling(times, frequencies[-2:], func_values[-2:], inf_correction_term=True, interpolation="pchip")
            - _fourier_integral_inf_correction(times=times, omega_end=2*np.pi*frequencies[-2], func_value_end=func_values[-2], func_derivative_end=func_derivative_values[-2])
        )

        # TODO: Allow for relative error
        integral_absolute_error = np.max(np.abs(new_integral_contributions))

        integral_values += new_integral_contributions

        logging.info(f"New highest frequency: {frequencies[-1]}\nFourier integral relative error between last two iterations: {integral_absolute_error}")

    return integral_values, frequencies
