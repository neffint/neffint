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

from .fixed_grid_fourier_integral import fourier_integral_fixed_sampling
from .utils import complex_pchip

# TODO: go over types
# TODO: Write something about units in documentation

FuncOutputType = Any # TODO: reconsider func output type
FuncType = Callable[[float], FuncOutputType]

MAX_FREQUENCY = 1e60

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


def bisect_intervals(interval_endpoints: np.ndarray, linear_bisection_mask: np.ndarray, logstep_towards_inf: float) -> np.ndarray:
    
    # NOTE: With normal (static) numpy arrays, testing shows it is faster to recalculate all midpoints
    # than to calculate one midpoint and insert into an array of existing midpoints
    midpoints = np.zeros(len(interval_endpoints)-1)

    left_ends = interval_endpoints[:-1]
    right_ends = interval_endpoints[1:]
    
    midpoints[~linear_bisection_mask] = np.sqrt(left_ends[~linear_bisection_mask] * right_ends[~linear_bisection_mask]) * np.sign(left_ends[~linear_bisection_mask]) # geometric/logarithmic mean
    midpoints[ linear_bisection_mask] =  1/2 * (left_ends[ linear_bisection_mask] + right_ends[ linear_bisection_mask]) # arithmetic mean

    # If the endpoints are +-infinity, "bisect" by multiplying the final finite point by a constant
    if interval_endpoints[0] == -np.inf:
        midpoints[0] = interval_endpoints[1]*logstep_towards_inf
    
    if interval_endpoints[-1] == np.inf:
        midpoints[-1] = interval_endpoints[-2]*logstep_towards_inf

    return midpoints


def integrate_interpolation_error(
    interval_endpoints: np.ndarray,
    linear_bisection_mask: np.ndarray,
    interpolation_error_at_midpoints: np.ndarray,
    logstep_towards_inf: float,
) -> np.ndarray:

    left_ends = interval_endpoints[:-1]
    right_ends = interval_endpoints[1:]

    interval_errors = np.zeros_like(interpolation_error_at_midpoints)

    # Logarithic Simpson's rule
    # Endpoints of intervals not included in formula, as they are interpolated exactly
    a =  left_ends[~linear_bisection_mask]
    b = right_ends[~linear_bisection_mask]
    interval_errors[~linear_bisection_mask] = 2/3 * np.sqrt(a*b) * np.sign(a) * np.log(b/a) * interpolation_error_at_midpoints[~linear_bisection_mask]

    # Normal (linear) Simpson's rule
    # Endpoints of intervals not included in formula, as they are interpolated exactly
    a =  left_ends[ linear_bisection_mask]
    b = right_ends[ linear_bisection_mask]
    interval_errors[ linear_bisection_mask] = 2/3 * (b - a) * interpolation_error_at_midpoints[linear_bisection_mask]
    
    if interval_endpoints[0] == -np.inf:
        if interval_endpoints[1] <= -MAX_FREQUENCY:
            logging.info(f"Minimum frequency (-{MAX_FREQUENCY}) reached, not adding any lower frequencies.")
            interval_errors[0] = 0
        else:
            b = interval_endpoints[1]
            # Create phantom point so that interval_error[0] corresponds to the geometric midpoint of a and b
            a = b*logstep_towards_inf**2
            interval_errors[0] = 2/3 * np.sqrt(a*b) * np.sign(a) * np.log(b/a) * interpolation_error_at_midpoints[0]
    
    if interval_endpoints[-1] == np.inf:
        if interval_endpoints[-2] >= MAX_FREQUENCY:
            logging.info(f"Maximum frequency ({MAX_FREQUENCY}) reached, not adding any higher frequencies.")
            interval_errors[-1] = 0
        else:
            a = interval_endpoints[-2]
            # Create phantom point so that interval_error[0] corresponds to the geometric midpoint of a and b
            b = a*logstep_towards_inf**2
            interval_errors[-1] = 2/3 * np.sqrt(a*b) * np.sign(a) * np.log(b/a) * interpolation_error_at_midpoints[-1]
    
    return interval_errors


def find_interval_errors(
    frequencies: np.ndarray,
    func_values: np.ndarray,
    func: FuncType,
    linear_bisection_condition: Callable[[np.ndarray], np.ndarray],
    interpolation_error_metric: Callable[[FuncOutputType, np.ndarray], float],
    logstep_towards_inf: float,
) -> Tuple[np.ndarray, np.ndarray]:
    
    # Make mask array where True indicates that the bisection of the interval of the same index should be done linearly
    linear_bisection_mask = linear_bisection_condition(frequencies[:-1])
    
    # If the interval contains 0 then linear bisection must be used
    linear_bisection_mask |= (frequencies[:-1]*frequencies[1:] <= 0)

    # Bisect all intervals either arithmetically or geometrically depending on frequency
    freq_midpoints = bisect_intervals(frequencies, linear_bisection_mask, logstep_towards_inf=logstep_towards_inf)
    
    # Compute func at bisections
    # TODO: Add possibility to disable points such as 0
    values_at_midpoints = np.array([func(freq) for freq in freq_midpoints])

    # Interpolate func at bisections using pchip
    # TODO: same for linear interp?
    # Note: From 2nd iteration and onwards, it would be slightly faster to only interpolate around the new point, but not much.
    # This approach then allows for much simpler code (i.e not having to pass arrays of midpoints and interpolations in and out of functions).
    # Benchmarking shows that this approach costs < 1 ms per iteration, which typically adds up to ~ 1 s for the entire algorithm
    # NOTE: In the case of infinities at the end extrapolation is done
    finite_mask = np.isfinite(frequencies) & np.isfinite(func_values)
    interpolated_values_at_midpoints = complex_pchip(xi=frequencies[finite_mask], zi=func_values[finite_mask], x=freq_midpoints, axis=0)

    # Find interpolation error at midpoints using a user defined function
    interpolation_error_at_midpoints = interpolation_error_metric(values_at_midpoints, interpolated_values_at_midpoints)
    
    # TODO: Treat inf and nan values of interpolation_error_at_midpoints

    # Integrate the interpolation error over the frequency range, find midpoint frequency of the interval with largest error
    interval_errors = integrate_interpolation_error(frequencies, linear_bisection_mask, interpolation_error_at_midpoints, logstep_towards_inf)

    return freq_midpoints, interval_errors


def improve_frequency_range(
    initial_frequencies: Sequence,
    func: FuncType, 
    interpolation_error_metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
    absolute_integral_tolerance: float,
    logstep_towards_inf: float = 2,
    bisection_mode_condition: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_iterations: Optional[int] = None
) -> np.ndarray:
    
    # NOTE: Could consider changing all insert and append to work in-place on a larger array. I.e. a dynamic array approach

    # Starting frequencies
    frequencies = np.asarray(initial_frequencies)
    assert len(frequencies) >= 2, "Need 2 or more frequencies in initial frequency range"
    
    # To avoid algorithmic problems, insert `-1` if the first frequency is -inf and the second is non-negative
    if frequencies[0] == -np.inf and frequencies[1] >= 0:
        frequencies = np.insert(frequencies, 1, -1)
    
    # Similarly, if the last frequency is +inf and the second last is non-positive, insert the frequency `1`
    if frequencies[-1] == np.inf and frequencies[-2] <= 0:
        frequencies = np.insert(frequencies, -1, 1)

    # Interpolation requires 2 finite frequencies to run
    if sum(np.isfinite(frequencies)) < 2:
        frequencies = np.union1d(frequencies, 2*frequencies)

    func = CachedFunc(func)

    # Set up sorted array of func outputs
    # TODO: Special handling of zero and +-inf, and maybe others
    func_values = np.array([func(freq) for freq in frequencies])
    
    # Default bisection mode: Only logarithmic bisection
    if bisection_mode_condition is None:
        bisection_mode_condition = lambda x: np.zeros_like(x, dtype=bool)
        
    logging.info(f"Starting adaptive bisection algorithm of frequency range: {frequencies[0]} to {frequencies[-1]}, "
                 f"starting with {len(frequencies)} frequencies.")
    
    # Initialize with dummy value before while loop
    total_interpolation_error = np.inf
    
    k = 0
    while total_interpolation_error > absolute_integral_tolerance:

        k += 1
        if max_iterations is not None and k >= max_iterations:
            break
           
        midpoint_freqs, interval_interpolation_errors = find_interval_errors(
            frequencies=frequencies,
            func_values=func_values,
            func=func,
            linear_bisection_condition=bisection_mode_condition,
            interpolation_error_metric=interpolation_error_metric,
            logstep_towards_inf=logstep_towards_inf,  
        )
         
        # Find total and max error
        total_interpolation_error = np.sum(interval_interpolation_errors)
        index_interval_max_error = np.argmax(interval_interpolation_errors)
        
        # Find midpoint frequency of highest interval error
        midpoint_freq_max_error = midpoint_freqs[index_interval_max_error]
        
        logging.info(f"Largest error in interval {index_interval_max_error} between frequencies"
                     f"{frequencies[index_interval_max_error]} and {frequencies[index_interval_max_error+1]}.\n"
                     f"Integrated interpolation error: {total_interpolation_error}\n"
                     f"Number of frequencies used to interpolate: {len(frequencies)}")

        # Insert midpoint from interval with largest error into frequencies and impedances
        frequencies = np.insert(frequencies, index_interval_max_error+1, midpoint_freq_max_error)
        func_values = np.insert(func_values, index_interval_max_error+1, func(midpoint_freq_max_error), axis=0)

    logging.info(f"Bisection algorithm converged. Final number of frequencies: {len(frequencies)}.\n"
                 f"Adding {len(midpoint_freqs)-1} already evaluated midpoint frequencies to frequency array.")

    # After bisection algorithm, return all frequencies with cached func results
    frequencies = sortednp.merge(frequencies, midpoint_freqs, duplicates=sortednp.DROP)
    func_values = np.array([func(freq) for freq in frequencies])

    return frequencies, func_values


def fourier_integral_adaptive(
    times: ArrayLike,
    initial_frequencies: Sequence,
    func: FuncType, 
    interpolation_error_metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
    absolute_integral_tolerance: float,
    interpolation: str = "pchip",
    logstep_towards_inf: float = 2,
    bisection_mode_condition: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_iterations: Optional[int] = None,
) -> np.ndarray:
    
    frequencies, func_values = improve_frequency_range(
        initial_frequencies=initial_frequencies,
        func=func, 
        interpolation_error_metric=interpolation_error_metric,
        absolute_integral_tolerance=absolute_integral_tolerance,
        logstep_towards_inf=logstep_towards_inf,
        bisection_mode_condition=bisection_mode_condition,
        max_iterations=max_iterations
    )
    
    return fourier_integral_fixed_sampling(
        times=times,
        frequencies=frequencies,
        func_values=func_values,
        pos_inf_correction_term=True,
        neg_inf_correction_term=False,
        interpolation=interpolation
    )
