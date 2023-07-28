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
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from .fixed_grid_fourier_integral import fourier_integral_fixed_sampling
from .utils import complex_pchip

MAX_FREQUENCY = 1e25

def _difference_norm(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Calculate a standard difference norm of ``a`` and ``b``
    
    If the input is 0- or 1-dimensional, the absolute difference is returned.
    
    Any dimensions after the first are collapsed using the 2-norm,
    so that the output is either 0-dimensional (for a and b 0-dimensional) or
    1D (for ``a`` and ``b`` of 1D or higher).

    :param a: Array 
    :type a: ArrayLike
    :param b: The other array to take the difference of
    :type b: ArrayLike
    :return: A 0D or 1D array with the difference norm of a and b
    :rtype: np.ndarray
    """

    a = np.asarray(a)
    b = np.asarray(b)
    
    difference = a - b
    
    # NOTE: By exchanging np.sum for np.mean here one would get the root mean square error
    return np.sqrt(np.sum(np.abs(difference)**2, axis=tuple(range(1, len(difference.shape)))))

class CachedFunc:
    """Wrapper class around a function to cache function calls for faster evaluation of the function with the same arguments.
    """
    
    def __init__(self, function: Callable[[float], ArrayLike]):
        """Create a wrapper around the input function that saves function calls in a dictionary for faster reevaluation with the same argument.

        :param function: The function to be wrapped. Must be a function of one parameter, which must be hashable.
        :type function: Callable[[float], ArrayLike]
        """
        
        self.function = function
        self.cache: Dict[float, ArrayLike] = dict()
    
    def __call__(self, x: float) -> ArrayLike:
        """Evaluate func
        
        Defaults to looking up the input variable x in the dictionary of cached calls,
        then evaluates the function itself if no cached output was stored for that input.

        :param x: The input variable of the wrapped function
        :type x: float
        :return: The outputs of the wrapped function for the given input
        :rtype: ArrayLike
        """
        if x not in self.cache.keys():
            self.cache[x] = self.function(x)
        return self.cache[x]


def _bisect_intervals(interval_endpoints: np.ndarray, linear_bisection_mask: np.ndarray, step_towards_inf_factor: float) -> np.ndarray:
    """Bisect the intervals between elements of ``interval_endpoints``
    
    The bisection is done either using the arithmetic or geometric mean of two neighboring points in ``interval_endpoint``. Which type depends on ``linear_bisection_mask``.
    For intervals containing 0, the corresponding value in linear_bisection_mask must be True
    
    If one of the endpoints of an interval is +-inf, the "midpoint" is generated as ``step_towards_inf_factor`` multiplied with the finite endpoint of the interval.
    In other words, this is the geometric midpoint between the finite point and a phantom point generated as ``step_towards_inf_factor``^2 multiplied by the finite point.
    
    It is assumed that if there are infinities in the array, the neighboring point is a finite number *of the same sign*.

    :param interval_endpoints: A sorted 1D array of floats with length N, denoting the edges of the intervals to be bisected
    :type interval_endpoints: np.ndarray
    :param linear_bisection_mask: A 1D array of bools with length N-1, where ``True`` denotes that the interval of the same index
    should be bisected linearly,and ``False`` geometrically
    :type linear_bisection_mask: np.ndarray
    :param step_towards_inf_factor: For intervals between edges (-inf, a) or (a, inf), where a is some finite number, the "midpoint" will be calculated as a*``step_towards_inf_factor``
    :type step_towards_inf_factor: float
    :return: A 1D array of floats with length N-1 containing the midpoints between the points in ``interval_endpoints``
    :rtype: np.ndarray
    """
    
    # NOTE: With normal (static) numpy arrays, testing shows it is faster to recalculate all midpoints
    # than to calculate one midpoint and insert into an array of existing midpoints
    midpoints = np.zeros(len(interval_endpoints)-1)

    left_ends = interval_endpoints[:-1]
    right_ends = interval_endpoints[1:]
    
    assert np.all(np.sign(left_ends[~linear_bisection_mask]) == np.sign(right_ends[~linear_bisection_mask])), "linear_bisection_mask must be True for intervals containing 0"
    midpoints[~linear_bisection_mask] = np.sqrt(left_ends[~linear_bisection_mask] * right_ends[~linear_bisection_mask]) * np.sign(left_ends[~linear_bisection_mask]) # geometric/logarithmic mean
    midpoints[ linear_bisection_mask] =  1/2 * (left_ends[ linear_bisection_mask] + right_ends[ linear_bisection_mask]) # arithmetic mean

    # If the endpoints are +-infinity, "bisect" by multiplying the final finite point by a constant
    # TODO: Add support for linear steps towars inf as well
    if interval_endpoints[0] == -np.inf:
        midpoints[0] = interval_endpoints[1]*step_towards_inf_factor
    
    if interval_endpoints[-1] == np.inf:
        midpoints[-1] = interval_endpoints[-2]*step_towards_inf_factor

    return midpoints


def _simpson_integral(a: ArrayLike, b: ArrayLike, y_mid: ArrayLike, geometric: bool) -> ArrayLike:
    """Calculate the Simpson's rule integral approximation of y(x), *assuming y(a) == y(b) == 0 at the interval endpoints a and b*
    
    The midpoint function value ``y_mid`` can be from a midpoint generated either as the arithmetic mean of a and b,
    in which case ``geometric`` should be set to ``False``. The normal Simpson's rule is then used: integral = 2/3 * (b-a) * y_mid
    
    If ``y_mid`` comes from the geometric midpoint, ``geometric`` should be ``True``, and a modified Simpson's rule is used:
    integral = 2/3 * sqrt(a*b) * sign(a) * log(b/a) * y_mid
    Note that ``geometric`` must be ``False`` if the interval contains 0, i.e. if sign(a) != sign(b)

    :param a: The lower endpoint(s) of the interval(s)
    :type a: ArrayLike
    :param b: The higher endpoint(s) of the interval(s)
    :type b: ArrayLike
    :param y_mid: The function value at the arithmetic or geometric midpoint of the interval. Can be complex
    :type y_mid: ArrayLike
    :param geometric: Flag set to ``True`` if ``y_mid`` comes from the geometric midpoint of the interval, and False if from the arithmetic midpoint
    :type geometric: bool
    :return: The Simpson's rule numerical integral over the interval(s)
    :rtype: ArrayLike
    """
    
    if geometric:
        assert np.all(np.sign(a) == np.sign(b)), "Geometrically bisected intervals can not contain 0"
        return 2/3 * np.sqrt(a * b) * np.sign(a) * np.log(b / a) * y_mid
    else:
        return 2/3 * (b - a) * y_mid
    

def _integrate_interpolation_error(
    interval_endpoints: np.ndarray,
    linear_bisection_mask: np.ndarray,
    interpolation_error_at_midpoints: np.ndarray,
    step_towards_inf_factor: float,
) -> np.ndarray:
    """Integrate numerically the error in each interval
    
    Takes in an array of the interpolation error at the midpoints of each interval and approximates the integral of said error in over the interval using Simpson's rule.
    
    The error at the endpoints of the intervals is assumed to be zero, since they are assumed to be the interpolation points.
    
    For linearly bisected intervals, Simpson's rule gives the integral as: 2/3 * (b-a) * err((a+b)/2), where a and b are the endpoints and err is the error
    
    For geometrically bisected intervals: 2/3 * sqrt(a*b) * log(b/a) * sign(a) * err(sqrt(a*b)*sign(a*b))
    
    If the frequency is higher than ``MAX_FREQUENCY`` or lower than ``-MAX_FREQUENY``, the integral is set to 0 to terminate the loop and avoid overflow.

    :param interval_endpoints: A sorted 1D array of floats with length N, denoting the edges of the intervals over which to integrate the error
    :type interval_endpoints: np.ndarray
    :param linear_bisection_mask: A 1D array of bools with length N-1, where ``True`` denotes that the interval of the same index was bisected linearly, and ``False`` geometrically.
    Must be True for intervals containing 0
    :type linear_bisection_mask: np.ndarray
    :param interpolation_error_at_midpoints: A 1D array of floats with length N-1 containing the interpolation error
    at the midpoints of the corresponding interval (assuming the endpoints were used to make the interpolation)
    :type interpolation_error_at_midpoints: np.ndarray
    :param step_towards_inf_factor: For intervals between edges (-inf, a) or (a, inf), where a is some finite number, the "midpoint" is calculated as a*``step_towards_inf_factor``,
    and the integration is performed over the interval (a*``step_towards_inf_factor``**2, a) or (a, a*``step_towards_inf_factor``**2), respectively
    :type step_towards_inf_factor: float
    :return: A 1D array of floats with length N-1 containing the approximated integrated interpolation error over each interval
    :rtype: np.ndarray
    """

    left_ends = interval_endpoints[:-1]
    right_ends = interval_endpoints[1:]

    interval_errors = np.zeros_like(interpolation_error_at_midpoints)

    # Logarithic Simpson's rule
    # Endpoints of intervals not included in formula, as they are interpolated exactly
    interval_errors[~linear_bisection_mask] = _simpson_integral(
        left_ends[ ~linear_bisection_mask],
        right_ends[~linear_bisection_mask],
        interpolation_error_at_midpoints[~linear_bisection_mask],
        geometric=True)

    # Normal (linear) Simpson's rule
    # Endpoints of intervals not included in formula, as they are interpolated exactly
    interval_errors[linear_bisection_mask] = _simpson_integral(
        left_ends[ linear_bisection_mask],
        right_ends[linear_bisection_mask],
        interpolation_error_at_midpoints[linear_bisection_mask],
        geometric=False)
    
    if interval_endpoints[0] == -np.inf:
        if interval_endpoints[1] <= -MAX_FREQUENCY:
            logging.info(f"Minimum frequency (-{MAX_FREQUENCY}) reached, not adding any lower frequencies.")
            interval_errors[0] = 0
        else:
            b = interval_endpoints[1]
            # Create phantom point so that interval_error[0] corresponds to the geometric midpoint of a and b
            a = b*step_towards_inf_factor**2
            interval_errors[0] = _simpson_integral(a, b, interpolation_error_at_midpoints[0], geometric=True)
    
    if interval_endpoints[-1] == np.inf:
        if interval_endpoints[-2] >= MAX_FREQUENCY:
            logging.info(f"Maximum frequency ({MAX_FREQUENCY}) reached, not adding any higher frequencies.")
            interval_errors[-1] = 0
        else:
            a = interval_endpoints[-2]
            # Create phantom point so that interval_error[0] corresponds to the geometric midpoint of a and b
            b = a*step_towards_inf_factor**2
            interval_errors[-1] = _simpson_integral(a, b, interpolation_error_at_midpoints[-1], geometric=True)
    
    return interval_errors


def _find_interval_errors(
    frequencies: np.ndarray,
    func_values: np.ndarray,
    func: Callable[[float], ArrayLike],
    linear_bisection_condition: Callable[[np.ndarray], np.ndarray],
    interpolation_error_norm: Callable[[ArrayLike, np.ndarray], float],
    step_towards_inf_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds the integrated interpolation error obtained when interpolating ``func`` on ``frequencies``
    
    Constructs a PCHIP interpolation of ``func`` based on ``frequencies`` and ``func_values``.
    Then bisects all frequency intervals either linearly or geometrically, depending on ``linear_bisection_condition``,
    and evaluates func and the interpolation on these midpoints.
    The difference between the two is passed through ``interpolation_error_norm``
    and then integrated over each interval using Simpson's method.

    :param frequencies: A 1D array of floats with length N containing frequencies [Hz]
    :type frequencies: np.ndarray
    :param func_values: An array of complex numbers with shape (N, X1, X2, ...) containing the values ``func`` evaluated at ``frequencies``
    :type func_values: np.ndarray
    :param func: The function to perform a Fourier integral on
    :type func: Callable[[float], ArrayLike]
    :param linear_bisection_condition: A callable that takes in the left ends of the frequency intervals [Hz] (length N-1)
    and returns an array of bools of length N-1, where ``True`` denotes that the following interval should be bisected linearly
    :type linear_bisection_condition: Callable[[np.ndarray], np.ndarray]
    :param interpolation_error_norm: A callable that takes in the values of ``func`` at the generated midpoints and the polynomial approximations of those values,
    both with shape (N-1, X1, X2, ...) and calculates some error metric between them, e.g. the absolute difference.
    The output must be 1D and with length N-1 (only the frequency dimension)
    :type interpolation_error_norm: Callable[[ArrayLike, np.ndarray], float]
    :param step_towards_inf_factor: If +-inf is included in the frequency range, the midpoint between it and its (finite) neighbor frequency is computed by multiplying that
    neighboring frequency with ``step_towards_inf_factor``
    :type step_towards_inf_factor: float
    :return: A 1D array of floats with length N-1 containing midpoint frequencies [Hz],
    as well as an array of the same shape containing the integrated interpolation error in the corresponding interval.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    
    # Make mask array where True indicates that the bisection of the interval of the same index should be done linearly
    linear_bisection_mask = linear_bisection_condition(frequencies[:-1])
    
    # If the interval contains 0 then linear bisection must be used
    linear_bisection_mask |= (frequencies[:-1]*frequencies[1:] <= 0)

    # Bisect all intervals either arithmetically or geometrically depending on frequency
    freq_midpoints = _bisect_intervals(frequencies, linear_bisection_mask, step_towards_inf_factor=step_towards_inf_factor)
    
    # Compute func at bisections
    # TODO: Add possibility to disable points such as 0
    values_at_midpoints = np.array([func(freq) for freq in freq_midpoints])

    # Interpolate func at bisections using pchip
    # TODO: same for linear interp?
    # NOTE: From 2nd iteration and onwards, it would be slightly faster to only interpolate around the new point, but not much.
    # The approach implemented below then allows for much simpler code (i.e not having to pass arrays of midpoints and interpolations in and out of functions).
    # Benchmarking shows that this approach costs < 1 ms per iteration, which typically adds up to ~ 1 s for the entire algorithm
    # NOTE: In the case of infinities at the end extrapolation is done
    finite_mask = np.isfinite(frequencies) & np.all(np.isfinite(func_values), axis=tuple(range(1, len(func_values.shape))))
    interpolated_values_at_midpoints = complex_pchip(xi=frequencies[finite_mask], zi=func_values[finite_mask], x=freq_midpoints, axis=0)

    # Find interpolation error at midpoints using a user defined function
    interpolation_error_at_midpoints = interpolation_error_norm(values_at_midpoints, interpolated_values_at_midpoints)
    
    # TODO: Treat inf and nan values of interpolation_error_at_midpoints

    # Integrate the interpolation error over the frequency range, find midpoint frequency of the interval with largest error
    interval_errors = _integrate_interpolation_error(frequencies, linear_bisection_mask, interpolation_error_at_midpoints, step_towards_inf_factor)

    return freq_midpoints, interval_errors


def improve_frequency_range(
    initial_frequencies: Sequence[float],
    func: Callable[[float], ArrayLike], 
    absolute_integral_tolerance: float,
    step_towards_inf_factor: float = 2,
    bisection_mode_condition: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    interpolation_error_norm: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]=None,
    max_iterations: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs an adaptive algorithm to improve a frequency range for calculation of the Fourier integral of ``func``
    
    The algorithm finds new frequencies by creating a [PCHIP]_ interpolaion of ``func`` on the frequencies, comparing that interpolation
    with the true function value the midpoints of all frequency subintervals, and using Simpson's rule to evaluate the integral of
    the difference between the interpolation and the true function.
    The midpoint of the subinterval with the largest interpolation error is then added to the ``frequencies`` array.
    
    This repeats either until ``max_iterations`` iterations have been performed, or until the interpolation error integral
    has a smaller total value than ``absolute_integral_tolerance``
    
    To avoid many repeated calls to ``func``, the outputs are cached, so that reevaluation at the same frequency only amounts to a dictionary lookup.
    
    ``func`` should only give finite outputs for all frequencies between the extremes of the input frequencies.
    
    The improved frequency range will never exceed the boundaries of the input.
    
    +-inf can be included in the frequency range, but user discretion is advised, as depending on ``func`` and ``step_towards_inf_factor``, adding frequencies towards inf
    can come at the cost of not adding frequencies close to the most important features of ``func``, since the frequency intervals added towards infinity can get very wide
    (they get gradually bigger), and this can overshadow shorter intervals where the error is larger.
    
    The starting frequency range must contain at least 2 finite frequencies, and +-infinity requires the adjacent frequency to be finite, non-zero, and of the same sign. 
    
    It should be noted that while the algorithm will run with as little as 2 initial frequencies, it is for the best results advised to use a frequency range that
    already captures the most essential features of ``func``, and thus use the algorithm to improve this range.
    
    Based on the algorithm outlined in section 1.6.3 in [N.Mounet-PhD]_.

    :param initial_frequencies: A 1D array of floats containing the starting frequencies [Hz]* for the algorithm. Must contain at least 2 frequencies. Can contain +-np.inf.
    :type initial_frequencies: Sequence[float]
    :param func: The function to be Fourier integrated, a function of frequency [Hz]* with a complex output. Does not need to be vectorized.
        Results will be cached to avoid repeat calls with the same input.
    :type func: Callable[[float], ArrayLike]
    :param interpolation_error_norm: A callable that takes in an array of func outputs (shape (N, X1, X2, ...)) and interpolated approximations
        of the same values (also shape (N, X1, X2, ...))
        The primary axis (of length N) corresponds to the frequency. The callable should return a norm of the difference between the two inputs,
        collapsed into a 1D array along the frequency axis, i.e. the output should have the shape (N,).
        If None (the default) is given, the norm will be calculated using the absolute difference for 1D arrays, or the 2-norm for higher dimension arrays
    :type interpolation_error_norm: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]
    :param absolute_integral_tolerance: The tolerance for the (approximate) integrated interpolation error. When the error drops below this value, the algorithm terminates.
    :type absolute_integral_tolerance: float
    :param step_towards_inf_factor: When one of the ends of the frequency array is +-inf, the bisection of the corresponding interval is calculated by multiplying the (finite)
        neighboring frequency with this number. Defaults to 2
    :type step_towards_inf_factor: float, optional
    :param bisection_mode_condition: A function that takes in a frequency array [Hz]* and returns an array of bools of the same size,
        where ``True`` denotes that the interval following that frequency should be bisected linearly,
        and ``False`` geometrically. Can be set to ``None``, in which case only geometric bisection is used. Defaults to ``None``
    :type bisection_mode_condition: Optional[Callable[[np.ndarray], np.ndarray]], optional
    :param max_iterations: The maximum amount of iterations to perform before the algorithm is terminated. If set to ``None``, the algorithm will only terminate
        after the tolerance has been met. Defaults to ``None``
    :type max_iterations: Optional[int], optional
    :return: The refined frequency range [Hz]*, and an array containing the corresponding outputs of ``func``
    :rtype: Tuple[np.ndarray, np.ndarray]
    
    \*Though Hz is used as units here, any frequency unit will work, just be mindful that if used in a Fourier integral the result will be in units 1/<frequency unit>
    """
    
    # NOTE: Could consider changing all insert and append to work in-place on a larger array. I.e. a dynamic array approach

    # Starting frequencies
    frequencies = np.asarray(initial_frequencies)
    assert sum(np.isfinite(frequencies)) >= 2, "Need 2 or more finite frequencies in initial frequency range"
    assert frequencies[0] != -np.inf or frequencies[1] < 0,  "-inf needs a finite, negative, non-zero adjacent frequency in the initial frequency range"
    assert frequencies[-1] != np.inf or frequencies[-2] > 0, "+inf needs a finite, positive, non-zero adjacent frequency in the initial frequency range"

    assert step_towards_inf_factor > 1 or sum(np.isinf(frequencies)) == 0, "step_towards_inf_factor must be greater than 1"

    func = CachedFunc(func)

    # Set up sorted array of func outputs
    # TODO: Special handling of zero and +-inf, and maybe others
    func_values = np.array([func(freq) for freq in frequencies])
    
    # Default bisection mode: Only logarithmic bisection
    if bisection_mode_condition is None:
        bisection_mode_condition = lambda x: np.zeros_like(x, dtype=bool)
    
    if interpolation_error_norm is None:
        interpolation_error_norm = _difference_norm
        
    logging.info(f"Starting adaptive bisection algorithm of frequency range: {frequencies[0]} to {frequencies[-1]}, "
                 f"starting with {len(frequencies)} frequencies.")
    
    # Initialize with dummy value before while loop
    total_interpolation_error = np.inf
    
    k = 0
    while total_interpolation_error > absolute_integral_tolerance:

        k += 1
        if max_iterations is not None and k >= max_iterations:
            break
           
        midpoint_freqs, interval_interpolation_errors = _find_interval_errors(
            frequencies=frequencies,
            func_values=func_values,
            func=func,
            linear_bisection_condition=bisection_mode_condition,
            interpolation_error_norm=interpolation_error_norm,
            step_towards_inf_factor=step_towards_inf_factor,  
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
    frequencies = np.union1d(frequencies, midpoint_freqs)
    func_values = np.array([func(freq) for freq in frequencies])

    return frequencies, func_values


def fourier_integral_adaptive(
    times: ArrayLike,
    initial_frequencies: Sequence,
    func: Callable[[float], ArrayLike], 
    absolute_integral_tolerance: float,
    interpolation: str = "pchip",
    step_towards_inf_factor: float = 2,
    bisection_mode_condition: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    interpolation_error_norm: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]=None,
    max_iterations: Optional[int] = None,
) -> np.ndarray:
    """Performs an adaptive algorithm to improve a frequency range and uses the improved frequency range to compute the Fourier integral of ``func``
    
    The algorithm finds new frequencies by creating a PCHIP interpolaion of ``func`` on the frequencies, comparing that interpolation
    with the true function value the midpoints of all frequency subintervals, and using Simpson's rule to evaluate the integral of
    the difference between the interpolation and the true function.
    The midpoint of the subinterval with the largest interpolation error is then added to the ``frequencies`` array.
    
    This repeats either until ``max_iterations`` iterations have been performed, or until the interpolation error integral has a
    smaller total value than ``absolute_integral_tolerance``
    
    To avoid many repeated calls to ``func``, the outputs are cached, so that reevaluation at the same frequency only amounts to a dictionary lookup.
    
    ``func`` should only give finite outputs for all frequencies between the extremes of the input frequencies.
    
    The improved frequency range will never exceed the boundaries of the input.
    
    +-inf can be included in the frequency range, but user discretion is advised, as depending on ``func`` and ``step_towards_inf_factor``, adding frequencies towards inf
    can come at the cost of adding frequencies close to the most important features of ``func``.
    
    It should be noted that while the algorithm will run with as little as 2 initial frequencies, it is for the best results advised to use a frequency range that
    already captures the most essential features of ``func``, and thus use the algorithm to improve this range.
    
    The Fourier integral itself is the integral from fmin to fmax of : exp(2*pi*j*f*t)*func(f)*df,
    
    where t is the time, fmin is the first frequency in the input ``frequencies``, fmax is the highest frequency and func(f) is the input function of frequency.
    
    It is computed using a Filon type algorithm using a piecewise cubic Hermite interpolating polynomial (pchip). 
    For details on implementation, see [N.Mounet-PhD]_.
    
    This function combines the functionality of ``neffint.improve_frequency_range`` and ``neffint.fourier_integral_fixed_sampling``, and only for convenience.
    If the improved frequencies or the corresponding function values are needed, use the two functions separately.

    Based on the algorithm outlined in section 1.6.3 in [N.Mounet-PhD]_.

    :param times: Float or 1D array of floats, the time(s) [s]* to compute the fourier integral for
    :type times: ArrayLike
    :param initial_frequencies: A 1D array of floats containing the starting frequencies [Hz]* for the algorithm. Must contain at least 2 frequencies. Can contain +-np.inf.
    :type initial_frequencies: Sequence[float]
    :param func: The function to be Fourier integrated, a function of frequency [Hz] with a complex output. Does not need to be vectorized.
        Results will be cached to avoid repeat calls with the same input.
    :type func: Callable[[float], ArrayLike]
    :param interpolation_error_norm: A callable that takes in an array of func outputs (shape (N, X1, X2, ...)) and interpolated approximations
        of the same values (also shape (N, X1, X2, ...))
        The primary axis (of length N) corresponds to the frequency. The callable should return a norm of the difference between the two inputs,
        collapsed into a 1D array along the frequency axis, i.e. the output should have the shape (N,).
        If None (the default) is given, the norm will be calculated using the absolute difference for 1D arrays, or the 2-norm for higher dimension arrays
    :type interpolation_error_norm: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]
    :param absolute_integral_tolerance: The tolerance for the (approximate) integrated interpolation error. When the error drops below this value, the algorithm terminates.
    :type absolute_integral_tolerance: float
    :param step_towards_inf_factor: When one of the ends of the frequency array is +-inf, the bisection of the corresponding interval is calculated by multiplying the (finite)
        neighboring frequency with this number. Defaults to 2
    :type step_towards_inf_factor: float, optional
    :param bisection_mode_condition: A function that takes in a frequency array [Hz]* and returns an array of bools of the same size,
        where ``True`` denotes that the interval following that frequency should be bisected linearly, and `False` geometrically.
        Can be set to ``None``, in which case only geometric bisection is used. Defaults to ``None``
    :type bisection_mode_condition: Optional[Callable[[np.ndarray], np.ndarray]], optional
    :param max_iterations: The maximum amount of iterations to perform before the algorithm is terminated. If set to ``None``, the algorithm will only terminate
        after the tolerance has been met. Defaults to ``None``
    :type max_iterations: Optional[int], optional
    :return: The fourier integral of `func` for the given input ``times``
    :rtype: np.ndarray
    
    \*Though the units s and Hz are used here, any coherent set of time and frequency units will work
    """
    
    
    frequencies, func_values = improve_frequency_range(
        initial_frequencies=initial_frequencies,
        func=func, 
        interpolation_error_norm=interpolation_error_norm,
        absolute_integral_tolerance=absolute_integral_tolerance,
        step_towards_inf_factor=step_towards_inf_factor,
        bisection_mode_condition=bisection_mode_condition,
        max_iterations=max_iterations
    )
    
    return fourier_integral_fixed_sampling(
        times=times,
        frequencies=frequencies,
        func_values=func_values,
        pos_inf_correction_term=(frequencies[-1] == np.inf),
        neg_inf_correction_term=(frequencies[0] == -np.inf),
        interpolation=interpolation
    )
