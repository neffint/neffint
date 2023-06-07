import time
from typing import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.integrate import simpson
from scipy.interpolate import pchip_interpolate

from neffint.adaptive_fourier_integral import (
    CachedFunc, adaptive_fourier_integral,
    add_points_until_interpolation_converged, bisect_intervals,
    find_interval_with_largest_error, integrate_interpolation_error)


def test_bisect_intervals():
    input_interval_endpoints = np.array([1., 2., 3., 4., 5.])
    input_linear_bisection_mask = np.array([False, True, True, False])

    expected_midpoints = np.array([np.sqrt(1*2), 2.5, 3.5, np.sqrt(4*5)])
    output_midpoints = bisect_intervals(input_interval_endpoints, input_linear_bisection_mask)

    assert output_midpoints == pytest.approx(expected_midpoints)


def test_integrate_interpolation_error_linear():
    input_frequencies = np.array([1., 2., 3., 5.]) # [1,5], uneven spacing
    input_linear_bisection_mask = np.array([True, True, True]) # all linear bisection
    input_midpoint_frequencies = bisect_intervals(input_frequencies, input_linear_bisection_mask)

    # f(x) = (x-1)(x-5) = x^2 - 6x + 5
    input_midpoint_func_values = (input_midpoint_frequencies-1)*(input_midpoint_frequencies-5)

    # On each interval, the interpolant g has the form a*(x - b), and interpolates the interval end points exactly
    # g1(x) = -3 * (x - 1)  ,  x in [1,2)
    # g2(x) = -1 * (x + 1)  ,  x in [2,3)
    # g3(x) =  2 * (x - 5)  ,  x in [3,5) 
    input_midpoint_interpolation_values = np.array([-1.5, -3.5, -2])
    input_interpolation_error = np.abs(input_midpoint_func_values - input_midpoint_interpolation_values)

    expected_func_integral = -32/3 # On intervals: (-5/3), (-11/3), (-16/3)
    expected_interpolation_integral = -9 # On intervals: (-3/2), (-7/2), (-4)

    # The error is a polynomial of degree 2 on each subinterval, so Simpson's method should be able to calculate the error exactly
    # Also, since in this case func is always larger than or equal to the interpolating function, we can complete the integrals first and then take their difference
    expected_total_interpolation_error = np.abs(expected_func_integral - expected_interpolation_integral) # On intervals: 1/6, 1/6, 2/6

    expected_index_max_error_interval = 2 # Largest error in final interval

    output_total_interpolation_error, output_index_max_error_interval = integrate_interpolation_error(input_frequencies, input_linear_bisection_mask, input_interpolation_error)

    assert output_total_interpolation_error == pytest.approx(expected_total_interpolation_error)
    assert output_index_max_error_interval == expected_index_max_error_interval


def test_integrate_interpolation_error_logarithmic():
    input_frequencies = np.exp(np.array([1., 2., 3., 5.]))
    input_linear_bisection_mask = np.array([False]*3) # all logarithmic
    input_midpoint_frequencies = bisect_intervals(input_frequencies, input_linear_bisection_mask)

    # substitute u = log(x)
    # f(u) = (u-1)(u-5)e^-u = (u^2 - 6u + 5)e^-u
    input_midpoint_func_values = (np.log(input_midpoint_frequencies) - 1) * (np.log(input_midpoint_frequencies) - 5) / input_midpoint_frequencies

    # On each interval, the interpolant g has the form a*(log(x) + b)/x, and interpolates the interval end points exactly
    # g1(x) = -3 * (log(x) - 1) / x  ,  log(x) in [1,2)
    # g2(x) = -1 * (log(x) + 1) / x  ,  log(x) in [2,3)
    # g3(x) =  2 * (log(x) - 5) / x  ,  log(x) in [3,5)
    input_midpoint_interpolation_values = np.array([
        -3 * (np.log(np.exp(1.5)) - 1) / np.exp(1.5),
        -1 * (np.log(np.exp(2.5)) + 1) / np.exp(2.5),
         2 * (np.log( np.exp(4))  - 5) / np.exp(4)
    ])

    input_interpolation_error = np.abs(input_midpoint_func_values - input_midpoint_interpolation_values)

    expected_func_integral = -32/3 # On intervals: (-5/3), (-11/3), (-16/3)
    expected_interpolation_integral = -9 # On intervals: (-3/2), (-7/2), (-4)

    # The integrand of the error integral is a polynomial of degree 2 on each subinterval when a log substitution is done,
    # so Simpson's method should be able to calculate the error exactly.
    # As above, the interpolant is smaller or equal to func at all x, so it is in this case fine to take
    # the difference of the integrals as opposed to the integral of the differences
    expected_total_interpolation_error = np.abs(expected_func_integral - expected_interpolation_integral)

    expected_index_max_error_interval = 2 # Largest error in final interval

    output_total_interpolation_error, output_index_max_error_interval = integrate_interpolation_error(input_frequencies, input_linear_bisection_mask, input_interpolation_error)

    assert output_total_interpolation_error == pytest.approx(expected_total_interpolation_error)
    assert output_index_max_error_interval == expected_index_max_error_interval


def test_find_interval_with_largest_error():
    input_frequencies = np.arange(1,11)
    input_bisection_mode_condition = lambda freq: freq > 0 # use only linear for simplicity

    def func(freq: float):
        result = np.zeros_like(freq)

        if 2 < freq < 3:
            # Add positive bulge between 2 and 3
            result +=  - (freq-2) * (freq-3)
        elif 6 < freq < 7:
            # Add large negative bulge between 6 and 7
            result += 3*(freq-6) * (freq-7)
        
        return result
    
    func = np.vectorize(func)

    # Absolute error
    input_error_metric = lambda func_vals, interp_vals: np.abs(func_vals - interp_vals)


    output_midpoint_freqs, output_max_error_interval_idx, output_total_interpolation_error = find_interval_with_largest_error(
        frequencies=input_frequencies,
        func_values=func(input_frequencies),
        linear_bisection_condition=input_bisection_mode_condition,
        func=func,
        interpolation_error_metric=input_error_metric
    )
    output_max_error_freq = output_midpoint_freqs[output_max_error_interval_idx]

    assert output_max_error_freq == pytest.approx(6.5)
    assert output_max_error_interval_idx == 5

    # Since the function is piecewise polynomial with degree <=2, we expect the error to be calculated exactly
    assert output_total_interpolation_error == pytest.approx(
        1/6 # small bulge contribution
        + 3/6 # large bulge contribution
    )


def test_add_points_until_interpolation_converged():
    # NOTE: This test does not test what the function *does* (minimize integrated midpoint interpolation error),
    # but rather what it *is intended to do* (find a frequency grid that one can use as nodes to create a good interpolation)
    input_frequencies = np.linspace(1.,10.,11)
    input_finer_frequencies = np.linspace(1.,10.,1000)
    input_bisection_mode_condition = lambda freq: freq > 0 # use only linear for simplicity, TODO: consider both
    input_tolerance = 1e-5
    
    # Some random smooth function
    func = lambda freq: np.sqrt(freq)*np.cos(2*np.pi*freq / 10) + 10*np.exp(-(freq-4)**2 / 0.1)

    # Use absolute error as metric
    error_metric = lambda func_vals, interp_vals: np.abs(func_vals - interp_vals)

    output_frequencies, output_func_values = add_points_until_interpolation_converged(
        starting_frequencies=input_frequencies,
        starting_func_values=func(input_frequencies),
        func=func,
        bisection_mode_condition=input_bisection_mode_condition,
        interpolation_error_metric=error_metric,
        absolute_error_tolerance=input_tolerance
    )

    output_interpolation_on_fine_grid = pchip_interpolate(output_frequencies, output_func_values, input_finer_frequencies)
    expected_interpolation_on_fine_grid = func(input_finer_frequencies)

    # NOTE: Absolute tolerance here because that is used in the add_points... function itself
    # TODO: Fix algorithm so modifying the tolerance is not necessary
    assert simpson(error_metric(output_interpolation_on_fine_grid, expected_interpolation_on_fine_grid)) < input_tolerance*1e3


@pytest.mark.parametrize(("input_func", "expected_transform"), [
    ( lambda f: 1 / np.sqrt( 2 * np.pi * f),                        lambda t: np.sqrt(np.pi / (2 * t)) ),
    # ( lambda f: np.array([[1,2],[3,4]]) / np.sqrt( 2 * np.pi * f),  lambda t: np.array([[1,2],[3,4]]) * np.sqrt(np.pi / (2 * t)) ), # Test dimension handling
    ( lambda f: 1 / (1 + (2 * np.pi * f)**2),                       lambda t: np.pi / 2 * np.exp(-t) ),
])
def test_adaptive_fourier_integral(input_func: Callable[[ArrayLike], ArrayLike], expected_transform: Callable[[ArrayLike], ArrayLike]):
    input_times = np.logspace(-10, 5, 100)
    input_starting_frequencies = np.logspace(6,12,3)
    input_interpolation_error_metric = lambda x, y: np.max(np.abs(x-y))

    output_transform_arr, output_frequencies = adaptive_fourier_integral(
    times=input_times,
    initial_frequencies=input_starting_frequencies,
    func=input_func, 
    interpolation_error_metric=input_interpolation_error_metric,
    )

    expected_transform_arr = np.array([expected_transform(time) for time in input_times])

    def relative_error(x, target):
        """Relative error"""
        if target == 0:
            return np.nan
        return abs(x-target)/abs(target)
    relative_error = np.vectorize(relative_error)


    import matplotlib.pyplot as plt
    absolute_error = lambda x, y: np.abs(x-y)
    plt.plot(input_times, relative_error(np.real(output_transform_arr), expected_transform_arr), label="Relative error")
    plt.plot(input_times, absolute_error(np.real(output_transform_arr), expected_transform_arr), label="Absolute error")
    # plt.plot(input_times, expected_transform_arr, label="Analytic")
    # plt.plot(input_times, np.real(output_transform_arr), label="Numeric")
    plt.loglog()
    plt.legend()
    plt.show()

    assert np.real(output_transform_arr) == pytest.approx(expected_transform_arr, 1e-4)


def test_cached_func():
    # NOTE: using time library might be a bad idea in unit tests...
    input_call_delay = 0.1 # [s]
    input_x = 3
    def input_func(x: float):
        time.sleep(input_call_delay)
        return 2*x
    
    cached_func = CachedFunc(input_func)

    t0 = time.time()
    output_result_first = cached_func(input_x)
    t1 = time.time()
    duration_first = t1 - t0

    t0 = time.time()
    output_result_second = cached_func(input_x)
    t1 = time.time()
    duration_second = t1 - t0

    assert output_result_first == output_result_second
    assert duration_first >= input_call_delay
    assert duration_second < input_call_delay
