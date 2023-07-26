import time
from typing import Callable, Sequence

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.integrate import simpson

from neffint.adaptive_fourier_integral import (CachedFunc, _bisect_intervals, _difference_norm,
                                               _find_interval_errors,
                                               fourier_integral_adaptive,
                                               improve_frequency_range,
                                               _integrate_interpolation_error)
from neffint.utils import complex_pchip


def test_bisect_intervals():
    input_interval_endpoints = np.array([-np.inf,      -5,     -2,      -1,     0,     1,      4,    np.inf])
    input_linear_bisection_mask = np.array([     False,   True,   False,   True,  True,  False,  True])
    input_step_towards_inf_factor = 2

    expected_midpoints = np.array([-10, -3.5, -np.sqrt(2), -0.5, 0.5, np.sqrt(4), 8])
    
    output_midpoints = _bisect_intervals(
        interval_endpoints=input_interval_endpoints,
        linear_bisection_mask=input_linear_bisection_mask,
        step_towards_inf_factor=input_step_towards_inf_factor
    )

    assert output_midpoints == pytest.approx(expected_midpoints)


def test_integrate_interpolation_error_linear():
    input_x = np.array([1., 2., 3., 5.]) # [1,5], uneven spacing
    input_linear_bisection_mask = np.array([True]*3) # all linear bisection
    input_midpoint_frequencies = _bisect_intervals(
        interval_endpoints=input_x,
        linear_bisection_mask=input_linear_bisection_mask,
        step_towards_inf_factor=2 # Not relevant for this test
    )

    # f(x) = (x-1)(x-5) = x^2 - 6x + 5
    input_midpoint_func_values = (input_midpoint_frequencies-1)*(input_midpoint_frequencies-5)

    # On each interval, the interpolant g has the form a*(x - b), and interpolates the interval end points exactly
    # g1(x) = -3 * (x - 1)  ,  x in [1,2)  =>  g1(1.5) = -1.5
    # g2(x) = -1 * (x + 1)  ,  x in [2,3)  =>  g2(2.5) = -3.5
    # g3(x) =  2 * (x - 5)  ,  x in [3,5)  =>  g3(4)   = -2
    input_midpoint_interpolation_values = np.array([-1.5, -3.5, -2])
    input_interpolation_error = np.abs(input_midpoint_func_values - input_midpoint_interpolation_values)
    
    expected_func_integral_by_interval = np.array([-5/3, -11/3, -16/3])
    expected_interpolation_integral_by_interval = np.array([-3/2, -7/2, -4])

    # The error is a polynomial of degree 2 on each subinterval, so Simpson's method should be able to calculate the error exactly
    # Also, since in this case func is always larger than or equal to the interpolating function, we can complete the integrals first and then take their difference
    expected_interpolation_error_by_inteval = np.abs(expected_func_integral_by_interval - expected_interpolation_integral_by_interval)

    output_interpolation_error_by_interval = _integrate_interpolation_error(
        interval_endpoints=input_x,
        linear_bisection_mask=input_linear_bisection_mask,
        interpolation_error_at_midpoints=input_interpolation_error,
        step_towards_inf_factor=2 # Not relevant for this test
    )

    assert output_interpolation_error_by_interval == pytest.approx(expected_interpolation_error_by_inteval)


def test_integrate_interpolation_error_logarithmic():
    input_x = np.exp(np.array([1., 2., 3., 5.]))
    input_linear_bisection_mask = np.array([False]*3) # all logarithmic
    input_midpoint_frequencies = _bisect_intervals(
        interval_endpoints=input_x,
        linear_bisection_mask=input_linear_bisection_mask,
        step_towards_inf_factor=2 # Not relevant for this test
    )

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

    expected_func_integral_by_interval = np.array([-5/3, -11/3, -16/3])
    expected_interpolation_integral_by_interval = np.array([-3/2, -7/2, -4])

    # The integrand of the error integral is a polynomial of degree 2 on each subinterval when a log substitution is done,
    # so Simpson's method should be able to calculate the error exactly.
    # As above, the interpolant is smaller or equal to func at all x, so it is in this case fine to take
    # the difference of the integrals as opposed to the integral of the differences
    expected_interpolation_error_by_interval = np.abs(expected_func_integral_by_interval - expected_interpolation_integral_by_interval)

    output_interpolation_error_by_interval = _integrate_interpolation_error(
        interval_endpoints=input_x,
        linear_bisection_mask=input_linear_bisection_mask,
        interpolation_error_at_midpoints=input_interpolation_error,
        step_towards_inf_factor=2 # Not relevant for this test
    )

    assert output_interpolation_error_by_interval == pytest.approx(expected_interpolation_error_by_interval)


@pytest.mark.parametrize("interval_containing_zero", ([0,1], [-1,0], [-1,1]))
def test_geometric_on_zero_raises_error(interval_containing_zero: Sequence[float]):
    interval_containing_zero = np.asarray(interval_containing_zero)
    bisection_mode = np.asarray([False]) # Use geometric
    
    with pytest.raises(AssertionError):
        _bisect_intervals(interval_endpoints=interval_containing_zero, linear_bisection_mask=bisection_mode, step_towards_inf_factor=2)
        
    with pytest.raises(AssertionError):
        _integrate_interpolation_error(
            interval_endpoints=interval_containing_zero,
            linear_bisection_mask=bisection_mode,
            interpolation_error_at_midpoints=np.asarray([1]),
            step_towards_inf_factor=2)


def test_find_interval_errors():
    input_frequencies = np.arange(1,11)
    input_bisection_mode_condition = lambda x: x > 0 # use only linear for simplicity

    def func(x: float):
        result = np.zeros_like(x)

        if 2 < x < 3:
            # Add positive bulge between 2 and 3
            result +=  - (x-2) * (x-3)
        elif 6 < x < 7:
            # Add large negative bulge between 6 and 7
            result += 3*(x-6) * (x-7)
        
        return result
    
    func = np.vectorize(func)

    # Absolute error
    input_error_metric = lambda func_vals, interp_vals: np.abs(func_vals - interp_vals)

    output_midpoint_frequencies, output_interpolation_error_by_interval = _find_interval_errors(
        frequencies=input_frequencies,
        func_values=func(input_frequencies),
        func=func,
        linear_bisection_condition=input_bisection_mode_condition,
        interpolation_error_norm=input_error_metric,
        step_towards_inf_factor=2 # Not relevant for this test
    )
    
    output_max_error_freq = output_midpoint_frequencies[np.argmax(output_interpolation_error_by_interval)]

    assert output_max_error_freq == pytest.approx(6.5)

    # Since the function is piecewise polynomial with degree <=2, we expect the error to be calculated exactly
    assert np.sum(output_interpolation_error_by_interval) == pytest.approx(
        1/6 # small bulge contribution
        + 3/6 # large bulge contribution
    )


def test_improve_frequency_range():
    # NOTE: This test does not test what the function *does* (minimize integrated midpoint interpolation error),
    # but rather what it *is intended to do* (find a frequency grid that one can use as nodes to create a good interpolation)
    input_frequencies = np.linspace(1.,10.,11)
    input_finer_frequencies = np.linspace(1.,10.,1000)
    input_bisection_mode_condition = None
    input_tolerance = 1e-3
    
    # Some random smooth function
    func = lambda freq: np.sqrt(freq)*np.cos(2*np.pi*freq / 10) + 2*np.exp(-(freq-4)**2 / 0.1)

    # Use absolute error as metric
    error_metric = lambda func_vals, interp_vals: np.abs(func_vals - interp_vals)

    output_frequencies, output_func_values = improve_frequency_range(
        initial_frequencies=input_frequencies,
        func=func, 
        interpolation_error_norm=error_metric,
        absolute_integral_tolerance=input_tolerance,
        step_towards_inf_factor=2, # Not relevant for this test
        bisection_mode_condition=input_bisection_mode_condition
    )

    output_interpolation_on_fine_grid = complex_pchip(output_frequencies, output_func_values, input_finer_frequencies)
    expected_interpolation_on_fine_grid = func(input_finer_frequencies)

    # NOTE: Absolute tolerance here because that is used in the add_points... function itself
    assert simpson(error_metric(output_interpolation_on_fine_grid, expected_interpolation_on_fine_grid), input_finer_frequencies) < input_tolerance


@pytest.mark.parametrize("bisection_mode_condition", (None, lambda freqs: np.ones_like(freqs, dtype=bool)))
@pytest.mark.parametrize(("input_func", "expected_transform"), [
    ( lambda f: 1 / np.sqrt( 2 * np.pi * f + 1e-50), lambda t: np.sqrt(np.pi / (2 * t)) ),
    ( lambda f: np.array([[1,2],[3,4]]) / np.sqrt( 2 * np.pi * f + 1e-50),  lambda t: np.array([[1,2],[3,4]]) * np.sqrt(np.pi / (2 * t)) ), # Test dimension handling
    ( lambda f: 1 / (1 + (2 * np.pi * f)**2),        lambda t: np.pi / 2 * np.exp(-t) ),
])
def test_fourier_integral_adaptive(input_func: Callable[[ArrayLike], ArrayLike], expected_transform: Callable[[ArrayLike], ArrayLike], bisection_mode_condition):
    input_times = np.logspace(-10, 5, 100)
    input_starting_frequencies = (0, 1, np.inf) # Start with very few frequencies
    input_absolute_tolerance = 1e0

    output_transform_arr = fourier_integral_adaptive(
        times=input_times,
        initial_frequencies=input_starting_frequencies,
        func=input_func, 
        absolute_integral_tolerance=input_absolute_tolerance,
        interpolation = "pchip",
        step_towards_inf_factor = 2,
        bisection_mode_condition = bisection_mode_condition,
        interpolation_error_norm=None,
        max_iterations = 1000
    )
    
    expected_transform_arr = np.array([expected_transform(time) for time in input_times])

    # Tolerance is increased by 1e3 compared to what it should be. This partially stems from limiting the number of iterations in the interest of time,
    # and partially because the initial frequency range is poor. One would get better results by giving better initial frequencies, but this approach also allows testing
    # that 
    assert np.real(output_transform_arr) == pytest.approx(expected_transform_arr, abs=input_absolute_tolerance*1e3)


def test_cached_func():
    # NOTE: using time library might be a bad idea in unit tests...
    input_call_delay = 0.5 # [s]
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

@pytest.mark.parametrize(["input_a", "input_b", "expected_difference"], [
    (-1, 2, 3),
    ([1,-2,-3], [-3,-2,1], [4,0,4]),
    (
        [[1,0,0,0],  [2,0,0,0], [0,0,3,0]],
        [[0,-1,0,0], [-2,1,0,0], [0,1,3,0]],
        [np.sqrt(2), np.sqrt(17), 1]
    )
])
def test_standard_difference_norm(input_a: ArrayLike, input_b: ArrayLike, expected_difference: ArrayLike):
    output_difference = _difference_norm(input_a, input_b)
    
    expected_difference = np.asarray(expected_difference)
    
    assert len(output_difference.shape) <= 1
    assert output_difference == pytest.approx(expected_difference)
