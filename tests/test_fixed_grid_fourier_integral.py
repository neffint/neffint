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

from typing import Callable, Tuple

import numpy as np
import pytest
from numpy.typing import ArrayLike

from neffint.fixed_grid_fourier_integral import (
    InterpolationMode, _fourier_integral_inf_correction, _lambda, _phi_and_psi,
    fourier_integral_fixed_sampling)


def test_lambda():
    """Test _lambda against calculation with the analytical formula using 200 decimal digit precision with mpmath."""
    input_x = np.array([1e-9, 1, 1e9, 1e18], dtype=np.float64)

    # Set tolerance to 10*machine precision
    comparison_tolerance = 10 * np.finfo(input_x.dtype).eps

    # Expected outputs generated with mpmath at 200 decimal digit precision
    expected_lambda = np.array([
        0.5+3.333333333333333e-10j,
        0.3817732906760362+0.3011686789397568j,
        5.458434492865867e-10-8.378871808180589e-10j,
        -9.92969320740405e-19-1.1837199021871074e-19j
    ])

    output_lambda = _lambda(input_x)

    assert output_lambda == pytest.approx(expected_lambda, rel=comparison_tolerance, abs=comparison_tolerance)


def test_phi_and_psi():
    """Test _phi_and_psi against calculation using the analytical formula using 200 decimal digit precision with mpmath."""
    input_x = np.array([1e-9, 1, 1e9, 1e18], dtype=np.float64)

    # Set tolerance to 10*machine precision
    comparison_tolerance = 10 * np.finfo(input_x.dtype).eps

    # Expected outputs generated with mpmath at 200 decimal digit precision
    expected_phi = np.array([
        0.5 + 3.5e-10j,
        0.37392456407295215539 + 0.31553567661778005804j,
        5.4584344944869956752e-10 - 8.3788718136390234542e-10j,
        -9.9296932074040507621e-19 - 1.1837199021871073261e-19j
    ])

    expected_psi = np.array([
        -0.0833333333333333333 - 5e-11j,
        -0.06739546857228461362 - 0.046145700566923663667j,
        8.3788717918052853757e-19 + 5.4584345480024828642e-19j,
        1.1837199021871073658e-37 - 9.9296932074040507374e-37j
    ])

    output_phi, output_psi = _phi_and_psi(input_x)

    assert output_phi == pytest.approx(expected_phi, rel=comparison_tolerance, abs=comparison_tolerance)
    assert output_psi == pytest.approx(expected_psi, rel=comparison_tolerance, abs=comparison_tolerance)


@pytest.mark.parametrize(("interpolation_mode", "rel_tol"), [
    (InterpolationMode.PCHIP.value, 1e-4),
    (InterpolationMode.LINEAR.value, 1e-3),
])
@pytest.mark.parametrize(("input_func", "expected_transform"), [
    ( lambda f: 1 / np.sqrt( 2 * np.pi * f),                        lambda t: np.sqrt(np.pi / (2 * t)) ),
    ( lambda f: np.array([[1,2],[3,4]]) / np.sqrt( 2 * np.pi * f),  lambda t: np.array([[1,2],[3,4]]) * np.sqrt(np.pi / (2 * t)) ), # Test dimension handling
    ( lambda f: 1 / (1 + (2 * np.pi * f)**2),                       lambda t: np.pi / 2 * np.exp(-t) ),
])
def test_fourier_integral_fixed_sampling(input_func: Callable[[ArrayLike], ArrayLike], expected_transform: Callable[[ArrayLike], ArrayLike], interpolation_mode: str, rel_tol: float):
    """Test Fourier integral accuracy on function with an analytically known Fourier transform on positive half range."""
    input_frequencies = np.logspace(-10,20,1000)
    input_times = np.logspace(-15, 0, 50)

    input_func_arr = np.array([input_func(freq) for freq in input_frequencies])

    output_transform_arr = fourier_integral_fixed_sampling(
        times=input_times,
        frequencies=input_frequencies,
        func_values=input_func_arr,
        pos_inf_correction_term=True,
        neg_inf_correction_term=False,
        interpolation=interpolation_mode
    )

    expected_transform_arr = np.array([expected_transform(time) for time in input_times])

    assert np.real(output_transform_arr) == pytest.approx(expected_transform_arr, rel=rel_tol)


@pytest.mark.parametrize(("interpolation_mode", "tolerances"), [
    (InterpolationMode.PCHIP.value, (1e-4,1e-4)),
    (InterpolationMode.LINEAR.value, (1e-3,5e-3)),
])
def test_full_range_fourier_integral_fixed_sampling(interpolation_mode: str, tolerances: Tuple[float, float]):
    """Test Fourier integral accuracy on function with an analytically known Fourier transform on full range."""
    input_func = lambda f: np.sqrt(np.pi/1e10) * np.exp(- np.pi**2 * f**2 / 1e10) # Gaussian
    expected_transform = lambda t: 2*np.pi*np.exp( - 1e10 * t**2)
    
    input_freqs_half_range = np.logspace(-10,20,1000)
    input_times_half_range = np.logspace(-15, 0, 50)
    
    # Add negative half ranges and sort
    input_frequencies = np.union1d(-input_freqs_half_range, input_freqs_half_range)
    input_times = np.union1d(-input_times_half_range, input_times_half_range)

    input_func_arr = np.array([input_func(freq) for freq in input_frequencies])

    output_transform_arr = fourier_integral_fixed_sampling(
        times=input_times,
        frequencies=input_frequencies,
        func_values=input_func_arr,
        pos_inf_correction_term=True,
        neg_inf_correction_term=True,
        interpolation=interpolation_mode
    )

    expected_transform_arr = np.array([expected_transform(time) for time in input_times])

    assert output_transform_arr == pytest.approx(expected_transform_arr, rel=tolerances[0], abs=tolerances[1])


@pytest.mark.parametrize(("input_positive_inf", "expected_correction_term"), [
    (True, 4.875060250580407690e-6 + 8.731196225215127567e-6j),
    (False, 4.87506025058040769e-6 - 8.731196225215127567e-6j),
])
def test_asymptotic_correction_term(input_positive_inf: bool, expected_correction_term: complex):
    """Test that asymptotic term computes correctly for both signs.
    
    Expected answers calculated in WolframAlpha as: integral from 1e10 to inf of 1/sqrt(x)*exp(+-i * x)dx"""
    
    input_times = 1
    input_omega_end = 1e10
    input_func_value_end = 1 / np.sqrt(input_omega_end)
    input_func_derivative_end = 0.5 * input_omega_end**-1.5
    
    input_omega_end *= (1 if input_positive_inf else -1)
    
    output_correction_term = _fourier_integral_inf_correction(
        times=input_times,
        omega_end=input_omega_end,
        func_value_end=input_func_value_end,
        func_derivative_end=input_func_derivative_end,
        positive_inf=input_positive_inf
    )
    
    assert output_correction_term == pytest.approx(expected_correction_term)
