from typing import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike

from neffint.fixed_grid_fourier_integral import (
    fourier_integral_fixed_sampling_pchip, phi_and_psi)


def test_phi_and_psi():
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

    output_phi, output_psi = phi_and_psi(input_x)

    assert output_phi == pytest.approx(expected_phi, rel=comparison_tolerance, abs=comparison_tolerance)
    assert output_psi == pytest.approx(expected_psi, rel=comparison_tolerance, abs=comparison_tolerance)


@pytest.mark.parametrize(("input_func", "expected_transform"), [
    ( lambda f: 1 / np.sqrt( 2 * np.pi * f),                        lambda t: np.sqrt(np.pi / (2 * t)) ),
    ( lambda f: np.array([[1,2],[3,4]]) / np.sqrt( 2 * np.pi * f),  lambda t: np.array([[1,2],[3,4]]) * np.sqrt(np.pi / (2 * t)) ), # Test dimension handling
    ( lambda f: 1 / (1 + (2 * np.pi * f)**2),                       lambda t: np.pi / 2 * np.exp(-t) ),
])
def test_fourier_integral(input_func: Callable[[ArrayLike], ArrayLike], expected_transform: Callable[[ArrayLike], ArrayLike]):
    input_frequencies = np.logspace(-10,20,1000)
    input_times = np.logspace(-15, 0, 50)

    input_func_arr = np.array([input_func(freq) for freq in input_frequencies])

    output_transform_arr = fourier_integral_fixed_sampling_pchip(
        times=input_times,
        frequencies=input_frequencies,
        func_values=input_func_arr,
        inf_correction_term=True,
    )

    expected_transform_arr = np.array([expected_transform(time) for time in input_times])

    assert np.real(output_transform_arr) == pytest.approx(expected_transform_arr, 1e-4)

# @pytest.mark.parametrize(("input_func", "expected_transform"), [
#     ( lambda f: 1 / np.sqrt( 2 * np.pi * f),                        lambda t: np.sqrt(np.pi / (2 * t)) ),
#     ( lambda f: np.array([[1,2],[3,4]]) / np.sqrt( 2 * np.pi * f),  lambda t: np.array([[1,2],[3,4]]) * np.sqrt(np.pi / (2 * t)) ), # Test dimension handling
#     ( lambda f: 1 / (1 + (2 * np.pi * f)**2),                       lambda t: np.pi / 2 * np.exp(-t) ),
# ])
# def test_fourier_integral_uneven_sampling(input_func: Callable[[ArrayLike], ArrayLike], expected_transform: Callable[[ArrayLike], ArrayLike]):
#     # Initialize a random number generator
#     rng = np.random.default_rng(seed=101)

#     # Logspace-esque distribution from 1e1 to 1e10
#     random_logspace = 10**(-10+30*rng.random(900))

#     # Gaussian spike at 1e4
#     random_spike = rng.normal(loc=1e4, scale=5e2, size=100)

#     # Sorted frequencies with an uneven density of frequencies
#     input_frequencies = np.union1d(random_logspace, random_spike)

#     input_times = np.logspace(-15, 0, 50)

#     input_func_arr = np.array([input_func(freq) for freq in input_frequencies])
#     input_func_derivative_arr = pchip_interpolate(input_frequencies, input_func_arr, input_frequencies, der=1, axis=0)

#     output_transform_arr = fourier_integral_fixed_sampling(
#         times=input_times,
#         omegas=input_frequencies,
#         func_values=input_func_arr,
#         func_derivatives=input_func_derivative_arr,
#         inf_correction_term=True,
#     )

#     expected_transform_arr = np.array([expected_transform(time) for time in input_times])

#     assert np.real(output_transform_arr) == pytest.approx(expected_transform_arr, 1e-4)