from typing import Sequence, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import pchip_interpolate


def complex_pchip(xi: ArrayLike, zi: ArrayLike, x: ArrayLike, derivative_order: Union[int, Sequence[int]] = 0, axis: int = 0) -> np.ndarray:
    """Compute the piecewise cubic hermite interpolating polynomial (PCHIP) characterized by the points (xi, zi), where xi are real and zi are complex,
    and evaluate this polynomial or its derivatives at the points x.
    The real and imaginary components are treated separately, using the implementation in scipy.interpolate for
    each component, as that implementation is only designed to take in real-valued functions.
    The PCHIP interpolation is a cubic interpolation that guarantees to preserve monotonicity on each subinterval of the input data. In this case,
    monotonicity is therefore preserved for both real and imaginary components.
    
    For more information about the PCHIP algorithm, see [PCHIP]_.

    :param xi: A 1D array of N real input points
    :type xi: ArrayLike
    :param func_values: An array of shape (N, X1, X2, ...) containing the output of the function to compute the interpolation for at each xi.
    :type func_values: ArrayLike
    :param x: A 1D array of M real inputs to evaluate the interpolation at
    :type x: ArrayLike
    :param derivative_order: The order(s) of derivatives to compute (`0` gives the function values). If a sequence, a list of arrays is returned, one for each order. Defaults to 0
    :type derivative_order: Union[int, Sequence[int]], optional
    :param axis: The axis in func_values that corresponds to xi, defaults to 0. By setting this argument, N does not need to be the first axis of func_values
    :type axis: int, optional
    :return: The computed PCHIP values or derivatives evaluated at the points x. If more than one derivative order is given, a list of arrays is returned
    :rtype: np.ndarray
    """
    return (
        (pchip_interpolate(xi, np.real(zi), x, der=derivative_order, axis=axis)
        + 1j*pchip_interpolate(xi, np.imag(zi), x, der=derivative_order, axis=axis))
    )
