Theory
======

Introduction
------------

The Neffint package is mainly designed for with computing Fourier integrals, i.e.

.. math::
    \int_{-\infty}^{\infty} \psi(\omega) e^{i \omega t} d\omega,

using a quadrature method largely based on section 1.6.3 in [N.Mounet-PhD]_, which builds on the work of [Filon]_.
The need for a specialized method arises from the fact that for large values of `t`,
the integrand :math:`\psi(\omega) e^{i \omega t}` oscillates with a very short period.
This can cause problems when using a standard Fast Fourier Transform (FFT) technique,
since it relies on the assumption that on, each frequency subinterval :math:`[\omega_k, \omega_{k+1}]` used in the computation,

.. math::
    \int_{\omega_k}^{\omega_{k+1}} \psi(\omega) e^{i \omega t} d\omega
    \approx
    (\omega_{k+1} - \omega_k) \psi(\omega_k) e^{i \omega t},

which for high `t` require a very fine frequency spacing to be correct.
In many cases this is unfeasable in terms of computing time and/or memory usage.
The same requirements for fine frequency grids also apply to other standard integration schemes, such as Simpson's method.

It is therefore clear that an alternative approach for computing Fourier integrals is needed.


The Neffint Method
------------------

When computing Fourier integrals using the Neffint method, the integration range is first split into main integration range and asymptotic regions:

.. math::
    \int_{-\infty}^{\infty} \psi(\omega) e^{i \omega t} d\omega
    \approx
    \int_{-\infty}^{\omega_{min}} \psi(\omega) e^{i \omega t} d\omega
    + \int_{\omega_{min}}^{\omega_{max}} \psi(\omega) e^{i \omega t} d\omega
    + \int_{\omega_{max}}^{\infty} \psi(\omega) e^{i \omega t} d\omega,

and compute each of the regions separately.


Integrating the asymptotic regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The asymptotic regions can be computed using a Taylor series of :math:`\psi(\omega)`
around :math:`\omega_{min}` or :math:`\omega_{min}`, which evaluates to

.. math::
    \int_{\omega_{max}}^{\infty} \psi(\omega) e^{i \omega t} d\omega
    =
    e^{i \omega_{max} t} \sum_{n=0}^\infty \frac{ i^{n+1} \psi^{(n)}(\omega_{max})}{t^{n+1}},

which can be truncated to desired precision and computed. For the region going to :math:`-\infty`,
one gets the same result only with oposite sign.

Often, one deals with functions that are known to be real in time domain,
in which case one only needs to integrate over half the frequency domain.
One can in that case neglect the negative asymptotic term and set :math:`\omega_{min} = 0`
(or arbitrarily close in the case of a singularity at `0`).


Integrating the main integration range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We split the main integration range into subintervals :math:`[\omega_k, \omega_{k+1}]`,
`not necessarily with equal spacing`.
On each of these subintervals, we replace :math:`\psi` with a polynomial :math:`p_k(\omega)`
that interpolates :math:`\psi` on that subinterval. This yields a new integral,

.. math::
    \int_{\omega_k}^{\omega_{k+1}} \psi(\omega) e^{i \omega t} d\omega
    \approx
    \int_{\omega_k}^{\omega_{k+1}} p_k(\omega)  e^{i \omega t} d\omega,

that can be computed analytically.

The form of this analytic expression depends on the choice of interpolating polynomial.
The special cases of linear interpolation and [PCHIP]_ interpolation is detailed in :ref:`appendix-polynomial-examples`.

With this procedure, one can accurately compute Fourier integrals
without the need for equidistant or densely spaced frequencies.
There are only three remaining arbitrary choices:

1. The choice of frequency endpoints :math:`\omega_{min}, \omega_{max}`
2. The choice of frequency subintervals :math:`[\omega_k, \omega_{k+1}]`
3. The choice of interpolating polynomial type

The choice of polynomial type is a matter of preference to the user, within the limits of what is supported by Neffint.
The other two points will be discussed in the next section.


Frequency Selection
-------------------

To select frequencies, we can make use of the inequality

.. Using \left and \right (og \big or similar) on the absolute value signs change the colour to make them almost invisible, for some reason
.. math::
    | \left( \int_{\omega_{min}}^{\omega_{max}} \psi(\omega) e^{i \omega t} d\omega - \int_{\omega_{min}}^{\omega_{max}} p(\omega) e^{i \omega t} d\omega \right) |
    \leq
    \int_{\omega_{min}}^{\omega_{max}} |\psi(\omega) - p(\omega)| d\omega,

where :math:`p(\omega)` is the function equal to :math:`p_k(\omega)` on each subinterval.
In other words, the integrated interpolation error sets an upper bound on the Fourier integral error,
and one can therefore find a good set of frequencies to use for the Fourier integration
by finding one that minimizes the integrated interpolation error. An algorithm for achieving this is outlined as follows:

First, define an initial sequence of frequencies :math:`\{\omega_{min}, \omega_1, \omega_2, \cdots, \omega_{max}\}`.
This could be anything from a arithmetic or geometric sequence to a more specialized initial guess.
Compute the value of :math:`\psi(\omega)` at each of these frequencies, and use those values to create an interpolating polynomial :math:`p_k(\omega)` in each subinterval.

Then, compute the midpoint frequency :math:`\omega_{k, k+1} = \frac{\omega_k + \omega_{k+1}}{2}` of each existing frequency subinterval.
The Simpson's rule approximation of the integrated interpolation error is given then as

.. math::
    \text{Total error} = \sum_k \frac{2(\omega_{k+1} - \omega_k)}{3} | \psi(\omega_{k, k+1}) - p_k(\omega_{k, k+1}) |,

where the contribution from the interval endpoints has been removed since they by construction of the interpolant are identically zero.
By iteratively adding the midpoint frequency from the interval with largest error, and recomputing the error with the two new intervals added,
the error should gradually shrink, and one can terminate the iteration when reaching a desired tolerance.

As an alternative to bisecting the subintervals using the arithmetic mean, as shown above, one could use the geometric mean instead:

.. math::
    \omega_{k, k+1} = \operatorname{sign}(\omega_k) \sqrt{\omega_k \omega_{k+1}} = \operatorname{sign}(\omega_k) e^{\frac{\log(\left|\omega_k\right|) + \log(\left|\omega_{k+1}\right|)}{2}}.

This can not be done for intervals containing zero, and requires Simpson's formula to be modified into

.. math::
    \text{Total error} = \sum_k \frac{2}{3} \log\left({\frac{\omega_{k+1}}{\omega_k}}\right) | \psi(\omega_{k, k+1}) - p_k(\omega_{k, k+1}) |\omega_{k, k+1}.

The calculation steps are shown in :ref:`appendix-logscale-simpson`.
One can also combine the two approaches, selecting either arithmetic or geometric bisection depending on the frequency.

Regarding the determination of good frequency end points :math:`\omega_{min}, \omega_{max}`, one can incorporate this into the bisection algorithm by also allowing the intervals
:math:`(-\infty, \omega_{min})` and :math:`(\omega_{max}, \infty)` to be bisected.
This can be done by creating a phantom frequency by taking the sum or product of :math:`\omega_{min}` or :math:`\omega_{max}` and some constant,
and using this phantom frequency for the bisection and error integration.


Appendix
--------

.. _appendix-polynomial-examples:

Appendix A - Integration of linear and PCHIP interpolants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned in `Integrating the main integration range`_, when performing the Fourier integral using Filon's method, any interpolating polynomial can in principle be used.
Here, we show the solution for a linear interpolant and a PCHIP (Piecewise Cubic Hermite Interpolating Polynomial - see [PCHIP]_) interpolant. Derivation of these results are given in [N.Mounet-PhD]_.

In both solutions,

.. math::
    \Delta_k = \omega_{k+1} - \omega_k

denotes the length of the frequency interval. It is also worth noting that while e.g. :math:`p_k(\omega)` is written in the expressions,
by construction, this is identical to :math:`\psi(\omega)` as well.
There is in practice therefore not necessary to compute the interpolant (i.e. slopes and constants on each interval for linear interpolation),
as knowing the function values is enough.

Linear
^^^^^^
When using a linear interpolation, the integral of each subinterval evaluates to

.. math::
    \begin{align}
        \int_{\omega_k}^{\omega_{k+1}} p_k(\omega) e^{i \omega t} d\omega
        = \Delta_k[
            & p_k(\omega_k) e^{i \omega_{k+1} t} \Lambda(-\Delta_k t) \\
            + & p_k(\omega_{k+1}) e^{i \omega_k t} \Lambda(\Delta_k t)
        ],
    \end{align}

where :math:`\Lambda(x)` is given by


.. math::
    \Lambda(x) = - \frac{i e^{ix}}{x} + \frac{e^{ix} -1}{x^2}.

PCHIP
^^^^^

When using PCHIP interpolation, the integral in each subinterval evaluates to

.. math::
    \begin{align}
        \int_{\omega_k}^{\omega_{k+1}} p_k(\omega) e^{i \omega t} d\omega
        = \Delta_k[ 
            & p_k(\omega_k) e^{i \omega_{k+1} t} \Phi(-\Delta_k t) \\
            + & p_k(\omega_{k+1}) e^{i \omega_k t} \Phi(\Delta_k t) \\
            - & \Delta_k p_k'(\omega_k) e^{i \omega_{k+1} t} \Psi(-\Delta_k t) \\
            + & \Delta_k p_k'(\omega_{k+1}) e^{i \omega_k t} \Psi(\Delta_k t)
        ],
    \end{align}

where :math:`\Phi` and :math:`\Psi` are given by

.. math::
    \begin{align}
        \Phi(x) & =
            - \frac{i e^{i x}}{x} 
            - \frac{6 i (e^{i x} + 1)}{x^3}
            + \frac{12(e^{i x} - 1)}{x^4}, \\
        \Psi(x) & = 
            \frac{e^{i x}}{x^2} 
            + \frac{2 i (2 e^{i x} + 1)}{x^3}
            - \frac{6(e^{i x} - 1)}{x^4}.
    \end{align}

Note that the derivatives of the interpolant enters into the expression. This derivative is determined by the PCHIP algorithm.

.. _appendix-logscale-simpson:

Appendix B - Log-scale Simpson's method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
    \begin{align*}
        \int_{x_0}^{y_0} f(x) dx &= \int_{u_0}^{u_1} f(e^u) e^u du \quad |\quad u = \log(x), u_0 = \log(x_0) , u_1 = \log(x_1) \\
        \\
        &\approx \frac{u_1 - u_0}{6} \left(f(e^{u_0})e^{u_0} + 4 f(e^{\frac{u_0 + u_1}{2}})e^{\frac{u_0 + u_1}{2}} + f(e^{u_1})e^{u_1} \right) \\
        \\
        &= \frac{\log(x_1) - \log(x_0)}{6} \Bigl( f(x_0)x_0 + 4 f(e^{\frac{\log(x_0) + \log(x_1)}{2}})e^{\frac{\log(x_0) + \log(x_1)}{2}}+ f(x_1)x_1 \Bigr) \\
        \\
        &=\frac{1}{6} \log\left({\frac{x_1}{x_0}}\right) \Bigl( f(x_0)x_0 + 4 f(\sqrt{x_0 x_1})\sqrt{x_0 x_1} + f(x_1)x_1 \Bigr)
    \end{align*}


.. [N.Mounet-PhD] N. Mounet. The LHC Transverse Coupled-Bunch Instability, PhD thesis 5305 (EPFL, 2012), http://infoscience.epfl.ch/record/174672/files/EPFL_TH5305.pdf
.. [Filon] L. N. G. Filon. On a quadrature formula for trigonometric integrals. Proc. Roy. Soc. Edinburgh, 49:38-47, 1928.
.. [PCHIP] F. N. Fritsch and J. Butland, A method for constructing local monotone piecewise cubic interpolants, SIAM J. Sci. Comput., 5(2), 300-304 (1984). DOI:10.1137/0905021.
