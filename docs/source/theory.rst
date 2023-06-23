Theory
======

Introduction
------------

The Neffint package is mainly designed for with computing Fourier integrals, i.e.

.. math::
    \int_{-\infty}^{\infty} \psi(\omega) e^{i \omega t} d\omega,

using a quadrature method largely based on section 1.6.3 in [N.Mounet-PhD]_, which builds on the work of Filon_.
The need for a specialized method arises from the fact that for large values of `t`, the integrand :math:`\psi(\omega) e^{i \omega t}` oscillates with a very short period.
This can cause problems when using a standard Fast Fourier Transform (FFT) technique,
since it relies on the assumption that on each frequency subinterval :math:`[\omega_k, \omega_{k+1}]`

.. math::
    \int_{\omega_k}^{\omega_{k+1}} \psi(\omega) e^{i \omega t} d\omega
    \approx
    (\omega_{k+1} - \omega_k) \psi(\omega_k) e^{i \omega t},

which for high `t` require a very fine frequency spacing to be correct. In many cases this is unfeasable in terms of computing time and/or memory use.
The same requirements for fine frequency grids also apply to other standard integration schemes, such as Simpson's method.
It is therefore clear that another approach for computing Fourier integrals is needed.


The Neffint Method
------------------

When computing Fourier integrals using the Neffint method, the integration range is first split into main integration range and asymptotic regions:

.. math::
    \int_{-\infty}^{\infty} \psi(\omega) e^{i \omega t} d\omega
    \approx
    \int_{-\infty}^{\omega_{min}} \psi(\omega) e^{i \omega t} d\omega
    + \int_{-\omega_{min}}^{\omega_{max}} \psi(\omega) e^{i \omega t} d\omega
    + \int_{\omega_{max}}^{\infty} \psi(\omega) e^{i \omega t} d\omega,

and compute each of the regions separately.


Integrating the asymptotic regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The asymptotic regions can be computed using a Taylor series of :math:`\psi(\omega)` around :math:`\omega_{min}` or :math:`\omega_{min}`, which evaluates to

.. math::
    \int_{\omega_{max}}^{\infty} \psi(\omega) e^{i \omega t} d\omega
    =
    e^{i \omega_{max} t} \sum_{n=0}^\infty \frac{ i^{n+1} \psi^{(n)}(\omega_{max})}{t^{n+1}},

which can be truncated to desired precision and computed. For the region going to :math:`-\infty`, one gets the same result only with oposite sign.

Often, one deals with functions that are known to be real in time domain, in which case one only needs to integrate over half the frequency domain.
One can in that case neglect the negative asymptotic term and set :math:`\omega_{min} = 0` (or arbitrarily close in the case of a singularity at `0`).


Integrating the main integration range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main integration range we will split into subintervals :math:`[\omega_k, \omega_{k+1}]`, not necessarily equidistant (!).
On each of these subintervals, we replace :math:`\psi` with an polynomial :math:`p_k(\omega)` that interpolates :math:`\psi` on that subinterval.
This yields a new integral,

.. math::
    \int_{\omega_k}^{\omega_{k+1}} \psi(\omega) e^{i \omega t} d\omega
    \approx
    \int_{\omega_k}^{\omega_{k+1}} p_k(\omega)  e^{i \omega t} d\omega,

that can be computed analytically.

The form of this analytic expression depends on the choice of interpolating polynomial.
The special cases of linear interpolation and PCHIP_ interpolation is detailed in :ref:`appendix-polynomial-examples`.

With this procedure, one can accurately compute Fourier integrals without the need for equidistant frequencies. There are only three remaining arbitrary choices:

1. The choice of frequency endpoints :math:`\omega_{min}, \omega_{max}`
2. The choice of frequency subintervals :math:`[\omega_k, \omega_{k+1}]`
3. The choice of interpolating polynomial type

These will all be adressed in the next section.


Code Implementation
-------------------

TODO


Appendix
--------

.. _appendix-polynomial-examples:

Appendix A - Integration of linear and PCHIP interpolants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~








.. [N.Mounet-PhD] N. Mounet. The LHC Transverse Coupled-Bunch Instability, PhD thesis 5305 (EPFL, 2012), http://infoscience.epfl.ch/record/174672/files/EPFL_TH5305.pdf
.. [Filon] L. N. G. Filon. On a quadrature formula for trigonometric integrals. Proc. Roy. Soc. Edinburgh, 49:38-47, 1928.
.. [PCHIP] TODO
