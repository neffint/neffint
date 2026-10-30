# Neffint
Neffint is an acronym for **N**on-**e**quidistant **F**ilon **F**ourier **int**egration. This is a python package for computing Fourier integrals using a method based on Filon's rule with non-equidistant grid spacing.

Neffint licensed under an Apache-2.0 license. See [HERE](https://www.apache.org/licenses/LICENSE-2.0) or the LICENSE file for details.

Documentation for the package is hosted at: https://neffint.readthedocs.io/en/latest

A presentation about Neffint given @[CERN-ABP-CEI meeting](https://indico.cern.ch/event/1283488/) is available [here](https://indico.cern.ch/event/1283488/contributions/5392363/attachments/2687780/4670873/ANIMATED_esvik_CEI_20230720.pdf).

## Repository structure

The following is a brief explanation of the repository structure:

- `neffint/` - root directory
    - `.github/workflows/` - Continuous integration jobs
    - `examples/` - directory for examples of use
    - `neffint/`- source code directory for the Python package
    - `tests/`- directory for code tests (pytest)
    - `LICENSE`, `README.md`, `.gitignore` - Standard github repository files
    - `pyproject.toml`, `setup.py` - Python packaging instructions
    - `release.sh` - PyPI publishing script

## Using the package

To install Neffint, simply run:

    pip install neffint

The primary functionality of the package is the ability to compute Fourier integrals on non-equidistant frequency grids. The `fourier_integral_fixed_sampling` function is a good starting point for this. This takes in an array of frequencies and a corresponding array of function values, and performs the integration for a range of time values.

The `improve_frequency_range` and `fourier_integral_adaptive` functions allow improving an initial frequency grid adaptively, for more precise integration.

The notebooks in the `examples/` folder give a more detailed introduction to using the package.

## Development

Neffint is an open source project, and as such, users should feel welcome to provide feedback or contribute to the code. Check out [CONTRIBUTING.md](https://github.com/ImpedanCEI/neffint/blob/main/CONTRIBUTING.md) for more details on how to contribute to the project.

