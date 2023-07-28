# Neffint
Neffint is an acronym for **N**on-**e**quidistant **F**ilon **F**ourier **int**egration. This is a python package for computing Fourier integrals using a method based on Filon's rule with non-equidistant grid spacing.

Neffint licensed under an Apache-2.0 license. See [HERE](https://www.apache.org/licenses/LICENSE-2.0) or the LICENSE file for details.

Documentation for the package is hosted at: https://neffint.readthedocs.io/en/latest

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

Neffint is an open source project, and as such, users should feel welcome to provide feedback or contribute to the code.

### Providing feedback

If you experience any problems with the package, feel free to put up a [Github issue](https://github.com/neffint/neffint/issues) giving a detailed explanation the problem. This description should mention:

- What the problem is
- Why it is a problem
- If the problem is with the code itself, enough information to reproduce the behaviour
- (Optional) Ideas or suggestions for fixing the issue

### Contributing to the code

#### Using Git and Github

To contribute to the code, clone the repository to your machine, and create a new branch for your contribution

    git clone git@github.com:neffint/neffint.git
    git checkout -b your-branch-name

Branch names should start with a category, followed by a `/` and a descriptive name. The categories used are:
- `feature` - for new code features
- `bugfix` - for bugfixes to existing code
- `misc` - for other changes, e.g. stuctural changes, documentation changes, addition of tests, etc. As a general rule these should not modify the behaviour of the code itself.

An example of a branch name: `feature/add-asymptotic-correction`

If desirable, sub-branches can be created to segment development into smaller work chunks. For large features this can be a good way to ensure more managable code review sizes, and thus improve the quality of the review.

The `main` branch is used for stable releases. This is a protected branch, meaning all changes to this branch must go through a pull request with review. A pull request is submitted on Github, under the "Pull requests" tab on the repository home page.

#### Coding guidelines

When making contributions to the code, keep in mind that the code should be understandable to future readers of the code (including reviewers). Try therefore to make the code as readable as possible. Some concrete suggestions include, but are not limited to:
- Use type hints in function declarations
- Write docstrings for functions, classes and modules
- Use descriptive names for variables, functions and classes

As far as possible and practical, the code should be split into functions and classes with a single responsibility, with corresponding unit tests checking that it performs that task correctly.

#### Publishing Releases

For publishing a release and uploading the release to PyPI, use the following procedure:

- Press the `"Create a new release"` link on the sidebar on Github.
- Create a new tag with the name `vX.Y.Z` on `main`, where `X`, `Y` and `Z` are the semantic version number of the release. This should be the same version number as `__version__` in `neffint/_version.py`. Give the release the same name.
- Press `"Generate release notes"` to generate release notes. No additional notes are needed.
- Press publish release. This will push a git tag to the latest commit to main, and run a CI job to build and publish the package to PyPI.

Alternatively, one can checkout to `main` and run `release.sh`, which will also push a git tag and build and publish the release. Using this alternative procedure, one should still preferably make a Github release from the pushed tag as well. The main differences are:

- When creating a Github release, the Github release and the git tag will automatically be the same, while one must manually verify that the version number in `neffint/_version.py` matches. When using the shell script, `neffint/_version.py` and the git tag are automatically the same, but one must manually check that the Github release matches.
- `release.sh` relies on using `setup.py`, a format setuptools is moving away from. The Github release workflow does not rely on the `setup.py` format.
- When using `release.sh`, one must have an authentication token locally on one's machine to upload to PyPI. The Github action uses a repository secret to authenticate with PyPI, so one only needs write access to the Github repository (for creating the release).
