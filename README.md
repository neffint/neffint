# Neffint
Neffint is an acronym for **N**on-**e**quidistant **F**ilon **F**ourier **int**egration. This is a python package for computing Fourier integrals using a method based on Filon's rule with non-equidistant grid spacing.

Neffint licensed under an Apache-2.0 license. See [HERE](https://www.apache.org/licenses/LICENSE-2.0) or the LICENSE file for details.

## Repository structure

The following is a brief explanation of the repository structure:

- `neffint/` - root directory
    - `examples/` - directory for examples of use
    - `neffint/`- source code directory for the Python package
    - `tests/`- directory for code tests (pytest)
    - `LICENSE`, `README.md`, `.gitignore` - Standard github repository files
    - `pyproject.toml`, `setup.py` - Python packaging instructions


## Development

Neffint is an open source project, and as such, users should feel welcome to provide feedback or contribute to the code.

### Providing feedback

If you experience any problems with the package, feel free to put up a [Github issue](https://github.com/neffint/neffint/issues) giving a detailed explanation the problem. This description should mention:

- What the problem is
- Why it is a problem
- If the problem is with the code itself, enough information to reproduce the behaviour
- (Optional) Ideas or suggestions for fixing the issue

### Contributing to the code

#### Coding guidelines

When making contributions to the code, keep in mind that the code should be understandable to future readers of the code (including reviewers). Try therefore to make the code as readable as possible. Some concrete suggestions include, but are not limited to:
- Use type hints in function declarations
- Write docstrings for functions, classes and modules
- Use descriptive names for variables, functions and classes

As far as possible and practical, the code should be split into functions and classes with a single responsibility, with corresponding unit tests checking that it performs that task correctly.

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
