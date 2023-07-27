from setuptools import setup, find_packages
from pathlib import Path

def get_version(path):
    with path.open("rt") as f:
        for line in f.readlines():
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")

# Minimum python version. Remember to also change in github workflows if increased
PY_VERSION_REQUIRED = ">=3.7"

PACKAGES_REQUIRED = {
    "core": [
        "numpy",
        "scipy",
    ],
    "test": [
        "pytest",
    ],
    "docs": [
        "sphinx",
        "sphinx-rtd-theme",
        "nbsphinx",
        "nbsphinx-link",
        "ipykernel",
    ],
}

root_dir = Path(__file__).parent.absolute()

# Read README.md to use as long description in package metadata
with (root_dir / "README.md").open("rt") as readme_file:
    long_description = readme_file.read().strip()

setup(
    name="neffint",
    version=get_version(root_dir / "neffint" / "_version.py"),
    description="Python package for computing Fourier integrals using a Filon type method with non-equidistant grid spacing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neffint/neffint/",
    author="Eskil Vik, Nicolas Mounet",
    packages=find_packages(), # finds all the packages in the repository
    python_requires=PY_VERSION_REQUIRED,
    install_requires=PACKAGES_REQUIRED["core"],
    extras_require=PACKAGES_REQUIRED
)
