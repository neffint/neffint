from setuptools import setup, find_packages
from pathlib import Path

# Use semantic versioning
VERSION = "1.0.0"

PY_VERSION_REQUIRED = ">=3.7"
PACKAGES_REQUIRED = {
    "core": [
        "numpy",
        "scipy"
    ],
    "test":[
        "pytest"
    ]
}

# Read README.md to use as long description in package metadata
root_dir = Path(__file__).parent.absolute()
with (root_dir / "README.md").open("rt") as readme_file:
    long_description = readme_file.read().strip()

setup(
    name="neffint",
    version=VERSION,
    description="A python package for computing Fourier integrals using a Filon type method with non-equidistant grid spacing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neffint/neffint/",
    author="Eskil Vik, Nicolas Mounet",
    packages=find_packages(), # finds all the packages in the repository
    python_requires=PY_VERSION_REQUIRED,
    install_requires=PACKAGES_REQUIRED["core"],
    extras_require=PACKAGES_REQUIRED
    )
