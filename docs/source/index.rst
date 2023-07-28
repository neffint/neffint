.. Neffint documentation master file, created by
   sphinx-quickstart on Thu Jun 22 18:09:09 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Neffint's documentation!
===================================

Neffint is an acronym for **N**\ on-\ **e**\ quidistant **F**\ ilon **F**\ ourier **int**\ egration. This is a python package for computing Fourier integrals using a method based on Filon's rule with non-equidistant grid spacing.

Neffint licensed under an Apache-2.0 license. See `HERE <https://www.apache.org/licenses/LICENSE-2.0>`_ for details.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   theory
   ex_fixed_grid_FI.ipynb
   ex_adaptive.ipynb
   api


Installation procedure
----------------------

To start using Neffint, install it using pip:

.. code-block:: console

   pip install neffint

To get an editable install, download the repository from Github and install with the ``-e`` flag.

.. code-block:: console

   git clone https://github.com/neffint/neffint.git
   cd neffint
   pip install -e .[test]

The ``[test]`` is optional, but installs the requirements for running the tests, which is needed for development.
To run the tests after installing locally with the above commands, simply run

.. code-block:: console

   pytest

from the root directory of the repository.

To compile the documentation locally, run

.. code-block:: console

   git clone https://github.com/neffint/neffint.git
   cd neffint
   pip install .[docs]
   sphinx-build -b html docs/source docs/build/html

.. note:: In case the sphinx build command fails because of an issue related to ``Pandoc``, you can try to install Pandoc manually
 (but not from PyPI as the package is outdated there). If using ``conda``, try the following command
  .. code-block:: console

    conda install pandoc

The documentation at https://neffint.readthedocs.io/ is built automatically by ReadTheDocs at every commit to the `main` branch.




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
