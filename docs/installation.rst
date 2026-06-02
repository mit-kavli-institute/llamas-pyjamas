Installation
============

Requirements
------------

LLAMAS Pyjamas requires **Python 3.11 or 3.12**. Its main dependencies (declared in
``pyproject.toml``) are:

* ``numpy``, ``scipy``, ``astropy``, ``matplotlib`` — core numerics and FITS I/O
* ``pypeit`` — wavelength-calibration algorithms
* ``ray`` + ``cloudpickle`` — parallel processing across cameras
* ``lacosmic`` — cosmic-ray rejection
* ``pyqt6`` — GUI tools (optional at runtime)

From source (recommended)
-------------------------

The project uses `Poetry <https://python-poetry.org/>`_ for dependency management:

.. code-block:: bash

   git clone https://github.com/mit-kavli-institute/llamas-pyjamas.git
   cd llamas-pyjamas
   poetry install

Alternatively, install into an existing environment with ``pip``:

.. code-block:: bash

   git clone https://github.com/mit-kavli-institute/llamas-pyjamas.git
   cd llamas-pyjamas
   pip install -e .

Using a conda environment
-------------------------

.. code-block:: bash

   conda create -n llamas python=3.12
   conda activate llamas
   pip install -e .

Verifying the installation
--------------------------

.. code-block:: python

   import llamas_pyjamas
   print(llamas_pyjamas.__version__)

Troubleshooting
---------------

* **PypeIt** has heavy, version-sensitive dependencies; if installation fails, install it
  on its own first (``pip install pypeit``) and re-run the project install.
* **Ray** signal-handler conflicts with PypeIt are handled automatically by the package
  ``__init__`` (a guarded ``signal.signal`` shim).
