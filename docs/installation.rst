Installation
============

Requirements
------------

LLAMAS Pyjamas requires Python 3.8 or later and has the following dependencies:

Core Dependencies
~~~~~~~~~~~~~~~~~

* numpy >= 1.21.0
* scipy >= 1.7.0
* matplotlib >= 3.5.0
* astropy >= 5.0.0

Spectroscopy Tools
~~~~~~~~~~~~~~~~~~

* pypeit (for wavelength calibration algorithms)

Data Handling
~~~~~~~~~~~~~

* h5py >= 3.0.0
* pandas >= 1.3.0

Parallel Processing
~~~~~~~~~~~~~~~~~~~

* ray >= 1.0.0 (optional, for parallel extraction)

GUI Components
~~~~~~~~~~~~~~

* PyQt5 >= 5.15.0 (optional, for GUI tools)

Installation Methods
--------------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/your-org/llamas-pyjamas.git
   cd llamas-pyjamas
   pip install -e .

This allows you to modify the code and see changes immediately.

Using pip
~~~~~~~~~

Once available on PyPI:

.. code-block:: bash

   pip install llamas-pyjamas

Conda Environment
~~~~~~~~~~~~~~~~~

For a clean installation, create a new conda environment:

.. code-block:: bash

   conda create -n llamas python=3.9
   conda activate llamas
   pip install -e .

Verification
------------

Test your installation by importing the package:

.. code-block:: python

   import llamas_pyjamas
   from llamas_pyjamas.Trace import traceLlamas
   from llamas_pyjamas.Extract import extractLlamas
   
   print("LLAMAS Pyjamas successfully installed!")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**pypeit installation fails**: pypeit has specific dependencies. Try installing it separately:

.. code-block:: bash

   pip install pypeit

**Qt/GUI issues**: If you don't need GUI functionality, the package will work without PyQt5.

**Ray import errors**: Ray is optional. The package will work without parallel processing capabilities.

Development Installation
------------------------

For developers who want to contribute:

.. code-block:: bash

   git clone https://github.com/your-org/llamas-pyjamas.git
   cd llamas-pyjamas
   pip install -e ".[dev]"

This installs additional development dependencies for testing and documentation.