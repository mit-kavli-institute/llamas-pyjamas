Installation
============

Requirements
------------

LLAMAS PyJamas requires Python 3.8+ and the following dependencies:

Core Dependencies
~~~~~~~~~~~~~~~~~

* numpy
* scipy  
* astropy
* matplotlib
* pypeit (for wavelength calibration)
* ray (for parallel processing)
* cloudpickle (for serialization)

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* pyqt5 (for GUI tools)
* jupyter (for notebook examples)

Installation from Source
-------------------------

1. Clone the repository::

    git clone <repository-url>
    cd llamas-pyjamas

2. Install dependencies::

    pip install -r requirements.txt

3. Install the package::

    pip install -e .

Configuration
-------------

Before running the pipeline, configure your paths in the config file:

1. Copy the example configuration::

    cp example_config.txt my_config.txt

2. Edit the configuration file to set your data paths::

    BASE_DIR=/path/to/your/data
    OUTPUT_DIR=/path/to/output
    CALIB_DIR=/path/to/calibrations

See the :doc:`quickstart` guide for more details on configuration options.