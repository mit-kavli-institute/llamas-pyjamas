Welcome to LLAMAS Pyjamas documentation!
=========================================

LLAMAS Pyjamas is a comprehensive Python package for processing and analyzing data from the 
LLAMAS (Large Lens Array Multi-Object Spectrograph) instrument. This package provides a 
complete pipeline for spectroscopic data reduction, including tracing, extraction, 
wavelength calibration, and analysis tools.

Features
--------

* **Complete Data Reduction Pipeline**: From raw CCD images to science-ready spectra
* **Multi-Channel Support**: Handles blue, green, and red spectrograph channels
* **Optimal Extraction**: Advanced fiber extraction algorithms for maximum signal-to-noise
* **Wavelength Calibration**: Thorium-Argon arc line identification and wavelength solutions
* **White Light Imaging**: Generate white light images and fiber maps from extracted spectra
* **Quality Assurance**: Comprehensive QA tools and visualizations
* **Parallel Processing**: Ray-based parallel processing for improved performance

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/llamas-pyjamas.git
   cd llamas-pyjamas
   
   # Install dependencies
   pip install -e .

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from llamas_pyjamas.Trace import traceLlamas
   from llamas_pyjamas.Extract import extractLlamas
   from llamas_pyjamas.Arc import arcLlamas
   
   # Trace fiber positions
   trace = traceLlamas.TraceLlamas()
   trace.traceSingleCamera(data_object)
   
   # Extract spectra
   extractor = extractLlamas.ExtractLlamas(trace, hdu_data, header)
   
   # Wavelength calibration
   arcLlamas.arcSolve(arc_extraction_file)

Package Overview
----------------

The LLAMAS Pyjamas package is organized into several modules:

Core Processing Modules
~~~~~~~~~~~~~~~~~~~~~~~

* :doc:`llamas_pyjamas.Trace <llamas_pyjamas.Trace>` - Fiber tracing and profile fitting
* :doc:`llamas_pyjamas.Extract <llamas_pyjamas.Extract>` - Spectral extraction algorithms
* :doc:`llamas_pyjamas.Arc <llamas_pyjamas.Arc>` - Wavelength calibration and arc line identification
* :doc:`llamas_pyjamas.Flat <llamas_pyjamas.Flat>` - Flat field processing and correction
* :doc:`llamas_pyjamas.Bias <llamas_pyjamas.Bias>` - Bias subtraction and dark current correction

Data Handling and I/O
~~~~~~~~~~~~~~~~~~~~~

* :doc:`llamas_pyjamas.File <llamas_pyjamas.File>` - FITS file I/O and data handling
* :doc:`llamas_pyjamas.Image <llamas_pyjamas.Image>` - White light image generation and processing

Analysis and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~

* :doc:`llamas_pyjamas.QA <llamas_pyjamas.QA>` - Quality assurance and diagnostic tools
* :doc:`llamas_pyjamas.Flux <llamas_pyjamas.Flux>` - Flux calibration and throughput analysis
* :doc:`llamas_pyjamas.Utils <llamas_pyjamas.Utils>` - Utility functions and helpers

Post-Processing
~~~~~~~~~~~~~~~

* :doc:`llamas_pyjamas.Postprocessing <llamas_pyjamas.Postprocessing>` - Advanced data products

User Interface
~~~~~~~~~~~~~~

* :doc:`llamas_pyjamas.GUI <llamas_pyjamas.GUI>` - Graphical user interface components

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   quickstart
   tutorials
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   llamas_pyjamas.Trace
   llamas_pyjamas.Extract
   llamas_pyjamas.Arc
   llamas_pyjamas.Flat
   llamas_pyjamas.Bias
   llamas_pyjamas.File
   llamas_pyjamas.Image
   llamas_pyjamas.QA
   llamas_pyjamas.Flux
   llamas_pyjamas.Utils
   llamas_pyjamas.Postprocessing
   llamas_pyjamas.GUI

.. toctree::
   :maxdepth: 1
   :caption: Development:

   contributing
   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

About LLAMAS
============

The Large Lens Array Multi-Object Spectrograph (LLAMAS) is an innovative instrument designed 
for efficient multi-object spectroscopy. This Python package provides the complete data 
reduction pipeline for LLAMAS observations, enabling astronomers to extract high-quality 
scientific spectra from raw CCD images.

For more information about the LLAMAS instrument and its scientific applications, please 
refer to the instrument documentation and published papers.

Support
=======

If you encounter any issues or have questions about using LLAMAS Pyjamas, please:

* Check the documentation and examples
* Search existing issues on GitHub
* Create a new issue with detailed information about your problem

License
=======

LLAMAS Pyjamas is distributed under the MIT License. See the LICENSE file for more details.