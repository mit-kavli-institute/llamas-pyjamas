llamas-pyjamas Documentation
============================

Welcome to the LLAMAS PyJamas documentation! This package provides a comprehensive 
data reduction pipeline for the LLAMAS (Large Lens Array Multi-Object Spectrograph) 
instrument.

LLAMAS PyJamas handles the complete data reduction workflow from raw observations 
to final data products including RSS files and 3D data cubes.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/modules
   examples

Features
--------

* Fiber trace generation from flat field observations
* Optimal and boxcar spectral extraction
* Wavelength calibration using arc lamp observations  
* Flux calibration and throughput corrections
* RSS (Row-Stacked Spectra) file generation
* 3D data cube construction
* Parallel processing support with Ray
* GUI tools for interactive data inspection

Quick Start
-----------

To get started with LLAMAS PyJamas:

1. Install the package and dependencies
2. Configure your reduction parameters
3. Run the reduction pipeline

See the :doc:`quickstart` guide for detailed instructions.

API Reference
=============

.. toctree::
   :maxdepth: 4
   
   api/llamas_pyjamas

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`