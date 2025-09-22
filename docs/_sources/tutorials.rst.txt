Tutorials
=========

This section provides detailed tutorials for using LLAMAS Pyjamas in various scenarios.

Basic Tutorials
---------------

.. toctree::
   :maxdepth: 2

   tutorial_basic_reduction
   tutorial_advanced_extraction
   tutorial_wavelength_calibration
   tutorial_whitelight_images

Tutorial 1: Basic Data Reduction
---------------------------------

This tutorial covers the fundamental steps of LLAMAS data reduction from raw FITS files to extracted spectra.

**Prerequisites:**
- Raw LLAMAS FITS files (bias, flat, science, arc)
- LLAMAS Pyjamas installed and working

**Learning Objectives:**
- Understand the LLAMAS data reduction pipeline
- Learn to process bias and flat field frames
- Perform fiber tracing and spectral extraction
- Generate quality assurance plots

**Tutorial Files:**
The tutorial uses sample data that can be downloaded from the LLAMAS data repository.

Step 1: Data Organization
~~~~~~~~~~~~~~~~~~~~~~~~~

First, organize your data files:

.. code-block:: bash

   data/
   ├── bias/
   │   ├── bias_001.fits
   │   ├── bias_002.fits
   │   └── bias_003.fits
   ├── flats/
   │   ├── flat_blue_001.fits
   │   ├── flat_green_001.fits
   │   └── flat_red_001.fits
   ├── science/
   │   └── target_observation.fits
   └── arcs/
       └── thar_calibration.fits

Step 2: Load Required Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   import numpy as np
   import matplotlib.pyplot as plt
   
   # LLAMAS modules
   from llamas_pyjamas.Bias import llamasBias
   from llamas_pyjamas.File import llamasIO
   from llamas_pyjamas.Trace import traceLlamas
   from llamas_pyjamas.Extract import extractLlamas
   from llamas_pyjamas.QA import llamasQA

Step 3: Process Bias Frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # List all bias files
   bias_files = ['data/bias/bias_001.fits', 'data/bias/bias_002.fits', 'data/bias/bias_003.fits']
   
   # Create master bias
   bias_processor = llamasBias.BiasLlamas()
   master_bias = bias_processor.master_bias(bias_files)
   
   # Quality check: plot master bias
   plt.figure(figsize=(10, 8))
   plt.imshow(master_bias, vmin=0, vmax=50, cmap='viridis')
   plt.colorbar(label='ADU')
   plt.title('Master Bias Frame')
   plt.show()

Continue with the remaining steps...

Tutorial 2: Advanced Extraction Techniques
-------------------------------------------

This tutorial explores advanced spectral extraction methods and optimization techniques.

**Topics Covered:**
- Optimal vs. boxcar extraction comparison
- Profile fitting optimization
- Handling problematic fibers
- Parallel processing with Ray

Tutorial 3: Wavelength Calibration
-----------------------------------

Learn to perform precise wavelength calibration using ThAr arc lamps.

**Topics Covered:**
- Arc line identification
- Wavelength solution fitting
- Quality assessment of calibrations
- Troubleshooting calibration issues

Tutorial 4: White Light Image Generation
-----------------------------------------

Create publication-quality white light images and fiber maps.

**Topics Covered:**
- Generating white light FITS files
- Creating fiber maps and overlays
- Advanced visualization techniques
- Exporting images for publication

Advanced Topics
---------------

.. toctree::
   :maxdepth: 2

   tutorial_custom_reductions
   tutorial_parallel_processing
   tutorial_qa_analysis
   tutorial_data_products

Tutorial 5: Custom Reduction Scripts
-------------------------------------

Learn to create custom reduction scripts for specific observing programs.

Tutorial 6: Parallel Processing
--------------------------------

Optimize processing speed using Ray for large datasets.

Tutorial 7: Quality Assurance Analysis
---------------------------------------

Comprehensive QA analysis and problem diagnosis.

Tutorial 8: Advanced Data Products
-----------------------------------

Generate advanced data products and perform scientific analysis.

Interactive Notebooks
----------------------

Interactive Jupyter notebooks are available in the ``llamas_pyjamas/Tutorials/`` directory:

- ``llamas_extraction_demo.ipynb`` - Basic extraction walkthrough
- ``LUT_and_master_cals.ipynb`` - Working with lookup tables and master calibrations
- ``cal_plots.ipynb`` - Calibration plotting and analysis

To run these notebooks:

.. code-block:: bash

   cd llamas_pyjamas/Tutorials
   jupyter notebook

Sample Data
-----------

Sample datasets for tutorials can be downloaded from:

- Basic reduction dataset (500 MB): ``tutorials/basic_reduction_data.tar.gz``
- Advanced extraction dataset (1.2 GB): ``tutorials/advanced_extraction_data.tar.gz``
- Full observing sequence (3.5 GB): ``tutorials/full_sequence_data.tar.gz``

Each dataset includes:
- Raw FITS files
- Expected output products
- Validation scripts
- Detailed README with observing conditions

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Fiber tracing fails
**Solution:** Check flat field quality and adjust detection thresholds

**Problem:** Poor wavelength calibration
**Solution:** Verify arc lamp exposure time and line identification

**Problem:** Extraction artifacts
**Solution:** Examine profile fits and adjust extraction parameters

**Problem:** Memory issues with large datasets
**Solution:** Enable parallel processing or process subsets

Getting Help
~~~~~~~~~~~~

If you encounter issues with the tutorials:

1. Check the troubleshooting section above
2. Review the API documentation for function details
3. Post questions on the GitHub issues page
4. Contact the development team