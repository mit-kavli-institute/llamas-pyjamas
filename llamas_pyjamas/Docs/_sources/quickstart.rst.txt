Quick Start Guide
=================

This guide will walk you through a complete LLAMAS data reduction from start to finish.

Basic Pipeline
--------------

The LLAMAS pipeline follows these main steps:

1. **Trace Generation**: Generate fiber traces from flat field observations
2. **Extraction**: Extract spectra from science observations  
3. **Wavelength Calibration**: Calculate wavelength solutions from arc lamps
4. **Flux Calibration**: Apply throughput corrections
5. **RSS Generation**: Create Row-Stacked Spectra files
6. **Cube Construction**: Build 3D data cubes

Running the Pipeline
--------------------

Complete Reduction
~~~~~~~~~~~~~~~~~~

To run the complete pipeline with a configuration file::

    python reduce.py --config my_config.txt

Step-by-Step Reduction
~~~~~~~~~~~~~~~~~~~~~~

You can also run individual steps:

1. Generate traces::

    from llamas_pyjamas.reduce import generate_traces
    generate_traces('red_flat.fits', 'green_flat.fits', 'blue_flat.fits', 'output/')

2. Extract science spectra::

    from llamas_pyjamas.reduce import run_extraction
    run_extraction('science.fits', 'output/', trace_dir='traces/')

3. Calculate wavelength solution::

    from llamas_pyjamas.reduce import calc_wavelength_soln
    calc_wavelength_soln('arc.fits', 'output/')

Configuration
-------------

The configuration file specifies data paths and processing parameters::

    # Data directories
    BASE_DIR=/data/llamas
    OUTPUT_DIR=/data/reduced
    DATA_DIR=/data/raw
    
    # Processing parameters
    BIAS_CORRECT=True
    PARALLEL_PROCESSES=4

See :doc:`installation` for more configuration details.

Example Notebook
----------------

Check out the example notebook at ``Docs/llamas_pyjamas_demo.ipynb`` for an interactive tutorial.