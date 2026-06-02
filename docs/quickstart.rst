Quick Start
===========

The LLAMAS pipeline is driven by a plain-text configuration file that points to your
calibration and science frames. Reduction is launched through :mod:`llamas_pyjamas.reduce`.

Pipeline stages
---------------

A full reduction runs these stages, in order:

1. **Bias subtraction** — build/apply a master bias for the detector read mode.
2. **Flat fielding** — trace and normalise twilight/dome flats; optionally apply the correction.
3. **Fibre tracing** — locate and trace fibre positions across each detector.
4. **Spectral extraction** — optimal extraction of 1D spectra from the traced fibres.
5. **Wavelength calibration** — solve and apply a wavelength solution from arc frames.
6. **RSS / white-light / cube construction** — assemble row-stacked spectra and data cubes.

Configuration file
------------------

Create a configuration file (e.g. ``my_reduction.txt``). Paths must be **complete
(absolute) paths**. If a file cannot be found or processed, the pipeline falls back to the
packaged master calibration files.

.. code-block:: text

   # Master bias frames (per read mode)
   slow_bias_file = /path/to/LLAMAS_..._CAL_mef.fits
   fast_bias_file = /path/to/LLAMAS_..._CAL_mef.fits

   # Twilight / dome flats per channel
   red_twilight_flat   = /path/to/flat_mef.fits
   green_twilight_flat = /path/to/flat_mef.fits
   blue_twilight_flat  = /path/to/flat_mef.fits

   red_flat_file   = /path/to/flat_mef.fits
   green_flat_file = /path/to/flat_mef.fits
   blue_flat_file  = /path/to/flat_mef.fits

   # Science frame(s) to reduce (comma-separate for batch processing)
   science_files = /path/to/LLAMAS_..._SCI_mef.fits

   # Processing options
   apply_flat_field_correction = True
   cube_method = simple

   # Output directories (created automatically if missing)
   trace_output_dir      = /path/to/output/traces
   extraction_output_dir = /path/to/output/extractions
   cube_output_dir       = /path/to/output/cubes

   # Optional: Ray parallelism (defaults to all CPU cores)
   ray_num_cpus = 8

Running the pipeline
--------------------

Run the reduction by passing the configuration file to the ``reduce`` module:

.. code-block:: bash

   python -m llamas_pyjamas.reduce my_reduction.txt

The pipeline validates the configuration before processing (checking required keys,
master-bias availability, and that input paths exist), then writes traces, extractions,
RSS files, and cubes to the configured output directories.

Next steps
----------

* Browse the :doc:`api/modules` reference for detailed module, class, and function documentation.
* Key entry points: :mod:`llamas_pyjamas.reduce` (orchestration),
  :mod:`llamas_pyjamas.Trace` (tracing), :mod:`llamas_pyjamas.Extract` (extraction),
  :mod:`llamas_pyjamas.Arc` (wavelength calibration), and :mod:`llamas_pyjamas.Cube`
  (cube construction).
