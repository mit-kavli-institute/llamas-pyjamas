Quick Start
===========

The LLAMAS pipeline is driven by a plain-text configuration file that points to your
calibration and science frames. Reduction is launched through :mod:`llamas_pyjamas.reduce`.

Pipeline stages
---------------

A full reduction runs these stages, in order:

1. **Validate** — check the calibration files and repair frames with missing camera extensions.
2. **Bias subtraction** — apply the master bias matching the detector read mode, plus a per-frame
   edge DC offset measured from unilluminated pixels.
3. **Fibre tracing** — locate and trace fibre positions across each detector.
4. **Pixel flat** — build the 2-D detector response from the lamp flats and divide it out.
5. **Spectral extraction** — boxcar (default) or optimal extraction of every fibre, with
   cosmic-ray removal, and a white-light image for quick inspection.
6. **Wavelength calibration** — transfer the ThAr arc solution onto each extraction.
7. **Sky subtraction** — build and subtract a per-fibre sky model (``sky_subtract``).
8. **Heliocentric correction** — shift the wavelength scale to the barycentric frame.
9. **RSS generation** — assemble the spectra into one row-stacked-spectra file per colour.
10. **Fibre flat** — apply the fibre-to-fibre throughput correction.
11. **Consolidate + QA** — collapse the per-stage files into one RSS per colour and run
    wavelength QA.

LLAMAS has no dome flats: the inputs are **lamp flats and twilight flats**.

Stacking multiple dithers of a field into a deep cube happens *afterwards*, interactively in the
CubeViewer or from the command line — see the end-to-end workflow guide linked below.

Configuration file
------------------

Create a configuration file (e.g. ``my_reduction.txt``). Paths must be **complete
(absolute) paths**. If a file cannot be found or processed, the pipeline falls back to the
packaged master calibration files.

.. code-block:: text

   # Master bias frames (per read mode)
   slow_bias_file = /path/to/LLAMAS_..._CAL_mef.fits
   fast_bias_file = /path/to/LLAMAS_..._CAL_mef.fits

   # Twilight flats per channel
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

* **Read the** `end-to-end workflow guide
  <https://github.com/mit-kavli-institute/llamas-pyjamas/blob/documentation/docs/workflow/README.md>`_
  — a task-oriented walkthrough from a directory of raw frames to a stacked, dithered field,
  covering registration, combining dithers and science extraction.
* Browse the :doc:`api/modules` reference for detailed module, class, and function documentation.
* Key entry points: :mod:`llamas_pyjamas.reduce` (orchestration),
  :mod:`llamas_pyjamas.Trace` (tracing), :mod:`llamas_pyjamas.Extract` (extraction),
  :mod:`llamas_pyjamas.Arc` (wavelength calibration), and :mod:`llamas_pyjamas.Cube`
  (cube construction).
