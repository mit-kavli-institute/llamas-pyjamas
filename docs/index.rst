LLAMAS Pyjamas
==============

**LLAMAS Pyjamas** is the Python data-reduction pipeline for the LLAMAS integral-field
spectrograph at the Magellan Telescopes. It takes raw multi-extension CCD frames through
bias subtraction, fibre tracing, optimal extraction, wavelength calibration, flat
fielding, and white-light / data-cube construction to produce science-ready spectra.

Features
--------

* **End-to-end reduction** — from raw CCD images to extracted, wavelength-calibrated spectra and cubes.
* **Multi-channel** — handles the blue, green, and red spectrograph channels.
* **Optimal extraction** — fibre profile fitting and optimal extraction for maximum S/N.
* **Wavelength calibration** — Thorium–Argon arc line identification and wavelength solutions.
* **White-light imaging & cubes** — reconstruct white-light images and IFU data cubes from extractions.
* **Quality assurance** — diagnostic plots and QA tooling.
* **Parallel processing** — Ray-based parallelism across cameras.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

API coverage
------------

The API Reference is generated automatically from the package docstrings. It covers the
importable pipeline subpackages — ``Arc``, ``Bias``, ``Combine``, ``Cube``, ``CubeViewer``,
``DataModel``, ``Extract``, ``File``, ``Flat``, ``Flux``, ``Image``, ``Masking``, ``QA``,
``Sky``, ``Trace``, ``Utils`` — and the top-level ``config``, ``constants``, and ``reduce``
modules.

Display-dependent code is not a barrier: ``PyQt6``, ``pyds9`` and the Qt matplotlib backend are
mocked at build time, so ``CubeViewer`` is documented in full.

The ``GUI``, ``Postprocessing`` and ``Scripts`` directories are **not** auto-documented — they
have no ``__init__.py`` and are collections of standalone scripts rather than importable
packages. The ``Tests`` package and the ``Sky/diagnosis`` investigation scripts are excluded as
development tooling, not public API.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
