"""
CubeViewer — an interactive viewer for reduced LLAMAS data products.

CubeViewer is a standalone PyQt6 application for postprocessing analysis. It displays
collapsed ("white-light") images in SAOImageDS9 over a user-selected wavelength window,
and plots the spectrum of whatever spatial element the user picks in the DS9 window.

DS9 owns the image display (stretch, colourmap, zoom, pan); CubeViewer owns the data,
the wavelength selection, and the spectral plot.

This package is deliberately independent of the QuickLook/observing-log GUI in
``llamas_pyjamas.GUI``. It imports read-only helpers from the reduction code
(``Image.WhiteLightModule``, ``Utils.deadfibers``) but modifies nothing.

Modules
-------
cubeViewDS9      DS9 transport over the XPA messaging system
cubeViewScene    SpectralScene interface — the geometry-agnostic data abstraction
cubeViewRSS      SpectralScene implementation for row-stacked spectra (fibres)
cubeViewCube     SpectralScene implementation for rectilinear cubes (spaxels)
cubeViewPick     DS9 crosshair polling, translated into scene queries
cubeViewSpecPlot Embedded matplotlib spectrum panel
cubeViewLlamas   Main window and application entry point
"""
