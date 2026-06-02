# Claude Code Documentation

This directory contains comprehensive feature documentation for the LLAMAS pipeline, specifically designed for use with Claude Code (claude.ai/code). Each file provides detailed technical information about a major pipeline feature.

## Purpose

These documentation files are created to help Claude Code understand and work with the LLAMAS pipeline features quickly and effectively. Each document contains:

- Overview and core functionality
- Key files and data structures  
- Usage patterns and code examples
- Pipeline integration details
- Configuration options and dependencies
- Performance notes and quality metrics

## Feature Documentation

### Core Data Reduction Features
- **FIBER_TRACING.md** - Fiber position identification and mapping
- **SPECTRUM_EXTRACTION.md** - 1D spectrum extraction with optimal/boxcar methods
- **WAVELENGTH_CALIBRATION.md** - Pixel-to-wavelength conversion using arc lamps
- **FLAT_FIELD_PROCESSING.md** - Pixel-to-pixel sensitivity corrections with per-fiber normalization
- **BIAS_CORRECTION.md** - Electronic offset removal from CCD readout

### Data Product Generation
- **RSS_FILE_GENERATION.md** - Row-Stacked Spectra FITS file creation
- **CUBE_CONSTRUCTION.md** - 3D data cube construction with advanced CRR method
- **WHITE_LIGHT_IMAGING.md** - 2D image reconstruction from spectroscopic data

### Data Validation and Quality Control
- **DATA_VALIDATION.md** - FITS structure validation, missing extension handling, trace fallback mechanisms
- **QUALITY_ASSURANCE.md** - Comprehensive QA visualization and validation
- **THROUGHPUT_ANALYSIS.md** - System efficiency and flux calibration calculations

### User Interface and Utilities
- **GUI_INTERFACE.md** - Interactive graphical interface for pipeline operations
- **POSTPROCESSING_UTILS.md** - Standalone FITS arithmetic and quick white light preview tools

## Usage

These files are referenced from the main `CLAUDE.md` file in the repository root and provide the detailed context needed for Claude Code to work effectively with each pipeline feature.

## Maintenance

When adding new features to the pipeline, create corresponding documentation files in this directory following the established format and update the references in `CLAUDE.md`.