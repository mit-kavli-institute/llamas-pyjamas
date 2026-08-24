# Fiber Tracing Feature

## Overview
The Fiber Tracing feature identifies and maps the spatial positions of individual fiber traces across the detector for all three color channels (red, green, blue). This is essential for extracting accurate 1D spectra from the 2D detector images.

## Core Functionality
- **Peak Detection**: Uses algorithms to identify fiber positions from flat field exposures
- **Profile Fitting**: Fits spatial profiles for each detected fiber
- **Multi-Channel Processing**: Handles red, green, and blue channels simultaneously
- **Ray Parallelization**: Distributed processing for multiple exposures
- **Lookup Table Generation**: Creates fiber position maps for extraction
- **Quality Assessment**: Validates trace quality and completeness

## Key Files
- `Trace/traceLlamasMaster.py` - Main production tracing code
  - `TraceLlamas` class: Core fiber tracing functionality
  - `TraceRay` class: Ray-optimized parallel processing version
  - `run_ray_tracing()`: Entry point for parallel processing
- `Trace/traceLlamas.py` - Legacy tracing code (deprecated)
- `Trace/traceLlamasMulti.py` - Multi-exposure tracing utilities

## Data Structures
- **TraceLlamas Object**: Contains fiber positions, profiles, and metadata
  - `traces`: Array of trace polynomials
  - `tracearr`: 2D trace position arrays
  - `nfibers`: Number of detected fibers
  - `channel`: Color channel (red/green/blue)
  - `bench`, `side`: Instrument configuration identifiers

## Usage Patterns
```python
# Basic tracing
from llamas_pyjamas.Trace.traceLlamasMaster import TraceLlamas
tracer = TraceLlamas(flat_field_file)
tracer.process_hdu_data(hdu_data, hdu_header)
tracer.saveTraces(output_dir)

# Parallel processing
from llamas_pyjamas.Trace.traceLlamasMaster import run_ray_tracing
run_ray_tracing(flat_field_file, outpath=output_dir, channel='red')
```

## Pipeline Integration
Called by `reduce.py:generate_traces()` which processes all three color channels:
- Takes flat field FITS files as input
- Generates trace files saved as pickle objects
- Creates lookup tables in `LUT/` directory
- Used by downstream extraction and QA modules

## Output Products
- **Trace Files**: Pickle files containing TraceLlamas objects
- **Lookup Tables**: JSON files with fiber position mappings
- **QA Plots**: Trace quality visualization (via QA module)

## Configuration
- Trace parameters set in class initialization
- Output directories configurable via pipeline config
- Ray parallelization settings adjustable
- Peak detection thresholds tunable per channel

## Dependencies
- Ray (for parallel processing)
- Astropy (FITS handling)
- NumPy/SciPy (numerical processing)
- Pypeit (core algorithms)
- Matplotlib (debugging plots)

## Performance Notes
- Ray parallelization scales well with CPU cores
- Memory usage ~1-2 GB per channel for typical datasets
- Processing time ~5-15 minutes per channel depending on fiber count
- Pickle serialization requires cloudpickle for Ray compatibility