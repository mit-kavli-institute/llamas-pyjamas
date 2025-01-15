
# Trace Module

This module handles the fibre tracing functionality for the LLAMAS instrument. It identifies and maps the positions of individual fibre traces across the detector.

## Core Functionality

The tracing module performs several key operations:
- Identifies fibre positions using peak detection algorithms
- Fits trace profiles for each fibre
- Creates fibre position lookup tables
- Saves trace information for extraction

## Key Files

### 

traceLlamasMaster.py


Current production version of the tracing code. Contains:
- 

TraceLlamas

 class: Main class for fibre tracing
- 

TraceRay

 class: Ray-optimised version for parallel processing
- 

run_ray_tracing()

: Entry point for parallel trace processing

### 

traceLlamas.py

 (Deprecated)
Original version of the tracing code. Retained for reference but should not be used for new development.

## Usage

The tracing module is typically used as part of the LLAMAS data reduction pipeline:

```python
from llamas_pyjamas.Trace.traceLlamasMaster import TraceLlamas

# Create trace object
tracer = TraceLlamas(fitsfile)

# Process data
tracer.process_hdu_data(hdu_data, hdu_header)

# Save traces
tracer.saveTraces()
```

For parallel processing of multiple exposures, use the Ray-enabled version:

```python
from llamas_pyjamas.Trace.traceLlamasMaster import run_ray_tracing

run_ray_tracing(fitsfile)
```