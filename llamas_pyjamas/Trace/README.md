
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

## Legacy trace sets

A *legacy* trace is a pickle written before the tracer checked fibre registration:
the commissioning-era master bundle, or per-night traces made by v1.0.0. You can
tell one by the missing `comb_ok` attribute.

**They load as-is.** A trace pickle carries its own copy of the tracer class, so an
old pickle unpickles under the current code. Every attribute the extraction, flat
and white-light code reads is present in old pickles too.

**They are audited when reused.** Since v1.0.1 the tracer will not save a comb it
cannot register. Before saving it runs three checks:
- slot resolution against the fibremap's live slots (`resolve_trace_slots`);
- comb regularity (`validate_trace_comb`);
- the fibre-throughput fingerprint (`check_fingerprint_registration`, against
  `LUT/trace_fingerprints.json`).

A legacy trace never went through these checks. Before v1.0.1 it was accepted on
fibre count alone, and a misregistered comb has the right count: the old 2A
mis-trim kept 298 fibres with up to 298 of them on the wrong lenslet. So
`validate_and_fix_trace_fibres` runs the same checks on any legacy trace it is
about to reuse (`audit_trace_registration`, using the trace positions and the flat
image stored in the pickle):

- **Pass**: the trace is used unchanged.
- **Fail**: the file is renamed to `<name>.pkl.legacy` and the mastercalib trace
  for that camera is copied in its place. The run reports it under
  "legacy trace(s) fail the fibre-registration check".

To replace a failing legacy trace with a fresh one, re-trace from that night's flats
with `clobber = true`. The `.legacy` file is not read by the pipeline. Delete it once
you no longer need it.

### The mastercalib fallback is per camera, and must be usable

A camera falls back to its mastercalib trace when its own trace is missing, has the
wrong fibre count, or is a misregistered legacy trace. The master is only used if it
is *usable* (`find_usable_master_trace`):
- it has that benchside's fibre count;
- if it is a legacy master, it passes the same registration audit.

A master that fails is never copied in. If that leaves a camera with no usable
trace, `reduce.py` lists the affected cameras and the reasons in the terminal and
asks:

```
Proceed without these cameras, or terminate the run? [p/t]:
```

- **p (proceed)**: those cameras are skipped. A bad trace of their own is set aside
  as `<name>.pkl.legacy`, so extraction finds no trace and their fibres are blank in
  the RSS. No misregistered fibres reach the data.
- **t (terminate)**: the whole run stops, and no files have been changed.
- **No terminal** (GUI, batch script, `nohup`): there is nobody to answer, so the
  run is terminated with a message naming the cameras.

When there are no traces at all to reuse, the mastercalib directory is used in place
only if `count_trace_fibres` passes. Any wrong count or failed registration fails
that check. Otherwise the fallback is made camera by camera as above.

**v1.0.0 cannot read traces written by v1.0.1 or later; upgrade.** New pickles
refer to `llamas_pyjamas.Bias.BiasCameraMissingError`, which v1.0.0 does not have.

`traceLlamasMaster_backup_combinecube.py` is an unmodified copy of the tracer as it
was on the `combinecube` branch (commit `dcc6fdc`), before the legacy audit was
added. It is kept for reference only.
