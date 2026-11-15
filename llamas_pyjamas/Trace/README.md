
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

A *legacy* (old-format) trace is a pickle written before the tracer checked fibre
registration: the commissioning-era master bundle, or per-night traces made by
v1.0.0. You can tell one by the missing `comb_ok` attribute; only the current tracer
sets it (`is_legacy_trace`).

**They load as-is.** A trace pickle carries its own copy of the tracer class, so an
old pickle unpickles under the current code.

**But an old night's own traces are never reused.** Since v1.0.1 the tracer will not
save a comb it cannot register. Before saving it runs three checks:
- slot resolution against the fibremap's live slots (`resolve_trace_slots`);
- comb regularity (`validate_trace_comb`);
- the fibre-throughput fingerprint (`check_fingerprint_registration`, against
  `LUT/trace_fingerprints.json`).

An old trace was accepted on fibre count alone, and a misregistered comb has the right
count: the old 2A mis-trim kept 298 fibres with up to 298 of them on the wrong
lenslet. Re-checking only the centre column is not enough either. An old trace can
still carry faults the current tracer fixes away from the centre, such as a faint
fibre creeping onto a bright neighbour.

So when `reduce.py` finds old-format traces in `trace_output_dir`, it does this:
1. It prints them ("written by an older pipeline version and are not reused").
2. It checks the flats are set (`red_flat_file`, `green_flat_file`, `blue_flat_file`).
   If any is missing, it **stops with nothing changed** and names the keys to set.
3. It renames the old files to `<name>.pkl.legacy`. If that name is taken, it uses
   `<name>.pkl.legacy.1`, `.2`, and so on; an earlier backup is never overwritten.
4. It **re-traces the night from its flats** with the current tracer
   (`traceLlamasMaster.run_ray_tracing`).

The `.legacy` files are not read by the pipeline. Delete them once you no longer
need them.

### Old-format masters are audited

The mastercalib fallback traces are different. An old-format master, e.g.
`comissioning_mastercalib`, is still usable if it passes the same checks the tracer
runs before saving (`audit_trace_registration`, using the trace positions and the
flat stored in the pickle). A copy of a master in a trace directory counts as that
master if it is byte-identical to it. So a fallback copied in by an earlier run is
re-checked as a master, not taken for an old night's trace.

### The mastercalib fallback is per camera, and must be usable

Every route that puts a master trace in place for a camera goes through
`find_usable_master_trace`. The master must have that benchside's fibre count and, if
it is old-format, pass the registration audit.

| Route | Where | The master is |
|---|---|---|
| a camera's own trace has the wrong count, or the camera has no trace, and its flat is set | `validate_and_fix_trace_fibres` | **shifted onto the night's flat** (below) |
| the same, without a flat for that camera | `validate_and_fix_trace_fibres` | copied as it is |
| a placeholder (missing) camera in the flats | `reduce.copy_mastercalib_traces_for_placeholders` | copied as it is (there is no data) |
| no traces and no flats: the mastercalib directory used in place | `count_trace_fibres`, then per camera | used as it is |

A resumed run with an empty `trace_output_dir` traces the night from its flats if
they are set. Only without flats does it go straight to mastercalib.

A master that fails is never copied in, and a master that cannot be placed on the
night's flat is never used unshifted. If that leaves a camera with no usable trace,
`reduce.py` lists the affected cameras and the reasons in the terminal and asks:

```
Proceed without these cameras, or terminate the run? [p/t]:
```

- **p (proceed)**: those cameras are skipped. A bad trace of their own is set aside
  as `<name>.pkl.legacy`, so extraction finds no trace and their fibres are blank in
  the RSS. No misregistered fibres reach the data.
- **t (terminate)**: the whole run stops, and no files have been changed.
- **No terminal** (GUI, batch script, `nohup`): there is nobody to answer, so the
  run is terminated with a message naming the cameras.

**Behaviour change:** before the shifted master, a camera that failed to trace got
the master unshifted. Now it gets the shifted master, or, if that cannot be placed
with confidence, the prompt above. So a non-interactive run can now stop where it
used to carry on with misplaced traces.

### Shifted mastercalib fallback

The fibres move between epochs. From the commissioning masters to the 2026 masters,
red and green shifted by 0–4.4 px. Blue shifted by up to 35 px, compressed the slit by
0.3–0.5% (5–9 px end to end), tilted by up to 17 px across x, and changed its
distortion by a further 0.2–0.4 px. The extraction window is ±2 px at a 6.4 px pitch,
so an unshifted master on another night extracts the neighbouring fibre, or one
several fibres away.

So when a camera falls back and its flat is set, `Trace/masterShift.py` moves the
master onto that flat (`build_shifted_trace`):

1. **Offset within one pitch.** On 15 columns × 6 slit segments, the tracer's own
   valley-subtracted comb is summed at the master positions shifted by δ, for δ
   across one pitch. A plane (shift + stretch + tilt) is fitted to those offsets
   modulo the pitch. The template is the master's *positions*, not its stored flat:
   fibre throughput alternates by ±10% between neighbours, so a flat template
   favours two-fibre aliases. That trap is what broke v1.0.0's `cross_correlate_combs`.
2. **Which fibre is which** (offsets of −10 to +10 fibres are tried). The comb repeats
   every pitch, so this is decided by evidence that doesn't:
   - **the slit ends**: every predicted trace must sit on a fibre, and the first live
     slot beyond each end must be dark. Dead slots are skipped, so the 2A dead-299
     leak does not count as a fibre. An end within a pitch of the detector edge has no
     vote. Exactly one offset must pass.
   - **the throughput fingerprint** (`LUT/trace_fingerprints.json`), which must confirm
     it: r ≥ 0.5, and beating every other offset by ≥ 0.15. The margin matters because
     a two-fibre alias can still correlate at r ≈ 0.5–0.77.
3. **Surface.** Each fibre's offset is measured at every column with a Gaussian
   template, and a bicubic surface in (x, y) is fitted to them. On the commissioning
   ↔ 2026 masters it fits to ~0.03 px; a plane leaves up to 0.4 px in blue.
4. **Gates.**
   - Every fibre must sit on its shifted trace: per-fibre median offset ≤ 0.3 px (the
     tracer's own `maxdev`), camera median ≤ 0.15 px, with ≥ 90% of the offsets
     measurable.
   - Every trace must be ≥ 3 px inside the detector over the traced columns.
   - The profile model is rebuilt on the night's flat (`profileFit`).
   - The comb check must pass, and so must `audit_trace_registration` against the
     night's flat.

The result is a normal current-format trace pickle, written as
`LLAMAS_{colour}_{bench}_{side}_traces.pkl`, or over the camera's own wrong-count trace.
It carries a `master_shift` record: the master, the flat, the surface, the whole-fibre
offset and the fingerprint r. A resumed run reuses it as a valid trace. The terminal
and the end-of-run report list it, e.g.
`✓ red3A: mastercalib trace shifted onto this night's flat (+4.40 px, stretch +0.3 px, tilt +1.0 px; fibres within 0.05 px; fingerprint r 0.98)`.

If any step fails, the camera goes to the prompt above with the reason.

**The fingerprint switch.** By default the fingerprint must confirm the slit-end match.
On a weak flat it may be inconclusive. If so, the reason names the switch, and so does
the message when the run terminates:

```
shifted_master_require_fingerprint = false
```

With it set to false, the slit ends alone decide. The fingerprint then only refuses
a confident mismatch, as the tracer's own check does.

**Verified** on the commissioning ↔ 2026 masters (48 cases, each master shifted onto
the other epoch's stored flat):
- the whole-fibre offset was right in every case, including blue shifts of ±27 and
  ±35 px;
- no case was accepted off the truth;
- three were refused:
  - blue1B and green4B, commissioning → 2026: those commissioning masters have fibres
    up to 0.5 px off their own flat;
  - red1B, 2026 → commissioning: a fibre on the commissioning flat comes within
    2.0 px of the detector edge, inside the 3 px gate.

72 negative controls (rolled flats, masked slit ends, the true offset removed, noise)
gave no wrong placement.

On the 2025-05-23 night, each 2026 master was shifted onto that night's flat and
compared with the night's own trace from the current tracer. That covers blue1A at
+38.6 px, blue4A at −29.2 px and a 16 px tilt. Across x = 200–1948, the traces agree to
a median of 0.01–0.03 px and a 95th percentile ≤ 0.12 px. The exception is red1B,
where the night's own trace of fibre 2 runs 15 px off its fibre at low x, and the
shifted master does not.

### Where the code lives

`Trace/traceValidation.py` holds the trace-set checks and the fallback
(`validate_and_fix_trace_fibres`, `find_usable_master_trace`,
`copy_mastercalib_trace`, `write_shifted_master`, `count_trace_fibres`,
`confirm_unusable_master_traces`, `set_aside_legacy_traces`,
`build_trace_fingerprints`). `Trace/masterShift.py` places a master on a night's flat;
it leaves `traceLlamasMaster.py` untouched, so re-traces still match the code that
built the 2026 masters. They used to be in
`Utils/utils.py`. They moved so they can import the tracer at module top: the tracer
imports `Utils`, so importing it from `utils.py` was circular.

`from llamas_pyjamas.Trace import ...` gives the current tracer (`traceLlamasMaster`).
The v1.0.0 tracer (`traceLlamasMulti`) is only reachable by its module path.

**v1.0.0 cannot read traces written by v1.0.1 or later; upgrade.** New pickles
refer to `llamas_pyjamas.Bias.BiasCameraMissingError`, which v1.0.0 does not have.

`traceLlamasMaster_backup_combinecube.py` is an unmodified copy of the tracer as it
was on the `combinecube` branch (commit `dcc6fdc`), before the legacy audit was
added. It is kept for reference only.
