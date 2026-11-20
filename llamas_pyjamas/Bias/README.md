# Bias Module

Bias and dark-current removal for LLAMAS. The CCDs have no overscan. Every raw frame
goes through **bias-first** (`biasFirst.py`) before anything else touches it. Bias-first
writes `{output_dir}/bias_corrected/{frame}_bias_corrected.fits`, and every later stage
reads that file.

## Order of operations, per camera

Bias-first works in **pipeline orientation**: the DATASEC trim plus the colour flip of
`File/llamasIO.process_fits_by_color` (blue is flipped in x and y, green in x). That is the
orientation of the master bias, the master dark and the trace masks. The result is flipped
back to raw orientation for writing, so later stages read the file exactly as they read a raw
frame.

1. **Master bias.** The READ-MDE-matched 2D master bias (`slow_bias_file` /
   `fast_bias_file`), found by `guiExtract.select_bias_for_extension`. If the file lacks the
   camera, a constant level is estimated from the frame itself instead.
2. **Master dark.** Applied to science, flux-standard and sky frames that have a matched dark.
   `scale × master dark` is subtracted (see below).
3. **Edge DC.** The sigma-clipped median of the unlit stripes above the top fibre and below
   the bottom fibre (more than 20 px from any fibre) is subtracted as a constant.
   - Without a dark, this constant is mostly dark current: 0–65 DN on 1350 s frames. It cannot
     remove structured dark signal, such as the glow inside the fibre stack.
   - After a matched dark, it is a residual of about ±1 DN. It still removes any drift of the
     bias since the darks were taken, plus the constant a scaled dark leaves behind.

Flats, arcs and twilights are short and never get a dark; they have steps 1 and 3 only.
Before BFVER 2, the raw frame was corrected directly, which mirrored the bias and the stripe
mask on green and blue cameras. At that time 17.7% of the blue mask lay on fibres.

## Darks

**Listing them.** Darks are listed in the config as `dark_files`, or tagged with the Dark
button in `Utils/reduxSetupGUI`.
- A listed file whose `PRODCATG` is not `CAL.R-DRK` stops the run at pre-flight.
- Darks are grouped by read mode (READ-MDE; a missing one counts as SLOW) and requested
  exposure (`REXPTIME`).

**Matching.** A science, standard or sky frame uses the group with its read mode and a
`REXPTIME` within `dark_exptime_tolerance` (default 0.10) of its own; the nearest group wins.
- `scale` = the frame's `DEXPTIME` / the median `DEXPTIME` of the darks. `DEXPTIME` is the
  total dark accumulation time, about 1 s more than the shutter time.
- If either side lacks `DEXPTIME`, the ratio of `REXPTIME` values is used instead.
- A scale more than 5% from 1 is warned about, since the exposure may have been cut short.
- A frame with no match is reduced with the bias and edge DC only, and the end-of-run
  report says so.

**Master dark** (`darkLlamas.build_master_dark`). Per camera, the dark frames are combined,
then the **same** master-bias extension that bias-first subtracts is removed.
- Because the darks contain the bias, a matched dark (scale ≈ 1) also removes any difference
  between that master bias and the night's true bias.
- Combining:

  | Darks | Method |
  |---|---|
  | 3 or more | Median |
  | 2 | Mean, except where the frames differ by more than 5σ; there the lower value is used (rejects cosmic rays) |
  | 1 | 5×5 median filter, so its cosmic rays are not subtracted as holes; hot pixels are then not corrected |

  Use at least 3.
- The result is written in raw orientation to
  `{output_dir}/darks/master_dark_{MODE}_{texp}s_{DARKID}.fits`, and reused while `DARKID` (a
  hash of the darks, the bias and the algorithm version) is unchanged.
- A camera is left out of the master dark, and reported, if it has:
  - no usable dark frames;
  - no master bias;
  - a constant master bias, i.e. a fill and not a measurement (the shipped masters' blue 1A
    and 4A are like this);
  - darks whose detector serials disagree.
- A QA grid of the master dark is written to `qa_output_dir`.

**Extension cross-check.** A dark extension is matched to the frame's camera by header and
re-confirmed, exactly as the master bias is (`Bias.match_camera_hdu` and
`Bias.verify_camera_hdu`):
- COLOR/BENCH/SIDE, then CAM_NAME, then a header-verified index;
- a re-check of bench, side and colour just before subtraction.

For a dark, the detector serial `CAMSN`, the read mode, the shape and the `DARKID` must also
agree. If any check fails, that camera gets no dark, with the reason in `DARKWHY`; it never
gets another camera's dark.

**Checks per camera**, on the unlit stripes (the reference notebook's checks):
- **`DARKRES`**: the stripe level after bias and dark. Flagged above 5 DN; this includes any
  bias drift since the darks.
- **`DARKNRAT`**: the stripe noise after / before the dark. Flagged above `1.1 × √(1 + c/N)`,
  because subtracting an N-frame master adds that much noise: c = π/2 for a median, 1 for a
  mean.
- **`DARKNLOW`**: the rise in pixels below level − 6σ. Flagged above 500.

Flags appear in `DARKFLAG` and in the end-of-run report. They are never acted on.

**Errors.** The subtracted dark is no longer in the counts, but its noise is still in the
data. At extraction (`guiExtract.process_trace`), the dark is reloaded through the same
cross-check:
- **L.A.Cosmic** gets a smoothed copy as `background`, so its noise model includes the dark,
  and the dark's hot pixels as `mask`. A full-resolution background would be added back
  before detection, turning every hot pixel into a false cosmic ray.
- **The per-fibre errors** gain `S/g + scale·(c/N)·S/g + scale²·(c/N)·A·RN²/g²`. Here S is
  the scaled dark summed over the extraction aperture, A the aperture pixels, g the gain and
  RN the read noise.

  On 4B blue at 1350 s this roughly doubles the error. The dark's shot noise was already
  missing before the dark existed, because the edge-DC constant removed the dark from the
  counts without its noise.

## Keywords

| Where | Keyword | Meaning |
|---|---|---|
| Primary | `BFVER` | Bias-first version (2: true orientation and dark) |
| Primary | `BIASFILE`, `BIASID` | Master bias used; identity from its name, size and mtime |
| Primary | `DARKID`, `DARKSCL` | Master dark and scale (`none`/0 if none) |
| Extension | `BIASSUB`, `BIASSRC`, `BIASLVL` | Bias applied, its source, median level |
| Extension | `DARKSUB`, `DARKWHY` | Dark applied; if not, why |
| Extension | `DARKFILE`, `DARKID`, `DARKSCL`, `DARKNCMB`, `DARKMETH`, `DARKLVL` | Dark used, frames combined, method, median DN removed |
| Extension | `EDGEDC0`, `DARKRES`, `DARKNRAT`, `DARKNLOW`, `DARKFLAG` | Checks above |
| Extension | `EDGEDC`, `EDGENPIX`, `EDGEDMIN`, `EDGESRC`, `EDGETRC` | Edge DC, its mask, and which traces drew it |
| Extension (extraction) | `DARKVAR` | Whether the dark's noise reached the errors |

## Resuming

`reduce.py` resumes by file name at every stage. Every product copies the primary header,
so before bias-first, `biasFirstResume.redo_stale_frames` compares `BFVER`, `BIASID`,
`DARKID` and `DARKSCL` with what this run would write. It checks:
- the bias-corrected file;
- the flat-corrected file;
- every RSS of each frame.

The RSS of a science, standard or sky frame is also checked for `CRID`, the cosmic-ray cleaning
(see `Masking/README.md`).

A frame is out of date if it was made by the older bias step, or if the master bias, the
dark, its scale or the cosmic-ray settings have changed. Its products are moved, never deleted, into
`{output_dir}/stale/<timestamp>/`, and the resume rebuilds them:
- **Science or standard frame:** that exposure. With `sky_empirical_2d`, all science frames
  are redone, because fields borrow donor frames.
- **Sky frame:** itself, plus all science frames when the sky model uses it.
- **Flat, arc or twilight:** everything, including traces, flat products, night arcs,
  pedestal templates and the QA tables.

Raw inputs, packaged calibrations and files made by the current run are never moved.

## Not covered

- The quick look (`Image/WhiteLightModule`, `GUI/obslog.py`) runs on raw frames and
  subtracts no dark.
- The shipped `slow_master_bias.fits` / `fast_master_bias.fits` are single raw frames from
  2026-02-20, with constant fills for blue 1A and 4A. A master bias built from the night's
  biases (setup GUI → Write Config) is better, and gives those cameras a dark.

## Files

- `biasFirst.py`: `bias_correct_frame`, provenance stamps, the dark checks.
- `darkLlamas.py`: grouping, matching, building and loading master darks.
- `biasFirstResume.py`: out-of-date frames on resume.
- `__init__.py`: bias exceptions, `match_camera_hdu` / `verify_camera_hdu`, and
  `generate_fallback_bias_hdu`.
- `biasChecking.py`: inter-fibre and stripe masks, edge-DC measurement.
- `biasCache.py`: the per-process cache of processed master biases. It lives here, not in
  `Trace/traceLlamasMaster.py`, because that module is pickled into every trace (see
  `Trace/README.md`).
- `llamasBias.py`: `BiasLlamas`, the median master bias (used by the setup GUI).
