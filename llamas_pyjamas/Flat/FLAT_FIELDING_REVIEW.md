# Flat-Fielding Review & Improvement Plan

> **Scope:** Code review of the LLAMAS flat-fielding subsystem (`llamas_pyjamas/Flat/`),
> focused on *why current performance is sub-optimal* and what to change.
> **Status:** Review only — no code changed. Intended as input for feature-update planning.
> **Date:** 2026-06-03

---

## TL;DR

The flat-fielding subsystem has two problems that compound each other:

1. **Algorithmic flaw** — the "pixel-to-pixel flat" is not actually a pixel-to-pixel
   flat. It is derived in collapsed 1D extracted space and stamped back onto 2D, so it
   carries *zero* cross-dispersion (spatial) pixel information and instead bakes
   lamp/extraction spectral structure into the correction.
2. **Architectural disorder** — ~6,000 lines across six overlapping modules, two
   near-identically-named fibre-flat implementations, a completely dead "PypeIt" path,
   and three mutually-contradictory status docs. Nobody can reason about which code
   actually runs, which is itself a root cause of poor, hard-to-improve results.

---

## What actually runs vs. what is dead code

The live pipeline (via `reduce.py`) uses:

- `flatLlamas.py` — pixel-to-pixel flat (Stage 1)
- `flatProcessing.py` — flat extraction (`produce_flat_extractions`)
- **`fibreFlat.py`** (camelCase, function-based) — fibre-to-fibre flat (Stage 2)

Dead or parallel code in the same directory:

| File | Lines | Imported by | Status |
|------|-------|-------------|--------|
| `fibre_flat.py` (underscore, class-based `FibreFlatField`) | 1,055 | only `Flat/__init__.py` | **Not used by pipeline** |
| `flatPypeit.py` | 1,261 | only the dead `fibre_flat.py` | **Dead** |
| `scattered2dLlamas.py` | 95 | nothing | **Dead** |

`flat_method` appears in **zero** `.py` files — there is no PypeIt method switch anywhere
in the code, despite what the docs claim (see below).

> ⚠️ Two files named `fibre_flat.py` and `fibreFlat.py` with overlapping responsibilities
> is a serious trap — an import typo silently selects the wrong implementation.

---

## 1. Core scientific problem: it isn't a real pixel flat

### How the map is built

`generate_pixel_flat_extension` (`flatLlamas.py:206-303`) and the B-spline variant in
`Thresholding._generate_single_pixel_map` (`flatLlamas.py:1239-1323`) both:

1. Take each fibre's **extracted 1D spectrum** (`counts[fib_idx, :]` — already a
   profile-weighted *sum* across the cross-dispersion direction).
2. Smooth it (median + Gaussian, or a B-spline) to get a "lamp envelope."
3. Form `ratio_1d = actual / smooth`.
4. Project the 1D ratio back to 2D:
   ```python
   sensitivity_map[fiber_mask] = ratio_1d[np.where(fiber_mask)[1]]   # flatLlamas.py:296
   ```

That last line is the crux: **every detector pixel belonging to a fibre at a given column
gets the same correction value.** There is no per-pixel information in the spatial
(cross-dispersion) direction at all.

### Why this hurts

- A pixel-to-pixel flat exists precisely to remove per-pixel QE/gain variation in 2D.
  This map is constant across each fibre's spatial profile, so it **cannot correct the
  thing it is named for.** The spatial dimension of the detector response is
  unrecoverable from a 1D-extracted spectrum.
- `actual / smooth` does **not** capture detector QE. With `filter_size=12` (the
  `process_pixel_flat_simple` default, `flatLlamas.py:730`) the median window is short
  enough to track *real* spectral structure — lamp emission/absorption lines, fringing
  (red), extraction wiggles. Dividing the science frame by this re-imprints or removes
  structure that is astrophysical/lamp, not instrumental → it can **inject** systematics.
- The red-channel comment at `flatLlamas.py:1157-1159` ("fringing should NOT be modeled
  in the flat") shows awareness of the hazard, but smoothing scale is the only safeguard,
  and the code paths disagree on it (12 vs 51 vs B-spline `bkspace=30/50`).

**Correct approach:** build the pixel flat in **2D detector space** — divide the raw 2D
flat frame by a smooth 2D model (spectral shape along each fibre's trace × the spatial
profile), leaving only per-pixel residuals. The current design collapses to 1D first and
never recovers.

### Order-of-operations / circularity

- The map is **derived** from extracted, wavelength-calibrated flat spectra (requires
  traces + `arcTransfer`, `flatLlamas.py:805-825`), but **applied** to the **raw 2D
  science frame before extraction** (`apply_flat_field_correction`, `flatLlamas.py:930`).
- The map is indexed by extracted-array column (`ratio_1d[c]`, `flatLlamas.py:1286-1288`)
  and stamped onto detector column `c`, silently assuming extracted spectral index ==
  detector x-pixel with no shift. Any trace / `xshift` / sub-pixel offset misaligns it.
- Requiring a full extract + arc-transfer of the flat *just to build a pixel flat* is
  heavy and fragile (aborts if wavelength transfer fails for any extension,
  `flatLlamas.py:822-824`).

### Clipping & thresholds throw away signal

- `CHANNEL_SIGNAL_THRESHOLDS = {'red':5000,'green':8000,'blue':5000}` (`flatLlamas.py:29`)
  are hardcoded absolute ADU. Pixels below threshold are set to 1.0 (no correction,
  `flatLlamas.py:290-292`) → faint fibres / faint wavelengths get **no flat correction**,
  and the threshold doesn't scale with exposure or gain.
- Clip ranges are inconsistent and aggressive: `process_pixel_flat_simple` clips to
  **(0.90, 1.10)** (`flatLlamas.py:731`); the B-spline path clips to **(0.5, 2.0)**
  (`flatLlamas.py:1271`). A ±10% clip cannot correct any QE feature larger than 10% and
  quietly hides bad pixels rather than masking them.
- The map is never explicitly renormalised to unit mean, so any net offset rides into the
  science flux scale.

---

## 2. Architecture & documentation disorder

The three status docs contradict each other and the code:

- `PYPEIT_INTEGRATION_COMPLETE.md` — claims "✅ FULLY IMPLEMENTED"; references
  `pypeit_integration.py` and `process_flat_with_pypeit()` — **neither exists**.
- `FLAT_FIELDING_IMPLEMENTATION_SUMMARY.md` — says PypeIt "falls back to the standard
  method"; references `flatLlamas_pypeit.py` — **wrong filename** (actual: `flatPypeit.py`).
- `CURRENT_IMPLEMENTATION_STATUS.md` — says it's a placeholder needing a translation layer.

All three are AI-authored (one signed "Author: Claude"), dated 2025-11-20, and describe a
`flat_method=pypeit` config switch and a `reduce.py:212-247` fallback that **do not exist
in the code**. They are actively misleading and should be removed.

**Why this matters for performance:** with six modules and no clear entry point, you can't
tune the one that runs, and you risk "fixing" dead code.

---

## 3. Fibre-to-fibre flat (`fibreFlat.py`) — the part that runs

Conceptually sounder (relative throughput per fibre), but with real issues:

- **Variance is not propagated.** `correction_var` is hardcoded to zeros with a `TODO`
  (`fibreFlat.py:274-275`). In the lamp-only branch the error array is simply divided by
  `C` (`fibreFlat.py:1446`) with no flat noise added → violates the project rule
  (CLAUDE.md: "propagate variance through every reduction step") and makes downstream S/N
  optimistic.
- **Lamp-only fallback injects spatial gradients** — explicitly acknowledged
  (`fibreFlat.py:209-213`). The per-benchside median reference
  (`_compute_benchside_reference`, `fibreFlat.py:63`) normalises each benchside to itself,
  so **real cross-benchside throughput differences between IFU blocks are not consistently
  handled** → spatial discontinuities in the cube. `QA/qa_flatfield.py` checks for exactly
  this (Moran's I, PCA, chromatic stability) — a known, recurring failure mode.
- **Twilight branch is fragile**: `_remove_twilight_gradient` drops to a 1st-order surface
  when >2 benchsides lack twilight data (`fibreFlat.py:403-410`), and the whole branch is
  wrapped in a broad `except Exception` that silently falls back to lamp-only
  (`reduce.py` ~2071). In practice you may be getting the gradient-injecting fallback
  without obvious notice.
- **Double-handling of lamp spectral shape**: both stages derive from the same
  `flat_smooth_models.fits`. Confirm the lamp's intrinsic spectral shape (not an instrument
  property) isn't being divided out in a way that tilts science continua.

---

## 4. Literal speed bottlenecks

- `_generate_single_pixel_map` loops pixel-by-pixel in Python
  (`for r, c in zip(rows, cols)`, `flatLlamas.py:1285-1289`) over every fibre of every
  extension — should be fully vectorised with boolean masks.
- `fit_spectrum_to_xshift` tries up to **4 sequential B-spline strategies per fibre**
  (`flatLlamas.py:422-475`), ×~300 fibres ×24 extensions, single-threaded — despite
  tracing/extraction being Ray-parallelised elsewhere.
- `generate_thresholds` / `_find_matching_trace` re-open and unpickle trace files inside
  nested loops (`flatLlamas.py:1381-1387`, `342-361`) → O(N²) pickle loads.
- `apply_fibre_flat_to_rss` corrects fibres in a per-row Python loop (`fibreFlat.py:1382`).

---

## Recommendations (priority order)

1. **Delete the fiction.** Remove or quarantine `flatPypeit.py`, `fibre_flat.py`
   (underscore), `scattered2dLlamas.py`, and the three status docs
   (`PYPEIT_INTEGRATION_COMPLETE.md`, `FLAT_FIELDING_IMPLEMENTATION_SUMMARY.md`,
   `CURRENT_IMPLEMENTATION_STATUS.md`). Keep one fibre-flat module. This alone makes the
   subsystem reviewable.
2. **Rebuild the pixel flat in 2D.** Construct per-pixel response from the raw 2D flat
   divided by a 2D smooth model (spectral B-spline along the trace × spatial profile) so it
   actually captures cross-dispersion QE. Normalise to unit median; *mask* bad pixels
   rather than clipping to 1.0.
3. **Stop fitting spectral structure.** Smoothing scale must be demonstrably wider than
   lamp lines/fringes; standardise it across both code paths (currently 12 vs 51 vs bspline).
4. **Propagate flat variance** through both stages (kill the `correction_var=0` TODO).
5. **Make thresholds/clips relative** (fraction of per-fibre median, not absolute ADU) so
   faint fibres/wavelengths still get corrected.
6. **Surface the lamp-only fallback loudly** — it injects spatial gradients; a silent
   `except Exception` hiding the twilight failure is dangerous for science.
7. **Optimise hot loops** — vectorise pixel projection, parallelise per-fibre fits, cache
   trace loads.

---

## File reference map

| Concern | Location |
|---------|----------|
| Pixel flat (simple median+gaussian) | `flatLlamas.py:206-303`, `727-927` |
| Pixel flat (B-spline / Thresholding) | `flatLlamas.py:364-519`, `1078-1420` |
| 1D→2D projection (the flaw) | `flatLlamas.py:296`, `1285-1289` |
| Apply pixel flat to science | `flatLlamas.py:930-1073` |
| Fibre flat (lamp-only) | `fibreFlat.py:196-295` |
| Fibre flat (twilight + gradient removal) | `fibreFlat.py:302-351`, `358-630`, `955+` |
| Apply fibre flat to RSS | `fibreFlat.py:1314-1502` |
| Variance TODO | `fibreFlat.py:274-275`, `1446` |
| Pipeline orchestration | `reduce.py` (~1832-1940 pixel flat, ~2031-2107 fibre flat) |
| QA checks | `QA/qa_flatfield.py` |
