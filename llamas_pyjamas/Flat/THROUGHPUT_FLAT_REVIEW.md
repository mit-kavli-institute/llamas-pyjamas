# Throughput & Detector-to-Detector Flat-Fielding Review

> **Scope:** Review of the LLAMAS flat-fielding subsystem (`llamas_pyjamas/Flat/`),
> focused on **fibre-to-fibre throughput** and **detector-to-detector (bench-side)
> differences**. Pixel-to-pixel QE is **deliberately out of scope** here (tracked
> separately).
> **Guiding principle:** flat fielding should follow proper IFU practice — corrections
> derived from, and applied to, the **raw detector frames**, not the extracted RSS.
> **Status:** Review only — no code changed. Input for feature-update planning.
> **Date:** 2026-06-03

---

## TL;DR

The throughput / detector-to-detector flat fielding underperforms for two structural
reasons:

1. **It works in the wrong domain.** The correction that actually runs is computed from,
   and applied to, **extracted RSS spectra** (post-extraction), not the raw frames. Proper
   IFU practice builds the throughput/illumination correction from the raw flat in 2D
   detector space and applies it to the raw science frame *before* extraction. Operating on
   RSS folds extraction weighting and artefacts into the "throughput" and breaks clean
   variance propagation.
2. **Detector-to-detector throughput is never actually tied together.** Every method here
   normalises each bench-side to *its own* reference, which mathematically removes the very
   cross-detector signal it should be measuring. There is no common, uniformly-illuminated
   reference linking the eight bench-sides (×3 channels).

A third-party repo (MISTY, `virajvman/MISTY`) produces visually better fibre-to-fibre
results, **but it is RSS-based and therefore not standard IFU practice** — it is included
below only as a performance comparison, *not* as the model to copy.

---

## What runs vs. what is dead code

Live pipeline path (via `reduce.py`):

- `flatProcessing.py` — flat extraction (`produce_flat_extractions`)
- `flatLlamas.py` — builds `flat_smooth_models.fits` (lamp envelope per fibre)
- **`fibreFlat.py`** (camelCase, function-based) — fibre-to-fibre correction, **applied to
  RSS files** via `apply_fibre_flat_to_rss` (`fibreFlat.py:1314`)

Dead / parallel code in the same directory:

| File | Lines | Imported by | Domain | Status |
|------|-------|-------------|--------|--------|
| `fibre_flat.py` (class `FibreFlatField`) | 1,055 | only `Flat/__init__.py` | **RSS** | Not wired into pipeline |
| `flatPypeit.py` | 1,261 | only the dead `fibre_flat.py` | — | Dead |
| `scattered2dLlamas.py` | 95 | nothing | — | Dead |

> ⚠️ Two modules named `fibre_flat.py` and `fibreFlat.py` with overlapping
> responsibilities is a real trap — an import typo silently picks the wrong one.
> `fibre_flat.py` is the RSS-based "synthetic central-fibre reference" program (the same
> *class* of approach as MISTY); it is **not** the one the pipeline calls.

---

## 1. Wrong domain: throughput is corrected on RSS, not raw frames

### What proper IFU practice looks like

The fibre-to-fibre throughput and bench-side illumination terms should be measured from a
**uniformly illuminated raw flat** (twilight/dome) by tracing each fibre on the raw detector,
integrating its profile, and building a normalised flat that is **divided into the raw
science frame before extraction**. Doing it this way:

- keeps the correction in the same 2D detector domain where the photons (and their
  Poisson + read-noise variance) actually live, so variance propagates cleanly;
- corrects illumination/throughput *consistently* with the pixel/detector flat in a single
  raw-domain step;
- avoids contaminating "throughput" with the extraction profile weights and any extraction
  artefacts.

### What the pipeline actually does

- The live correction is computed from `flat_smooth_models.fits` (per-fibre lamp envelopes)
  and **applied to extracted RSS files**: `apply_fibre_flat_to_rss` divides `FLUX`/`ERROR`
  per RSS row (`fibreFlat.py:1314-1502`). This is a **post-extraction** correction.
- Consequence: whatever the optimal extraction did to each fibre (profile weighting,
  masking, sub-pixel resampling) is already baked into the spectra before the throughput
  correction sees them. The "throughput" you measure is therefore extraction-dependent, not
  a pure instrument property.
- It also splits flat fielding across two domains: the pixel/`flatLlamas` step touches raw
  frames, while the throughput step touches RSS — so the two corrections are derived and
  applied inconsistently.

**Implication for planning:** the target architecture should derive the
throughput/illumination flat from raw flats and apply it in the raw (or extraction-input)
domain, unifying it with the detector flat rather than bolting a separate RSS-domain step
onto the end.

---

## 2. Detector-to-detector throughput is never tied together

This is the crux of the "detector-to-detector differences" problem.

- **Live lamp-only branch** (`compute_fibre_flat_lamp_only`, `fibreFlat.py:196`): builds a
  **per-bench-side median reference** (`_compute_benchside_reference`, `fibreFlat.py:63`) and
  forms `C_i(λ) = smooth_i(λ) / S̄_benchside(λ)`. Each bench-side is normalised to itself, so
  the absolute throughput differences *between* bench-sides/detectors are not measured — they
  are divided out by construction. The function's own warning admits it "will contain
  artificial spatial illumination gradients" (`fibreFlat.py:209-213`).
- **The RSS-based `fibre_flat.py`** does the same thing structurally: synthetic reference =
  median of the N central fibres **per bench-side** (`fibre_flat.py:6-11`, `319-372`). Better
  fibre-to-fibre flattening *within* a bench-side, but still no cross-detector tie.
- **Twilight branch** (`compute_fibre_flat_twilight` + `_remove_twilight_gradient`,
  `fibreFlat.py:358-630`) is the only code that *attempts* a cross-bench solution: it maps
  fibre integrals to IFU coordinates and fits a low-order 2D polynomial across bench-sides.
  But it is fragile — it drops to a 1st-order surface when >2 bench-sides lack twilight data
  (`fibreFlat.py:403-410`), and the whole branch is wrapped in a broad `except Exception`
  that silently falls back to the gradient-injecting lamp-only path (`reduce.py` ~2071).

**Why this can't be fixed by smoothing or tuning:** detector-to-detector throughput ratios
are only observable with a source that illuminates *all* bench-sides uniformly (twilight sky,
ideally; or a characterised dome flat). Any method that uses a per-bench-side reference is
blind to them. Proper IFU practice ties the eight bench-sides (per channel) to a **single
common reference** from such a uniform exposure, in the raw domain.

Your QA module already looks for the symptoms — `QA/qa_flatfield.py` tests cube spatial
autocorrelation (Moran's I), PCA residuals, and chromatic stability — i.e. exactly the
bench-to-bench discontinuities this gap produces.

---

## 3. Variance is not propagated

- `correction_var` is hardcoded to zeros with a `TODO` in the lamp-only path
  (`fibreFlat.py:274-275`); the error array is simply divided by `C` (`fibreFlat.py:1446`),
  adding no flat noise.
- The RSS-based `fibre_flat.py` likewise notes the flat error is "currently unused"
  (`fibre_flat.py:274`).
- This violates the project rule (CLAUDE.md: *"propagate variance through every reduction
  step"*) and makes downstream S/N optimistic. Deriving the flat in the raw domain (where the
  flat's own photon statistics are available) makes correct variance propagation natural
  rather than an afterthought.

---

## 4. The MISTY comparison (context only — not the target)

`virajvman/MISTY` (`data_reduction/fiber_norm/flat_fielding.py`) implements an RSS-based
fibre normalisation that reportedly produces better-looking fibre-to-fibre results. The
in-repo `fibre_flat.py` is the same class of approach (synthetic central-fibre reference +
Savitzky-Golay smoothing on extracted spectra).

It is included here as a **performance benchmark, not a methodological model**: it operates
on RSS, which is not standard IFU flat-fielding practice and carries the same domain and
variance-propagation drawbacks described in §1 and §3. The goal is to *match or beat* its
fibre-to-fibre quality **while** doing the correction properly in the raw domain — not to
adopt its RSS-based design.

---

## Recommendations (priority order)

1. **Move throughput/illumination flat fielding into the raw domain.** Derive the
   fibre-throughput and bench-side illumination correction from raw, uniformly-illuminated
   flats (twilight preferred) by tracing/integrating fibre profiles on the raw frame, and
   apply it before/at extraction — unified with the detector flat, not as a separate
   RSS-domain step.
2. **Tie detectors together with a common reference.** Replace per-bench-side
   self-normalisation with a single cross-bench-side reference per channel, anchored to a
   uniform-illumination exposure, so real detector-to-detector throughput ratios are
   *measured and corrected* rather than divided out.
3. **Make the twilight (cross-bench) path the primary route, not a fragile fallback.**
   Remove the silent `except → lamp-only` behaviour; if twilight coverage is insufficient,
   fail loudly or degrade explicitly with a recorded flag.
4. **Propagate flat variance** (kill the `correction_var = 0` TODO); natural once the flat is
   built in the raw photon domain.
5. **Consolidate the modules.** Decide on one implementation, delete/quarantine the rest
   (`fibre_flat.py`, `flatPypeit.py`, `scattered2dLlamas.py`) and the stale PypeIt status
   docs, so the throughput code is reviewable and tunable.
6. **Benchmark against MISTY** on the same data — target equal-or-better fibre-to-fibre RMS
   and cube spatial uniformity (via `QA/qa_flatfield.py`) using the raw-domain method.

---

## File reference map

| Concern | Location |
|---------|----------|
| Live throughput correction, applied to **RSS** | `fibreFlat.py:1314-1502` (`apply_fibre_flat_to_rss`) |
| Per-bench-side reference (no cross-detector tie) | `fibreFlat.py:63-115`, `196-295` |
| Twilight cross-bench gradient fit (fragile) | `fibreFlat.py:358-630`, `955+` |
| Lamp envelope source (`flat_smooth_models.fits`) | `flatLlamas.py:677-927` |
| Variance-not-propagated TODO | `fibreFlat.py:274-275`, `1446` |
| RSS-based program (non-standard; comparison only) | `fibre_flat.py` (`compute_fibre_flat` ~254, `apply_fibre_flat` ~470) |
| Pipeline orchestration (fibre flat) | `reduce.py` (~2031-2107) |
| QA checks for the symptoms | `QA/qa_flatfield.py` |
