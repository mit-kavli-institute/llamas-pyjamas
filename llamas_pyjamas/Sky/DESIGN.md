# Sky subtraction — design & development plan (`sky-refine`)

Status: **plan / in progress.** Branch `sky-refine` (off `rs-dev`). This document is the reference for
the sky-subtraction refactor and the striping diagnosis. It is intentionally architecture-first: we make
sky subtraction pluggable (this is a facility instrument — different programmes subtract sky differently)
*without changing today's default behaviour*, then diagnose the residual striping with the current model.

---

## 1. Where we are today (verified against the code and real may26 data)

Two sky stages **both run** in the production config — the "current sky model" is their composition:

1. **Base B-spline** — `skyLlamas.py:skyModel_1d` (`sky_subtract=True`). A per-*camera*, pypeit-style
   B-spline sky model (Kelson-inspired) fit to the pooled sky-fibre counts vs `xshift`, evaluated on every
   fibre. Runs on the extraction pickles, **before** RSS assembly and **before** the fibre flat. Writes each
   science extraction's `.sky` attribute → the RSS `SKYSUB` / `SKY` / `COUNTS` planes.
2. **Framework refinement** — `skySubtract.py:subtract_sky_all_colors` (`sky_framework=True`, **active in
   production**). A per-*fibre* derivative-augmented OH refinement (α·S + β·S′ + γ·S″,
   `skyScale.py:scale_sky_per_fiber`), optionally followed by ZAP-style PCA residual cleaning
   (`skyResidual.py`). Runs on the fibre-flat (`_FF`) RSS, **after** the flat. Appends a `SKYRESID` plane.

A may26 green RSS header confirms both ran: `SKYSUB=True, SKYSUB2=True, SKYMETH=scaled, SKYSEL=stratified,
SKYDVORD=2, SKYPCANC=0, SKYNMASK=1194`. So the striping we see in stacked images is the residual left after
**B-spline → derivative-OH scaling**, and it is flat-amplified (the fibre flat runs between the two stages).

### Structural facts that shape the design

- **Selection is already factored** into `skySelect.py:select_sky_fibres` (returns a boolean per-fibre mask;
  dispatch over `stratified`/`quantile`/`dimmest`/`middle-third`/`skymap`/`frame`/`all`). This is the natural
  "mask provider" seam — **but** its result is recomputed inline in two places that *disagree*:
  `skyModel_1d` (base) and `skyMask.py:build_sky_fiber_mask` (framework). The framework collapses
  `stratified`/`quantile`/`dimmest` to a single white-light percentile cut, so `SKYSEL=stratified` is recorded
  while a ~60th-percentile cut (~1194 fibres) actually ran. **This is a latent bug to fix during unification.**
- **The estimator is not abstracted** — the base B-spline is inlined in `skyModel_1d`; the refinement chain is
  hard-wired in `subtract_sky_rss`. No registry, nothing subclassable.
- **No persisted sky mask** — the selection is computed and thrown away, so it can't be inspected, reused, or
  user-overridden.
- **Blank/offset-field sky is only partial** — the `'frame'` method + `sky_frame_files` +
  `reduce.py:_extract_sky_frame` can take sky from a separate exposure, but multi-frame combination is not
  implemented (first frame only), and the framework path does not consume a separate frame directly.
- **Gotchas:** fibre arrays are *live-indexed* (convert via `deadfibers.live_fibre_ids` before `FiberMap_LUT`);
  and per-fibre RA/Dec are **absent** in the base-path sky stage (they are only assigned at RSS assembly). A
  spatial / external-image (e.g. LSST) provider is therefore natural in the RSS/framework domain, not the base.

---

## 2. Design principles

- **Facility-general and pluggable**, with **sensible defaults that reproduce today's behaviour exactly**.
  Every option exposed; nothing hard-wired to one programme's needs.
- Three orthogonal seams: **sky-fibre selection**, **sky-model estimation**, and **sky source** (in-field vs
  a separate blank/offset field).
- **Do not change the sky-fibre selection method now.** Make it swappable; keep the current object-flux-based
  selection as the default.

### The estimator is a base **plus refinements**, not a menu

For sky estimated from **in-exposure fibres we always run the pypeit-style B-spline base** (Kelson et al.-
style). The derivative and PCA stages are **optional refinements layered on top of it — not alternatives**:

- **Derivative refinement** (α·S + β·S′ + γ·S″) corrects the **changing shape of the LSF across the slit**
  caused by optical aberrations. On by default in production.
- **PCA refinement** is a **pure empirical fit to remove leftover residuals — a last resort.** Off by default.

So the estimator abstraction is a **base estimator + an ordered, individually-toggleable refinement chain**,
never a `method = bspline | scaled | pca` mutually-exclusive choice. (A separate blank/offset *source* may use
a different estimator; the "always B-spline" rule is specific to in-exposure sky.)

---

## 3. Target architecture (three seams)

### 3a. `SkyMask` — a first-class, persisted object
A live-indexed boolean-per-fibre mask **plus provenance** (method, parameters, source exposure). Written as an
**RSS extension** so it is inspectable, reusable, and **user-overridable** — this persisted mask is the hook
that later lets people define sky fibres externally (an LSST broadband image, a hand-drawn region, an explicit
list). A single **provider registry** (wrapping `skySelect`) produces it, and **both** sky stages consume the
*same* mask, ending the current base/framework disagreement.

### 3b. `SkyEstimator` — base + refinement chain
A base estimator (B-spline for in-exposure sky) followed by an ordered list of optional refinements
(derivative, then PCA), each independently toggleable, each with typed config and header provenance. New
refinements or a new base (e.g. offset-field-direct) register without touching the core.

### 3c. `SkySource` — in-field vs offset/blank field
Where the sky *sample* comes from: the same exposure (today) or a separate blank/offset exposure. Generalises
the existing `'frame'` path and is designed for multi-frame combination later.

Config keeps every current key and default; new knobs default to exactly today's behaviour.

---

## 4. Phasing

### Phase 1 — architecture, thinner first cut (behaviour-preserving) ← we are here
- Introduce the first-class, **persisted `SkyMask`** (RSS extension + provenance).
- **Unify** the two inline selections behind one provider (wrapping `skySelect`), **fixing the
  stratified-collapse bug** so the recorded method is the method actually used.
- **Regression test:** the default pipeline output is **bit-identical** on a may26 exposure.
- Prove the seam with a trivial **manual-mask provider** — no change to the default result.
- **Defer** the full `SkyEstimator` (base + refinement chain) and `SkySource` registries to Phase 3 — the
  diagnosis may reshape the estimator interface, so we don't over-build it now.

### Phase 2 — diagnose the striping (current model)
- Quantify the striping on a stacked field (a repeatable metric).
- Decompose the `SKYRESID` / `SKY` / `SKYSUB` / `COUNTS` planes; determine whether the coherent additive
  residual originates in the base B-spline or the framework refinement (toggle the framework and compare).
- Correlate the residual against fibre throughput (flat amplification), slit-y position, bench/side,
  wavelength (between-line continuum vs OH lines), and across dithers.
- Build on the prior finding that the striping is an **additive, flat-amplified, coherent-across-dithers**
  residual (diffuse scattered continuum into low-throughput fibres). Deliverable: a diagnostic toolkit +
  a root-cause memo.

### Phase 3 — later (informed by Phase 2)
- The full `SkyEstimator` base+refinement-chain abstraction and model improvements from the diagnosis.
- New selection providers: external broadband-image (e.g. LSST) masks, manual / GUI-defined masks.
- Full offset/blank-field sky, including multi-frame combination.
- Cross-camera sky sharing (requires fixing the per-camera throughput normalisation noted in the diagnosis
  work).

---

## 5. Testing & validation
- A **bit-identical regression harness** on a real may26 exposure guards Phase 1.
- Reuse the existing `test_sky_select.py` (14 tests) and `test_sky_framework.py`; add mask-persistence and
  provider-unification tests, then diagnostic tests in Phase 2.

## 6. Key risks / gotchas
- Live-vs-physical fibre indexing (convert before any `FiberMap_LUT` use).
- RA/Dec absent in the base-path sky stage — spatial/external-image providers live in the RSS/framework domain
  or require plumbing RA/Dec earlier.
- Two stages both run — don't double-count; unify their mask; the framework's stratified-collapse is fixed as
  part of the unification.
