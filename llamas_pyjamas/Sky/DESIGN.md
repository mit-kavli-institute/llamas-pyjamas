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

### Phase 1 — architecture, thinner first cut (behaviour-preserving) — ✓ DONE
- Persisted first-class `SkyMask` + `build_sky_mask` provider; both stages unified onto it; honest
  provenance (the "stratified-collapse" turned out to be inert — it feeds only the PCA basis, which is
  off — so it was recorded honestly rather than changed). Manual-mask seam proven. Tests green.

### Phase 2 — diagnose the striping — ✓ DONE (see `Sky/diagnosis/DIAGNOSIS.md`)
- Root cause (all channels): **one additive per-camera / per-fibre sky-subtraction residual** — a
  positive interior floor (worst benchsides 1A/2A green, 1A red) + negative slit-edge over-subtraction
  dips — coherent across dithers, so the two rotation groups cross-hatch it into diagonal stripes.
  Additive, not a scaling error; a fixed instrumental pattern across J1613/J2151/J0958. **Blue** is the
  same residual **amplified by the large blue flux calibration** (worst at the blue end). One root
  cause ⇒ one fix, applied in the counts/pkl domain before flux calibration, helps every channel.

### Phase 3 — the striping fix (branch `pedestal-fix`) — HYPOTHESIS UNDER TEST

**Framing (important).** The Phase 2 diagnosis — one additive per-camera sky-subtraction residual
(positive interior floor + slit-edge over-subtraction), amplified by flux-cal in blue — is the
**best-available theory, not proven**. Phase 3 is therefore built to be **falsifiable and reversible**:

- it lives on a **separate branch (`pedestal-fix`)** off `sky-refine`;
- every correction is **config-gated, default OFF** (the pipeline is unchanged unless enabled);
- it is judged by **whether it actually reduces the measured striping** — if it doesn't, we abandon
  the branch and lose nothing.

All corrections act in the **counts / pkl / xshift domain, *before* flux calibration** — that is what
protects blue (where flux-cal otherwise amplifies the residual). RSS stays write-once after sky is done.

**The two corrections** (built as refinements on a `SkyEstimator` = B-spline base + ordered,
individually-toggleable refinement chain — the abstraction deferred from Phase 1):

1. **Per-camera additive continuum pedestal.** Estimate the positive additive floor from *genuinely
   blank* fibres in line-free pixels (per camera; optionally smooth along the slit) and subtract it.
   Targets the +floor (worst 1A/2A in green, 1A in red).
   - **CENTRAL RISK — could eat real Lyα.** A per-camera additive pedestal can subtract the very diffuse
     emission we want. Mitigations: (a) measure the pedestal only from fibres/regions known blank
     (field edge, away from the QSO pair — or a user sky mask / offset field); (b) restrict it to a
     smooth, low-spatial-frequency instrumental component distinct from compact emission; (c) keep it
     optional and gated behind the signal-preservation guardrail below. This risk is the main reason
     the theory must be *tested*, not assumed.

2. **Slit-edge / across-slit LSF refinement in the xshift domain.** Relocate the derivative refinement
   (α·S + β·S′ + γ·S″) into the pkl/xshift domain so it corrects the edge over-subtraction (the negative
   dips) at the source, where xshift lives (per the domain principle).

**Sub-phases:** 3a build the `SkyEstimator` base+chain scaffold + the pedestal refinement (gated);
3b the edge-LSF derivative refinement in xshift; 3c validate on may26.

**Validation / falsification (the whole point of the branch):**
- **Primary metric:** does the stacked-image striping RMS (blank regions) drop across *all* channels and
  fields? Reuse `Sky/diagnosis/{sky_residual_diag,sky_residual_verify,blue_striping_investigation}.py`
  as the before/after test.
- **Guardrail — real signal preserved:** QSO point-source spectra unchanged; no over-subtraction of
  diffuse flux; explicitly check the pedestal does not remove real Lyα (test on/off on a field region).
- **Not bit-identical:** the pipeline has run-to-run non-determinism (WAVE/xshift jitter), so validate
  with the striping-RMS + signal-preservation metrics, not bit-for-bit.
- **Reversible:** default OFF, separate branch; compare on/off on the same reduction.

**Config keys (defaults reproduce today's behaviour):** `sky_pedestal` (false), `sky_pedestal_scope`
(`camera` | `fibre-along-slit`), `sky_pedestal_source` (`blank` | `edge` | `mask` | `offset`),
`sky_edge_refine` (false) + `sky_refine_domain` (`xshift`).

### Phase 3a concept-check result — the additive pedestal does NOT fix the striping

Tested on J1613 green (FLAM white-light, existing RSS; `Sky/diagnosis/pedestal_concept_check.py` →
`figures/pedestal_concept_qa.png`):

- **per-camera constant pedestal: −1%** striping (no effect);
- **per-slit-position profile: +13%**, but what it removes is a **smooth broad field** (positive
  interior, negative edges), **not** the diagonal stripes — which persist. QSO continuum preserved
  (0.1 %).

**The diagonal striping is not a removable additive continuum floor.** The additive pedestal (any
scope) removes ≤13 % and, worse, subtracts a broad smooth component — the real-diffuse-continuum
over-subtraction risk — for little gain. **Phase 3a (pedestal) is deprioritised.**

**Single-frame test (RS's idea — decisive; `singleframe_test.py` → `figures/singleframe_qa.png`):**
the striping is **worse in a single frame (3.4e-19) than in the 8-frame stack (1.6e-19)** — co-adding
*reduces* it. And the **raw fibres (pre-grid) already show the diagonal bands**. So the striping is:
- **NOT a co-add / cross-hatch effect** (stacking helps, ~partial averaging as dithers move the
  fibre-fixed pattern on the sky) — this revises the earlier "coherent across dithers" claim;
- **NOT a gridding artifact** (it's in the fibre values before resampling);
- a **per-fibre, per-frame banded systematic** in the sky-subtracted data.

Multiplicative per-camera renorm gave only +16% (per-camera SKY-scale spread ~12%), additive slit
profile +13% — both minor. **The bulk of the striping is a regular per-fibre band pattern in each
exposure.** Neither an additive pedestal nor a per-camera scale is the fix.

**Moiré test (RS's hypothesis; `moire_test.py` → `figures/moire_qa.png`):** the high-pass stripes are
*identical* at output pixel scales 0.3 / 0.5 / 0.75″ (same positions, same ~arcsec spacing), and the
banding is present in the **raw fibre values** (no grid), regular with a **~4–5″ period** at 145°. So it
is **NOT an output-grid moiré** (that would change with pixscale) — it is a **real, periodic per-fibre
banding fixed on the IFU/sky**.

**Hypothesis scoreboard (keep several alive):**
- ✗ additive per-camera pedestal (concept check: ≤13 %, wrong pattern)
- ✗ multiplicative per-camera scale (+16 %, minor; ~0 residual has little leverage)
- ✗ output-grid moiré (pixscale-invariant)
- ✓ real periodic per-fibre banding (~4–5″) fixed on the IFU — CONFIRMED, origin open
- ? origin candidates: (a) the IFU↔slit interleaving mapping a per-benchside residual into regular
  diagonal sky bands; (b) a per-lenslet-row IFU throughput/flat pattern; (c) an **extraction-level**
  beat (fibre traces vs detector pixels) imprinting periodic per-fibre throughput.

**COUNTS-vs-SKYSUB + IFU/benchside test (`counts_ifu_test.py` → `figures/counts_ifu_qa.png`) — ROOT
CAUSE FOUND:**
- the banding **is already in COUNTS** (pre-sky, pre-flux-cal) at **~5 %** (7.4 counts); the SKY model
  has nearly the same ~5 % (it inherits throughput) and absorbs ~half, leaving ~3.8 counts in SKYSUB —
  the striping. SKYSUB/FLAM banding correlates with the COUNTS pattern (+0.32), **not** with the sky
  model (~0). So it is an **upstream flat/throughput artifact, not a sky-subtraction error**.
- it is organized by **benchside / IFU-row block**: each benchside is a contiguous block of IFU rows
  that rotates into a diagonal band on the sky (the stripes). FLAM-vs-COUNTS shows discrete per-benchside
  throughput clusters.

**Root cause (confirmed):** a **per-benchside (IFU-row-block) ~5 % throughput/flat gain banding in the
extracted COUNTS**, which maps to the diagonal sky stripes, is only half-removed by sky subtraction, and
is amplified by flux-cal in blue. This is the **per-camera throughput NORMALISATION** issue surfacing:
the fibre flat is normalised per camera, so the absolute per-benchside gain is discarded → per-benchside
gain steps → banding.

### Verified mechanism + corrected fix (`throughput-norm` branch)

Per-benchside verification (`verify_throughput.py` → `figures/throughput_qa.png`): the per-benchside
FLAM residual is **inversely correlated with throughput** (`corr(benchside COUNTS, FLAM residual) =
−0.72`; blank COUNTS spread ~10 %). So the striping is the **fibre-flat (÷throughput) AMPLIFYING a
per-benchside ADDITIVE floor** — `FLAM_residual ∝ floor / throughput` — worst in the low-throughput
benchsides (1A, 3B). It is **not** a multiplicative flat error (scaling the ~0 sky-subtracted signal
would give ~0, as RS noted); the flat is doing its job — it is amplifying a real additive floor.

**Correction to the concept-check verdict:** that test subtracted the pedestal in **FLAM (post-flat)**,
where the floor is already multiplied per-fibre by 1/throughput, so a per-camera constant could not
match it (→ 0 %). The floor is **additive in the pre-flat counts**; removing it *there* (before
÷throughput) makes it vanish cleanly — which is exactly what `Sky/skyPedestal.py` does (pkl domain,
pre-flat). **So the pedestal was tested in the wrong domain and is NOT falsified.**

**⇒ Corrected fix direction:** remove the per-benchside additive floor **pre-flat, in the counts/pkl
domain** (the pedestal, correctly applied) — not a flat-normalisation change and not a post-flat
correction. **Definitive test:** a faithful **re-reduction with `sky_pedestal=True`** (which applies
pre-flat) and measure the striping. (Origin of the additive floor itself = the diffuse scattered
continuum, cf. [[between-line-residual-rootcause]].)

### Faithful re-reduction result — PEDESTAL FALSIFIED (definitive)

Re-reduced 3 J1613 dithers with `sky_pedestal` off vs on in the real pipeline (pre-flat, correct
domain; flat/traces reused; `SKYPED=True`, SKY changed ~6 counts, so it genuinely ran). Striping
(SKYSUB white-light high-pass): **per-frame off 7.07 → on 7.12 (−1 %); 3-frame stack off 4.02 →
on 5.01 (−25 %, WORSE)** (`figures/pedestal_result_qa.png`). So the per-camera additive pedestal does
**not** reduce the striping even applied correctly — it slightly worsens the stack. **Phase 3a is
falsified.** (Trajectory, for honesty: falsified by the post-flat concept check → temporarily
un-falsified on the "wrong domain" argument → re-falsified by this faithful test. The faithful test is
the arbiter.)

**Why:** the striping is a finer **per-fibre** banding (benchside/IFU-row-organized, present in COUNTS),
not a removable per-camera *additive* floor. The verification's flat-amplification-of-a-floor was a real
mechanism, but the floor is not per-camera-constant — it varies per fibre within a camera (edge dips +
gradient), so a per-camera additive term misses it (and the noisy per-camera estimate adds stack
structure). **The fix needs per-fibre granularity in the flat/extraction domain, not a sky pedestal.**
`banding-fix` is a diagnosis + falsification record; `skyPedestal.py` stays config-gated (default OFF),
nothing merged. Next direction (needs decision): a per-fibre-along-slit flat/throughput correction, or
investigate why the fibre flat leaves the per-fibre benchside banding in COUNTS.

### Deferred beyond the striping fix
- New selection providers: external broadband-image (e.g. LSST) masks, manual / GUI-defined masks.
- Full offset/blank-field sky, including multi-frame combination.
- Cross-camera sky sharing (requires fixing the per-camera throughput normalisation).

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
