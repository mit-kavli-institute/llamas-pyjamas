# Sky-subtraction striping — Phase 2 diagnosis (root cause)

Field: **J1613+0808**, green, 8 dithers (the deepest may26 field, worst striping). Analysis is on the
delivered RSS (`SKY`/`SKYSUB`/`COUNTS`/`MASK`) plus the combined green cube. Scripts:
`llamas_pyjamas/Sky/diagnosis/sky_residual_diag.py`.

## Root cause (summary)

The diagonal striping in stacked images is a **per-camera / per-fibre ADDITIVE sky-subtraction
residual**, coherent across dithers, that the rotated dither groups cross-hatch into diagonal stripes.
It has two parts:

1. a **positive interior floor** in each camera's fibres (~+1 count median, up to **+4–5 in benchsides
   1A and 2A**, ~0 in 4A/2B) — additive light the sky model does not capture; and
2. strong **negative dips at the slit/camera edges** (edge fibres over-subtract, down to ≈ −10 counts).

It is **additive, not a sky-scaling error**: on blank fibres the residual does **not** scale with the
sky-model level (`corr(floor, SKY) = −0.07`). An earlier apparent flux-dependence was an artefact of
object continuum contaminating source fibres (see caveat below), and vanishes on blank fibres.

In the stacked diffuse regime the striping is ~**1.9e-19 FLAM RMS — about 50% of the faint diffuse
signal** — so it is the limiting systematic for Lyα work, exactly as expected.

## Evidence

![Blank-fibre residual and stack striping](figures/blank_residual.png)

*Left: the sky-subtraction residual on blank fibres, per fibre — a repeating per-camera pattern
(positive interior floor + negative edge dips). Right: the stacked white-light high-pass in blank,
source-masked regions — the diagonal cross-hatch of the two rotation groups.*

![Per-fibre floor, flat test, OH vs continuum](figures/perfibre_floor.png)

*The residual floor is structured by benchside (1A/2A/2B elevated); OH-line residuals (~10 counts) are
a separate, fairly uniform component.*

Key measurements (J1613 green, blank fibres):
- residual floor median **+1.09**, MAD 1.4; per-benchside offsets 1A +4.2, 2A +4.8, 3B +1.6, … 4A ≈ 0.
- `corr(floor, SKY level) = −0.07` → additive, not multiplicative/throughput scaling.
- stacked blank-region high-pass RMS **1.9e-19 FLAM** (~50% of the core diffuse signal).

## What this points to (Phase 3 fix directions)

- **Slit-edge over-subtraction** (the negative dips): the sky-model amplitude/LSF is wrong for
  edge fibres — an across-slit LSF/throughput effect. This is what the **derivative refinement**
  (α·S + β·S′ + γ·S″) targets; but per RS's domain principle it should run in the **pkl/xshift
  domain**, not on the RSS in wavelength space.
- **Positive additive floor** (worst 1A/2A): a diffuse additive component (scattered continuum /
  instrumental floor) the OH-anchored sky model doesn't remove — needs a per-camera/per-fibre
  **continuum pedestal** term, or upstream scattered-light handling. Echoes the earlier
  between-line-residual root cause (diffuse scattered continuum, benches 1–2).
- Both are **per-camera coherent**, which is precisely why they survive co-adding and cross-hatch into
  stripes; a correct per-camera treatment should remove most of the striping.

## Caveats

- **Metric confound (resolved):** a "between-line floor" measured on *all* fibres is contaminated by
  real object continuum in source fibres (`corr(floor, object excess) = +0.88`). The clean signal is
  on **blank fibres only** — used for all conclusions here.
- **Run-to-run non-determinism:** the pipeline is not bit-reproducible at the extraction/wavelength
  level (WAVE/xshift jitter), so residual measurements carry some run-to-run scatter; the per-camera
  *pattern* is robust across the 8 dithers.
- Verified on J1613 green; should be checked on the other channels/fields before finalising a fix
  (blue is skipped by the derivative stage; red has different OH).
