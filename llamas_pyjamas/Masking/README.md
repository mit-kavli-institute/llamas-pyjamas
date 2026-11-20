# Masking: cosmic-ray cleaning

`cosmicLlamas.py` flags cosmic rays with L.A.Cosmic (the `lacosmic` package). It runs in
`guiExtract.process_trace` on each camera. That happens after bias, dark, edge DC and the pixel
flat, and before extraction, which treats the flagged pixels as missing data (see "Cleaned
pixels are missing data" below).

**Only science, flux-standard and sky frames are cleaned.** Lamp flats, twilights and arcs
never are. On a lamp flat, L.A.Cosmic takes the bright, sharp fibre cores for cosmic rays and
lowers them. Up to 460,000 pixels per green or blue detector were changed on 20261015_16. This
left about 9,400 false narrow dips in the green fibre flat, and each one becomes a fake emission
bump in the science. Any cosmic ray in a calibration frame is left as it is. A lamp flat is
the single exposure configured for its colour, and a 0.07–0.5 s exposure collects few of them.

## Settings

| Config key | Default | Meaning |
|---|---|---|
| `remove_cosmic_rays` | `true` | Master switch |
| `cosmic_ray_channels` | `blue, green` | Channels to clean: any of `red`, `green`, `blue`, or `none` |
| `cosmic_ray_threshold_{c}` | red 10, green/blue 6.5 | Detection threshold in σ (L.A.Cosmic `cr_threshold`). Higher flags fewer, brighter cosmic rays. |
| `cosmic_ray_contrast_{c}` | red 9, green/blue 5 | Minimum contrast of a candidate against the fine-structure image (`contrast`). Higher protects sharp real features, such as sky lines. |
| `cosmic_ray_neighbor_threshold_{c}` | 0.7 | Threshold in σ for pixels next to a cosmic ray (`neighbor_threshold`) |
| `cosmic_ray_maxiter` | 2 | L.A.Cosmic iterations (all channels) |

- **Where to set them:** in the config, or in the setup GUI under **Configuration Options →
  Bias / Cosmic rays**. Unticking every channel there writes `cosmic_ray_channels = none`.
- **Defaults:** they come from `constants.LACOSMIC_DEFAULTS` and `LACOSMIC_COLOR_OVERRIDES`, so a
  config without these keys behaves exactly as before.
- **Not settable:** gain and read noise are per-detector values from the lab table (`CAMSN`), not
  settings.
- **Pre-flight errors:** an unknown `cosmic_ray_*` key, an unknown channel, `none` together with a
  channel, a non-positive threshold, or a `maxiter` that isn't a whole number of at least 1.
- **Pre-flight warnings:** a threshold set for a channel that is off, or a neighbour threshold
  above the detection threshold.

Only science, flux-standard and sky frames follow these settings. Lamp flats, twilights and
arcs are never cleaned (`Utils/recipes.py`).

## Cleaned pixels are missing data

L.A.Cosmic returns a flag image and an image with the flagged pixels replaced by the median of
their neighbours. **Only the flags are used.** The extraction treats every flagged pixel as
missing data (`Extract/README.md`): a boxcar sample whose aperture holds one has no data, and a
Horne sample is measured from the rest of its profile, or has no data when less than half of
it is left. Such samples are NaN in the RSS with `MASK` bit 256 (`MASK_CR_MISSING`), and the
cube and spectrum resamplers never interpolate across them, so other fibres and other exposures
fill them.

Before 2026-11-15 the replacement values were extracted as if they were data. Wherever
L.A.Cosmic flagged a real sharp feature, the product was biased, with nothing in the RSS to
show it, and the lowered counts also lowered the error, so inverse-variance co-adds weighted
those samples up.

**What a false flag costs.** Data, not bias. A real feature flagged in one exposure is filled
by the other exposures; one flagged in every exposure leaves a gap that the coverage maps show.
`CRMISSF` in each RSS gives the share of samples lost to cosmic rays.

## Red

Red was switched off in April 2026 (`25e637d`) because bright OH sky lines were flagged as cosmic
rays. A test on 2026-10-16 compared two consecutive 1350 s dithers. Sky lines and hot pixels stay
at the same detector pixels from frame to frame, but cosmic rays don't, so a pixel flagged in both
frames is a false detection.

| Red, per camera | Flags per frame | Same pixel in both frames |
|---|---|---|
| No dark, default thresholds (6.5 / 5) | 7,600–33,000 | 14–33% |
| No dark, red thresholds (10 / 9) | 3,500–6,700 | 4–20% |
| Dark subtracted, red thresholds | 3,300–5,000 | 0.9–7.5% |
| Green, for comparison | 3,600–9,600 | 9–15% |
| Blue, for comparison | 2,000–3,400 | 0.1–1.1% |

With the dark subtracted (`dark_files`), red behaves about as well as green. In 3A, 3B and 4A,
some bright sky-line cores are still clipped, each by about 500–700 DN. If red is switched on, the
run warns about every frame that has no dark.

## Darks

When bias-first has subtracted a master dark (`DARKSUB`), the dark is no longer in the data, but
its noise is. L.A.Cosmic therefore gets:
- **a smoothed copy of the dark as `background`**, so its noise model includes it. It adds the
  background back before detection, so the full-resolution dark would turn every hot pixel into a
  false cosmic ray.
- **the dark's hot pixels as `mask`**, so their residuals are neither flagged nor used for
  replacement.

Without this, a dark-subtracted blue frame gave about 48,800 false flags instead of about 3,000.
See `Bias/README.md`.

## Headers and resuming

| Where | Keyword | Meaning |
|---|---|---|
| Extension (in the extraction) | `CRCLEAN` | Whether cleaning ran; if not, the reason is in `CRWHY` (`channel off`, `remove_cosmic_rays false`) |
| Extension (in the extraction) | `CRNPIX`, `CRTHRESH`, `CRCONTR`, `CRNEIGH` | Pixels flagged, and the thresholds used |
| Primary (extraction and RSS) | `CRID` | Identity of the cleaning: a hash of the enabled channels' thresholds and `maxiter`, or `off` |
| Primary (extraction and RSS) | `CRCHAN` | Channels cleaned, e.g. `green,blue` |
| Primary (extraction and RSS) | `CRMODE` | How flagged pixels are used: `mask` (missing data), or `off` |
| Extension (in the extraction) | `CRNMISS`, `CRNPART` | Samples lost to cosmic rays, and Horne samples measured around them |
| Primary (RSS) | `CRMISS`, `CRPART`, `CRMISSF` | The same, summed over the channel, and `CRMISS` as a % of samples with counts |

A resumed run compares each science, standard and sky RSS's `CRID` with the current settings. A
frame whose settings changed is redone: its products move into `{output_dir}/stale/<timestamp>/`,
as for a changed dark.
- **Old RSS files:** one without `CRID` counts as the old default cleaning (`LEGACY_CRID`), so
  existing reductions are only redone if the settings now differ from that default.
- **What changes the ID:** the thresholds of a channel that is off don't count, so tuning a
  disabled red never triggers a redo.
- **How flagged pixels are used is part of the ID.** Frames cleaned before cosmic-ray pixels
  became missing data (and RSS files without `CRID`) are redone on resume; frames with cleaning
  off (`CRID = off`) are not.
- **Masks:** the cosmic-ray masks are written to `{mask_output_dir or extraction_output_dir}/masks/`,
  in pipeline orientation (`CRORIENT`). Each extension records its camera (`COLOR`, `BENCH`,
  `SIDE`, `CAMSN`, from the camera's own header) with `CRNPIX`, `CRNMISS`, `CRNPART` and the
  thresholds; masks written before 2026-11-15 have only `EXTVER` and `CRNPIX`.
- **Run summary:** the end of a run reports, per channel, the median pixels flagged per frame
  and the share of samples lost to cosmic rays (`CRMISSF`, median and maximum over frames). It
  is a reported number, with no threshold.
