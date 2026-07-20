# Stage 2 — Running the reduction (`reduce.py`)

⟵ [Setup & config](01-setup-and-config.md) · Next: [Registration ⟶](03-registration.md)

With a config file in hand, one command reduces the whole night from raw MEF frames to
**per-exposure, flux-calibrated RSS** (row-stacked spectra).

Source: [`llamas_pyjamas/reduce.py`](../../llamas_pyjamas/reduce.py)

## Run it

```bash
cd llamas-pyjamas/llamas_pyjamas
python reduce.py /path/to/raw_night/llamas_redux_config.txt
```

The config path is a **positional** argument (there is no `--config` flag). Equivalently:
`python -m llamas_pyjamas.reduce /path/to/config.txt`.

The pipeline is **resume-aware**: it reuses on-disk products from a previous run, so re-running after
an interruption picks up where it left off. Set `clobber = true` in the config to force everything
from scratch. Heavy steps run in parallel via [Ray](https://www.ray.io/) (tune with `ray_num_cpus`).

## What happens, in order

Each stage writes a product to disk and stamps its headers so later runs can skip it.

| # | Stage | What it does | Key output |
|---|-------|--------------|------------|
| 0 | **Validate** | checks calib files/keys; repairs MEFs with missing camera extensions | — |
| 1 | **Bias subtraction** | applies the mode-matched master bias + a per-frame edge-DC offset to *every* frame | `bias_corrected/*_bias_corrected.fits` |
| 2 | **Fibre tracing** | locates each fibre on the detector from the flats (falls back to packaged master traces) | `traces/LLAMAS_{color}_{bench}_{side}_traces.pkl` |
| 3 | **Arc refinement** *(optional)* | refines the wavelength solution from night arcs (`refine_arc=true`); else uses the packaged reference arc | `arcs/…` |
| 4 | **Pixel flat** | builds the 2-D detector flat and divides it into the science frames | `extractions/flat/pixel_maps.fits` |
| 5 | **Extraction** | pulls a spectrum from every fibre (`boxcar` default; `horne`/`optimal` available); removes cosmic rays; renders the white-light image | `extractions/*_extract.pkl`, `*_whitelight_fullpipeline.fits` |
| 6 | **Wavelength cal** | transfers the arc solution onto each extraction | `*_corrected_extractions.pkl` |
| 7 | **Sky subtraction** | builds and subtracts a 1-D sky model per fibre (`sky_subtract=true`) | `*_sky1d_extractions.pkl` |
| 8 | **Heliocentric shift** | corrects the wavelength scale to the heliocentric/barycentric frame (`wave_frame`) | header `VELFRAME/HELIOVEL/VELCORR` |
| 9 | **RSS generation** | assembles the per-fibre spectra into one RSS file per colour + provisional per-fibre RA/DEC | `*_RSS_{red,green,blue}.fits` |
| 10 | **Fibre-flat** | applies the twilight fibre-to-fibre throughput correction | `*_RSS_{color}_FF.fits` |
| 11 | **Consolidate + QA** | collapses the per-stage files into one clean RSS per colour and runs wavelength QA | final `*_RSS_{color}.fits`, `QA/` |

Flux calibration (building a sensitivity function from the standard-star exposures and applying it) is
handled with the standards you tagged in stage 1; the calibrated flux lands in the RSS `FLAM`
extension (below).

## The per-exposure product: the RSS file

The deliverable for each exposure is three FITS files — `*_RSS_red.fits`, `*_RSS_green.fits`,
`*_RSS_blue.fits`. Each holds **one row per fibre** (≈2389 fibres × 2048 wavelengths per channel) as a
stack of image planes plus two fibre tables:

| HDU | Name | Contents |
|-----|------|----------|
| 0 | `PRIMARY` | header: pointing, rotator, exposure, wavelength range, `VELFRAME`, `FIBAREA` |
| 1 | `SKYSUB` | sky-subtracted counts (the working flux plane) |
| 2 | `ERROR` | 1-σ uncertainty per fibre |
| 3 | `MASK` | data-quality mask |
| 4 | `COUNTS` | extracted counts *before* sky subtraction |
| 5 | `SKY` | the sky model per fibre |
| 6 | `WAVE` | per-fibre wavelength array (native, non-resampled) |
| 7 | `FWHM` | per-fibre resolution |
| 8 | `FIBERMAP` | table: `FIBER_ID, BENCHSIDE, FIBER_TYPE, RA, DEC` |
| 9 | `SKYRESID` | sky-subtraction residual estimate |
| 10 | `FLAM` | **flux-calibrated** spectrum (erg s⁻¹ cm⁻² Å⁻¹) |
| 11 | `FLAM_ERR` | flux-calibration uncertainty |
| 12 | `FIBERWCS` | table: per-fibre `RA/DEC` + detector `X/Y` + astrometry provenance |

Two design points worth knowing downstream:

- **Wavelengths are per-fibre and never resampled here.** The single spectral resample happens only
  when a cube is built ([stage 4](04-combining-dithers.md)) — *resample once, at the end*.
- **`FIBERWCS` is separate from `FIBERMAP`** so the astrometry can be re-solved without touching the
  data. Stage 9 writes only a *provisional* header-based WCS; [stage 3](03-registration.md) replaces
  it with a Gaia-anchored one.

## What to check

- **Wavelength QA** in `reduced/QA/` — per-frame HTML/PNG showing the arc solution quality.
- **White-light images** (`*_whitelight_fullpipeline.fits`) — a quick look that extraction and tracing
  worked; open in DS9 or the CubeViewer.
- **The log** in `reduced/logs/llamas_pipeline_*.log` — every stage records what it reused vs rebuilt.

⟵ [Setup & config](01-setup-and-config.md) · Next: [Registration ⟶](03-registration.md)
