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

**A resumed run refuses products made by an older version of the pipeline.** Before any work it
checks the flat products, the fibre-flat corrections and every RSS in the output folder. It
lists the ones that cannot be reused, then stops. Re-run with `clobber = true` to rebuild them.

The checks:
- calibration products must carry their current recipe stamp (`RECIPE`/`RCPHASH`; see
  `Utils/recipes.py`) and have been extracted with this run's `extraction_method` (`EXTMETH`);
- a base RSS must record its throughput (FIBERMAP `THRUPUT`);
- a fibre-flat-fielded RSS must be at the current `FFVER`.

Reusing older products would mix the old fibre-flat errors into the new output: the throughput
divided twice, fibres shifted by one above a dead fibre, and cosmic-ray-cleaned lamp flats.

Science, standard and sky frames are not refused but **redone**: their products move to
`stale/<timestamp>/` when the bias, dark, cosmic-ray settings (`CRID`, which includes how
flagged pixels are used) or extraction method (`EXTMETH`) differ from this run's.

## Disk space

A night needs a lot of disk. Each frame's extraction pickle carries all 24 cameras' traces
(about 3 GB), and every frame leaves float32 copies of itself and three RSS stages. A 15-frame night
with 3 twilights and 3 arcs needs roughly 160 GB at its peak. Outputs go to `output_dir`. If that is
not set, they go to `reduced/` next to the first science file.

Ray fails earlier than a full disk. It refuses to create or spill objects once its scratch disk is
more than **95% full**, i.e. with about 50 GB still free on a 1 TB disk. On macOS that disk is
usually the same one as your outputs.

**Pre-flight check.** Before anything is written, the pipeline estimates what the run will write
and compares it with the space really free on each disk it writes to. Products that already exist
cost nothing, so a resumed run needs less. It prints a `DISK SPACE` report and then:

- **enough room:** it carries on;
- **space may run out:** it asks `Proceed anyway, or terminate the run? [p/t]`. Without a terminal
  (a batch or background run) it terminates;
- **not even the calibrations fit:** it stops without asking.

The estimate comes from measured product sizes (`Utils/diskSpace.py`):

| Product | Size at peak |
|---|---|
| Bias-corrected or flat-corrected copy of a frame | 2 × the raw file (float32) |
| Extraction pickle | the night's 24 trace pickles × 1.05 (about 3 GB) |
| Science, standard or sky frame | its extraction, 3 RSS stages × 3 colours × 0.25 GB, a cosmic-ray mask, and 0.12 GB per cube colour with `generate_cubes` |
| Twilight flat | 3 extraction pickles, 2 flat-corrected copies, a 3-colour RSS |
| Arc (night arcs) | 1 extraction pickle |
| Traces, flat products | 24 × 120 MB; 5 GB |
| Ray scratch | the object store (30% of RAM) + 1 extraction pickle |

| Key | Default | Meaning |
|---|---|---|
| `disk_margin` | 1.2 | Safety factor on the estimate |
| `min_free_gb` | 5 | GB kept free on a disk that holds no Ray scratch |
| `skip_disk_check` | false | Skip the check and its prompt |
| `scratch_dir` | unset | Put Ray's spill folder on a larger disk |

**Finder versus the real free space.** Finder's "available" figure counts purgeable space:
cloud-drive copies (Box, iCloud, Google Drive) and snapshots, which macOS frees only on demand.
Ray and this check count only the space that is really free, as `df -h` shows it.

If a run still fills the disk, it ends with `❌ Failed after …: out of disk space: N GB free on …`.

## What happens, in order

Each stage writes a product to disk and stamps its headers so later runs can skip it.

| # | Stage | What it does | Key output |
|---|-------|--------------|------------|
| 0 | **Validate** | checks calib files/keys; repairs MEFs with missing camera extensions | — |
| 1 | **Bias subtraction** | applies the mode-matched master bias + a per-frame edge-DC offset to *every* frame | `bias_corrected/*_bias_corrected.fits` |
| 2 | **Fibre tracing** | locates each fibre on the detector from the flats (falls back to packaged master traces) | `traces/LLAMAS_{color}_{bench}_{side}_traces.pkl` |
| 3 | **Arc refinement** *(optional)* | refines the wavelength solution from night arcs (`refine_arc=true`); else uses the packaged reference arc | `arcs/…` |
| 4 | **Pixel flat** | builds the 2-D detector flat and divides it into the science frames | `extractions/flat/pixel_maps.fits` |
| 5 | **Extraction** | pulls a spectrum from every fibre (`horne` default, alias `optimal`; `boxcar` available); cosmic-ray pixels are missing data; renders the white-light image | `extractions/*_extract.pkl`, `*_whitelight_fullpipeline.fits` |
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

## What the terminal shows

By default (`terminal_verbose = false`) the terminal shows one line per stage, with a spinner while
it runs and a tick and elapsed time when it finishes. Everything else goes to the log file.

- **Warnings** print as they happen. Repeats of the same warning within a stage (for example the
  same message for each of the 24 cameras) print once, followed by `×N more like … — see log`
  when the stage finishes.
- **Fibre-flat summary.** After the wavelength and sky stage, a short table shows how flat the sky
  is within each benchside after the fibre flat (see the [Flat README](../../llamas_pyjamas/Flat/README.md)),
  with any checks that fired listed under it.
- **Run report.** Every run ends with a few lines, in verbose mode too:

```
✅ Done in 42m10s
   Output      15 science frames → 45 RSS, 30 cubes in /…/extractions
   Traces      24 cameras, existing traces
   Fibre flat  twilight · sky scatter after flat 0.8–2.1% (before 2.0–4.8%)
   Warnings    3 (Sky 2, Pipeline 1) — see log
   Log         /…/logs/llamas_pipeline_20261028_101500.log
```

A run that stops early ends with `❌ Failed after …` and the reason (for example a pre-flight
config error, which is also printed above it).

## What to check

- **Wavelength QA** in `reduced/QA/` — per-frame HTML/PNG showing the arc solution quality.
- **White-light images** (`*_whitelight_fullpipeline.fits`) — a quick look that extraction and tracing
  worked; open in DS9 or the CubeViewer.
- **The log** in `reduced/logs/llamas_pipeline_*.log` — every stage records what it reused vs rebuilt.

⟵ [Setup & config](01-setup-and-config.md) · Next: [Registration ⟶](03-registration.md)
