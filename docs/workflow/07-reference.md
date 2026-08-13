# Reference & troubleshooting

⟵ [Back to overview](README.md)

Quick reference for the whole workflow: commands, directory layout, file formats, key config
parameters, and common problems. Deep details live in each stage page and in the source; this page is
for looking things up.

## Command cheat-sheet

| Task | Command |
|------|---------|
| Classify a raw night, write a config | `python -m llamas_pyjamas.Utils.reduxSetupGUI RAW_DIR -o config.txt` |
| Reduce a night | `python reduce.py config.txt` *(from `llamas_pyjamas/`)* |
| Open the CubeViewer | `python -m llamas_pyjamas.CubeViewer [file]` |
| Combine a field → cube | `python -m llamas_pyjamas.Combine.combineField --dir REDUCED --object J2151 --cube` |
| Combine a field → image | `python -m llamas_pyjamas.Combine.combineField --dir REDUCED --object J2151 --band LO HI --png` |
| Build/update a master bias | `python -m llamas_pyjamas.Scripts.update_bias_master RAW_DIR` |
| Fetch flux-standard catalogue | `python -m llamas_pyjamas.Scripts.fetch_standards` |
| Repair a MEF with missing cameras | `python -m llamas_pyjamas.DataModel.validate in.fits -o fixed.fits` |

The CubeViewer menus: **File** (open RSS/cube) · **WCS** ([registration](03-registration.md)) ·
**Combine** ([stacking](04-combining-dithers.md), narrowband) · **Extraction**
([optimal spectra, Gaia anchor](05-science-products.md)) · **Sensitivity** (flux calibration).

## Directory layout of a reduction

Rooted at `output_dir` (default `<raw_night>/reduced/`):

```
reduced/
├── bias_corrected/     bias-subtracted frames
├── traces/             fibre traces: LLAMAS_{color}_{bench}_{side}_traces.pkl (24)
├── arcs/               extracted night arcs (only if refine_arc)
├── extractions/        ← the main per-exposure products live here:
│   ├── {name}_RSS_{red,green,blue}.fits     per-channel RSS  ← key deliverable
│   ├── {name}_whitelight_fullpipeline.fits  per-exposure white-light image
│   ├── flat/           pixel_maps.fits, twilight/, dome_rss/
│   └── masks/          cosmic-ray masks
├── cubes/              per-exposure cubes (only if generate_cubes=true)
├── combined/           stacked products: <field>_cube_{blue,green,red}.fits, mosaics
├── QA/                 wavelength-calibration QA (HTML/PNG/CSV)
├── QA_registration/    {..}_reg_qa.png registration QA plots
└── logs/               llamas_pipeline_*.log
```

## File formats

**Per-exposure RSS** (`*_RSS_{color}.fits`) — one row per fibre (≈2389 × 2048 per channel):

`PRIMARY` · `SKYSUB` · `ERROR` · `MASK` · `COUNTS` · `SKY` · `WAVE` · `FWHM` · `FIBERMAP` ·
`SKYRESID` · `FLAM` · `FLAM_ERR` · `FIBERWCS`.
Use `FLAM`/`FLAM_ERR` for flux-calibrated science; `WAVE` is per-fibre and native (unresampled);
`FIBERWCS` carries the registered per-fibre RA/DEC + detector X/Y. Full table in
[stage 2](02-running-the-reduction.md#the-per-exposure-product-the-rss-file). Fibre solid angle is in
the `FIBAREA` primary keyword (≈0.44 arcsec²); surface brightness = flux / `FIBAREA`.

**Combined cube** (`combined/<field>_cube_{color}.fits`): `PRIMARY` (data, 3-D WCS) · `VAR` ·
`COVERAGE` · `NEXP` · `WAVELENGTH` table. The header lists the contributing RSS files
(`NRSSFILE`, `RSSFILn`) so opening the cube rebuilds the super-RSS for extraction.

## Config keys

The config is `key = value` text (`#` comments; comma-separated lists; booleans `true/false`; **do not
quote paths**). The authoritative, annotated list is
[`example_config.txt`](../../llamas_pyjamas/example_config.txt). The ones you touch most:

| Key | Meaning |
|-----|---------|
| `science_files` | comma-separated science exposures (set by the setup GUI) |
| `red_flat_file` / `green_flat_file` / `blue_flat_file` | per-colour flats (**required**) |
| `red_twilight_flat` / … | per-colour twilights (fibre-to-fibre throughput; recommended) |
| `red_arc_file` / … , `refine_arc`, `generate_new_wavelength_soln` | wavelength-solution controls |
| `flux_standard_files` | standard-star exposures for the sensitivity function |
| `slow_bias_file` / `fast_bias_file` | master bias per readout mode |
| `output_dir`, `trace_output_dir`, `extraction_output_dir`, `cube_output_dir` | output locations |
| `extraction_method` (`boxcar`/`horne`/`optimal`), `boxcar_halfwidth` | extraction |
| `sky_subtract`, `sky_selection_method` | sky subtraction |
| `wave_frame` (`heliocentric`/…) | wavelength reference frame |
| `apply_fibre_flat`, `twilight_throughput` | throughput corrections |
| `generate_cubes` | per-exposure cubes (default off; stack later instead) |
| `clobber` | rebuild everything instead of resuming |
| `ray_num_cpus` | parallelism |

## `combineField`

```
--dir DIR --object NAME        discover a field's RSS by OBJECT prefix (or list RSS files directly)
--cube                         build an (RA,DEC,wave) cube (else a 2-D image)
--band LO HI                   wavelength window for an image (default: full range)
--channels {blue,green,red}    channels to combine (default: all)
--units {sb,flux}              surface brightness (default) or flux
--weight {ivar,uniform,exptime}
--kernel {gaussian,tophat}     --fwhm ARCSEC   --pixscale ARCSEC
--scale-transparency           scale exposures to a common throughput (in-field source)
  --scale-radius, --scale-sources, --scale-source RA DEC
--keep-bad-fibres              keep strongly-negative broken fibres (default: masked)
--dwave A                      cube wavelength step (default: native median)
-o OUT.fits                    --png (preview)
```

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `reduce.py` errors on `--config` | the arg is **positional**: `python reduce.py config.txt` |
| A frame won't register / rotation looks wrong | fix it in the CubeViewer **WCS ▸ Refine WCS interactively…**, or **Auto-register block** holding the correct rotation ([stage 3](03-registration.md)) |
| Missing-camera / extension errors on raw frames | run `python -m llamas_pyjamas.DataModel.validate` on the MEF first |
| Only the green cube appears | build with the CubeViewer combine (all channels on a shared grid) or `combineField` without `--channels`; opening one cube auto-loads its colour siblings |
| Negative "holes" in the co-add | broken fibres — masked by default; do **not** pass `--keep-bad-fibres` |
| Co-add looks noisier than a single exposure (red) | coherent sky-subtraction residuals, not a weighting error; they don't average down ([stage 4 caveat](04-combining-dithers.md#a-caveat-sky-subtraction-striping)) |
| Displayed image looks soft / PSF too big | the gridding kernel softens the *rendered* image only; lower the combine **Kernel FWHM**; point-source extraction is unaffected ([stage 4](04-combining-dithers.md)) |
| "latex could not be found" when anchoring to Gaia | resolved in current code (gaiaxpy no longer forces LaTeX); update the package |
| Gaia anchor says *approximate* | the source has no Gaia XP spectrum (faint); it fell back to broadband photometry ([stage 5](05-science-products.md#absolute-flux-anchor-gaia)) |

## Other documentation

- Repository [`README.md`](../../README.md) — install, auxiliary-file downloads, QuickLook GUI.
- [`example_config.txt`](../../llamas_pyjamas/example_config.txt) — every config key, annotated.
- Sphinx API docs — `cd docs && make html`.
- Tutorial notebooks — [`llamas_pyjamas/Tutorials/`](../../llamas_pyjamas/Tutorials/).

⟵ [Back to overview](README.md)
