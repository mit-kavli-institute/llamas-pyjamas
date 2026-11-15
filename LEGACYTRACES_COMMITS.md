# `legacytraces` commit summary

This file lists every commit on `legacytraces` that is not on `main`, so any change can be
found and rolled back. Update it with each new commit.

The branch starts from `main` at `3739110`. It went through the `mapping-fix`,
`sensfunc-standard-match` and `combinecube` work, which was merged in along the way.

## How to roll back

**Undo one commit and keep everything after it:**
```
git revert <sha>
```
- This adds a new commit that reverses the old one, and is safe on a shared branch.
- Read that commit's **Depends on / depended on by** notes first. Later commits that edit the
  same lines will conflict, or will break if their base is removed.
- Revert dependent commits newest-first.

**Go back to a commit and drop everything after it:**
```
git reset --hard <sha>
```
- **Destructive.** It discards every later commit **and all uncommitted work**.
- Commit or stash first, or make a safety branch: `git branch backup-before-reset`.

**See exactly what a commit changed:**
```
git show --stat <sha>
git show <sha> -- <path>
```

**Products.** Several commits change pipeline outputs. After rolling one back, re-run the
affected nights with `clobber = true` so that no products from the other version are reused.

## 1. Commits made before this summary (2026-09-03 to 2026-10-28)

| SHA | Date | Subject | What it does | Notes |
|---|---|---|---|---|
| `3452610` | 09-03 | Fix benchside 2A fibre misregistration and the live/fibremap index mixup | `resolve_trace_slots`: benchside 2A had silently renumbered 129–298 of its fibres. Also fixes live vs fibremap indexing. | Trace, Utils, reduce.py, tests |
| `0ece50a` | 09-03 | Recognise flux standards in headers repaired from the TEL block | The flux-standard crossmatch reads sexagesimal and TEL-block pointings, so "Build Sensitivity Function" works on GD108 frames. | Flux, CubeViewer |
| `01ab939` | 09-03 | Merge branch 'mapping-fix' into sensfunc-standard-match | Merge of the two commits above. | merge |
| `c810a2c` | 09-04 | Stop the slot resolver dropping a live fibre instead of an end ghost | Fixes the slot resolver on benchsides with no dead-fibre entry: green 3B, 3A and 1B had been one pitch off. | builds on `3452610` |
| `3c071ef` | 09-04 | Merge trace-slot resolver fix into sensfunc-standard-match | Merge of `c810a2c`. | merge |
| `45714eb` | 09-29 | Make master tracing robust: fingerprint registration, bias fallback, centroid walk | Fibre fingerprint registration, bias fallback, and a centroid walk in the tracer, plus the fingerprint LUT. | Trace, Bias, LUT (+7.8k lines, mostly LUT) |
| `128a2ab` | 10-21 | Add the RSS stage contract and make the stage producers safe | `exposure_key`, `rss_stage`, `final_rss`, `check_stage_consistency` in `File/llamasRSS.py`. Stage producers refuse to re-apply a stage. | base of the stage work below |
| `3bc8cd4` | 10-21 | Mask zero-count pixels and stop counting masked samples as zero | One bad-pixel rule for every RSS reader (`bad_pixel_mask`). Zero-count pixels are masked. | 19 files: Cube, Combine, CubeViewer |
| `5f4ae43` | 10-21 | Guard the combiner and flux calibration against wrong-stage inputs | The combiner needs FLAM; flux calibration checks the stage. | Combine, Flux |
| `0f1bb9f` | 10-21 | Write FLAM on each exposure's final stage | `apply_fluxcal --stage`; FLAM lands on the file the combiner uses. | Flux, CubeViewer |
| `695f2f3` | 10-21 | Register on each exposure's final stage and pick it in the GUI | WCS registration reads and writes each channel's final stage. | CubeViewer |
| `04ad102` | 10-21 | Combine each exposure from its final stage | `build_super_rss` resolves every exposure to its most-corrected stage. | Combine, docs |
| `dcc6fdc` | 10-21 | Expose the SKYRESID sky penalty on combineField | `--sky-penalty`. Raises if the SKYRESID extension is missing. | Combine |
| `6f602e7` | 10-27 | Audit legacy traces on reuse and make the mastercalib fallback per camera | Old trace pickles (no `comb_ok`) are audited. The mastercalib fallback works per camera. | Trace, Utils, reduce.py |
| `8ef92f2` | 10-28 | Remove Lyman-alpha references from docs and trim example_config comments | Comments and docs only. | |
| `71dc88f` … `55db52c` (20 commits) | 10-28 | Trim verbose comments in Trace/, reduce.py, Extract/, File/, Image/, Arc/, Bias/, DataModel/, Utils/, Flat/, Flux/, Scripts/, Sky/, Combine/, Cube/, CubeViewer/, QA/, Postprocessing/, GUI/, Tests/ | **Comments and docstrings only.** Code was verified unchanged by AST comparison. Each commit touches one folder. | safe to revert singly |
| `868ae4e` | 10-28 | Summarise reduce.py warnings between phases and end runs with a short report | The run-report summary. `assess_fibre_flat` (the colour × benchside sky-scatter table) replaces the per-benchside throughput warning. Flat and twilight RSS are made without sky subtraction. | Flat, Utils, reduce.py |

## 2. Uncommitted work committed on 2026-11-14

This work existed only in the working tree and was split into feature commits so each can be
rolled back. Mixed files were split hunk by hunk. In each commit's state:
- the package imports;
- that feature's tests pass;
- the full `Tests/` suite shows only the two failures that were already there:
  `test_wcs_llamas::test_pointing_from_header_decimal_rot_and_none`, and
  `test_flat_processing` failing to import `LlamasFlatFielding`.

| SHA | Subject | What it does | Depends on / depended on by |
|---|---|---|---|
| `cd72b11` | Shift mastercalib traces onto the night's flats when a camera fails to trace | See below. | Depended on by `c069aed` and `7df3aed` (same lines of `reduce.py`). |
| `c069aed` | Subtract master darks at bias-first and add per-channel cosmic-ray settings | See below. | On `cd72b11`. Depended on by `7df3aed` (`reduce.py`, `guiExtract.py`, `traceLlamasMaster.py`, `llamasIO.py`). |
| `7df3aed` | Stop trace pickles carrying the master bias and check disk space at pre-flight | See below. | On `cd72b11` and `c069aed`. |
| `5d1557f` | Add the twilight benchside tie and lamp-throughput modules (not yet wired in) | `Flat/twilightTie.py`, `Flat/lampThroughput.py` and their tests in `Test/`. `reduce.py` imports neither. | Independent. |
| `69e1bb7` | Add the FEG21 flux table for the standard EG21 | `LUT/standards/flux/feg21.dat` and its `index.dat` entry. | Independent. |

**`cd72b11`: shifted-master trace fallback.**
- A camera that fails to trace gets its master trace shifted onto the night's flat
  (`Trace/masterShift.py`), guarded by the fibre fingerprint.
- Trace-set checks move from `Utils/utils.py` to the new `Trace/traceValidation.py`.
- Old-format night traces are set aside and the night is re-traced.
- The proceed/terminate prompt moves to `Utils/userPrompt.py`.
- New config key: `shifted_master_require_fingerprint`.
- `traceValidation.py` already contains `slim_bloated_traces`, which is first used by
  `7df3aed`.
- In this commit's state, `masterShift` still writes a `_trimmed` copy of each flat (the
  behaviour from before `7df3aed`).

**`c069aed`: master darks and per-channel cosmic-ray settings.**
- **Darks.** `dark_files` are combined into master darks per read mode and exposure
  (`Bias/darkLlamas.py`). They are subtracted at bias-first from science, standard and sky
  frames only. Their noise is added to the errors.
- **Resume.** Stale frames are moved aside on resume (`Bias/biasFirstResume.py`).
- **Camera matching** is by header (`Bias.match_camera_hdu`). The setup GUI gains a Dark
  role.
- **Cosmic rays.** Per-channel L.A.Cosmic settings, with red off by default. Extractions are
  stamped CRCLEAN/CRWHY per camera and CRID/CRCHAN on the primary header. A resumed run
  re-reduces frames cleaned with other settings.
- **Why one commit.** Darks and cosmic rays share code: the cosmic-ray noise model uses the
  dark, and the stale-frame check covers both.
- **Products.** Science RSS change (dark subtracted). Re-run with clobber after reverting.

**`7df3aed`: master bias out of trace pickles; pre-flight disk check.**
- The master-bias cache moves to `Bias/biasCache.py`. It lived in `traceLlamasMaster`, which
  is pickled by value, so every trace carried both master biases (1.7 GB traces, 14 GB
  extractions). `slim_bloated_traces` repairs old pickles.
- The new pre-flight disk-space check (`Utils/diskSpace.py`) asks proceed/terminate.
- `GUI_extract` raises the real error.
- No `_trimmed` files are written.
- `output_dir` defaults to `<night>/reduced`.
- **After reverting:** outputs default back to the package folder again, and new traces will
  be bloated again.

The commit that adds this file comes after these five. Its SHA appears in `git log`.

## 3. Fibre-flat fixes (plan of 2026-11-14)

Background: on 20261015_16 the fibre flat raised the sky scatter within each benchside from
1.4–4.8% to 9–19%. Three causes were found:
- the fibre throughput was divided out twice;
- fibre ids were off by one above a dead fibre;
- the lamp flats were cosmic-ray cleaned, which planted false dips in them.

Each step below is one commit. Find a step's SHA with `git log --oneline --grep "<subject>"`;
SHAs are filled in after each phase.

| Step | Subject | What it does | Depends on / products |
|---|---|---|---|
| R0 + A0 `6e55b00` | Key fibre-flat tables by physical fibre id and add processing recipes | See below. | Needs a clobber re-run: older flat products are refused. |
| A1 `a8691ee` | Apply the fibre throughput once: divide the RSS by C_i over the throughput it already removed | See below. | On R0 + A0. Needs a clobber re-run; sensfuncs must be rebuilt from re-reduced standards. |
| A2 `3def65f` | Stop cosmic-ray cleaning the lamp flats and stamp the flat products with their recipe | See below. | On R0 + A0. Changes the pixel flat, so every science extraction changes: clobber re-run. |
| A3 `84cca12` | Check the fibre flat against the throughput, the twilight and its own narrow structure | See below. | On A1, A2. QA only; no product data change. |
| A4 `54f4410` | Refuse to resume from products an older pipeline version made | See below. | On A0–A2. A resumed run on an old output folder now stops; use clobber. |
| B1 `267bb2d` | Reduce each twilight once and take both throughputs from it, per colour | See below. | On A0–A4. Changes the RSS throughput for colours whose twilight was not the newest file: clobber re-run. |
| B2 `7ed8748` | Combine each lamp flat's set of exposures before tracing and flat fielding | See below. | On A2. Changes traces and flats: clobber re-run. New key `lamp_flat_combine` (default true). |
| B3 `a1f1ccf` | Normalise the Horne profile over the whole aperture so masked pixels do not bias the flux | See below. | Changes Horne fluxes and errors where pixels are masked: clobber re-run; rebuild sensfuncs. |
| B4 `b81ea8c` | Make every extraction name its recipe and check each calibration product's recipe | See below. | API change: `process_trace`, `GUI_extract` and `run_extraction` need `recipe=`. Old arcs are refused on resume. |
| — | Add fibre-flat acceptance and arc cosmic-ray check scripts | `Scripts/ffcheck/fibreflat_acceptance.py <output_dir>` prints the per-benchside acceptance numbers; `arc_cr_check.py` checks the arcs' recipe and cosmic-ray stamps. | Read-only scripts. |

**R0 + A0: physical fibre ids and processing recipes.**
- `Utils/recipes.py` adds the per-type processing recipes and their `RECIPE`/`RCPHASH` header
  stamps. The rest of the plan uses them.
- The smooth models (`Flat/flatLlamas.smooth_model_table`) and the corrections are keyed by
  **physical** fibre id, as the RSS is.
- `fibreFlat` maps each id to its live twilight row with `Utils/deadfibers.live_row_index`.
- Fixes the 2A/2B off-by-one, where 28 + 248 fibres got their neighbour's flat and the last
  fibre got none. `twilightTie` is fixed the same way.
- Both products carry `FIBIDS='physical'`, and older files are refused.

**A1: throughput applied once.**
- The RSS writer records each row's throughput divisor in a FIBERMAP `THRUPUT` column and
  its source in `TPSRC`.
- `apply_fibre_flat_to_rss` divides by `C_i / THRUPUT`, so the fibre throughput is divided
  out once instead of twice. Simulated on 20261015_16, this takes the sky scatter from
  9–19% to 1.6–5%.
- `_FF` files carry `FFVER = 2`, and an RSS without `THRUPUT` is refused.
- Flux calibration refuses a sensfunc whose `ffver` differs from the science's. CubeViewer
  records `ffver` when it builds a sensfunc.

**A2: no cosmic-ray cleaning on lamp flats.**
- `Flat/flatProcessing.py` passes `remove_cosmic_rays=False`. Before, `process_trace`
  defaulted to cleaning, so L.A.Cosmic lowered bright green and blue fibre cores. That left
  about 9,400 false dips in the green flat (a fake-emission risk), and 1.8% of green
  pixel-flat pixels at the 0.90 clip floor.
- The flat products carry the `lamp_flat` recipe, `CRCLEAN=False` and `FLATFIL<c>`.
- The corrections file carries the `twilight` recipe, `TWIFIL<c>` and `SMRCPHS`.

**A3: QA, layered and never tuned to pass.**
- **Sky scatter at three levels:** raw (`FFSB`), RSS throughput only (`FFST`) and after the
  flat (`FFSA`). A relative warning fires when the flat does worse than the throughput alone.
- **Narrow structure in the lamp shape `W_i`:** `WINRMS` and `WINDIP`, with a provisional 1%
  ceiling.
- **Twilight cross-check of the `W_i` dips:** `WIDIPN` and `WIDIPTW`.
- **Pixel-flat clip-floor share:** `FCLIPLO` per detector, with relative warnings within and
  across colours (`assess_pixel_flat`, reported after the flat stage).
- Each of these would have caught the bugs fixed in A0–A2.

**A4: stale products refused on resume.**
- Before any work, `reduce.stale_products` lists every reused product that older code made:
  - flat products or corrections without their current recipe stamp;
  - a base RSS without `THRUPUT`;
  - an `_FF` or consolidated RSS below `FFVER` 2.
- If there are any, the run stops and asks for `clobber = true`.
- On the 20261015_16 products from before these fixes, it refuses all 49.

**B1: one twilight reduction.**
- `Flat/fibreFlat.reduce_twilights` reduces each distinct twilight once: pixel flat,
  `twilight` recipe, wavelength calibration. It saves `<flat>/twilight/<stem>_twilight.pkl`,
  reused while its `.json` sidecar matches the recipe, the frame and the pixel flat.
- The RSS throughput (`reduce.twilight_throughputs` / `apply_twilight_throughput`) and the
  fibre flat's T_i (`load_twilights`) both read those pickles, each camera from its own
  colour's twilight.
- Before, path (i) (`_process_flat_for_rss`) fed the throughput from the newest pickle for
  every colour. It was skipped on resume and wrote RSS files nothing read. Path (ii)
  (`reduce_twilight_flat`, now removed) reduced the twilights again for T_i on every run.
- The dome-flat RSS is still made when no twilight is configured.

**B2: lamp-flat sets combined.**
- `Flat/flatCombine.py` finds each configured flat's set of identical exposures from the raw
  headers. On 20261015_16 it finds 16-34-26.8/44.5/35-08.4 and 16-38-38.0/39-02.9/39-31.4.
- Every exposure is bias-corrected, then the set is combined: matched by header, scaled,
  and median-combined via `Bias/darkLlamas.combine_frames`.
- The result is `bias_corrected/<stem>_x<N>_bias_corrected.fits`, which tracing and the flat
  stage use.
- Cosmic rays in lamp flats are now rejected by the combine, not by L.A.Cosmic.
- A comma list in `{colour}_flat_file` gives the set explicitly, and
  `lamp_flat_combine = false` turns combining off.

**B3: Horne normalisation.**
- `Extract/extractLlamas.py` normalises the profile over the whole aperture. It used to
  renormalise over the valid pixels only, which returned F × the valid profile share, biased
  low wherever pixels were NaN or masked.
- Each column's variance now grows by the information lost to masking:
  `Σ_all(P²/V) / Σ_valid(P²/V)`.
- `Tests/test_horne_masking.py`: the old code fails 3 of its 4 cases.
- Measured on 20261015_16 traces, the tracer zeroes no profile pixels (0.000%), so the
  tracer is unchanged.

**B4: recipes enforced.**
- `process_trace`, `GUI_extract` and `run_extraction` need `recipe=` (no science default).
  `remove_cosmic_rays` is derived from it and can only turn cleaning off.
- Every call site names its frame type. The GUI quick look picks the recipe from the
  header, so arcs and flats are no longer cosmic-ray cleaned there.
- Extractions are stamped with their recipe and get a `.recipe.json` sidecar.
- Consumers check their inputs: pixel flat, smooth models, corrections, twilights, night
  arcs.
- Stale arcs are refused on resume.
- `reduce.report_recipes` prints the recipe of every calibration product at the end of a
  run, plus the science cosmic-ray summary.

## Not committed (left untracked on purpose)

- **Backups:** `*_backup_*` files and `backup.txt`. These are the copies made before
  rewriting core files.
- **Notebooks:** `Tutorials/*.ipynb`, `Tutorials/cosmic_ray_removal.png`,
  `cal_plots.ipynb`.
- **QA scratch:** `QA/baselines/scripts/*.json`; logs in `llamas_pyjamas/logs/`.
- **Scratch scripts at the package root:** `check_arc_simple.py`, `check_detector_order.py`,
  `check_reference_arc.py`, `example_pixel_to_fiber.py`, `sky_subtract_spectra.py`,
  `test_normalized_flat_fix.py`, `verify_cube_position.py`, `testing_omegaCen.txt`,
  `testing_sunburst.txt`; `Postprocessing/test_green_whitelight.py`.
- **`Utils/readmode.py`:** nothing imports it.
- **Stale tests:** `Test/test_fibre_flat.py` and `Test/test_qa_flatfield.py`. They import
  `_resolve_correction_row` and `qa_bench_boundary_step`, which no longer exist.
- **`CLAUDE.md`.**
- **`LLAMAS-CRR/` and `LLAMAS-sky-framework/`** are separate git worktrees of this repository
  (branches `crr-cube-gen` and `flux-cal`), not part of `legacytraces`.
