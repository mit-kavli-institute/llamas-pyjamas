# Archive

Point-in-time notes — implementation summaries, fix write-ups, status reports — kept as a record
of decisions and of bugs that were found and fixed. They are filed by the month they were written.

> **Historical only. Do not follow the instructions in these files.**
> Each one describes the code as it was on the date in its folder name. Several document config
> keys, modules or functions that have since been renamed, reversed or deleted. For anything you
> intend to *run*, use the published site
> (<https://mit-kavli-institute.github.io/llamas-pyjamas/>), the workflow guide in
> [`docs/workflow/`](../../docs/workflow/README.md), or the code itself.

## Contents

### `2025-09/`
- **`BUILD_INSTRUCTIONS.md`** — how the Sphinx docs were built at the time. Superseded by
  [`docs/README.md`](../../docs/README.md); the theme has since changed from `sphinx-rtd-theme`
  to `furo`, the Python floor moved to 3.11+, and deployment is via GitHub Pages, not
  Read the Docs.

### `2025-11/`
- **`FLAT_FIELDING_IMPLEMENTATION_SUMMARY.md`** — an attempt to add a second, PypeIt-based flat
  method selectable with a `flat_method` config key. **The experiment was abandoned**: neither
  the `flat_method` key nor the `flatLlamas_pypeit.py` module it describes exists any more. Kept
  as the record of what was tried.

### `2025-12/`
- **`SIMPLE_CUBE_INTEGRATION.md`** — the integration of `Cube/simple_cube_constructor.py`.
  It landed, but the decision it announces was later **reversed**: the doc says `simple` became
  the default cube method, while `reduce.py` now defaults to `traditional`. Useful for
  understanding why the simple constructor exists.
- **`OLD_RSS_WAVELENGTH_ISSUES.md`** — diagnosis of a `(2048, 2389)` vs `(2389, 2048)`
  transposition in an older `llamasRSS.py`. Fixed. Worth keeping in case the symptom recurs.

### `2026-01/`
- **`WHITELIGHT_FIX_INSTRUCTIONS.md`** — the rationale for deriving the white-light grid from the
  fibre extent instead of a hard-coded 53×53. **Shipped**, as
  `Image/WhiteLightModule.py::whitelight_grid()`. Its second half proposed spaxel masking in
  `Cube/cubeConstruct.py`, which was never actioned.
- **`FLUX_CALIBRATION_SUMMARY.md`** — the first end-to-end flux-calibration run (LTT 1788). A
  one-off analysis using paths outside the repo; the production path is now `Flux/sensFunc.py`
  and `Flux/fluxStandards.py`.
