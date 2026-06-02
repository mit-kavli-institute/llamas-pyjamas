# Reference Material

Supporting documentation and reference material for `llamas-pyjamas` that is **not**
part of the Sphinx documentation build. These files were relocated here from the
former `llamas_pyjamas/Docs/` grab-bag so they no longer live inside the Python package.

The published, auto-generated documentation lives in [`../docs/`](../docs/) (Sphinx).

## Contents

- **`claude_docs/`** — Feature-oriented notes written to help Claude Code understand the
  pipeline quickly (bias correction, cube construction, fibre tracing, flat fielding,
  extraction, wavelength calibration, etc.). Includes `WAVELENGTH_CALIBRATION_ISSUES.md`,
  a deep-dive on arc/wavelength-calibration issues (merged in from the old `claude-docs/`).
- **`plans/`** — Implementation plans (`crr_cube_plan.md`, `tracing_diagnostics_plan.md`)
  and the Liu et al. 2020 paper PDF referenced by the CRR cube work.
- **`notebooks/`** — `llamas_pyjamas_demo.ipynb`, a demo/tutorial notebook.
- **`resources/`** — `Extraction routine.pdf` and `Llamas Template.docx`.
- **`old_sphinx_source/`** — The previous hand-written Sphinx `source/` (`conf.py`,
  `*.rst`). Kept for reference only; it is **superseded** by the rebuilt `docs/` project
  and is not used by any build.
