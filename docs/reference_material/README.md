# Reference Material

Superseded and archival material for `llamas-pyjamas`. **Not current documentation**, and not
part of the published site.

> **This directory sits inside the Sphinx source tree but is excluded from the build.**
> `docs/conf.py` lists `reference_material` in `exclude_patterns`, which prunes it from Sphinx's
> source discovery. That entry is load-bearing: `old_sphinx_source/` holds 14 `.rst` files that
> Sphinx would otherwise build and publish under `/sphinx/`, duplicating the real API reference.
> Do not remove it.

Current documentation lives in three places:

- **Published site** — <https://mit-kavli-institute.github.io/llamas-pyjamas/>
- **End-to-end workflow guide** — [`docs/workflow/`](../workflow/README.md)
- **API reference** — [`docs/`](..) (Sphinx), published under `/sphinx/`

## Contents

- **`archive/`** — Point-in-time notes (implementation summaries, fix write-ups, status reports),
  filed by the month they were written. Historical record only; several describe config keys and
  modules that no longer exist. See [`archive/README.md`](archive/README.md).

- **`claude_docs/`** — Narrative feature notes on each pipeline stage (bias, tracing, extraction,
  flat fielding, wavelength calibration, cube construction, white light, QA, throughput, GUI).
  Background reading rather than instructions. Two of them — `GUI_INTERFACE.md` and
  `THROUGHPUT_ANALYSIS.md` — cover `GUI/` and `Flux/`, which are excluded from the Sphinx build,
  so for those they are the only prose documentation that exists.

- **`plans/`** — Implementation plans:
  - `crr_cube_plan.md` — **implemented.** Every module it proposes exists (`Cube/crr_kernels.py`,
    `crr_weights.py`, `crr_cube_constructor.py`, `crr_parallel.py`, `crr_cli.py`). Its planned
    `test_crr_cube.py` and `config/crr_config.yaml` were never written.
  - `tracing_diagnostics_plan.md` — **abandoned.** None of the functions it proposes exists
    anywhere in the package, and it is written generically for MaNGA/MUSE rather than LLAMAS.
  - `WAVELENGTH_CALIBRATION_ISSUES.md` — **implemented.** A deep-dive remediation plan for
    wavelength-solution validation; shipped as `Arc/arcValidation.py`, called from
    `Arc/arcLlamasMulti.py`.

- **`papers/`** — `Liu_2020_AJ_159_22.pdf`, *Covariance-regularized Reconstruction of Data Cubes
  in Integral Field Spectroscopy* (Liu et al. 2020, AJ 159, 22). This is a live citation, not an
  archive curiosity: `Cube/crr_cube_constructor.py` references it throughout and stamps
  `CUBEREF = 'Liu et al. (2020)'` into every CRR cube header.

- **`notebooks/`** — `llamas_pyjamas_demo.ipynb`, an early demo notebook. Superseded by
  [`../llamas_pyjamas/Tutorials/llamas_extraction_demo.ipynb`](../../llamas_pyjamas/Tutorials/llamas_extraction_demo.ipynb),
  which uses the production tracing and extraction path rather than the deprecated one.

- **`resources/`** — `Extraction routine.pdf` (a 2024 working note on the extraction routine) and
  `Llamas Template.docx` (an empty document-control template).

- **`old_sphinx_source/`** — The previous hand-written Sphinx source (`conf.py`, `*.rst`).
  **Superseded** by the rebuilt `docs/` project and excluded from the build. Kept for reference; note
  it is wrong in several places (Python 3.8, PyQt5, `sphinx_rtd_theme`, and a `sys.path` depth
  that would not import the package).

## History

Most of this directory was relocated here from the former `llamas_pyjamas/Docs/` grab-bag so it
no longer lives inside the Python package (commit `e120a95`). `archive/` and `papers/` were added
in a later cleanup pass.
