# Feature notes

Narrative, feature-oriented notes on the LLAMAS pipeline, written to give a quick conceptual
overview of each major stage. They describe *what a stage does and why*, which the auto-generated
API reference does not.

**These are reference material, not current documentation.** They were written between 2025 and
mid-2026 and are not kept in step with the code. Where a note disagrees with the code, the code
wins. For current material see:

- the published site — <https://mit-kavli-institute.github.io/llamas-pyjamas/>
- the end-to-end workflow guide — [`docs/workflow/`](../../workflow/README.md)
- the API reference — [`docs/`](../..) (Sphinx), published under `/sphinx/`

## Contents

### Core reduction
- `FIBER_TRACING.md` — fibre position identification and mapping
- `SPECTRUM_EXTRACTION.md` — 1D extraction, optimal and boxcar
- `WAVELENGTH_CALIBRATION.md` — pixel→wavelength from ThAr arcs
- `FLAT_FIELD_PROCESSING.md` — pixel-to-pixel sensitivity and per-fibre normalisation
- `BIAS_CORRECTION.md` — electronic offset removal

### Data products
- `RSS_FILE_GENERATION.md` — row-stacked-spectra FITS creation
- `CUBE_CONSTRUCTION.md` — 3D cube construction, including the CRR method
- `WHITE_LIGHT_IMAGING.md` — 2D image reconstruction from spectra

### Quality and throughput
- `QUALITY_ASSURANCE.md` — QA visualisation and validation
- `THROUGHPUT_ANALYSIS.md` — system efficiency and flux-calibration calculations

### Interface
- `GUI_INTERFACE.md` — the interactive extraction GUI

## Which of these still earn their keep

`GUI_INTERFACE.md` and `THROUGHPUT_ANALYSIS.md` cover `GUI/` and `Flux/`, which are **excluded
from the Sphinx API build** — for those two, this is the only prose documentation that exists.
`CUBE_CONSTRUCTION.md` is the only explanation of *why* covariance-regularised reconstruction is
used (see [`../papers/Liu_2020_AJ_159_22.pdf`](../papers/Liu_2020_AJ_159_22.pdf)).

The rest overlap the Sphinx API reference and the per-module `README.md` files, and should be
read as background rather than as instructions.
