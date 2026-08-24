# Extract — optimal 1D extraction

Pulls a 1-D spectrum out of every traced fibre, propagating variance alongside the flux.

| Module | Role |
|---|---|
| `extractLlamas.py` | Everything. `ExtractLlamas` (the extractor), `ExtractLlamasRay` (its Ray-parallel subclass), `save_extractions` / `load_extractions` / `sort_extractions`, and `effective_aperture_pix()` |

`reduce.py` imports `ExtractLlamas` and `save_extractions` from here.

## Methods

`boxcar` is the default; `horne` / `optimal` profile-weighted extraction is available. The
error model is method-aware — the read-noise aperture follows the extraction aperture rather than
a fixed pixel count.

## Output

Extractions are pickled as `*_extract.pkl`, one per camera, and later consolidated into the
per-colour RSS files. Dead fibres are **dropped**, not zero-filled, so an extraction holds one row
per *live* fibre; `../Utils/deadfibers.py` converts between live-row and fibre-map indexing.

## See also

- Pipeline context: [`docs/workflow/02-running-the-reduction.md`](../../docs/workflow/02-running-the-reduction.md)
- API reference: <https://mit-kavli-institute.github.io/llamas-pyjamas/sphinx/api/llamas_pyjamas.Extract.html>
