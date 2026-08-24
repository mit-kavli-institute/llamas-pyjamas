# Arc — wavelength calibration

Turns ThAr arc-lamp exposures into a per-fibre pixel→wavelength solution, and transfers that
solution onto science extractions.

## What runs

`reduce.py` imports **`arcLlamasMulti`** (`import llamas_pyjamas.Arc.arcLlamasMulti as arc`).
That is the production path — the Ray-parallel implementation. `arcLlamas.py` is the original
serial version, kept as the reference implementation.

| Module | Role |
|---|---|
| `arcLlamasMulti.py` | **Production.** Ray-parallel arc solution: `reidentifyArc`, `shiftArcXRay`, `fiberRelativeThroughputRay`, `arcSolveRay`, `arcTransfer` |
| `arcLlamas.py` | Serial equivalents: `reidentifyArc`, `shiftArcX`, `refineArcX`, `fiberRelativeThroughput`, `arcSolve`, `arcTransfer` |
| `arcValidation.py` | `validate_wavelength_solution()` — QA gate on a fitted solution; called from `arcTransfer(..., enable_validation=True)` |
| `arcSurface.py` | `refineArcX2D()` — 2-D surface refinement of the x-shift across a detector |
| `arcReidentify.py`, `nistArc.py` | Standalone scripts. Not importable (they run code at import and reference a removed module) and excluded from the API build |

## Reference data

The reference arc lives in `../LUT/LLAMAS_reference_arc.pkl` (download location in the repository
[`README.md`](../../README.md)). Line lists are in `../LUT/ThAr_MagE_lines.dat`; per-channel peak
catalogues in `../LUT/{red,green,blue}_peaks.csv`.

## See also

- [`RAY_PARALLELISM.md`](RAY_PARALLELISM.md) — how the Ray arc path is parallelised, and why.
- API reference: <https://mit-kavli-institute.github.io/llamas-pyjamas/sphinx/api/llamas_pyjamas.Arc.html>
