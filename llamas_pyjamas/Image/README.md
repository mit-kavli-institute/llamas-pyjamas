# Image — white-light image reconstruction

Collapses the extracted spectra along wavelength and lays the fibres out on sky to make a 2-D
white-light image. It is the fastest visual check that tracing and extraction worked.

| Module | Role |
|---|---|
| `WhiteLightModule.py` | The reconstruction. `WhiteLight()`, `WhiteLightFits()`, `WhiteLightFromRSS()`, `WhiteLightQuickLook()`, `WhiteLightHex()`, plus `whitelight_grid()`, `hex_tile_image()`, `FiberMap()` and `color_isolation()` |
| `processWhiteLight.py` | Post-processing: `remove_striping()`, `quartile_bias()` |

## The output grid

`whitelight_grid()` derives the image extent **from the fibre map** rather than hard-coding it.
An earlier version assumed a 53 × 53 field while the fibres only reach x ≈ 46.0, y ≈ 44.2, which
left roughly a quarter of every frame as NaN padding beyond the last fibre.

Fibres are laid out in **contiguous horizontal stripes by bench-side**, ordered 1A, 2A, 3A, 4A,
4B, 3B, 2B, 1B from the top of the field down — so a striping artefact that respects those bands
points at a per-bench-side problem (bias level, throughput tie), not at the reconstruction.

Fibre positions come from `../LUT/LLAMAS_FiberMap_rev04.dat`.

## Striping

Visible striping usually means the master bias was taken well away from the science date. The
usual remedy is to rebuild it with `../Scripts/update_bias_master.py`; `remove_striping()` is
the cosmetic fallback.

## See also

API reference: <https://mit-kavli-institute.github.io/llamas-pyjamas/sphinx/api/llamas_pyjamas.Image.html>
