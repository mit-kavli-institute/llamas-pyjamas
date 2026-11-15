# Utils Module

This module provides utility functions and helper routines used throughout the LLAMAS data reduction pipeline.

## Core Functionality

The utils module provides several essential functions:
- Plotting and visualization routines
- Data manipulation helpers
- Statistical analysis tools
- File handling utilities
- Configuration management

## Key Files

### 

utils.py


Main utilities file containing:
- Plotting functions for traces and spectra
- ZScale implementation for image display
- Data verification routines
- File handling helpers

Notable functions include:
- `plot_traces()`: Basic trace plotting
- 

plot_traces_on_image()

: Overlay traces on raw data
- `verify_header()`: FITS header validation
- `write_fits()`: FITS file output handler

## Usage

Common usage patterns:

```python
from llamas_pyjamas.Utils.utils import plot_traces_on_image

# Plot traces overlaid on raw data
plot_traces_on_image(trace_object, raw_data, zscale=True)

# Write output to FITS file
from llamas_pyjamas.Utils.utils import write_fits
write_fits(data, header, output_file)
```

The utils module is designed to be imported and used by other modules in the pipeline rather than run directly.

## Processing recipes (`recipes.py`)

**Every extraction names the recipe of its frame type.** There is no science default a
calibration frame could inherit:

| Recipe | Frame | Cosmic rays | Pixel flat | Dark | Sky subtraction |
|---|---|---|---|---|---|
| `lamp_flat` | LDLS lamp flat (traces, pixel flat, lamp shape) | never | no | no | no |
| `twilight` | twilight flat (fibre throughput) | never | yes | no | no |
| `arc` | arc lamp (wavelength solution) | never | no | no | no |
| `dome_flat` | lamp flat for a fibre-flat RSS, used when no twilight is configured | never | yes | no | no |
| `science`, `standard` | science and flux standards | per the config channels | yes | yes | yes |
| `sky_frame` | dedicated sky frame | per the config channels | yes | yes | no |

**How the recipe is enforced:**
- `process_trace`, `GUI_extract` and `run_extraction` need `recipe=`. Cleaning cosmic rays is
  derived from it (`cosmic_rays_for`). A caller can turn cleaning off, never on where the
  recipe forbids it.
- The GUI quick look picks the recipe from the frame's header (`recipe_for_frame`).

**How it is recorded:**
- Products carry `RECIPE`, `RCPHASH` (a hash of the recipe's fields and its product format
  `fmt`), `PIPEVER` and `FIBIDS` (`stamp_recipe`).
- An extraction pickle also gets a small `<pickle>.recipe.json` with the same stamps, so they
  can be checked without loading the pickle.

**How it is checked:**
- Each consumer checks its input with `check_recipe`: the pixel flat, the smooth models, the
  fibre-flat corrections, the reduced twilights and the night arcs. A batch run raises
  `CalibrationRecipeError`.
- A resumed run refuses stale products before any work (`reduce.stale_products`).
- At the end of the run, `reduce.report_recipes` prints the recipe of every calibration
  product, plus the science frames' cosmic-ray summary.

**When to bump `fmt`:** whenever a code change makes a recipe's products different. Their
hash then changes, and older products are rebuilt instead of reused.

