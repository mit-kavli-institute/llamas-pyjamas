"""Shared flat-field utilities for LLAMAS.

Contains utility functions shared between fibreFlat.py, fibre_flat.py, and
other flat-fielding modules.
"""

import os

# Path to the LLAMAS spatial fibre map (bench, fiber, xpos, ypos)
_FIBERMAP_LUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'LUT', 'LLAMAS_FiberMap_rev04.dat'
)


def _load_fibermap_lut(lut_path=None):
    """Load the LLAMAS spatial fibre map from LLAMAS_FiberMap_rev04.dat.

    Returns a dict keyed by bench-side string (e.g. ``'1A'``) mapping to a
    list of ``(fiber_id, xpos)`` tuples.

    Parameters
    ----------
    lut_path : str, optional
        Override path to the .dat file.  Defaults to ``_FIBERMAP_LUT_PATH``.

    Returns
    -------
    dict
        ``{benchside: [(fiber_id, xpos), ...]}``
    """
    path = lut_path or _FIBERMAP_LUT_PATH
    lut = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip blank lines, comments, and the header row
            if not line or line.startswith('#'):
                continue
            if line.startswith('|') and 'bench' in line.lower():
                continue
            # Format: | bench | fiber | xindex | yindex | xpos | ypos |
            parts = [p.strip() for p in line.strip('|').split('|')]
            if len(parts) < 5:
                continue
            try:
                bench_str = parts[0].strip()   # e.g. '1A'
                fiber_id  = int(parts[1])
                xpos      = float(parts[4])
                lut.setdefault(bench_str, []).append((fiber_id, xpos))
            except (ValueError, IndexError):
                continue
    return lut
