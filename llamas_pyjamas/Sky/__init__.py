"""
llamas_pyjamas.Sky
==================
Sky modelling and subtraction for the LLAMAS IFU pipeline.

Two layers:

* **Base model** (``skyLlamas``) — ``skyModel_1d`` / ``refineSkyX`` build a 1-D
  B-spline sky model into each fibre's ``sky`` attribute (pickle/extraction
  domain).  Imported lazily; not re-exported here because it pulls heavy
  dependencies (pypeit).

* **Subtraction framework** (``skySubtract`` + ``skyMask`` / ``skyScale`` /
  ``skyResidual`` / ``skyQA``) — a post-fibre-flat, per-colour, FITS-level stage
  that refines the base model (source masking → per-fibre OH scaling → ZAP-style
  PCA residual cleaning) and writes ``..._RSS_{color}_FF_SKYSUB.fits``.

Public API
----------
SkyMask                 -- first-class sky-fibre mask (boolean + provenance, FITS I/O)
build_sky_mask          -- the shared sky-fibre mask provider (all methods)
SkySubtractConfig       -- framework tunables
subtract_sky_rss        -- refine one FF RSS file
subtract_sky_all_colors -- driver over the three colour FF files
build_sky_fiber_mask    -- framework sky-fibre selection (returns a SkyMask)
"""

from llamas_pyjamas.Sky.skySelect import SkyMask, build_sky_mask
from llamas_pyjamas.Sky.skyConfig import SkySubtractConfig
from llamas_pyjamas.Sky.skyMask import build_sky_fiber_mask
from llamas_pyjamas.Sky.skySubtract import (subtract_sky_rss,
                                            subtract_sky_all_colors)

__all__ = [
    "SkyMask",
    "build_sky_mask",
    "SkySubtractConfig",
    "build_sky_fiber_mask",
    "subtract_sky_rss",
    "subtract_sky_all_colors",
]
