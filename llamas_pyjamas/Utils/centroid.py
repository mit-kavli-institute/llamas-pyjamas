"""Flux-weighted centroiding of fibres, in fibre space (not on rendered images).

The white-light FITS images render each fibre as a Voronoi hexagon -- a display artifact, not a
physical measurement. The correct centroid of a source is the flux-weighted mean of the
contributing FIBRES' own positions (fibre-map X/Y, or a local tangent plane), background-
subtracted and iterated so the window recenters on the source.

This module is the shared engine for astrometric registration and CubeViewer's interactive
refine. It is deliberately coordinate-agnostic and data-source-agnostic: the caller collapses the
S/N-optimal wavelength band to per-fibre fluxes and supplies the fibre positions in whatever
planar coordinate it wants the centroid in (fibre-map units are ideal -- planar, no projection).
For sky centroids, centroid in fibre-map X/Y and convert the result through the WCS.

This astrometric centroid is distinct from the optimal (variance/PSF) weighting used for
point-source spectral *extraction* -- that is a separate operation.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Centroid:
    """Result of :func:`fibre_centroid` (all in the caller's input coordinate)."""
    x: float
    y: float
    n_fibres: int          #: fibres with positive weight in the final window
    flux_sum: float        #: summed background-subtracted flux in the final window
    background: float      #: background level subtracted
    converged: bool        #: True if the last window shift fell below ``tol``
    shift: float           #: distance the centre moved on the last iteration


def _median_background(flux: np.ndarray) -> float:
    finite = flux[np.isfinite(flux)]
    return float(np.median(finite)) if finite.size else 0.0


def fibre_centroid(x, y, flux, *, guess=None, radius: float = 1.5, power: float = 1.0,
                   iterations: int = 3, background: Optional[float] = None,
                   min_fibres: int = 3, tol: float = 1e-3) -> Optional[Centroid]:
    """Flux-weighted centroid of fibres around a source, in the input planar coordinate.

    ``centre = sum(w_i * pos_i) / sum(w_i)`` over the fibres within ``radius`` of the current
    centre, with ``w_i = max(flux_i - background, 0) ** power``. The window is recentred and the
    centroid recomputed ``iterations`` times (removes the bias when the source sits between
    fibres).

    Parameters
    ----------
    x, y, flux : array_like
        Per-fibre positions (same length) and their collapsed flux in the chosen band.
    guess : (float, float), optional
        Starting centre; defaults to the brightest fibre.
    radius : float
        Window radius in the units of ``x``/``y`` (fibre-map units: ~1.5 = two rings).
    power : float
        Weight exponent: 1 = flux (unbiased for a symmetric PSF), 2 = flux^2 (sharpens toward
        the peak, suppresses background/wing bias).
    iterations : int
        Window re-centring passes.
    background : float, optional
        Level subtracted before weighting; default is the median of all finite fluxes. Pass a
        local sky estimate in crowded fields. **Not** subtracting a background biases the centroid
        toward the window centre, so a background is always applied.
    min_fibres : int
        Minimum positive-weight fibres required; returns None below this.
    tol : float
        Convergence threshold on the per-iteration centre shift.

    Returns
    -------
    Centroid or None
        None if there are too few usable fibres or no positive flux in the window.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    f = np.asarray(flux, dtype=float)
    if not (x.shape == y.shape == f.shape):
        raise ValueError('x, y, flux must have the same shape')
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
    if good.sum() < min_fibres:
        return None
    x, y, f = x[good], y[good], f[good]

    bg = _median_background(f) if background is None else float(background)
    if guess is None:
        i = int(np.argmax(f))
        cx, cy = float(x[i]), float(y[i])
    else:
        cx, cy = float(guess[0]), float(guess[1])

    r2 = float(radius) * float(radius)
    converged, shift, n, fsum = False, float('inf'), 0, 0.0
    for _ in range(max(1, int(iterations))):
        sel = ((x - cx) ** 2 + (y - cy) ** 2) <= r2
        resid = np.clip(f[sel] - bg, 0.0, None)
        w = resid ** power
        wsum = float(w.sum())
        if wsum <= 0 or int((w > 0).sum()) < min_fibres:
            return None
        nx = float((w * x[sel]).sum() / wsum)
        ny = float((w * y[sel]).sum() / wsum)
        shift = float(np.hypot(nx - cx, ny - cy))
        cx, cy = nx, ny
        n = int((w > 0).sum())
        fsum = float(resid.sum())
        if shift < tol:
            converged = True
            break
    return Centroid(cx, cy, n, fsum, bg, converged, shift)
