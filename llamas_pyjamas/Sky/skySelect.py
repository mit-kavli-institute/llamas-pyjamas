"""
llamas_pyjamas.Sky.skySelect
=============================
Single source of truth for *which fibres are treated as sky* when building the
sky model.  Shared by the base 1-D model (``Sky.skyLlamas.skyModel_1d``) and the
advanced framework's PCA-basis mask (``Sky.skyMask.build_sky_fiber_mask``) so the
two stages always agree on what "sky" means.

User-facing selection methods (config key ``sky_selection_method``)
-------------------------------------------------------------------
``dimmest``      (default) the ``n_fibres`` faintest finite fibres.  Restores the
                 original "take the dimmest fibres" intent.
``middle-third`` legacy: the central third of fibres ranked by brightness
                 (reproduces the pre-existing ``skyModel_1d`` behaviour exactly).
``skymap``       fibres whose spatial position falls in a user-supplied sky
                 region — a 2-D FITS *mask* or *flux image* (see ``load_sky_map``).
``frame``/``all`` every finite fibre.  Used when the data source is itself a
                 dedicated blank-sky exposure, so all fibres are sky.
``manual``       an explicit, user-supplied set of sky fibres (a boolean mask or
                 a list of live fibre indices) — the seam for future providers
                 that define sky fibres externally (e.g. from an LSST broadband
                 image or a hand-drawn region).  Handled by :func:`build_sky_mask`.

The first-class :class:`SkyMask` (a boolean mask + provenance, with FITS
serialisation) is the reusable, inspectable, persistable representation of a
selection; :func:`build_sky_mask` is the single provider that produces one for
every method, so the base 1-D model and the framework agree on what "sky" means.

Public API
----------
SkyMask                                              boolean mask + provenance (+ FITS I/O)
build_sky_mask(brightness, finite, *, method, ...) -> SkyMask
select_sky_fibres(brightness, finite, *, method, n_fibres, in_sky_region) -> bool[n]
load_sky_map(path) -> SkyMap
fibres_in_sky_region(benchsides, fibers, skymap, *, ra, dec, faint_percentile) -> bool[n]
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Methods understood by select_sky_fibres / build_sky_mask / SkySubtractConfig.
# 'manual' is handled by build_sky_mask (an explicit mask/id list), not by the
# brightness-based select_sky_fibres.
VALID_METHODS = ("stratified", "quantile", "dimmest", "middle-third",
                 "skymap", "frame", "all", "manual")

# Rank band for the 'quantile' method: fibres whose white-light brightness rank
# (among finite fibres) falls in [QUANTILE_LO, QUANTILE_HI). Sits just above the
# dead/vignetted low tail that 'dimmest' camps on, while staying far below any
# object flux. ~5% of ~300 fibres => ~15 sky fibres per camera.
QUANTILE_LO = 0.05
QUANTILE_HI = 0.10

# 'stratified' method: bin fibres by slit position (trace-y) and take the
# faintest few per bin, so the sky model is built from fibres spanning the
# whole slit. This removes a width bias: the plain faint selections pick the
# low-throughput SLIT-EDGE fibres, so the pooled model carries edge-LSF and,
# applied slit-wide, leaves ~2x larger residuals on OH lines (measured green,
# 2026-07). Object guard: fibres brighter than STRAT_CAP_PCT (global) are
# excluded first, so a compact object cannot enter any bin.
STRAT_NBINS = 8        # trace-y bins across the slit
STRAT_PER_BIN = 5      # faintest fibres kept per bin (=> ~30-40 sky fibres,
                       # enough for robust cross-fibre outlier rejection)
STRAT_CAP_PCT = 60.0   # exclude fibres above this global brightness percentile

# A selection that leaves fewer than this many fibres is treated as degenerate
# and falls back (mirrors the relaxation in skyMask.build_sky_fiber_mask).
MIN_SKY_FIBRES = 3

# Desired floor on the number of fibres a 'dimmest' fit should use. Low-signal
# cameras (e.g. faint blue) may have very few positive fibres; below this floor
# the caller broadens the selection (e.g. to middle-third) for a sturdier fit.
MIN_SKY_FIT_FIBRES = 10


# ----------------------------------------------------------------------------
# Core fibre selector
# ----------------------------------------------------------------------------
def select_sky_fibres(brightness, finite, *, method="dimmest", n_fibres=20,
                      in_sky_region=None, q_lo=QUANTILE_LO, q_hi=QUANTILE_HI,
                      fiber_y=None):
    """Return a boolean mask of the fibres to use for the sky estimate.

    Parameters
    ----------
    brightness : array_like (n_fiber,)
        Per-fibre white-light brightness proxy (e.g. throughput-corrected sum
        over wavelength).  May contain NaN/inf for dead/edge fibres.
    finite : array_like[bool] (n_fiber,)
        True where the fibre is usable (finite brightness, not dead).
    method : str
        One of :data:`VALID_METHODS`.  Unknown values fall back to ``'dimmest'``.
    n_fibres : int
        Number of fibres for the ``'dimmest'`` method.
    in_sky_region : array_like[bool] (n_fiber,), optional
        Required for ``'skymap'`` — True where the fibre sits in the user's sky
        region (from :func:`fibres_in_sky_region`).

    Returns
    -------
    np.ndarray[bool] (n_fiber,)
        True for fibres contributing to the sky model.
    """
    brightness = np.asarray(brightness, dtype=float)
    finite = np.asarray(finite, dtype=bool)
    n = brightness.size
    method = (method or "dimmest").lower()

    if method in ("frame", "all"):
        return _fallback_if_degenerate(finite.copy(), finite, brightness, n_fibres,
                                       label=method)

    if method == "skymap":
        if in_sky_region is None:
            logger.warning("skySelect: method='skymap' but no in_sky_region given; "
                           "falling back to dimmest-%d", n_fibres)
            return select_sky_fibres(brightness, finite, method="dimmest",
                                     n_fibres=n_fibres)
        region = np.asarray(in_sky_region, dtype=bool)
        mask = finite & region
        return _fallback_if_degenerate(mask, finite, brightness, n_fibres,
                                       label="skymap")

    if method == "middle-third":
        # Parity with the legacy skyModel_1d: rank ALL fibres by descending
        # brightness and take the central third (indices n//3 : 2n//3).  The set
        # of fibres is what matters — the downstream fit re-sorts by wavelength.
        order = np.argsort(-brightness)
        sky_start = n // 3
        sky_end = 2 * n // 3
        mask = np.zeros(n, dtype=bool)
        mask[order[sky_start:sky_end]] = True
        return _fallback_if_degenerate(mask, finite, brightness, n_fibres,
                                       label="middle-third")

    if method == "stratified":
        # Faintest few per trace-y bin (spans the slit), object-capped. Needs
        # per-fibre slit position; without it, fall back to quantile.
        if fiber_y is None:
            logger.warning("skySelect: method='stratified' but no fiber_y given; "
                           "falling back to 'quantile'")
            return select_sky_fibres(brightness, finite, method="quantile",
                                     n_fibres=n_fibres, q_lo=q_lo, q_hi=q_hi)
        mask = _stratified(finite, brightness, np.asarray(fiber_y, dtype=float))
        return _fallback_if_degenerate(mask, finite, brightness, n_fibres,
                                       label="stratified")

    if method == "quantile":
        # Fibres whose brightness RANK among finite fibres lies in
        # [QUANTILE_LO, QUANTILE_HI).  Unlike 'dimmest' this skips the
        # dead/vignetted low tail while staying far below any object flux.
        mask = _quantile_band(finite, brightness, q_lo, q_hi)
        return _fallback_if_degenerate(mask, finite, brightness, n_fibres,
                                       label="quantile")

    # Default: dimmest-N among finite fibres.
    if method != "dimmest":
        logger.warning("skySelect: unknown method %r; using 'dimmest'", method)
    mask = _dimmest_n(finite, brightness, n_fibres)
    return _fallback_if_degenerate(mask, finite, brightness, n_fibres,
                                   label="dimmest")


def _stratified(finite, brightness, fiber_y,
                nbin=STRAT_NBINS, per=STRAT_PER_BIN, cap_pct=STRAT_CAP_PCT):
    """Mask of the faintest ``per`` fibres in each of ``nbin`` trace-y bins,
    after excluding fibres brighter than the global ``cap_pct`` percentile.

    Spans the slit (removes the edge-LSF bias of the plain faint selections)
    while the global cap keeps a compact object out of every bin.
    """
    n = brightness.size
    mask = np.zeros(n, dtype=bool)
    pool = finite & np.isfinite(fiber_y)
    if pool.sum() == 0:
        return mask
    cap = np.nanpercentile(brightness[pool], cap_pct)
    pool = pool & (brightness <= cap)
    if pool.sum() == 0:
        return mask
    yv = fiber_y[pool]
    edges = np.linspace(np.nanmin(yv), np.nanmax(yv), nbin + 1)
    idx_pool = np.where(pool)[0]
    for k in range(nbin):
        lo, hi = edges[k], edges[k + 1]
        inb = idx_pool[(fiber_y[idx_pool] >= lo) &
                       (fiber_y[idx_pool] <= hi if k == nbin - 1
                        else fiber_y[idx_pool] < hi)]
        if inb.size == 0:
            continue
        faint = inb[np.argsort(brightness[inb])][:per]
        mask[faint] = True
    return mask


def _quantile_band(finite, brightness, q_lo, q_hi):
    """Boolean mask of finite fibres in the [q_lo, q_hi) brightness-rank band."""
    n = brightness.size
    idx_finite = np.where(finite)[0]
    mask = np.zeros(n, dtype=bool)
    nf = idx_finite.size
    if nf == 0:
        return mask
    order = idx_finite[np.argsort(brightness[idx_finite])]  # ascending
    i0 = int(np.floor(q_lo * nf))
    i1 = max(i0 + 1, int(np.ceil(q_hi * nf)))
    mask[order[i0:i1]] = True
    return mask


def _dimmest_n(finite, brightness, n_fibres):
    """Boolean mask of the ``n_fibres`` faintest finite fibres."""
    n = brightness.size
    idx_finite = np.where(finite)[0]
    mask = np.zeros(n, dtype=bool)
    if idx_finite.size == 0:
        return mask
    order = idx_finite[np.argsort(brightness[idx_finite])]  # ascending: faintest first
    take = max(1, int(n_fibres))
    mask[order[:take]] = True
    return mask


def _fallback_if_degenerate(mask, finite, brightness, n_fibres, *, label):
    """If ``mask`` selects too few fibres, relax to dimmest-N, then all finite."""
    if mask.sum() >= MIN_SKY_FIBRES:
        return mask
    dim = _dimmest_n(finite, brightness, max(MIN_SKY_FIBRES, int(n_fibres)))
    if dim.sum() >= MIN_SKY_FIBRES:
        logger.warning("skySelect: '%s' left %d sky fibres (<%d); falling back to "
                       "dimmest-%d", label, int(mask.sum()), MIN_SKY_FIBRES, int(n_fibres))
        return dim
    logger.warning("skySelect: '%s' degenerate (%d fibres); falling back to all "
                   "finite (%d)", label, int(mask.sum()), int(finite.sum()))
    return finite.copy()


# ----------------------------------------------------------------------------
# SkyMask: a first-class, persistable "which fibres are sky" object
# ----------------------------------------------------------------------------
@dataclass
class SkyMask:
    """The set of fibres treated as sky, plus how it was chosen.

    A thin, reusable wrapper around the boolean mask returned by
    :func:`select_sky_fibres`, carrying provenance so a selection can be
    inspected, persisted (as an RSS extension), reused, and user-overridden.

    Attributes
    ----------
    mask : np.ndarray[bool] (n_fiber,)
        True for fibres contributing to the sky model.  **Live-indexed** (row
        order), matching the RSS/extraction fibre arrays — not physical
        fibremap position.
    method : str
        The selection method that produced it (one of :data:`VALID_METHODS`).
    provenance : dict
        Parameters that produced the mask (e.g. ``n_fibres``, ``source``,
        quantile band, sky-map path).  Serialised to the FITS header as JSON.
    """
    mask: np.ndarray
    method: str = "unknown"
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mask = np.asarray(self.mask, dtype=bool)

    @property
    def n_sky(self) -> int:
        """Number of sky fibres."""
        return int(self.mask.sum())

    @property
    def n_fiber(self) -> int:
        """Total number of fibres the mask spans."""
        return int(self.mask.size)

    def ids(self) -> np.ndarray:
        """Live indices of the sky fibres."""
        return np.where(self.mask)[0]

    def to_hdu(self, name="SKYMASK"):
        """Serialise to a FITS ``ImageHDU`` (uint8 mask + provenance header)."""
        from astropy.io import fits
        hdu = fits.ImageHDU(self.mask.astype(np.uint8), name=name)
        h = hdu.header
        h["SKYMETH"] = (self.method, "sky-fibre selection method")
        h["SKYNSKY"] = (self.n_sky, "number of sky fibres")
        h["SKYNFIB"] = (self.n_fiber, "total fibres in the mask")
        h["SKYPROV"] = (json.dumps(self.provenance, default=str),
                        "sky-mask selection provenance (JSON)")
        return hdu

    @classmethod
    def from_hdu(cls, hdu):
        """Reconstruct a :class:`SkyMask` from a FITS HDU written by :meth:`to_hdu`."""
        mask = np.asarray(hdu.data, dtype=bool)
        h = hdu.header
        prov = {}
        raw = h.get("SKYPROV")
        if raw:
            try:
                prov = json.loads(raw)
            except (ValueError, TypeError):
                logger.warning("SkyMask.from_hdu: could not parse SKYPROV; dropping provenance")
        return cls(mask=mask, method=h.get("SKYMETH", "unknown"), provenance=prov)


def _as_bool_mask(explicit, n_fiber=None):
    """Coerce an explicit selection into a boolean mask.

    A **boolean** array is taken as the mask directly.  An **integer** array is
    taken as live fibre indices, requiring ``n_fiber`` to size the mask.
    """
    arr = np.asarray(explicit)
    if arr.dtype == bool:
        return arr.copy()
    ids = arr.astype(int)
    if n_fiber is None:
        raise ValueError("build_sky_mask(method='manual'): integer ids need n_fiber "
                         "(or pass a boolean mask instead)")
    mask = np.zeros(int(n_fiber), dtype=bool)
    if ids.size:
        if ids.min() < 0 or ids.max() >= n_fiber:
            raise ValueError("build_sky_mask(method='manual'): fibre id out of range "
                             f"[0, {n_fiber})")
        mask[ids] = True
    return mask


def build_sky_mask(brightness=None, finite=None, *, method="dimmest", n_fibres=20,
                   in_sky_region=None, q_lo=QUANTILE_LO, q_hi=QUANTILE_HI, fiber_y=None,
                   explicit=None, n_fiber=None, source="in-exposure") -> SkyMask:
    """Canonical sky-fibre mask provider — the single seam both sky stages use.

    For every brightness-based method this simply wraps :func:`select_sky_fibres`
    (so the resulting mask is *identical*) and attaches provenance.  For
    ``method='manual'`` it uses ``explicit`` (a boolean mask, or live fibre
    indices with ``n_fiber``) directly — the hook for externally-defined sky
    fibres (e.g. from an LSST image or a hand-drawn region).

    Returns
    -------
    SkyMask
    """
    method_l = (method or "dimmest").lower()

    if method_l == "manual":
        if explicit is None:
            raise ValueError("build_sky_mask(method='manual') requires 'explicit'")
        nf = n_fiber
        if nf is None and finite is not None:
            nf = np.asarray(finite).size
        elif nf is None and brightness is not None:
            nf = np.asarray(brightness).size
        mask = _as_bool_mask(explicit, n_fiber=nf)
        prov = {"source": source, "n_requested": int(mask.sum())}
        return SkyMask(mask=mask, method="manual", provenance=prov)

    if brightness is None or finite is None:
        raise ValueError(f"build_sky_mask(method={method_l!r}) requires brightness and finite")

    mask = select_sky_fibres(brightness, finite, method=method, n_fibres=n_fibres,
                             in_sky_region=in_sky_region, q_lo=q_lo, q_hi=q_hi,
                             fiber_y=fiber_y)
    prov = {"source": source, "n_fibres": int(n_fibres)}
    if method_l == "quantile":
        prov.update(q_lo=float(q_lo), q_hi=float(q_hi))
    if method_l == "skymap":
        prov["skymap"] = True
    if method_l == "stratified":
        prov["fiber_y"] = fiber_y is not None
    return SkyMask(mask=mask, method=method_l, provenance=prov)


# ----------------------------------------------------------------------------
# Sky-map loading and fibre→region mapping (method='skymap')
# ----------------------------------------------------------------------------
@dataclass
class SkyMap:
    """A loaded 2-D sky map.

    Attributes
    ----------
    data : np.ndarray (ny, nx)
        The 2-D map (a cube is collapsed over wavelength on load).
    is_mask : bool
        True if the map is a discrete sky/source mask (sky = truthy pixels);
        False if it is a flux image to be thresholded at ``faint_percentile``.
    wcs : astropy.wcs.WCS or None
        Celestial WCS if the header carried one; enables the RA/Dec path.
    faint_percentile : float
        For flux images: pixels at or below this percentile (of in-field finite
        pixels) are treated as sky.
    """
    data: np.ndarray
    is_mask: bool
    wcs: Optional[object] = None
    faint_percentile: float = 40.0


def load_sky_map(path, faint_percentile=40.0):
    """Load a user sky map from FITS, auto-detecting mask vs flux image.

    Accepts a 2-D image or a 3-D cube (collapsed via ``nansum`` over the first
    axis).  A map is treated as a discrete *mask* when its data are integer/bool
    or have <=2 distinct finite values; otherwise it is a *flux image*.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(path) as hdul:
        hdu = next((h for h in hdul
                    if getattr(h, "data", None) is not None and np.ndim(h.data) >= 2),
                   None)
        if hdu is None:
            raise ValueError(f"load_sky_map: no >=2D image found in {path}")
        data = np.asarray(hdu.data, dtype=float)
        header = hdu.header.copy()

    # Collapse a cube (wavelength on the leading FITS axis) to 2-D.
    while data.ndim > 2:
        data = np.nansum(data, axis=0)

    # Capture a celestial WCS if present.
    wcs = None
    try:
        w = WCS(header)
        if w.has_celestial:
            wcs = w.celestial
    except Exception as exc:  # pragma: no cover - header weirdness
        logger.debug("load_sky_map: no usable WCS (%s)", exc)

    is_mask = _looks_like_mask(data)
    logger.info("load_sky_map: %s -> %s (%s), shape=%s, wcs=%s",
                path, "mask" if is_mask else "flux-image",
                data.dtype, data.shape, "yes" if wcs is not None else "no")
    return SkyMap(data=data, is_mask=is_mask, wcs=wcs,
                  faint_percentile=float(faint_percentile))


def _looks_like_mask(data):
    """Heuristic: discrete mask (int/bool or <=2 unique finite values)."""
    finite_vals = data[np.isfinite(data)]
    if finite_vals.size == 0:
        return False
    if np.array_equal(finite_vals, np.round(finite_vals)):
        # integer-valued: a mask if it has at most two levels (e.g. 0/1).
        if np.unique(finite_vals).size <= 2:
            return True
    return False


def fibres_in_sky_region(benchsides, fibers, skymap, *, ra=None, dec=None):
    """Boolean mask: True where a fibre's position falls in the sky region.

    Two coordinate paths:

    * **WCS** — used when ``skymap.wcs`` is set and per-fibre ``ra``/``dec`` are
      supplied (e.g. the sky map is a non-sky-subtracted cube).
    * **Fibre grid** — otherwise, map each fibre's ``FiberMap_LUT`` position into
      the image by linear scaling of the fibre bounding box onto the pixel grid.
      This assumes the map is sampled on the fibre-position grid with the usual
      FITS orientation (no axis flips); supply a WCS for unambiguous results.
    """
    benchsides = np.asarray(benchsides)
    fibers = np.asarray(fibers)
    n = benchsides.size
    ny, nx = skymap.data.shape

    rows = np.full(n, -1, dtype=int)
    cols = np.full(n, -1, dtype=int)

    if skymap.wcs is not None and ra is not None and dec is not None:
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)
        good = np.isfinite(ra) & np.isfinite(dec)
        if good.any():
            cx, cy = skymap.wcs.world_to_pixel_values(ra[good], dec[good])
            cols[good] = np.round(cx).astype(int)
            rows[good] = np.round(cy).astype(int)
    else:
        x, y, valid = _fibre_grid_positions(benchsides, fibers)
        if valid.any():
            xs, ys = x[valid], y[valid]
            xspan = max(xs.max() - xs.min(), 1e-9)
            yspan = max(ys.max() - ys.min(), 1e-9)
            cols[valid] = np.round((xs - xs.min()) / xspan * (nx - 1)).astype(int)
            rows[valid] = np.round((ys - ys.min()) / yspan * (ny - 1)).astype(int)

    in_bounds = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
    region = np.zeros(n, dtype=bool)
    if not in_bounds.any():
        logger.warning("fibres_in_sky_region: no fibres mapped into the sky map")
        return region

    pix = np.full(n, np.nan)
    pix[in_bounds] = skymap.data[rows[in_bounds], cols[in_bounds]]

    if skymap.is_mask:
        region = in_bounds & np.isfinite(pix) & (pix > 0)
    else:
        infield = np.isfinite(skymap.data) & (skymap.data != 0)
        vals = skymap.data[infield]
        if vals.size == 0:
            return region
        thresh = np.percentile(vals, skymap.faint_percentile)
        region = in_bounds & np.isfinite(pix) & (pix <= thresh)
    logger.info("fibres_in_sky_region: %d/%d fibres flagged sky", int(region.sum()), n)
    return region


def _fibre_grid_positions(benchsides, fibers):
    """Look up (x, y) fibre-grid positions; returns (x, y, valid_mask)."""
    from llamas_pyjamas.Image.WhiteLightModule import FiberMap_LUT

    n = len(benchsides)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    for i in range(n):
        bs = str(benchsides[i]).strip()
        try:
            xi, yi = FiberMap_LUT(bs, int(fibers[i]))
        except Exception:
            continue
        if xi == -1 and yi == -1:
            continue
        x[i], y[i] = xi, yi
    valid = np.isfinite(x) & np.isfinite(y)
    return x, y, valid
