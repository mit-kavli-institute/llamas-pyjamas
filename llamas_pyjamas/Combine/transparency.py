"""
Transparency scaling: put a field's exposures on one photometric system (Phase 4).

Under variable weather each dither has a different throughput, so the same source yields different
counts frame to frame. Left uncorrected this prints as a dither-patterned mosaic in the co-add
(the diagonal striping) and biases any surface-brightness measurement. This module measures each
exposure's relative transparency from the bright in-field point source(s) -- the QSO pair here --
and returns per-exposure scales for :meth:`SuperRSS.apply_scales`.

The in-field source doubles as a photometric anchor: its aperture flux is proportional to the
exposure's throughput, so scaling every exposure's source flux to a common reference makes them
consistent. Because :meth:`SuperRSS.apply_scales` scales variance by scale^2, an inverse-variance
co-add then rescales AND down-weights a hazy exposure together. Absolute zero-point (aperture loss)
is a later, separate step (anchor to Gaia); this is the RELATIVE, exposure-to-exposure correction.

Only *relative* stability is assumed of the reference source over the ~hours session -- fine even
for a (variable) quasar. Using several sources and combining per-source scales by median guards
against one source landing on a gap in a given dither.

Functions
---------
find_reference_sources   Auto-locate the brightest compact point sources in a field's co-add
measure_reference_flux   Per-exposure aperture flux at given source positions
transparency_scales      Per-exposure scale map from the in-field reference source(s)
"""

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

logger = logging.getLogger(__name__)


def _full_band(super_rss, channels):
    lo, hi = np.inf, -np.inf
    for c in (channels or list(super_rss.channels)):
        st = super_rss.channels.get(c)
        if st is None:
            continue
        w = st.wave[np.isfinite(st.wave)]
        if w.size:
            lo, hi = min(lo, float(w.min())), max(hi, float(w.max()))
    return lo, hi


def find_reference_sources(super_rss, *, n=2, band=None, channels=None, pixscale=0.5,
                           min_sep_arcsec=3.0, snr_min=8.0) -> List[SkyCoord]:
    """The ``n`` brightest compact peaks in a uniform-weight co-add of the field, as SkyCoords.

    Uses a plain (uniform, flux) co-add so peaks are real source flux, finds local maxima above
    ``snr_min`` MAD over the background, and enforces ``min_sep_arcsec`` between accepted peaks
    (so the two QSOs of a pair are picked, not one QSO twice)."""
    from scipy.ndimage import maximum_filter
    from llamas_pyjamas.Combine.coadd import combine_image
    lo, hi = band or _full_band(super_rss, channels)
    img = combine_image(super_rss, lo, hi, channels=channels, units='flux', weighting='uniform',
                        kernel='gaussian', kernel_fwhm=0.9, pixscale=pixscale, min_coverage=1)
    data = img.data
    finite = np.isfinite(data)
    if not finite.any():
        return []
    bg = float(np.median(data[finite]))
    mad = float(np.median(np.abs(data[finite] - bg))) * 1.4826 or 1.0
    d = np.where(finite, data, -np.inf)
    size = max(3, int(round(min_sep_arcsec / pixscale)))
    peaks = (d == maximum_filter(d, size=size)) & finite & (data > bg + snr_min * mad)
    ys, xs = np.nonzero(peaks)
    if ys.size == 0:
        return []
    order = np.argsort(-data[ys, xs])
    ys, xs = ys[order], xs[order]

    chosen_yx: List = []
    min_sep_pix = min_sep_arcsec / pixscale
    for y, x in zip(ys, xs):
        if all(np.hypot(y - yy, x - xx) >= min_sep_pix for yy, xx in chosen_yx):
            chosen_yx.append((y, x))
        if len(chosen_yx) >= n:
            break
    coords = [img.wcs.pixel_to_world(x, y) for y, x in chosen_yx]
    logger.info('reference sources: %d found', len(coords))
    return coords


def measure_reference_flux(super_rss, sources: Sequence[SkyCoord], *, radius_arcsec=2.0,
                           band=None, channels=None) -> Dict[int, List[float]]:
    """Aperture flux at each source, per exposure. Returns ``{exposure_index: [flux_per_source]}``
    -- summed collapse-band flux of the fibres within ``radius_arcsec`` of the source, for each
    exposure separately (a source not covered in an exposure -> 0 for that source)."""
    lo, hi = band or _full_band(super_rss, channels)
    ft = super_rss.collapse_band(lo, hi, channels=channels)
    coords = SkyCoord(ft.ra * u.deg, ft.dec * u.deg)
    out: Dict[int, List[float]] = {e: [] for e in range(super_rss.n_exposures)}
    for src in sources:
        within = src.separation(coords).arcsec < radius_arcsec
        for e in range(super_rss.n_exposures):
            m = within & (ft.exposure == e)
            out[e].append(float(np.nansum(ft.value[m])) if m.any() else 0.0)
    return out


def transparency_scales(super_rss, *, sources: Optional[Sequence[SkyCoord]] = None, n_sources=2,
                        radius_arcsec=2.0, band=None, channels=None, reference='median'
                        ) -> Dict[str, float]:
    """Per-exposure photometric scale from the in-field reference source(s).

    For each source, scale_e = (reference flux) / (exposure e's aperture flux), with the reference
    the ``'median'`` (or ``'max'``) of that source's fluxes across exposures. Per-exposure scales
    are the MEDIAN over sources (robust to one source being partially covered). Returns
    ``{exposure_id: scale}`` for :meth:`SuperRSS.apply_scales`. Exposures with no source flux get
    scale 1.0 (left as-is)."""
    if sources is None:
        sources = find_reference_sources(super_rss, n=n_sources, band=band, channels=channels)
    if not sources:
        logger.warning('no reference sources found; no transparency scaling applied')
        return {e.exposure_id: 1.0 for e in super_rss.exposures}

    flux_by_exp = measure_reference_flux(super_rss, sources, radius_arcsec=radius_arcsec,
                                         band=band, channels=channels)
    n_src = len(sources)
    per_source = np.array([[flux_by_exp[e][s] for e in range(super_rss.n_exposures)]
                           for s in range(n_src)], dtype=float)   # (n_src, n_exp)

    scales = {}
    for e in range(super_rss.n_exposures):
        s_e = []
        for s in range(n_src):
            fluxes = per_source[s]
            valid = fluxes[fluxes > 0]
            ref = (np.median(valid) if reference == 'median' else valid.max()) if valid.size else 0
            if fluxes[e] > 0 and ref > 0:
                s_e.append(ref / fluxes[e])
        scales[super_rss.exposures[e].exposure_id] = float(np.median(s_e)) if s_e else 1.0
    logger.info('transparency scales: %s',
                ', '.join(f'{v:.3f}' for v in scales.values()))
    return scales
