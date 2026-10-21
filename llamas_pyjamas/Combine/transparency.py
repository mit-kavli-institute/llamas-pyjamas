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

from llamas_pyjamas.Utils.maskedSums import band_integral, common_coverage

logger = logging.getLogger(__name__)


class TransparencyError(ValueError):
    """An exposure has no complete reference-source measurement, so it cannot be put on the
    same photometric system as the others (leaving it at 1.0 while scaling the rest would mix
    systems in one co-add)."""


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
    -- the band-integrated flux of every fibre (and every channel covering the band) within
    ``radius_arcsec`` of the source, for each exposure separately.

    Complete-coverage: if ANY fibre-channel inside the aperture lacks a usable band integral
    (masked, too little data in the window), the measurement is NaN rather than a partial sum --
    a fibre silently missing from one exposure's sum would make that exposure look faint and be
    scaled up. A source the exposure does not cover is also NaN."""
    lo, hi = band or _full_band(super_rss, channels)
    ras, decs, vals, exps = [], [], [], []
    for c in super_rss._resolve_channels(channels):
        st = super_rss.channels[c]
        common = common_coverage(st.wave, ~st.mask & np.isfinite(st.flux))
        bi = band_integral(st.flux, st.wave, st.mask, lo, hi, common=common)
        if not np.isfinite(bi.value).any():
            continue                                  # this channel does not cover the band
        placed = np.isfinite(st.ra) & np.isfinite(st.dec)
        ras.append(st.ra[placed]); decs.append(st.dec[placed])
        vals.append(bi.value[placed]); exps.append(st.exposure[placed])
    out: Dict[int, List[float]] = {e: [] for e in range(super_rss.n_exposures)}
    if not ras:
        for e in out:
            out[e] = [np.nan] * len(sources)
        return out
    ra, dec, value, expo = (np.concatenate(x) for x in (ras, decs, vals, exps))
    coords = SkyCoord(ra * u.deg, dec * u.deg)
    for src in sources:
        within = src.separation(coords).arcsec < radius_arcsec
        for e in range(super_rss.n_exposures):
            m = within & (expo == e)
            complete = m.any() and bool(np.all(np.isfinite(value[m])))
            out[e].append(float(np.sum(value[m])) if complete else np.nan)
    return out


def transparency_scales(super_rss, *, sources: Optional[Sequence[SkyCoord]] = None, n_sources=2,
                        radius_arcsec=2.0, band=None, channels=None, reference='median',
                        max_scale=3.0) -> Dict[str, float]:
    """Per-exposure photometric scale from the in-field reference source(s).

    For each source, scale_e = (reference flux) / (exposure e's aperture flux), with the reference
    the ``'median'`` (or ``'max'``) of that source's fluxes across exposures. Per-exposure scales
    are the MEDIAN over sources (robust to one source being partially covered). Returns
    ``{exposure_id: scale}`` for :meth:`SuperRSS.apply_scales`. A source measurement with any
    aperture fibre missing is not used (see :func:`measure_reference_flux`); an exposure left with
    no usable measurement raises :class:`TransparencyError` rather than staying at 1.0 while the
    others are scaled.

    Robustness (``max_scale``, default 3.0): the aperture flux is a raw SUM over the fibres within
    ``radius_arcsec`` of the source, so for a faint/compact source it is unstable across exposures
    with differing coverage — a near-zero flux makes ``ref/flux`` explode. Any per-source ratio
    outside ``[1/max_scale, max_scale]`` is discarded, and if the surviving per-exposure scales still
    span more than ``max_scale**2`` the reference photometry is deemed unreliable and NO scaling is
    applied (all 1.0). Applying no scaling is far safer than applying a catastrophic one — a scale of
    e.g. 300x multiplies that whole exposure (source AND background) and wrecks the co-add."""
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

    max_scale = float(max_scale)
    lo_r, hi_r = 1.0 / max_scale, max_scale
    scales = {}
    missing: List[str] = []
    unreliable = False                                      # any exploded ratio == broken photometry
    for e in range(super_rss.n_exposures):
        s_e = []
        n_measured = 0
        for s in range(n_src):
            fluxes = per_source[s]
            valid = fluxes[np.isfinite(fluxes) & (fluxes > 0)]
            ref = (np.median(valid) if reference == 'median' else valid.max()) if valid.size else 0
            if np.isfinite(fluxes[e]) and fluxes[e] > 0 and ref > 0:
                n_measured += 1
                r = ref / fluxes[e]
                if lo_r <= r <= hi_r:
                    s_e.append(r)
                else:
                    unreliable = True                       # a real transparency ratio is never this extreme
        if n_measured == 0:
            missing.append(super_rss.exposures[e].exposure_id)
        scales[super_rss.exposures[e].exposure_id] = float(np.median(s_e)) if s_e else 1.0

    # A single out-of-band ratio (e.g. 300x, 0.03x) is proof the aperture photometry is unreliable for
    # this field (faint/compact source, per-exposure coverage differences) — so DON'T trust the plausible
    # -looking survivors either: a lone 0.5 among garbage would still imprint a 2x per-frame level step.
    # No scaling is far safer than partial/wrong scaling; FLAM is already per-second so exposures still
    # co-add coherently, just without a transparency correction.
    if missing:
        raise TransparencyError(
            f'no complete reference-source measurement for exposure(s) {missing} (source off the '
            f'field, or an aperture fibre without usable data); cannot put them on the same '
            f'photometric system. Give --scale-source positions they all cover, a larger/smaller '
            f'--scale-radius, or combine without --scale-transparency.')
    vals = np.array(list(scales.values()), float)
    if unreliable or not (vals.size and np.all(np.isfinite(vals))):
        logger.warning('transparency scales unreliable (%s); leaving all exposures UNSCALED '
                       '(reference source photometry unstable — likely too faint/compact)',
                       ', '.join(f'{v:.3g}' for v in scales.values()))
        return {e.exposure_id: 1.0 for e in super_rss.exposures}
    logger.info('transparency scales: %s',
                ', '.join(f'{v:.3f}' for v in scales.values()))
    return scales
