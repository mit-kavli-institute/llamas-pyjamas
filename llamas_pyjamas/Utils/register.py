"""Astrometric registration of LLAMAS exposures (Phase 3).

Refines the rough header WCS into a star-tied solution and writes per-fibre RA/DEC back into the
RSS (FIBERMAP + FIBERWCS). Ties together the fibre-space centroiding (Utils.centroid), the WCS
primitives (Utils.wcsLlamas) and the RSS astrometry writer (File.llamasRSS.apply_fibre_astrometry).

Per exposure:
  1. detect bright compact sources in fibre space (never on the rendered hexagons);
  2. live Gaia cone-search seeded by the rough WCS;
  3. robust match (a common translation offset, so the ~few-arcsec pointing error and mismatches
     are handled);
  4. solve -- translation always; an overall rotation only if it stays within a configurable cap
     of the calibrated value (else translation only), since the rotator is stable;
  5. recompute per-fibre RA/DEC from the refined WCS and overwrite FIBERMAP/FIBERWCS for every
     channel of the exposure, with updated provenance.

TIER 1 (Gaia present) is implemented here. Tier 2 (relative dither-to-dither when no Gaia) and
Tier 3 (keep rough) fallback: no match -> the rough solution is left in place (Tier 3); the
multi-frame relative path is a follow-up increment.
"""

import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits

from llamas_pyjamas.Utils.centroid import fibre_centroid
from llamas_pyjamas.Utils.wcsLlamas import (ARCSEC_PER_FIBRE, IFU_PA_OFFSET, celestial_wcs,
                                            fit_wcs_from_stars, pointing_from_header,
                                            register_pointing)

logger = logging.getLogger(__name__)

GAIA_TAP = 'https://gea.esac.esa.int/tap-server/tap/sync'


@dataclass
class RegistrationResult:
    tier: str                      #: 'gaia' | 'header'
    method: str                    #: 'astrometric' | 'rough-header'
    refined: bool
    n_stars: int
    rms_arcsec: float
    rotation_deg: float            #: PA(+x) actually used
    rotation_refined: bool         #: True if rotation was solved (within cap), else calibrated
    files: List[str] = field(default_factory=list)


def query_gaia(ra_deg, dec_deg, radius_arcsec=45.0, mag_limit=20.5, timeout=30):
    """Live Gaia DR3 cone-search. Returns a SkyCoord of sources (empty on any failure)."""
    adql = (f"SELECT ra,dec,phot_g_mean_mag FROM gaiadr3.gaia_source WHERE "
            f"1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra_deg},{dec_deg},"
            f"{radius_arcsec / 3600.0})) AND phot_g_mean_mag<{mag_limit} "
            f"ORDER BY phot_g_mean_mag")
    url = GAIA_TAP + '?' + urllib.parse.urlencode(
        {'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv', 'QUERY': adql})
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            lines = resp.read().decode().strip().splitlines()[1:]
    except Exception as exc:                            # noqa: BLE001 - network optional
        logger.warning('Gaia query failed (%s); no absolute reference this frame', exc)
        return SkyCoord([], [], unit='deg')
    if not lines:
        return SkyCoord([], [], unit='deg')
    ra, dec = [], []
    for ln in lines:
        parts = ln.split(',')
        ra.append(float(parts[0]))
        dec.append(float(parts[1]))
    return SkyCoord(ra * u.deg, dec * u.deg)


def per_fibre_flux(hdul, band=None):
    """Per-fibre collapsed flux from an RSS HDUList (SKYSUB summed over `band` (lo,hi) A)."""
    from llamas_pyjamas.File.llamasRSS import skysub_extname
    flux = np.asarray(hdul[skysub_extname(hdul)].data, dtype=float)
    wave = np.asarray(hdul['WAVE'].data, dtype=float)
    if band is not None:
        lo, hi = band
        flux = np.where((wave >= lo) & (wave <= hi), flux, np.nan)
    return np.nansum(flux, axis=1)


def _fibre_xy(fiber_ids, benchsides):
    from llamas_pyjamas.Image.WhiteLightModule import FiberMap_LUT
    n = len(fiber_ids)
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    for i, (b, f) in enumerate(zip(benchsides, fiber_ids)):
        x, y = FiberMap_LUT(str(b).strip(), int(f))
        if not (x < 0 and y < 0):
            xs[i], ys[i] = float(x), float(y)
    return xs, ys


def detect_fibre_sources(x, y, flux, *, nsigma=8.0, min_sep=2.0, max_sources=12,
                         radius=1.5, power=2.0, contrast_sigma=5.0,
                         annulus=(2.0, 4.5)):
    """Greedy detection of compact point sources in fibre space -> list of ``Centroid``.

    Stricter than a plain threshold, to reject the many spurious peaks in a banded sky-residual
    background: a candidate is kept only if it is a *local* peak -- its core flux exceeds the
    MEDIAN of a surrounding annulus by ``contrast_sigma`` * MAD. A broad band lights up the
    annulus too, so its contrast is low and it is rejected; a real star sits on local background.

    ``nsigma`` global threshold to seed candidates, ``max_sources`` cap, ``radius``/``power`` feed
    ``fibre_centroid``, ``min_sep`` masks each claimed source.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    f = np.asarray(flux, float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
    if good.sum() < 3:
        return []
    vals = f[good]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826 or 1.0
    thresh = med + nsigma * mad
    r_in, r_out = annulus

    remaining = f.copy()
    remaining[~good] = np.nan
    sources = []
    while len(sources) < max_sources:
        if not np.isfinite(np.nanmax(remaining)) or np.nanmax(remaining) < thresh:
            break
        i = int(np.nanargmax(remaining))
        c = fibre_centroid(x, y, f, guess=(x[i], y[i]), radius=radius, power=power,
                           background=med)
        if c is None:
            remaining[i] = np.nan
            continue
        # local contrast: core peak vs the median of a surrounding annulus (banding rejection)
        rr = np.hypot(x - c.x, y - c.y)
        core = np.nanmax(f[rr <= 1.0]) if np.any(rr <= 1.0) else f[i]
        ann = f[(rr > r_in) & (rr <= r_out) & good]
        ann_med = float(np.median(ann)) if ann.size else med
        if (core - ann_med) >= contrast_sigma * mad:
            sources.append(c)
        remaining[(x - c.x) ** 2 + (y - c.y) ** 2 <= min_sep ** 2] = np.nan
    return sources


def _match_common_offset(det_sky, gaia, coarse_tol=9.0, tight_tol=2.5):
    """Match detected sources to Gaia via a shared translation offset (robust to the pointing
    error). Returns (det_index, gaia_index) lists. `det_sky`, `gaia` are SkyCoord arrays.

    ``coarse_tol`` seeds the common offset and must admit the *real* TCS pointing error, which on
    these data runs 4-6" for several fields (GD108 ~1" but J1613/J2151/Feige110 ~4-6", internally
    consistent across stars and dithers). ``tight_tol`` is the true discriminator: after removing
    the common offset, genuine matches agree to <2.5" while a wrong-star pairing scatters."""
    if len(det_sky) == 0 or len(gaia) == 0:
        return [], []
    idx, sep, _ = det_sky.match_to_catalog_sky(gaia)
    cand = sep.arcsec < coarse_tol
    if cand.sum() == 0:
        return [], []
    cosd = np.cos(np.deg2rad(det_sky.dec.deg))
    dra = (gaia[idx].ra.deg - det_sky.ra.deg) * cosd
    ddec = gaia[idx].dec.deg - det_sky.dec.deg
    # Seed the common offset from the DENSEST cluster of candidate offset vectors, not their
    # median: when most detections are spurious (sky-residual banding) the median is dragged off
    # the real pointing offset, and the tight refinement below then misses the true star cluster.
    # Each candidate offset votes; keep the one the most other candidates agree with (< tight_tol).
    cra, cdec = dra[cand], ddec[cand]
    best_n, off_ra, off_dec = -1, float(np.median(cra)), float(np.median(cdec))
    for k in range(len(cra)):
        near = np.hypot(cra - cra[k], cdec - cdec[k]) * 3600.0 < tight_tol
        if int(near.sum()) > best_n:
            best_n = int(near.sum())
            off_ra, off_dec = float(np.median(cra[near])), float(np.median(cdec[near]))
    shifted = SkyCoord((det_sky.ra.deg + off_ra / np.cos(np.deg2rad(det_sky.dec.deg))) * u.deg,
                       (det_sky.dec.deg + off_dec) * u.deg)
    idx2, sep2, _ = shifted.match_to_catalog_sky(gaia)
    keep = sep2.arcsec < tight_tol
    det_i, gaia_i, used = [], [], set()
    for di in np.where(keep)[0]:
        gi = int(idx2[di])
        if gi in used:                                  # one Gaia star -> one detection
            continue
        used.add(gi)
        det_i.append(int(di))
        gaia_i.append(gi)
    return det_i, gaia_i


def _axis_pa(wcs):
    c0 = wcs.pixel_to_world(0.0, 0.0)
    return c0.position_angle(wcs.pixel_to_world(1.0, 0.0)).deg


def _image_wcs_from_fibremap(fm_wcs, step):
    """Convert a fibre-map WCS (1 unit = one fibre spacing) to a white-light *image* WCS, where
    one image pixel = ``step`` fibre-map units (image pixel p_1idx -> fibre-map (p-1)*step). Exact:
    CD_img = CD_fm*step, CRPIX_img = (CRPIX_fm-1)/step + 1, CRVAL unchanged."""
    from astropy.wcs import WCS as _WCS
    out = _WCS(naxis=2)
    out.wcs.ctype = list(fm_wcs.wcs.ctype)
    out.wcs.cunit = ['deg', 'deg']
    out.wcs.crval = list(fm_wcs.wcs.crval)
    out.wcs.crpix = [(c - 1.0) / step + 1.0 for c in fm_wcs.wcs.crpix]
    out.wcs.cd = fm_wcs.pixel_scale_matrix * step
    return out


def _rms_arcsec(wcs, xy, gaia):
    pred = wcs.pixel_to_world(np.array([p[0] for p in xy]), np.array([p[1] for p in xy]))
    return float(np.sqrt(np.mean(pred.separation(gaia).arcsec ** 2)))


def solve_wcs(matched_xy, matched_gaia, rough_wcs, *, refine_rotation=False, max_rot_deg=3.0,
              max_shift_arcsec=8.0, tol_resid_arcsec=2.5):
    """Refine `rough_wcs` from matched fibre-space centroids + Gaia coords.

    TRANSLATION ONLY by default, holding the (stable, calibrated) rotation -- the rotator does not
    change frame-to-frame within a pointing, so re-solving rotation per frame just injects centroid
    noise and lets dithers disagree. The translation is the robust MEDIAN offset with iterative
    outlier rejection (`tol_resid_arcsec`), so a single mis-matched star is dropped rather than
    dragging the solution. Returns None (-> caller falls back to rough) when no consistent set
    survives or the shift exceeds `max_shift_arcsec` (a gross mis-match).

    ``refine_rotation`` (default OFF, and meant to be solved once per BLOCK, not per frame) allows
    a full fit within ``max_rot_deg`` of the calibrated rotation.

    Returns (wcs, rms_arcsec, rotation_deg, rotation_refined, n_used) or None.
    """
    xy = np.atleast_2d(np.asarray(matched_xy, dtype=float))
    gaia = matched_gaia.reshape((1,)) if matched_gaia.isscalar else matched_gaia
    pred = rough_wcs.pixel_to_world(xy[:, 0], xy[:, 1])
    pred = pred.reshape((1,)) if pred.isscalar else pred
    cosd = np.cos(np.deg2rad(np.atleast_1d(pred.dec.deg)))
    dra = (np.atleast_1d(gaia.ra.deg) - np.atleast_1d(pred.ra.deg)) * cosd   # true-angle offset
    ddec = np.atleast_1d(gaia.dec.deg) - np.atleast_1d(pred.dec.deg)
    keep = np.ones(len(xy), dtype=bool)
    for _ in range(3):                                         # iterative outlier rejection
        mra, mdec = np.median(dra[keep]), np.median(ddec[keep])
        resid = np.hypot(dra - mra, ddec - mdec) * 3600.0
        nk = resid < tol_resid_arcsec
        if nk.sum() == 0 or (nk == keep).all():
            keep = nk if nk.sum() else keep
            break
        keep = nk
    if keep.sum() == 0:
        logger.warning('No self-consistent matched stars; falling back to rough')
        return None
    mra, mdec = float(np.median(dra[keep])), float(np.median(ddec[keep]))
    shift = float(np.hypot(mra, mdec)) * 3600.0                # deg -> arcsec
    if shift > max_shift_arcsec:
        logger.warning('Solved shift %.1f" exceeds cap %.1f"; falling back to rough',
                       shift, max_shift_arcsec)
        return None

    wcs = rough_wcs.deepcopy()
    dec0 = rough_wcs.wcs.crval[1]
    wcs.wcs.crval = [rough_wcs.wcs.crval[0] + mra / np.cos(np.deg2rad(dec0)),
                     rough_wcs.wcs.crval[1] + mdec]
    n_used = int(keep.sum())
    rms = _rms_arcsec(wcs, [tuple(p) for p in xy[keep]], gaia[keep])

    if refine_rotation and n_used >= 2:                       # block-level use; off per-frame
        try:
            full = fit_wcs_from_stars([tuple(p) for p in xy[keep]], gaia[keep])
            drot = ((_axis_pa(full) - _axis_pa(rough_wcs) + 180) % 360) - 180
            if abs(drot) <= max_rot_deg:
                return (full, _rms_arcsec(full, [tuple(p) for p in xy[keep]], gaia[keep]),
                        _axis_pa(full), True, n_used)
            logger.warning('Rotation solve %.2f deg exceeds cap %.2f; translation only',
                           drot, max_rot_deg)
        except Exception as exc:                              # noqa: BLE001
            logger.warning('Rotation fit failed (%s); translation only', exc)
    return wcs, rms, _axis_pa(wcs), False, n_used


def _single_source_is_dominant(x, y, flux, sources, di, min_snr=25.0):
    """True if the one matched detection is an obvious bright star (a standard on-axis): its core
    peak stands >= ``min_snr`` * MAD above the field median. Used to trust a *single*-star solve.

    A flux-RATIO test fails here: a very bright standard bleeds into a halo that fragments into
    several near-equal-flux detections (they tie, so no one 'dominates'). Absolute peak SNR does
    not have that problem -- a real standard is thousands of sigma over background, while a lone
    peak in the sky-residual banding sits only ~8-15 sigma up and is (correctly) not trusted."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    f = np.asarray(flux, float)
    good = np.isfinite(f)
    med = float(np.median(f[good]))
    mad = float(np.median(np.abs(f[good] - med))) * 1.4826 or 1.0
    s = sources[di[0]]
    rr = np.hypot(x - s.x, y - s.y)
    sel = (rr <= 1.0) & good
    core = float(np.nanmax(f[sel])) if np.any(sel) else s.flux_sum
    return (core - med) / mad >= min_snr


def _rough_wcs(ra, dec, pa, extra_rot=0.0):
    """The rough header WCS at the fibre-map scale, optionally with a block rotation added to PA."""
    from llamas_pyjamas.Image.WhiteLightModule import FIELD_XMAX, FIELD_YMAX
    return celestial_wcs(ra, dec, crpix=(FIELD_XMAX / 2.0 + 1.0, FIELD_YMAX / 2.0 + 1.0),
                         arcsec_per_pixel=ARCSEC_PER_FIBRE, pa_deg=pa + extra_rot)


def _detect_and_query(det_path, band, mag_limit, radius_arcsec):
    """Load a detection frame -> (ra, dec, pa, xs, ys, flux, sources, gaia). Fibre-space detection
    is rotation-independent, so these can be reused across rotation hypotheses."""
    with fits.open(det_path) as hd:
        ra, dec, pa = pointing_from_header(hd[0].header)
        fmap = hd['FIBERMAP'].data
        xs, ys = _fibre_xy(list(fmap['FIBER_ID']), list(fmap['BENCHSIDE']))
        flux = per_fibre_flux(hd, band=band)
    sources = detect_fibre_sources(xs, ys, flux) if ra is not None else []
    gaia = (query_gaia(ra, dec, radius_arcsec=radius_arcsec, mag_limit=mag_limit)
            if ra is not None else SkyCoord([], [], unit='deg'))
    return ra, dec, pa, xs, ys, flux, sources, gaia


def _write_frame_solution(det_path, siblings, wcs, prov):
    """Write a solved WCS into every channel's FIBERMAP/FIBERWCS and the exposure white-light image
    (pipeline product only; the external quicklook is left alone). Returns the files written."""
    from llamas_pyjamas.File.llamasRSS import apply_fibre_astrometry
    written = []
    for path in siblings.values():
        with fits.open(path, mode='update') as hh:
            fm = hh['FIBERMAP'].data
            sx, sy = _fibre_xy(list(fm['FIBER_ID']), list(fm['BENCHSIDE']))
            finite = np.isfinite(sx) & np.isfinite(sy)
            ras = np.full(len(sx), np.nan)
            decs = np.full(len(sx), np.nan)
            if finite.any():
                sky = wcs.pixel_to_world(sx[finite], sy[finite])
                ras[finite] = sky.ra.deg
                decs[finite] = sky.dec.deg
            apply_fibre_astrometry(hh, ras=ras, decs=decs, xs=sx, ys=sy, prov=prov)
            hh.flush()
        written.append(path)

    import glob as _glob
    prefix = os.path.basename(det_path).split('_RSS_')[0]
    for wl in _glob.glob(os.path.join(os.path.dirname(det_path),
                                      f'{prefix}_whitelight_fullpipeline.fits')):
        with fits.open(wl, mode='update') as wh:
            for ext in wh:
                if getattr(ext, 'data', None) is None or ext.name not in ('BLUE', 'GREEN', 'RED'):
                    continue
                step = 1.0 / float(ext.header.get('PIXUNIT', 10))
                ext.header.update(_image_wcs_from_fibremap(wcs, step).to_header())
                ext.header['WCSMETH'] = prov['method']
                ext.header['WCSREFIN'] = bool(prov['refined'])
            wh.flush()
        written.append(wl)
    return written


def register_exposure(rss_path, *, mag_limit=20.5, radius_arcsec=45.0, band=None,
                      refine_rotation=False, max_rot_deg=3.0, max_shift_arcsec=8.0, min_stars=1):
    """Register one exposure (all channel siblings) and write refined per-fibre RA/DEC in place.

    Detection uses the green channel (or the first sibling); the solved WCS is applied to every
    channel. Falls back to the rough solution (Tier 3) if no Gaia match. Returns a
    :class:`RegistrationResult`.
    """
    from llamas_pyjamas.CubeViewer.cubeViewRSS import channel_siblings
    from llamas_pyjamas.Image.WhiteLightModule import FIELD_XMAX, FIELD_YMAX
    from llamas_pyjamas.File.llamasRSS import apply_fibre_astrometry

    siblings = channel_siblings(rss_path)
    det_path = siblings.get('green') or next(iter(siblings.values()))
    cx, cy = FIELD_XMAX / 2.0, FIELD_YMAX / 2.0

    with fits.open(det_path) as hd:
        hdr = hd[0].header
        ra, dec, pa = pointing_from_header(hdr)
        fmap = hd['FIBERMAP'].data
        xs, ys = _fibre_xy(list(fmap['FIBER_ID']), list(fmap['BENCHSIDE']))
        flux = per_fibre_flux(hd, band=band)

    prov = {'method': 'rough-header', 'tier': 'header', 'refined': False,
            'pa_offset': float(IFU_PA_OFFSET), 'catalog': '', 'rms': float('nan'), 'nstars': 0}
    rot_used, rot_refined, n_stars, rms = pa if pa is not None else float('nan'), False, 0, float('nan')

    if ra is None:
        logger.warning('%s: no pointing; cannot register', rss_path)
        wcs = None
    else:
        wcs = celestial_wcs(ra, dec, crpix=(cx + 1.0, cy + 1.0),
                            arcsec_per_pixel=ARCSEC_PER_FIBRE, pa_deg=pa)
        rot_used = _axis_pa(wcs)
        sources = detect_fibre_sources(xs, ys, flux)
        gaia = query_gaia(ra, dec, radius_arcsec=radius_arcsec, mag_limit=mag_limit)
        if sources and len(gaia) > 0:
            det_xy = [(s.x, s.y) for s in sources]
            det_sky = wcs.pixel_to_world(np.array([s.x for s in sources]),
                                         np.array([s.y for s in sources]))
            di, gi = _match_common_offset(det_sky, gaia)
            # Consensus guard: the real pointing error is 4-6" for several fields, so we cannot use
            # the shift MAGNITUDE to reject bad matches. Trust a solve only when >=2 stars agree on
            # a common offset (they already do, via _match_common_offset's tight_tol), OR a single
            # match that is a dominant source (a standard on-axis). A lone faint match among the
            # banding is not trusted -> fall back to rough.
            consensus = len(di) >= 2 or (len(di) == 1 and
                                         _single_source_is_dominant(xs, ys, flux, sources, di))
            solved = None
            if len(di) >= max(min_stars, 1) and consensus:
                solved = solve_wcs([det_xy[k] for k in di], gaia[gi], wcs,
                                   refine_rotation=refine_rotation, max_rot_deg=max_rot_deg,
                                   max_shift_arcsec=max_shift_arcsec)
            elif len(di) == 1:
                logger.info('%s: single non-dominant match; not trusted, keeping rough', rss_path)
            if solved is not None:
                wcs, rms, rot_used, rot_refined, n_stars = solved
                prov = {'method': 'astrometric', 'tier': 'gaia', 'refined': True,
                        'pa_offset': float(IFU_PA_OFFSET), 'catalog': 'GaiaDR3',
                        'rms': float(rms), 'nstars': int(n_stars)}
                logger.info('%s: Tier-1 Gaia, %d stars, RMS %.2f", rot %s',
                            rss_path, n_stars, rms, 'refined' if rot_refined else 'held')
            else:
                # keep the rough WCS (Tier 3); do not corrupt the frame with a bad match
                wcs = celestial_wcs(ra, dec, crpix=(cx + 1.0, cy + 1.0),
                                    arcsec_per_pixel=ARCSEC_PER_FIBRE, pa_deg=pa)
                logger.info('%s: no confident Gaia match; keeping rough (Tier 3)', rss_path)
        else:
            logger.info('%s: no sources or no Gaia; keeping rough (Tier 3)', rss_path)

    written = _write_frame_solution(det_path, siblings, wcs, prov) if wcs is not None else []

    return RegistrationResult(tier=prov['tier'], method=prov['method'], refined=prov['refined'],
                              n_stars=n_stars, rms_arcsec=rms, rotation_deg=rot_used,
                              rotation_refined=rot_refined, files=written)


def _solve_frame_translation(ra, dec, pa, xs, ys, flux, sources, gaia, *, extra_rot=0.0,
                             max_shift_arcsec=8.0):
    """Match + translation-only solve for one frame against a rough WCS that already carries the
    block rotation ``extra_rot``. Returns (wcs, prov, rot_used, n_stars, rms). Falls back to the
    (block-rotated) rough WCS when there is no confident match -- so even unmatched frames inherit
    the better block rotation."""
    wcs = _rough_wcs(ra, dec, pa, extra_rot)
    prov = {'method': 'rough-header', 'tier': 'header', 'refined': False,
            'pa_offset': float(IFU_PA_OFFSET), 'catalog': '', 'rms': float('nan'), 'nstars': 0}
    rot_used, n_stars, rms = _axis_pa(wcs), 0, float('nan')
    if sources and len(gaia) > 0:
        det_xy = [(s.x, s.y) for s in sources]
        det_sky = wcs.pixel_to_world(np.array([s.x for s in sources]),
                                     np.array([s.y for s in sources]))
        di, gi = _match_common_offset(det_sky, gaia)
        consensus = len(di) >= 2 or (len(di) == 1 and
                                     _single_source_is_dominant(xs, ys, flux, sources, di))
        solved = solve_wcs([det_xy[k] for k in di], gaia[gi], wcs, refine_rotation=False,
                           max_shift_arcsec=max_shift_arcsec) if (di and consensus) else None
        if solved is not None:
            wcs, rms, rot_used, _rr, n_stars = solved
            prov = {'method': 'astrometric', 'tier': 'gaia', 'refined': True,
                    'pa_offset': float(IFU_PA_OFFSET), 'catalog': 'GaiaDR3',
                    'rms': float(rms), 'nstars': int(n_stars)}
        else:
            wcs = _rough_wcs(ra, dec, pa, extra_rot)          # inherit the block rotation
    return wcs, prov, rot_used, n_stars, rms


def register_block(rss_paths, *, mag_limit=20.5, radius_arcsec=45.0, band=None,
                   max_rot_deg=3.0, max_shift_arcsec=8.0, fixed_rotation=None):
    """Register a block of exposures that share ONE pointing + rotator (e.g. all dithers of one
    OBJECT at one rotator position) and write the refined per-fibre RA/DEC in place.

    Solves a SINGLE rotation for the whole block -- the median of the per-frame Gaia rotation fits,
    clamped to +-``max_rot_deg`` of the calibrated value -- then HOLDS it and applies per-frame
    TRANSLATION only. The rotator is fixed within a block, so one rotation is physically correct;
    a per-frame 2-star rotation fit is noisy (~0.5 deg scatter) and would let dithers disagree,
    whereas the block median is stable. This removes the residual rotation (~1-2 deg on these data)
    that a translation-only solve leaves as a ~1.5" error at the field edge. Frames without a
    confident match still inherit the block rotation (a better rough). The caller groups exposures
    into blocks (typically by OBJECT, which encodes the rotator incl. the deliberate _180 flip).

    ``fixed_rotation`` (degrees, relative to the calibrated PA) skips the rotation FIT and uses the
    given value as the block rotation. This is how the interactive tool applies a rotation the user
    solved once by hand to the whole block, letting the auto per-frame translation do the rest.

    Returns {rss_path: RegistrationResult}.
    """
    from llamas_pyjamas.CubeViewer.cubeViewRSS import channel_siblings

    # -- pass 1: per-frame detect/query + fit each frame's own rotation where 2+ stars match
    # (skipped entirely when the caller supplies a fixed block rotation)
    frames, drots = [], []
    for p in rss_paths:
        sib = channel_siblings(p)
        dp = sib.get('green') or next(iter(sib.values()))
        ra, dec, pa, xs, ys, flux, sources, gaia = _detect_and_query(dp, band, mag_limit,
                                                                      radius_arcsec)
        if fixed_rotation is None and ra is not None and sources and len(gaia) > 0:
            w0 = _rough_wcs(ra, dec, pa)
            det_sky = w0.pixel_to_world(np.array([s.x for s in sources]),
                                        np.array([s.y for s in sources]))
            di, gi = _match_common_offset(det_sky, gaia)
            if len(di) >= 2:
                ro = solve_wcs([(sources[k].x, sources[k].y) for k in di], gaia[gi], w0,
                               refine_rotation=True, max_rot_deg=max_rot_deg,
                               max_shift_arcsec=max_shift_arcsec)
                if ro is not None and ro[3]:
                    drots.append(((ro[2] - _axis_pa(w0) + 180) % 360) - 180)
        frames.append(dict(path=p, sib=sib, dp=dp, ra=ra, dec=dec, pa=pa, xs=xs, ys=ys,
                           flux=flux, sources=sources, gaia=gaia))
    if fixed_rotation is not None:
        block_rot = float(np.clip(fixed_rotation, -max_rot_deg, max_rot_deg))
        logger.info('block of %d frames: rotation %.2f deg (fixed by caller)', len(frames),
                    block_rot)
    else:
        block_rot = float(np.clip(np.median(drots), -max_rot_deg, max_rot_deg)) if drots else 0.0
        logger.info('block of %d frames: rotation %.2f deg from %d fits', len(frames), block_rot,
                    len(drots))

    # -- pass 2: hold the block rotation, solve per-frame translation, write
    results = {}
    for f in frames:
        if f['ra'] is None:
            results[f['path']] = RegistrationResult('header', 'rough-header', False, 0,
                                                    float('nan'), float('nan'), False, [])
            continue
        wcs, prov, rot_used, n_stars, rms = _solve_frame_translation(
            f['ra'], f['dec'], f['pa'], f['xs'], f['ys'], f['flux'], f['sources'], f['gaia'],
            extra_rot=block_rot, max_shift_arcsec=max_shift_arcsec)
        written = _write_frame_solution(f['dp'], f['sib'], wcs, prov)
        # rotation was set by the block solve (held), not by a per-frame rotation fit
        results[f['path']] = RegistrationResult(prov['tier'], prov['method'], prov['refined'],
                                                n_stars, rms, rot_used, bool(block_rot), written)
    return results


def _block_key(rss_path):
    """Group key for a block: OBJECT + rotator. OBJECT already encodes the deliberate _180 flip,
    and TEL ROT (rounded) separates any other rotator change under the same target name."""
    with fits.open(rss_path) as h:
        obj = str(h[0].header.get('OBJECT', '')).strip()
        try:
            rot = round(float(h[0].header.get('TEL ROT', 0.0)))
        except (TypeError, ValueError):
            rot = 0
    return f'{obj}@{rot}'


def register_exposures(rss_paths, **kwargs):
    """Register many exposures, auto-grouped into blocks by (OBJECT, rotator) so each pointing's
    dithers share one solved rotation (see :func:`register_block`). Returns {rss_path: result}."""
    from collections import defaultdict
    blocks = defaultdict(list)
    for p in rss_paths:
        blocks[_block_key(p)].append(p)
    results = {}
    for key, paths in blocks.items():
        logger.info('registering block %s (%d frames)', key, len(paths))
        results.update(register_block(paths, **kwargs))
    return results


def reset_rough(rss_path):
    """Revert an exposure's stored WCS to the rough header (TCS) pointing, discarding any star-tied
    solution -- the bail-out when a refinement is worse than the raw pointing. Rewrites FIBERMAP/
    FIBERWCS and the white-light image for every channel. Returns the files written."""
    from llamas_pyjamas.CubeViewer.cubeViewRSS import channel_siblings
    siblings = channel_siblings(rss_path) or {'': rss_path}
    det = siblings.get('green') or next(iter(siblings.values()))
    with fits.open(det) as hd:
        ra, dec, pa = pointing_from_header(hd[0].header)
    if ra is None:
        return []
    wcs = _rough_wcs(ra, dec, pa)
    prov = {'method': 'rough-header', 'tier': 'header', 'refined': False,
            'pa_offset': float(IFU_PA_OFFSET), 'catalog': '', 'rms': float('nan'), 'nstars': 0}
    return _write_frame_solution(det, siblings, wcs, prov)
