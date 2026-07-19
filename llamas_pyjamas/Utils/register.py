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


def detect_fibre_sources(x, y, flux, *, nsigma=5.0, min_sep=2.0, max_sources=30,
                         radius=1.5, power=2.0):
    """Greedy peak detection in fibre space -> list of ``Centroid`` (brightest first).

    Background = median, noise = MAD. Claims the brightest fibre above threshold, centroids it,
    masks fibres within ``min_sep``, and repeats. ``radius``/``power`` feed ``fibre_centroid``.
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

    remaining = f.copy()
    remaining[~good] = np.nan
    sources = []
    for _ in range(max_sources):
        if np.nanmax(remaining) < thresh:
            break
        i = int(np.nanargmax(remaining))
        c = fibre_centroid(x, y, f, guess=(x[i], y[i]), radius=radius, power=power,
                           background=med)
        if c is not None:
            sources.append(c)
            remaining[(x - c.x) ** 2 + (y - c.y) ** 2 <= min_sep ** 2] = np.nan
        else:
            remaining[i] = np.nan
    return sources


def _match_common_offset(det_sky, gaia, coarse_tol=18.0, tight_tol=2.5):
    """Match detected sources to Gaia via a shared translation offset (robust to the pointing
    error). Returns (det_index, gaia_index) lists. `det_sky`, `gaia` are SkyCoord arrays."""
    if len(det_sky) == 0 or len(gaia) == 0:
        return [], []
    idx, sep, _ = det_sky.match_to_catalog_sky(gaia)
    cand = sep.arcsec < coarse_tol
    if cand.sum() == 0:
        return [], []
    cosd = np.cos(np.deg2rad(det_sky.dec.deg))
    dra = (gaia[idx].ra.deg - det_sky.ra.deg) * cosd
    ddec = gaia[idx].dec.deg - det_sky.dec.deg
    off_ra = float(np.median(dra[cand]))
    off_dec = float(np.median(ddec[cand]))
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


def solve_wcs(matched_xy, matched_gaia, rough_wcs, *, refine_rotation=True, max_rot_deg=5.0):
    """Refine `rough_wcs` from matched fibre-space centroids + Gaia coords.

    Translation always (register_pointing). A rotation is solved (fit_wcs_from_stars) only with
    >=2 stars AND if it stays within `max_rot_deg` of the rough rotation; otherwise translation
    only. Returns (wcs, rms_arcsec, rotation_deg, rotation_refined).
    """
    translated = register_pointing(rough_wcs, matched_xy, matched_gaia)
    rot0 = _axis_pa(rough_wcs)
    if refine_rotation and len(matched_xy) >= 2:
        try:
            full = fit_wcs_from_stars(matched_xy, matched_gaia)
            drot = ((_axis_pa(full) - rot0 + 180) % 360) - 180
            if abs(drot) <= max_rot_deg:
                return full, _rms_arcsec(full, matched_xy, matched_gaia), _axis_pa(full), True
            logger.warning('Rotation solve %.2f deg exceeds cap %.2f; translation only',
                           drot, max_rot_deg)
        except Exception as exc:                        # noqa: BLE001
            logger.warning('Rotation fit failed (%s); translation only', exc)
    return translated, _rms_arcsec(translated, matched_xy, matched_gaia), rot0, False


def register_exposure(rss_path, *, mag_limit=20.5, radius_arcsec=45.0, band=None,
                      refine_rotation=True, max_rot_deg=5.0, min_stars=1):
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
            if len(di) >= min_stars:
                m_xy = [det_xy[k] for k in di]
                m_gaia = gaia[gi]
                wcs, rms, rot_used, rot_refined = solve_wcs(
                    m_xy, m_gaia, wcs, refine_rotation=refine_rotation, max_rot_deg=max_rot_deg)
                n_stars = len(di)
                prov = {'method': 'astrometric', 'tier': 'gaia', 'refined': True,
                        'pa_offset': float(IFU_PA_OFFSET), 'catalog': 'GaiaDR3',
                        'rms': float(rms), 'nstars': int(n_stars)}
                logger.info('%s: Tier-1 Gaia solve, %d stars, RMS %.2f", rot %s',
                            rss_path, n_stars, rms, 'refined' if rot_refined else 'held')
            else:
                logger.info('%s: no confident Gaia match; keeping rough (Tier 3)', rss_path)
        else:
            logger.info('%s: no sources or no Gaia; keeping rough (Tier 3)', rss_path)

    written = []
    if wcs is not None:
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

        # Update the exposure's white-light image WCS (what the user inspects in DS9) to match
        # the refined solution. Only the pipeline product; the external quicklook is left alone.
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

    return RegistrationResult(tier=prov['tier'], method=prov['method'], refined=prov['refined'],
                              n_stars=n_stars, rms_arcsec=rms, rotation_deg=rot_used,
                              rotation_refined=rot_refined, files=written)
