"""Visual QA for astrometric registration.

Overlays, on the smashed green white-light image, the fitted stellar centroids and the Gaia
catalogue positions (projected through the rough header WCS), so one can see at a glance whether:
  * the detected centroids actually land on the visible stars (is centroiding working?),
  * the Gaia predictions fall near those stars (is a match feasible / what is the pointing error?),
  * which detections matched Gaia and the offset that was solved.

Intended for the formal QA package. ``plot_registration_qa(rss_path, outpath)`` writes a PNG.
"""

import logging
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval

from llamas_pyjamas.Utils.register import (detect_fibre_sources, per_fibre_flux, _fibre_xy,
                                           query_gaia, _match_common_offset)
from llamas_pyjamas.Utils.wcsLlamas import (ARCSEC_PER_FIBRE, celestial_wcs, pointing_from_header)

logger = logging.getLogger(__name__)


def _whitelight_green(exposure_dir, prefix):
    for suff in ('_whitelight_fullpipeline.fits', '_whitelight.fits'):
        p = os.path.join(exposure_dir, prefix + suff)
        if os.path.exists(p):
            with fits.open(p) as h:
                if 'GREEN' in h:
                    return np.asarray(h['GREEN'].data, dtype=float), int(h['GREEN'].header.get('PIXUNIT', 10))
    return None, 10


def plot_registration_qa(rss_path, outpath=None, band=None, mag_limit=20.5, radius_arcsec=45.0):
    """Write a registration-QA PNG for one exposure (uses its green RSS + white light)."""
    d = os.path.dirname(rss_path)
    prefix = os.path.basename(rss_path).split('_RSS_')[0]
    img, pixunit = _whitelight_green(d, prefix)
    if img is None:
        logger.warning('No green white-light for %s; skipping QA', prefix)
        return None

    with fits.open(rss_path) as h:
        hdr = h[0].header
        obj = str(hdr.get('OBJECT', ''))
        ra, dec, pa = pointing_from_header(hdr)
        fmap = h['FIBERMAP'].data
        xs, ys = _fibre_xy(list(fmap['FIBER_ID']), list(fmap['BENCHSIDE']))
        flux = per_fibre_flux(h, band=band)
        fw = h['FIBERWCS'].data if 'FIBERWCS' in h else None
        fwh = h['FIBERWCS'].header if 'FIBERWCS' in h else {}

    sources = detect_fibre_sources(xs, ys, flux)
    # image pixel = fibre-map * pixunit (image p_1idx -> fibre-map (p-1)/pixunit)
    det_ix = np.array([s.x * pixunit for s in sources])
    det_iy = np.array([s.y * pixunit for s in sources])

    # rough fibre-map WCS -> project Gaia into the image; and the match used
    from llamas_pyjamas.Image.WhiteLightModule import FIELD_XMAX, FIELD_YMAX
    gaia_ix = gaia_iy = None
    matched_det = []
    if ra is not None:
        rough = celestial_wcs(ra, dec, crpix=(FIELD_XMAX / 2 + 1, FIELD_YMAX / 2 + 1),
                              arcsec_per_pixel=ARCSEC_PER_FIBRE, pa_deg=pa)
        gaia = query_gaia(ra, dec, radius_arcsec=radius_arcsec, mag_limit=mag_limit)
        if len(gaia):
            gx, gy = rough.world_to_pixel(gaia)                # fibre-map coords
            gaia_ix, gaia_iy = np.atleast_1d(gx) * pixunit, np.atleast_1d(gy) * pixunit
            if sources:
                det_sky = rough.pixel_to_world(np.array([s.x for s in sources]),
                                               np.array([s.y for s in sources]))
                di, gi = _match_common_offset(det_sky, gaia)
                matched_det = di

    fig, ax = plt.subplots(figsize=(9, 8.5))
    vmin, vmax = ZScaleInterval().get_limits(img[np.isfinite(img)]) if np.isfinite(img).any() else (0, 1)
    ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    if gaia_ix is not None:
        ax.scatter(gaia_ix, gaia_iy, s=170, facecolors='none', edgecolors='cyan', lw=1.4,
                   label=f'Gaia (rough WCS), N={len(gaia_ix)}')
    if len(det_ix):
        ax.scatter(det_ix, det_iy, marker='+', c='red', s=90, lw=1.6,
                   label=f'fitted centroids, N={len(det_ix)}')
        for k in matched_det:
            ax.scatter(det_ix[k], det_iy[k], marker='o', s=230, facecolors='none',
                       edgecolors='yellow', lw=1.8)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(0, img.shape[0])
    tier = fwh.get('WCSMETH', '?') if fwh else '?'
    rms = fwh.get('WCSRMS', -1) if fwh else -1
    nst = fwh.get('WCSNSTAR', 0) if fwh else 0
    ax.set_title(f'{prefix}\n{obj}   solve={tier}  Nstar={nst}  RMS={rms:.2f}"   '
                 f'(yellow=matched)', fontsize=10)
    ax.set_xlabel('image x (green white-light)')
    ax.set_ylabel('image y')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.6)
    fig.tight_layout()
    if outpath is None:
        outpath = os.path.join(d, f'{prefix}_registration_qa.png')
    fig.savefig(outpath, dpi=110)
    plt.close(fig)
    return outpath
