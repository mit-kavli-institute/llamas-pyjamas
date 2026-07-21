"""
llamas_pyjamas.Sky.skyFloorTemplate
===================================
Static per-camera additive floor TEMPLATE, built by combining MANY science frames across different
fields and dithers with positive-outlier rejection (no dedicated blank field needed: objects move
between dithers/fields, so each fibre is blank in most frames; object flux is rejected as a positive
outlier in the per-(fibre, wavelength-bin) combine across frames).

Motivation (Sky/DESIGN.md): each camera carries a fixed, additive, smooth along-slit continuum floor
(stable across dithers AND fields, corr 0.85-0.98) which the per-camera pooled sky fit mean-removes,
tiling the IFU in benchside bands -> the stacked-image striping. A static template x one per-frame
amplitude is the safe correction (the shape comes from OTHER frames/fields, so it cannot absorb real
diffuse emission in the target field).

Domain: the RSS COUNTS and SKY planes are PRE-fibre-flat, so COUNTS - SKY is the pkl-domain floor;
the template applies directly where the pedestal acts (added to .sky before the flat).

The per-frame amplitudes are also a PHYSICAL DIAGNOSTIC: amplitude ~ frame sky level => scattered
sky/slit light in the spectrograph; ~ exposure time => dark-like; ~ constant => electronic.

Public API
----------
build_floor_template(rss_files, channel='green', ...) -> (templates, diag)
save_template / load_template                            FITS round-trip
"""

import logging
import numpy as np
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)

CONT_WINDOW = 51        # wavelength median-filter width (px) -- matches skyPedestal
NLBIN = 64              # downsampled wavelength bins for the template (floor is smooth in lambda)
MIN_OK = 200            # min unmasked pixels for a fibre-frame to contribute
MIN_FRAMES = 5          # min surviving frames per (fibre, bin); else interpolate along slit
CLIP_SIGMA = 2.5        # positive-outlier rejection threshold (objects)


def _frame_floor(C, S, msk):
    """Per-fibre smooth continuum residual (nfib, NLBIN) + broadband object excess (nfib,)."""
    ok = (msk == 0) & np.isfinite(C) & np.isfinite(S)
    nfib, nwave = C.shape
    edges = np.linspace(0, nwave, NLBIN + 1).astype(int)
    cont = np.full((nfib, NLBIN), np.nan)
    obj = np.full(nfib, np.nan)
    for i in range(nfib):
        m = ok[i]
        if m.sum() < MIN_OK:
            continue
        r = np.where(m, C[i] - S[i], np.nan)
        sm = median_filter(np.nan_to_num(r, nan=0.0), size=CONT_WINDOW, mode="nearest")
        sm[~m] = np.nan
        for k in range(NLBIN):
            seg = sm[edges[k]:edges[k + 1]]
            if np.isfinite(seg).sum() > 5:
                cont[i, k] = np.nanmedian(seg)
        obj[i] = np.nanmedian(C[i][m]) - np.nanmedian(S[i][m])
    return cont, obj


def build_floor_template(rss_files, channel='green'):
    """Build per-camera (benchside) floor templates from many RSS frames.

    Per frame: drop object-contaminated fibre-frames (broadband excess above the camera's blank
    population); per (fibre, wavelength-bin): reject POSITIVE outliers across frames (CLIP_SIGMA
    above the median -- residual object/transient light), then average.

    Returns
    -------
    templates : dict  benchside -> (nfib_cam, NLBIN) template (counts)
    diag : dict with per-frame amplitudes, sky levels, exptimes, objects, rejection stats
    """
    from astropy.io import fits
    stacks = {}                    # cam -> list of (nfib, NLBIN) per frame (NaN = dropped)
    meta = []
    for f in rss_files:
        try:
            with fits.open(f) as h:
                hdr = h[0].header
                C = np.asarray(h['COUNTS'].data, float)
                S = np.asarray(h['SKY'].data, float)
                msk = np.asarray(h['MASK'].data)
                bs = np.array([str(b).strip() for b in h['FIBERMAP'].data['BENCHSIDE']])
        except Exception as exc:                          # noqa: BLE001
            logger.warning('skyFloorTemplate: skipping %s (%s)', f, exc)
            continue
        if not np.any(np.isfinite(S) & (S != 0)):
            logger.info('skyFloorTemplate: %s has no sky model; skipped', f)
            continue
        cont, obj = _frame_floor(C, S, msk)
        skylev = float(np.nanmedian(S[np.isfinite(S) & (S != 0)]))
        expt = hdr.get('SEXPTIME', hdr.get('DEXPTIME', hdr.get('EXPTIME', 0.0)))
        meta.append(dict(file=f, object=str(hdr.get('OBJECT', '')), skylev=skylev,
                         exptime=float(expt or 0.0)))
        for cam in sorted(set(bs)):
            mcam = bs == cam
            cc = cont[mcam].copy()
            oo = obj[mcam]
            # object prescreen: drop fibre-frames far above the camera's blank population
            fin = np.isfinite(oo)
            if fin.sum() > 20:
                low = oo[fin][oo[fin] <= np.nanpercentile(oo[fin], 60)]
                mad = 1.4826 * np.nanmedian(np.abs(low - np.nanmedian(low)))
                bad = fin & (oo > np.nanmedian(low) + 5 * max(mad, 1e-3))
                cc[bad] = np.nan
            stacks.setdefault(cam, []).append(cc)
    if not meta:
        raise ValueError('skyFloorTemplate: no usable frames')

    templates = {}
    diag = dict(frames=meta, amplitudes={}, reject_frac={})
    for cam, lst in stacks.items():
        A = np.stack(lst)                                  # (nframes, nfib, NLBIN)
        med = np.nanmedian(A, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(A - med[None]), axis=0)
        pos_out = A > med[None] + CLIP_SIGMA * np.maximum(mad[None], 1e-3)
        Ac = np.where(pos_out, np.nan, A)                  # reject positive outliers (objects)
        nsurv = np.isfinite(Ac).sum(axis=0)
        T = np.nanmean(Ac, axis=0)
        T[nsurv < MIN_FRAMES] = np.nan
        # fill remaining holes by along-slit interpolation per bin
        for k in range(T.shape[1]):
            col = T[:, k]; g = np.isfinite(col)
            if 5 < g.sum() < col.size:
                idx = np.arange(col.size)
                T[:, k] = np.interp(idx, idx[g], col[g])
        templates[cam] = T
        fin = np.isfinite(A)
        diag['reject_frac'][cam] = float(np.nanmean(pos_out[fin])) if fin.any() else 0.0
        # per-frame amplitude of the template in each frame (physical diagnostic)
        amps = []
        for fA in Ac:
            m = np.isfinite(fA) & np.isfinite(T)
            amps.append(float(np.nansum(fA[m] * T[m]) / np.nansum(T[m] ** 2)) if m.sum() > 100 else np.nan)
        diag['amplitudes'][cam] = np.array(amps)
    logger.info('skyFloorTemplate: built %d camera templates from %d frames',
                len(templates), len(meta))
    return templates, diag


def save_template(path, templates, diag, channel='green'):
    """Write templates + per-frame amplitude table to FITS."""
    from astropy.io import fits
    hdus = [fits.PrimaryHDU()]
    hdus[0].header['CHANNEL'] = channel
    hdus[0].header['NFRAMES'] = len(diag['frames'])
    hdus[0].header['NLBIN'] = NLBIN
    for cam, T in sorted(templates.items()):
        hd = fits.ImageHDU(T.astype(np.float32), name=f'{channel}_{cam}'.upper())
        rej = diag['reject_frac'].get(cam, 0.0)
        hd.header['REJFRAC'] = float(rej) if np.isfinite(rej) else 0.0
        hdus.append(hd)
    cams = sorted(templates)
    cols = [fits.Column(name='FILE', format='120A',
                        array=[m['file'][-120:] for m in diag['frames']]),
            fits.Column(name='OBJECT', format='24A',
                        array=[m['object'][:24] for m in diag['frames']]),
            fits.Column(name='SKYLEV', format='D', array=[m['skylev'] for m in diag['frames']]),
            fits.Column(name='EXPTIME', format='D', array=[m['exptime'] for m in diag['frames']])]
    for cam in cams:
        cols.append(fits.Column(name=f'AMP_{cam}', format='D', array=diag['amplitudes'][cam]))
    hdus.append(fits.BinTableHDU.from_columns(cols, name='AMPLITUDES'))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    logger.info('skyFloorTemplate: wrote %s', path)
    return path


def load_template(path, channel='green'):
    """Load templates written by :func:`save_template`. Returns dict benchside -> (nfib, NLBIN)."""
    from astropy.io import fits
    out = {}
    with fits.open(path) as h:
        for hd in h[1:]:
            name = hd.name.upper()
            if name.startswith(channel.upper() + '_') and hd.is_image and hd.data is not None:
                out[name[len(channel) + 1:]] = np.asarray(hd.data, float)
    return out
