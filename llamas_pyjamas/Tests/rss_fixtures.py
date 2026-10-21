"""Synthetic RSS files for the final-stage tests (not a test module itself).

:func:`write_rss` writes one loadable channel file shaped like the pipeline's products: PRIMARY
(science keys), SKYSUB, ERROR, COUNTS, SKY, MASK, WAVE, FWHM, FIBERMAP and FIBERWCS, plus the
stage markers (FIBRFLAT / SKYSUB2 + SKYRESID + SKYMASK) and optional FLAM/FLAM_ERR and NOFLAT.
The sky holds OH-like lines and the first fibres are bright, so the real fibre-flat and
sky-framework code can run on it.
"""

import os

import numpy as np
from astropy.io import fits

from llamas_pyjamas.File.llamasRSS import _fiberwcs_hdu

N_FIB = 60
N_WAVE = 400
N_BRIGHT = 8
BENCHSIDES = np.array(['1A'] * 30 + ['1B'] * 30)
FIBER_IDS = np.concatenate([np.arange(30), np.arange(30)]).astype(np.int32)
RA0, DEC0 = 150.0, 2.0
WAVE_RANGE = {'blue': (3500.0, 4800.0), 'green': (4700.0, 7000.0), 'red': (6800.0, 9800.0)}
CHANNELS = ('blue', 'green', 'red')

LONG = 'LLAMAS_2026-10-16_00-31-08.4_SCI24_mef_bias_corrected_flat_corrected_extract'
EXP_ID = 'LLAMAS_2026-10-16_00-31-08.4'

SCIENCE_KEYS = {
    'OBJECT': 'TESTFIELD', 'SEXPTIME': 600.0, 'AIRMASS': 1.2, 'RA': RA0, 'DEC': DEC0,
    'TEL ROT': 0.0, 'MJD-OBS': 61000.25, 'OBS TYPE': 'SCI24', 'FIBAREA': 0.4418,
    'VELFRAME': 'heliocentric', 'HELIOVEL': 12.3, 'VELCORR': 1.000041,
}


def fibre_positions(offset_arcsec=0.0):
    """Rough (header) per-fibre RA/DEC on a small grid; ``offset_arcsec`` shifts RA."""
    i = np.arange(N_FIB)
    ra = RA0 + (i % 10) * 0.75 / 3600.0 + offset_arcsec / 3600.0
    dec = DEC0 + (i // 10) * 0.65 / 3600.0
    return ra, dec


def _sky(nwave):
    x = np.arange(nwave)
    s = np.zeros(nwave)
    for c in (60, 130, 200, 275, 340):
        s += 100.0 * np.exp(-0.5 * ((x - c * nwave / 400.0) / 2.0) ** 2)
    return s


def write_rss(path, *, stage='base', channel='green', registered=False, wcs_method=None,
              wcs_refined=None, fiberwcs=True, flam=False, noflat=False, rsstage=None,
              header=None, mask_bits=None, zero_counts=None, seed=0, nfib=N_FIB,
              nwave=N_WAVE):
    """Write a synthetic RSS channel file and return its path.

    ``stage`` sets the structural markers; ``rsstage`` writes an RSSSTAGE card (default: the
    stage itself; pass ``False`` for a legacy file without one). ``registered`` writes a refined
    FIBERWCS 1" off the rough positions. ``mask_bits`` is ``{bit: (rows, cols)}`` OR-ed into
    MASK; ``zero_counts`` is ``(rows, cols)`` set to COUNTS == 0 with SKYSUB = -SKY there.
    """
    rng = np.random.default_rng(seed)
    lo, hi = WAVE_RANGE[channel]
    wave = np.tile(np.linspace(lo, hi, nwave), (nfib, 1))
    sky = np.tile(_sky(nwave), (nfib, 1))
    cont = np.full((nfib, nwave), 5.0)
    cont[:N_BRIGHT] = 500.0
    counts = cont + sky + rng.normal(0, 1, (nfib, nwave))
    skysub = counts - sky
    err = np.full((nfib, nwave), 1.0)
    mask = np.zeros((nfib, nwave), dtype=np.int16)
    if zero_counts is not None:
        counts[zero_counts] = 0.0
        skysub[zero_counts] = -sky[zero_counts]
    for bit, sl in (mask_bits or {}).items():
        mask[sl] |= np.int16(bit)

    phdu = fits.PrimaryHDU()
    for k, v in SCIENCE_KEYS.items():
        phdu.header[k] = v
    phdu.header['CHANNEL'] = channel
    if stage in ('ff', 'ff_skysub'):
        phdu.header['FIBRFLAT'] = True
        phdu.header['FFLATFIL'] = 'fibre_flat_corrections.fits'
        phdu.header['FFMETHD'] = 'twilight'
    if rsstage is None:
        rsstage = stage
    if rsstage:
        phdu.header['RSSSTAGE'] = rsstage
    for k, v in (header or {}).items():
        phdu.header[k] = v

    def img(data, name, dtype=np.float32):
        h = fits.ImageHDU(np.asarray(data, dtype=dtype))
        h.header['EXTNAME'] = name
        h.header['CHANNEL'] = channel
        return h

    sky_hdu = img(skysub, 'SKYSUB')
    if stage == 'ff_skysub':
        sky_hdu.header['SKYSUB2'] = True
    hdus = [phdu, sky_hdu, img(err, 'ERROR')]
    if noflat:
        hdus.append(img(skysub, 'NOFLAT'))
    hdus += [img(counts, 'COUNTS'), img(sky, 'SKY'), img(mask, 'MASK', np.int16),
             img(wave, 'WAVE', np.float64), img(np.full((nfib, nwave), 2.5), 'FWHM')]

    ra, dec = fibre_positions()
    fm = fits.BinTableHDU.from_columns([
        fits.Column(name='FIBER_ID', format='J', array=FIBER_IDS[:nfib]),
        fits.Column(name='BENCHSIDE', format='10A', array=BENCHSIDES[:nfib]),
        fits.Column(name='FIBER_TYPE', format='10A', array=np.array(['UNKNOWN'] * nfib)),
        fits.Column(name='RA', format='D', array=ra[:nfib]),
        fits.Column(name='DEC', format='D', array=dec[:nfib]),
    ])
    fm.header['EXTNAME'] = 'FIBERMAP'
    hdus.append(fm)
    if fiberwcs:
        if registered:
            ra_w, dec_w = fibre_positions(offset_arcsec=1.0)
            method = wcs_method or 'astrometric'
            refined = True if wcs_refined is None else wcs_refined
        else:
            ra_w, dec_w = ra, dec
            method = wcs_method or 'rough-header'
            refined = False if wcs_refined is None else wcs_refined
        prov = {'method': method, 'tier': 'gaia' if refined else 'header', 'refined': refined,
                'pa_offset': 0.0, 'catalog': 'GaiaDR3' if refined else '', 'rms': 0.1,
                'nstars': 3 if refined else 0}
        hdus.append(_fiberwcs_hdu(FIBER_IDS[:nfib], BENCHSIDES[:nfib], ra_w[:nfib], dec_w[:nfib],
                                  np.arange(nfib, dtype=float), np.zeros(nfib), prov))
        if registered:            # registration also rewrites FIBERMAP RA/DEC
            hdus[-2].data['RA'][:] = ra_w[:nfib]
            hdus[-2].data['DEC'][:] = dec_w[:nfib]
    if stage == 'ff_skysub':
        resid = img(np.zeros((nfib, nwave)), 'SKYRESID')
        resid.header['SKYSUB2'] = True
        hdus.append(resid)
        sm = fits.ImageHDU(np.zeros(nfib, dtype=np.uint8))
        sm.header['EXTNAME'] = 'SKYMASK'
        hdus.append(sm)
    if flam:
        f = img(skysub / 600.0 * 1e-17, 'FLAM')
        f.header['FLUXCAL'] = True
        f.header['RSSSTAGE'] = stage
        hdus.append(f)
        hdus.append(img(err / 600.0 * 1e-17, 'FLAM_ERR'))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def stage_name(directory, channel, stage, base=LONG, consolidated=False):
    """Pipeline filename for one channel/stage (or the consolidated name)."""
    if consolidated:
        return os.path.join(directory, f'{EXP_ID}_RSS_{channel}.fits')
    suffix = {'base': '', 'ff': '_FF', 'ff_skysub': '_FF_SKYSUB'}[stage]
    return os.path.join(directory, f'{base}_RSS_{channel}{suffix}.fits')


def write_exposure(directory, stages=('base', 'ff', 'ff_skysub'), channels=CHANNELS,
                   base=LONG, per_stage=None, **kwargs):
    """Write every (stage, channel) file of one exposure; ``per_stage[stage]`` overrides kwargs.
    Returns ``{(stage, channel): path}``."""
    out = {}
    for stage in stages:
        kw = dict(kwargs)
        kw.update((per_stage or {}).get(stage, {}))
        if stage != 'base':
            kw.setdefault('fiberwcs', False)       # legacy FF stages carry no FIBERWCS
        for ch in channels:
            out[(stage, ch)] = write_rss(stage_name(directory, ch, stage, base=base),
                                         stage=stage, channel=ch, **kw)
    return out
