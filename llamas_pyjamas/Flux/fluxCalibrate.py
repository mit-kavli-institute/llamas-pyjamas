"""
Apply a sensitivity function to a science RSS to produce flux-calibrated spectra.

Phase III of the flux-calibration chain. Given the sensitivity function built from a standard
(:mod:`llamas_pyjamas.Flux.sensFunc`), this converts a science exposure's sky-subtracted counts
into physical flux and writes it back into the RSS as new ``FLAM`` / ``FLAM_ERR`` extensions —
in place, no new files. The instrumental planes are left untouched.

The calibration, per fibre and wavelength::

    FLAM(lambda) = ( SKYSUB(lambda) / t_exp ) * S(lambda) * E(lambda)

where ``S`` is the sensitivity function for this channel and ``E`` is the **differential**
atmospheric-extinction correction between the science and standard airmasses,

    E(lambda) = 10 ** ( 0.4 * k(lambda) * (X_science - X_standard) )

The standard's own extinction is already folded into ``S`` (the sensfunc was built without
correcting the standard to zero airmass), so only the difference in airmass is applied here.
That is why the sensfunc must record the standard's airmass; without it, extinction is skipped
with a warning and the calibration is only exact when the two airmasses match. The extinction
curve k(lambda) comes from pypeit's bundled data — the La Silla / CTIO curve, the nearest match
to Las Campanas, selected by observatory coordinates.

The sky-subtracted plane is read as ``SKYSUB`` with a fallback to ``FLUX``, so this works both
before and after the planned ``FLUX -> SKYSUB`` rename.

Functions
---------
load_lco_extinction   The La Silla/CTIO extinction table for Las Campanas
skysub_extname        Which extension holds the sky-subtracted plane (SKYSUB or FLUX)
apply_sensfunc        Add FLAM/FLAM_ERR to a science RSS HDUList
flux_calibrate_file   Read a science RSS, calibrate, write FLAM/FLAM_ERR back in place
"""

import logging
import os
from typing import Optional

import numpy as np
from astropy.io import fits

from llamas_pyjamas.File.llamasRSS import skysub_extname
from llamas_pyjamas.Flux.sensFunc import SensFunc

logger = logging.getLogger(__name__)

# Las Campanas Observatory (Magellan). pypeit picks the nearest bundled curve (La Silla) —
# essentially identical extinction for a site ~30 km away at the same altitude.
LCO_LONGITUDE = -70.692
LCO_LATITUDE = -29.015

FLAM_BUNIT = 'erg/s/cm2/Angstrom'

_EXT_CACHE = None


def load_lco_extinction():
    """The atmospheric extinction table (wave, mag_ext) for Las Campanas, cached.

    Reuses pypeit's bundled curves via coordinate lookup; no data is bundled here.
    """
    global _EXT_CACHE
    if _EXT_CACHE is None:
        from pypeit.core.flux_calib import load_extinction_data
        _EXT_CACHE = load_extinction_data(LCO_LONGITUDE, LCO_LATITUDE, 'closest')
    return _EXT_CACHE


def _header_value(header, *keys, default=None):
    for key in keys:
        if key in header:
            return header[key]
        hierarch = 'HIERARCH ' + key
        if hierarch in header:
            return header[hierarch]
    return default


def _differential_extinction(wave: np.ndarray, x_sci: float, x_std: Optional[float],
                             extinct) -> np.ndarray:
    """E(lambda) = 10^(0.4 k (X_sci - X_std)) on `wave`; 1.0 where it cannot be computed.

    Returns an array the shape of `wave`. If the standard airmass is unknown the factor is
    unity (extinction skipped) — the caller warns.
    """
    if x_std is None or x_sci is None:
        return np.ones_like(wave, dtype=float)
    from pypeit.core.flux_calib import extinction_correction
    # extinction_correction returns 10^(0.4 k X); the differential is the ratio.
    flat = np.asarray(wave, dtype=float).ravel()
    corr_sci = extinction_correction(flat, float(x_sci), extinct)
    corr_std = extinction_correction(flat, float(x_std), extinct)
    return (corr_sci / corr_std).reshape(np.shape(wave))


def apply_sensfunc(hdul: fits.HDUList, sensfunc: SensFunc,
                   apply_extinction: bool = True, extinct=None,
                   sensfile: str = '') -> fits.HDUList:
    """Add FLAM / FLAM_ERR extensions to a science RSS HDUList, in place.

    Parameters
    ----------
    hdul : HDUList
        An open single-channel science RSS (primary ``CHANNEL`` set; SKYSUB/FLUX, ERROR, WAVE).
    sensfunc : SensFunc
        The sensitivity function; must cover this channel.
    apply_extinction : bool
        Apply the differential extinction correction (needs the standard airmass in the
        sensfunc and the science airmass in the header).
    extinct : optional
        Pre-loaded extinction table; :func:`load_lco_extinction` if None.
    sensfile : str
        Path of the sensfunc, recorded in provenance.

    Returns
    -------
    HDUList
        The same object, with FLAM and FLAM_ERR appended (replacing any existing ones).
    """
    header = hdul[0].header
    channel = str(header.get('CHANNEL', '')).strip().lower()
    if channel not in sensfunc.channels:
        raise ValueError(f"sensfunc has no '{channel}' channel (has: "
                         f"{', '.join(sensfunc.channels)})")

    sky_name = skysub_extname(hdul)
    skysub = np.asarray(hdul[sky_name].data, dtype=float)
    wave = np.asarray(hdul['WAVE'].data, dtype=float)
    error = (np.asarray(hdul['ERROR'].data, dtype=float)
             if 'ERROR' in {h.name for h in hdul} else None)

    exptime = _header_value(header, 'SEXPTIME', 'REXPTIME', 'EXPTIME')
    if exptime is None:
        raise ValueError('no exposure time in header (SEXPTIME/REXPTIME/EXPTIME)')
    exptime = float(exptime)

    x_sci = _header_value(header, 'AIRMASS', 'TEL AIRMASS')
    x_std = sensfunc.meta.get('airmass')
    try:
        x_sci = float(x_sci) if x_sci is not None else None
    except (TypeError, ValueError):
        x_sci = None

    # S(lambda) on the science grid, per fibre. np.interp is 1-D; ravel/reshape covers the 2-D
    # per-fibre WAVE array in one call.
    ch = sensfunc.channels[channel]
    s_grid = np.interp(wave.ravel(), ch.wave, ch.sens,
                       left=np.nan, right=np.nan).reshape(wave.shape)

    if apply_extinction and (x_std is None or x_sci is None):
        logger.warning('extinction skipped: %s airmass unknown (sci=%s, std=%s); '
                       'calibration exact only at equal airmass',
                       channel, x_sci, x_std)
    if apply_extinction and x_std is not None and x_sci is not None:
        extinct = extinct if extinct is not None else load_lco_extinction()
        e_grid = _differential_extinction(wave, x_sci, x_std, extinct)
    else:
        e_grid = 1.0

    with np.errstate(invalid='ignore'):
        flam = (skysub / exptime) * s_grid * e_grid
        flam_err = (error / exptime) * s_grid * e_grid if error is not None else None

    prov = {
        'FLUXCAL': (True, 'Flux calibration applied'),
        'BUNIT': FLAM_BUNIT,
        'SENSFILE': (os.path.basename(sensfile)[:68], 'Sensitivity function used'),
        'STDNAME': (str(sensfunc.meta.get('standard', ''))[:68], 'Standard star'),
        'STDAIR': (x_std if x_std is not None else 0.0, 'Standard airmass'),
        'SCIAIR': (x_sci if x_sci is not None else 0.0, 'Science airmass'),
        'EXTCORR': (bool(apply_extinction and x_std is not None and x_sci is not None),
                    'Differential extinction applied'),
    }

    _replace_or_append(hdul, 'FLAM', flam, prov)
    if flam_err is not None:
        err_prov = dict(prov)
        err_prov['FLUXCAL'] = (True, 'Flux-calibrated 1-sigma error')
        _replace_or_append(hdul, 'FLAM_ERR', flam_err, err_prov)
    logger.info('Flux-calibrated %s with %s (extinction=%s)',
                channel, sensfunc.meta.get('standard', '?'),
                bool(apply_extinction and x_std is not None and x_sci is not None))
    return hdul


def _replace_or_append(hdul: fits.HDUList, name: str, data: np.ndarray, header_items) -> None:
    """Add an ImageHDU named `name`, replacing an existing one so re-runs stay idempotent."""
    hdu = fits.ImageHDU(data=np.asarray(data, dtype=np.float32), name=name)
    for key, val in header_items.items():
        hdu.header[key] = val
    existing = [i for i, h in enumerate(hdul) if h.name == name]
    if existing:
        hdul[existing[0]] = hdu
    else:
        hdul.append(hdu)


def flux_calibrate_file(rss_path: str, sensfunc, apply_extinction: bool = True,
                        out_path: Optional[str] = None, extinct=None) -> str:
    """Flux-calibrate a science RSS on disk, writing FLAM/FLAM_ERR back into it.

    Parameters
    ----------
    rss_path : str
        Science RSS to calibrate.
    sensfunc : SensFunc or str
        A loaded SensFunc, or a path to one.
    apply_extinction : bool
    out_path : str, optional
        Where to write; overwrites `rss_path` in place if None.
    extinct : optional
        Pre-loaded extinction table.

    Returns
    -------
    str
        The path written.
    """
    if isinstance(sensfunc, str):
        sensfile = sensfunc
        sensfunc = SensFunc.load(sensfunc)
    else:
        sensfile = ''

    with fits.open(rss_path) as hdul:
        apply_sensfunc(hdul, sensfunc, apply_extinction=apply_extinction,
                       extinct=extinct, sensfile=sensfile)
        target = out_path or rss_path
        hdul.writeto(target, overwrite=True)
    logger.info('Wrote flux-calibrated RSS to %s', target)
    return target
