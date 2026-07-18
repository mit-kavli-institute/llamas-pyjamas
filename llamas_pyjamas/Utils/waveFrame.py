"""Heliocentric / barycentric wavelength-frame correction.

The arc solution is a *vacuum* wavelength scale in the observatory (topocentric) frame. Science
and standard-star exposures are shifted into the heliocentric (or barycentric) rest frame by
multiplying the per-fibre ``WAVE`` array by pypeit's velocity factor
``vel_corr = sqrt((1 + v/c)/(1 - v/c))``.

Two properties make this safe and simple:

* **Per exposure, uniform across fibres.** The factor depends only on the target and the
  mid-exposure time, so every fibre of an exposure is scaled identically.
* **Applied after sky subtraction.** :func:`stamp_and_factor` is called at RSS assembly, once
  sky subtraction has already run in the observed frame. OH sky lines are atmospheric and stay
  in the observed frame; shifting ``WAVE`` afterwards never disturbs them. (Sky subtraction
  itself never touches ``WAVE``, so the numbers are identical to applying at arc transfer.)

The correction is recorded in the primary header and is **reversible**: ``VELFRAME`` names the
frame, ``HELIOVEL`` is the applied velocity in km/s, and ``VELCORR`` is the factor ``WAVE`` was
multiplied by. A user who needs the observed frame again -- e.g. for telluric correction, whose
bands are atmospheric and must stay topocentric -- divides ``WAVE`` back by ``VELCORR``.

Only ``OBS TYPE`` science frames (which includes standard stars) are corrected; arcs, flats and
other calibrations are left in the observed frame.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Las Campanas Observatory (Magellan / LLAMAS). Longitude/latitude match
# Flux.fluxCalibrate; elevation ~2380 m (its effect on the correction is far below LLAMAS's
# resolution but the pypeit API takes it).
LCO_LONGITUDE = -70.692
LCO_LATITUDE = -29.015
LCO_ELEVATION = 2380.0

#: Accepted reference frames (anything else, or a falsy/"none" value, disables the correction).
VALID_FRAMES = ('heliocentric', 'barycentric')


def _is_disabled(frame) -> bool:
    return not frame or str(frame).strip().lower() in ('none', 'off', 'false', 'observed')


def _radec(header):
    """SkyCoord for the pointing, or None. Decimal RA/DEC, else sexagesimal TEL RA/TEL DEC."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ra, dec = header.get('RA'), header.get('DEC')
    try:
        ra, dec = float(ra), float(dec)
        if np.isfinite(ra) and np.isfinite(dec):
            return SkyCoord(ra * u.deg, dec * u.deg)
    except (TypeError, ValueError):
        pass
    tra, tdec = header.get('TEL RA'), header.get('TEL DEC')
    if tra and tdec:
        try:
            return SkyCoord(str(tra), str(tdec), unit=(u.hourangle, u.deg))
        except Exception:                               # noqa: BLE001 - bad string => no coord
            return None
    return None


def _obs_time(header):
    """astropy Time at mid-exposure (UTC), or None.

    Prefers ``MJD-OBS`` (unambiguous UTC). Falls back to ``TEL DATE-OBS`` + ``UTC`` -- note the
    plain ``DATE-OBS`` here is the *local* Chile date, so it is deliberately not used. Half the
    exposure time (``SEXPTIME``/``REXPTIME``/``EXPTIME``) is added to reach mid-exposure.
    """
    from astropy.time import Time
    import astropy.units as u

    t = None
    mjd = header.get('MJD-OBS')
    if mjd is not None:
        try:
            t = Time(float(mjd), format='mjd', scale='utc')
        except (TypeError, ValueError):
            t = None
    if t is None:
        date, utc = header.get('TEL DATE-OBS'), header.get('UTC')
        if date and utc:
            try:
                t = Time(f'{str(date).strip()}T{str(utc).strip()}', format='isot', scale='utc')
            except (ValueError, TypeError):
                t = None
    if t is None:
        return None

    exptime = header.get('SEXPTIME', header.get('REXPTIME', header.get('EXPTIME')))
    try:
        t = t + (float(exptime) / 2.0) * u.s
    except (TypeError, ValueError):
        pass                                            # start-of-exposure time is close enough
    return t


def velocity_and_factor(header, frame='heliocentric'):
    """Return ``(vel_kms, vel_corr)`` for an exposure, or None if it cannot be computed.

    Uses pypeit's ``geomotion_correct`` at the LCO site. ``vel_corr`` is the multiplicative
    factor to apply to a wavelength array; ``vel_kms`` is the corresponding velocity (the sign
    is pypeit's convention). No header mutation, no guards -- see :func:`stamp_and_factor`.
    """
    coord, time = _radec(header), _obs_time(header)
    if coord is None or time is None:
        return None
    try:
        from pypeit.core.wave import geomotion_correct
        vel, vel_corr = geomotion_correct(coord, time, LCO_LONGITUDE, LCO_LATITUDE,
                                          LCO_ELEVATION, str(frame).lower())
        return float(vel), float(vel_corr)
    except Exception as exc:                            # noqa: BLE001 - never fail the reduction
        logger.warning('velocity correction failed (%s); WAVE left in observed frame', exc)
        return None


def stamp_and_factor(header, frame='heliocentric'):
    """Decide, compute, record. Returns ``(vel_kms, vel_corr)`` to apply to ``WAVE``.

    Returns ``(0.0, 1.0)`` -- i.e. no change -- when the correction is disabled, the frame is
    not an ``OBS TYPE`` science exposure (arcs/flats stay observed), the pointing/time are
    missing, or it has already been applied (idempotent, keyed on ``VELFRAME``). Otherwise the
    header is stamped with ``VELFRAME``/``HELIOVEL``/``VELCORR`` and the factor is returned.
    """
    if header is None or _is_disabled(frame):
        return 0.0, 1.0
    frame = str(frame).strip().lower()
    if frame not in VALID_FRAMES:
        logger.warning("Unknown wave_frame %r; leaving WAVE in the observed frame", frame)
        return 0.0, 1.0
    if not str(header.get('OBS TYPE', '')).upper().startswith('SCI'):
        return 0.0, 1.0                                 # calibration frame: no correction
    if header.get('VELFRAME'):                          # already corrected -> do not re-scale
        return float(header.get('HELIOVEL', 0.0) or 0.0), 1.0

    result = velocity_and_factor(header, frame)
    if result is None:
        logger.warning('wave_frame=%s requested but pointing/time unavailable; '
                       'WAVE left in the observed frame', frame)
        return 0.0, 1.0
    vel, vel_corr = result
    header['VELFRAME'] = (frame, 'Wavelength reference frame')
    header['HELIOVEL'] = (round(vel, 4), f'Applied {frame} velocity [km/s]')
    header['VELCORR'] = (vel_corr, 'WAVE scaled by this; divide out for observed frame')
    logger.info('Applied %s correction: v=%.4f km/s, WAVE *= %.8f', frame, vel, vel_corr)
    return vel, vel_corr
