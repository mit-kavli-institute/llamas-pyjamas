"""Celestial (RA/DEC tangent-plane) WCS for LLAMAS images.

A first-guess sky WCS from the header pointing, so white-light images and cubes carry RA/DEC.
This is the groundwork for registering and stacking dithered exposures; a later phase will
centroid stars on the image and refine the WCS before combining.

Conventions
-----------
* **North up, East left.** ``CDELT1 < 0`` (RA decreases with +x, so East is to the left) and
  ``CDELT2 > 0`` (DEC increases with +y, North up). Field rotation is applied as a PC matrix.
* **Plate scale.** The LLAMAS fibre spacing on sky is 0.75" between fibres along a row. The
  fibre-map coordinates already carry the hexagonal row compression (rows are sqrt(3)/2 apart),
  so a single isotropic scale of 0.75"/fibre-map-unit gives 0.75" along rows and ~0.65" between
  rows automatically -- no separate y scale.

The pointing gives the field centre and the plate scale/rotation give the orientation, but the
IFU-to-sky **parity** (:data:`IFU_PARITY`) and the **rotation sign** are an initial guess: they
are not yet validated against a known field. Phase 2 (centroid stars -> fit the WCS) will pin
them down; until then a wrong parity only mirrors/rotates the frame and is a one-constant flip.
"""

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
from astropy.wcs import WCS

logger = logging.getLogger(__name__)

#: On-sky fibre spacing along a lattice row, per fibre-map unit (arcsec). Row-to-row spacing is
#: this times sqrt(3)/2 (~0.65"), already encoded in the fibre-map y-coordinates.
ARCSEC_PER_FIBRE = 0.75

#: Sign of the RA (x) axis. -1 puts East to the left (standard astronomical display). Flip to
#: +1 if a known field shows the image mirrored -- to be validated in phase 2.
IFU_PARITY = -1


def celestial_wcs(ra_deg: float, dec_deg: float, crpix: Sequence[float],
                  arcsec_per_pixel: float, pa_deg: float = 0.0) -> WCS:
    """Build a 2-D RA/DEC tangent-plane WCS (North up, East left).

    Parameters
    ----------
    ra_deg, dec_deg : float
        Field-centre coordinates -> CRVAL.
    crpix : (float, float)
        1-indexed reference pixel (x, y) that maps to the field centre.
    arcsec_per_pixel : float
        Isotropic plate scale of the image pixels.
    pa_deg : float
        Field position angle (degrees), applied as a PC rotation. Sign is provisional.
    """
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.cunit = ['deg', 'deg']
    w.wcs.crval = [float(ra_deg), float(dec_deg)]
    w.wcs.crpix = [float(crpix[0]), float(crpix[1])]
    scale = float(arcsec_per_pixel) / 3600.0
    w.wcs.cdelt = [IFU_PARITY * scale, scale]          # E-left, N-up at PA=0
    theta = np.deg2rad(float(pa_deg or 0.0))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    w.wcs.pc = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return w


def pointing_from_header(header) -> Tuple[Optional[float], Optional[float], float]:
    """Return (ra_deg, dec_deg, pa_deg) from a primary header, or (None, None, 0.0).

    Prefers decimal ``RA``/``DEC``; falls back to sexagesimal HIERARCH ``TEL RA``/``TEL DEC``
    (hourangle/deg). Rotation from ``TEL PA`` then ``TEL ROT``. Matches
    ``reduce._pointing_from_header`` but returns None (not 0,0) when the pointing is absent, so
    callers can fall back to a non-celestial WCS rather than mislabel the field as RA=DEC=0.
    """
    if header is None:
        return None, None, 0.0
    pa = header.get('TEL PA', header.get('TEL ROT', 0.0))
    try:
        pa = float(pa)
    except (TypeError, ValueError):
        pa = 0.0

    ra, dec = header.get('RA'), header.get('DEC')
    try:
        ra, dec = float(ra), float(dec)
        if np.isfinite(ra) and np.isfinite(dec):
            return ra, dec, pa
    except (TypeError, ValueError):
        pass

    tra, tdec = header.get('TEL RA'), header.get('TEL DEC')
    if tra and tdec:
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            c = SkyCoord(str(tra), str(tdec), unit=(u.hourangle, u.deg))
            return float(c.ra.deg), float(c.dec.deg), pa
        except Exception:                               # noqa: BLE001 - bad string => no pointing
            return None, None, pa
    return None, None, pa
