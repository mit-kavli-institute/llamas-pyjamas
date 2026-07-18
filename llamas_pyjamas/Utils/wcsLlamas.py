"""Celestial (RA/DEC tangent-plane) WCS for LLAMAS images.

A first-guess sky WCS from the header pointing, so white-light images and cubes carry RA/DEC.
This is the groundwork for registering and stacking dithered exposures; a later phase will
centroid stars on the image and refine the WCS before combining.

Two ways to get a WCS:

* **Rough / header** (:func:`celestial_wcs`) -- for any field, from the header pointing + rotator
  angle. This is the only option for sparse fields with too few stars to solve.
* **Astrometric** (:func:`fit_wcs_from_stars`, :func:`register_pointing`) -- fit the WCS from
  stars matched to a catalogue (Gaia/PanSTARRS). Accurate; needed for dither stacking because the
  header pointing is not reliable.

Calibration (from the J1613 field, TEL_ROT=212, tied to 2 Gaia stars and confirmed vs PanSTARRS)
-----------------------------------------------------------------------------------------------
* The LLAMAS field is **mirror-imaged** on sky -- the correct WCS has ``det(CD) > 0`` (:data:`IFU_MIRRORED`).
* **The rotator keyword is the sky PA of the fibre +x axis:** image +x points at PA = TEL_ROT
  (E of N), +y at TEL_ROT-90. Implemented as a PC rotation of ``pa_deg + IFU_PA_OFFSET``.
* **Plate scale** 0.75"/fibre-map-unit; the fibre-map y-coords already carry the sqrt(3)/2 hex
  row compression, so one isotropic scale gives 0.75" along rows and ~0.65" between rows.

PROVISIONAL: the offset/parity come from a SINGLE rotator angle (212) -- the sign of the TEL_ROT
dependence and the exact offset are unconfirmed (needs a starry field at a very different rotator
angle). The rotator is stable frame-to-frame, so once the mapping is trusted, sparse-field
registration should fix rotation from it and solve only the RA/DEC offset. ``IFU_PA_OFFSET`` is
the master knob (config ``wcs_pa_offset``) so the eventual telescope-side fix drops straight in.
"""

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
from astropy.wcs import WCS

logger = logging.getLogger(__name__)

#: On-sky fibre spacing along a lattice row, per fibre-map unit (arcsec). Row-to-row spacing is
#: this times sqrt(3)/2 (~0.65"), already encoded in the fibre-map y-coordinates.
ARCSEC_PER_FIBRE = 0.75

#: The LLAMAS field is mirror-imaged on sky: the correct WCS has det(CD) > 0. (Confirmed vs
#: PanSTARRS on the J1613 field.) False would give a standard N-up/E-left (det<0) frame.
IFU_MIRRORED = True

#: Degrees added to the header rotator angle (TEL_ROT/TEL_PA) to get the WCS rotation, so that
#: the fibre +x axis lands at sky PA = TEL_ROT. Calibrated from ONE field (J1613, TEL_ROT=212) ->
#: provisional, +-3 deg, and the sign of the TEL_ROT dependence is unconfirmed. Master knob for
#: the eventual TCS fix (config ``wcs_pa_offset``).
IFU_PA_OFFSET = 23.05


def celestial_wcs(ra_deg: float, dec_deg: float, crpix: Sequence[float],
                  arcsec_per_pixel: float, pa_deg: float = 0.0,
                  pa_offset: Optional[float] = None, mirrored: Optional[bool] = None) -> WCS:
    """Build a 2-D RA/DEC tangent-plane WCS from the header pointing + rotator angle.

    LLAMAS convention (see module docstring): mirrored field (``det>0`` when ``mirrored``), with
    the fibre +x axis at sky PA = ``pa_deg + pa_offset``.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Field-centre coordinates -> CRVAL.
    crpix : (float, float)
        1-indexed reference pixel (x, y) that maps to the field centre.
    arcsec_per_pixel : float
        Isotropic plate scale of the image pixels.
    pa_deg : float
        Header rotator angle (TEL_ROT / TEL_PA), degrees.
    pa_offset : float
        Calibration offset added to ``pa_deg`` (default :data:`IFU_PA_OFFSET`).
    mirrored : bool
        Mirror parity (default :data:`IFU_MIRRORED`); True => det(CD) > 0.
    """
    # Resolve calibration knobs at call time so a config override (which rebinds the module
    # constants) takes effect for every caller.
    if pa_offset is None:
        pa_offset = IFU_PA_OFFSET
    if mirrored is None:
        mirrored = IFU_MIRRORED
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.cunit = ['deg', 'deg']
    w.wcs.crval = [float(ra_deg), float(dec_deg)]
    w.wcs.crpix = [float(crpix[0]), float(crpix[1])]
    scale = float(arcsec_per_pixel) / 3600.0
    # mirrored field: cdelt=[+s,+s] (det>0); standard sky: cdelt=[-s,+s] (det<0, N-up/E-left).
    w.wcs.cdelt = [scale if mirrored else -scale, scale]
    theta = np.deg2rad(float(pa_deg or 0.0) + float(pa_offset))
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
    # TEL PA / TEL ROT is the field position angle, degrees east of North -- the angle this WCS
    # applies as a rotation. NOTE (per RS): the telescope control software has had bugs writing
    # this correctly; it *should* be right but the value is not yet trusted. Phase 2 (star
    # centroids -> fit the WCS) must verify both the header value and our applied sign/parity.
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


def fit_wcs_from_stars(pixels, skycoords, mirrored: bool = IFU_MIRRORED) -> WCS:
    """Fit a TAN WCS from matched ``(pixel, SkyCoord)`` pairs (pixels 0-indexed).

    Three or more stars: a full linear fit (astropy ``fit_wcs_from_points``). Exactly two: a
    similarity solve (scale + rotation + shift) in the ``mirrored`` parity branch -- two nearly
    collinear stars leave a mirror ambiguity, so ``mirrored`` selects it (LLAMAS = True). The
    solution reproduces the input stars exactly; verify against a third source / catalogue.
    """
    from astropy.coordinates import SkyCoord
    px = np.asarray([[float(p[0]), float(p[1])] for p in pixels], dtype=float)
    sc = SkyCoord(skycoords)
    if len(px) < 2:
        raise ValueError('fit_wcs_from_stars needs at least 2 stars')
    if len(px) >= 3:
        from astropy.wcs.utils import fit_wcs_from_points
        return fit_wcs_from_points((px[:, 0], px[:, 1]), sc, projection='TAN')

    c0 = sc[0]
    d = px[1] - px[0]
    cosd = np.cos(c0.dec.radian)
    dsky = np.array([(sc[1].ra.deg - c0.ra.deg) * cosd, sc[1].dec.deg - c0.dec.deg])
    dx, dy = d
    if mirrored:                                        # det>0: CD = [[a,-b],[b,a]] (scaled rotation)
        a, b = np.linalg.solve([[dx, -dy], [dy, dx]], dsky)
        cd = np.array([[a, -b], [b, a]])
    else:                                               # det<0: standard-sky reflection branch
        a, b = np.linalg.solve([[dx, dy], [dy, -dx]], dsky)
        cd = np.array([[a, b], [b, -a]])
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [px[0, 0] + 1.0, px[0, 1] + 1.0]      # 1-indexed CRPIX at star 0
    w.wcs.crval = [c0.ra.deg, c0.dec.deg]
    w.wcs.cd = cd
    return w


def register_pointing(wcs: WCS, pixels, skycoords) -> WCS:
    """Return a copy of `wcs` with CRVAL shifted so matched stars fit the catalogue.

    Keeps the CD matrix (the trusted, rotator-derived rotation + scale) and solves only the
    translation -- the sparse-field strategy: fix rotation from the calibration, register the
    RA/DEC offset from one or more stars. `pixels` are 0-indexed.
    """
    from astropy.coordinates import SkyCoord
    sc = SkyCoord(skycoords)
    px = np.asarray([[float(p[0]), float(p[1])] for p in pixels], dtype=float)
    pred = wcs.pixel_to_world(px[:, 0], px[:, 1])
    dra = float(np.mean(sc.ra.deg - pred.ra.deg))
    ddec = float(np.mean(sc.dec.deg - pred.dec.deg))
    out = wcs.deepcopy()
    out.wcs.crval = [wcs.wcs.crval[0] + dra, wcs.wcs.crval[1] + ddec]
    return out
