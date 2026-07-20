"""
Absolute flux anchoring to Gaia DR3 XP spectra (Phase 4).

The stacked, flux-calibrated (FLAM) spectra still carry an aperture loss + any zero-point drift.
Anchoring the extracted in-field point source to its Gaia DR3 XP low-resolution, flux-calibrated
SED fixes both at once, using a source already in the frame -- no separate standard. Because we
compare the extracted spectrum to Gaia's actual SED (not a magnitude through a bandpass), it is
SED-independent: it works for a quasar or a star.

The scale is the (robust, wavelength-binned) ratio Gaia/extracted; its wavelength dependence is a
diagnostic (a slope = a residual throughput/colour error in the sensfunc, or -- for a variable
quasar -- an epoch mismatch between the Gaia and observation epochs).

Split: ``anchor_scale`` (the SED comparison; pure, testable) vs ``gaia_xp_sed`` /
``nearest_gaia_source`` (network + gaiaxpy; verified live).

When a source has no XP spectrum (faint, roughly G > 17.6) we fall back to a coarse reference SED
built from Gaia DR3 G/BP/RP broadband photometry (Vega mag -> flux density at each band pivot, SVO
GAIA3 zero points). That fixes the flux LEVEL (and gross colour) but not the detailed SED shape, so
it is flagged as approximate (result ``method='gaia_phot'`` vs ``'xp'``).

Functions
---------
nearest_gaia_source   Gaia source_id nearest a position (TAP cone search)
gaia_xp_sed           Gaia DR3 XP flux-calibrated SED (wave A, FLAM) via gaiaxpy
gaia_photometry       Gaia DR3 mean G/BP/RP magnitudes for a source_id (TAP)
gaia_phot_sed         Coarse reference SED from broadband photometry (XP fallback)
anchor_scale          Robust scale to bring an extracted spectrum onto a reference SED
flux_anchor           Orchestrate: extracted spectra + position -> scale + diagnostics
"""

import logging
import urllib.parse
import urllib.request
from typing import Dict, Optional

import numpy as np

from llamas_pyjamas.Utils.register import GAIA_TAP

logger = logging.getLogger(__name__)

CHANNEL_ORDER = ('blue', 'green', 'red')

# SVO Filter Profile Service, GAIA/GAIA3 (DR3), Vega system: band -> (pivot wavelength [A],
# F_lambda for a zero-magnitude source [erg/s/cm^2/A]). Used by the broadband photometry fallback
# anchor (for sources with no XP spectrum): f_lambda = F0 * 10**(-0.4 * mag_Vega). Gaia mean mags
# (phot_*_mean_mag) are on the Vega system, so these are the right zero points.
GAIA_VEGA_ZP = {
    'BP': (5109.71, 4.07852e-9),
    'G':  (6217.59, 2.50386e-9),
    'RP': (7769.02, 1.26902e-9),
}


class NoXPSpectrumError(RuntimeError):
    """Raised when a Gaia source has no published XP (BP/RP) spectrum (typically G > ~17.6)."""


def nearest_gaia_source(ra_deg, dec_deg, *, radius_arcsec=5.0, mag_limit=21.0, timeout=30):
    """Gaia DR3 ``source_id`` nearest ``(ra_deg, dec_deg)`` within ``radius_arcsec`` (the anchor
    source), or None. Live TAP cone search (urllib), ordered by separation."""
    adql = (f"SELECT source_id, ra, dec, phot_g_mean_mag, "
            f"DISTANCE(POINT('ICRS',ra,dec),POINT('ICRS',{ra_deg},{dec_deg})) AS d "
            f"FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec),"
            f"CIRCLE('ICRS',{ra_deg},{dec_deg},{radius_arcsec / 3600.0})) "
            f"AND phot_g_mean_mag<{mag_limit} ORDER BY d")
    url = GAIA_TAP + '?' + urllib.parse.urlencode(
        {'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv', 'QUERY': adql})
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            lines = resp.read().decode().strip().splitlines()[1:]
    except Exception as exc:                             # noqa: BLE001
        logger.warning('Gaia source query failed: %s', exc)
        return None
    if not lines:
        return None
    parts = lines[0].split(',')
    return int(parts[0])                                 # nearest source_id


def gaia_xp_sed(source_id, *, timeout=120):
    """Gaia DR3 XP flux-calibrated SED for a source: returns ``(wave_A, flam)`` with flam in
    erg/s/cm^2/A. Needs ``gaiaxpy`` (conda install -c conda-forge gaiaxpy) + network.

    gaiaxpy flips matplotlib's ``text.usetex`` to True on import (for its own plots), which would
    make every other matplotlib canvas (e.g. the CubeViewer spectrum panel) require a LaTeX install;
    we save and restore it so importing gaiaxpy can't poison the shared matplotlib state."""
    import matplotlib
    saved_usetex = matplotlib.rcParams.get('text.usetex', False)
    try:
        try:
            import gaiaxpy
        except ImportError as exc:                       # noqa: BLE001
            raise ImportError('gaiaxpy is required for the Gaia XP flux anchor '
                              '(conda install -c conda-forge gaiaxpy)') from exc
        try:
            calibrated, sampling = gaiaxpy.calibrate([int(source_id)], save_file=False)
        except Exception as exc:                          # noqa: BLE001
            # gaiaxpy raises a cryptic "No continuous BP/RP data found" when the source has no
            # published XP spectrum -- true for faint sources (roughly G > 17.6). Make it actionable.
            raise NoXPSpectrumError(
                f'Gaia source {source_id} has no XP (BP/RP) spectrum -- it is likely fainter than '
                f'the XP publication limit (~G>17.6). Use a brighter in-field source for the anchor.'
            ) from exc
        if calibrated is None or len(calibrated) == 0:
            raise NoXPSpectrumError(
                f'Gaia source {source_id} has no XP (BP/RP) spectrum (empty gaiaxpy result); '
                f'it is likely fainter than the XP publication limit (~G>17.6).')
        row = calibrated.iloc[0]
        wave_A = np.asarray(sampling, dtype=float) * 10.0        # nm -> Angstrom
        flam = np.asarray(row['flux'], dtype=float) * 100.0     # W m^-2 nm^-1 -> erg s^-1 cm^-2 A^-1
        return wave_A, flam
    finally:
        matplotlib.rcParams['text.usetex'] = saved_usetex       # undo gaiaxpy's global change


def gaia_photometry(source_id, *, timeout=30) -> Dict[str, float]:
    """Gaia DR3 mean G/BP/RP magnitudes (Vega) for ``source_id`` as ``{'G':.., 'BP':.., 'RP':..}``
    (bands with no measurement are omitted). Live TAP query; returns ``{}`` on failure."""
    adql = (f"SELECT phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag "
            f"FROM gaiadr3.gaia_source WHERE source_id={int(source_id)}")
    url = GAIA_TAP + '?' + urllib.parse.urlencode(
        {'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv', 'QUERY': adql})
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            lines = resp.read().decode().strip().splitlines()
    except Exception as exc:                             # noqa: BLE001
        logger.warning('Gaia photometry query failed: %s', exc)
        return {}
    if len(lines) < 2:
        return {}
    out = {}
    for band, v in zip(('G', 'BP', 'RP'), lines[1].split(',')):
        v = v.strip()
        if v:
            try:
                out[band] = float(v)
            except ValueError:
                pass
    return out


def _phot_to_sed(mags: Dict[str, float]):
    """Pure conversion of Gaia G/BP/RP Vega magnitudes -> a coarse reference SED ``(wave_A, flam)``,
    sorted by wavelength, via the SVO GAIA3 Vega zero points (``f_lambda = F0 * 10**(-0.4*mag)``).
    Needs >=2 usable bands (so anchor_scale has a colour baseline); raises NoXPSpectrumError
    otherwise. Split out from the network fetch so the photometric conversion is unit-testable."""
    pts = []
    for band, mag in mags.items():
        zp = GAIA_VEGA_ZP.get(band)
        if zp is not None and mag is not None and np.isfinite(mag):
            wl, f0 = zp
            pts.append((wl, f0 * 10.0 ** (-0.4 * float(mag))))
    if len(pts) < 2:
        raise NoXPSpectrumError(
            'need >=2 Gaia broadband magnitudes for a photometric fallback anchor, '
            f'got {sorted(mags)}')
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def gaia_phot_sed(source_id, *, timeout=30):
    """Approximate reference SED from Gaia DR3 broadband photometry, for sources with no XP spectrum
    (the faint-source fallback RS asked for). Fetches G/BP/RP and converts each to a flux density at
    its band pivot wavelength -> a 2-3 point SED. This is a FALLBACK: it pins the overall flux LEVEL
    (and gross colour across BP..RP) but not the detailed SED shape, and it only overlaps the
    extracted spectrum between the bluest and reddest band pivots (~5110-7769 A for all three), so
    the anchor is driven by the green/red region. anchor_scale's colour scatter reflects the
    crudeness. Raises NoXPSpectrumError if <2 bands are available."""
    return _phot_to_sed(gaia_photometry(source_id, timeout=timeout))


def anchor_scale(spec_by_channel: Dict[str, tuple], ref_wave, ref_flam, *, nbins=24) -> Optional[dict]:
    """Robust multiplicative scale to bring the extracted spectrum onto the reference SED.

    ``spec_by_channel`` maps channel -> (wave, flux, var) (from optimal_spectrum). Concatenates the
    channels, interpolates the reference SED onto that grid, and takes the MEDIAN over wavelength
    bins of (reference / extracted) -- robust to emission lines and noise. Returns
    ``{'scale', 'scatter', 'nbins', 'lo', 'hi', 'ratios'}`` (scale multiplies the extracted flux to
    match Gaia; scatter = fractional bin-to-bin variation = throughput/colour/variability
    diagnostic) or None if there is too little overlap."""
    ews, efs = [], []
    for c in CHANNEL_ORDER:
        if c in spec_by_channel:
            w, f, _v = spec_by_channel[c]
            ews.append(np.asarray(w, float))
            efs.append(np.asarray(f, float))
    if not ews:
        return None
    ew = np.concatenate(ews)
    ef = np.concatenate(efs)
    order = np.argsort(ew)
    ew, ef = ew[order], ef[order]
    ref_wave = np.asarray(ref_wave, float)
    ref_flam = np.asarray(ref_flam, float)
    lo = max(float(ew.min()), float(ref_wave.min()))
    hi = min(float(ew.max()), float(ref_wave.max()))
    if not (hi > lo):
        return None
    gi = np.interp(ew, ref_wave, ref_flam, left=np.nan, right=np.nan)
    good = np.isfinite(ef) & np.isfinite(gi) & (ef > 0) & (gi > 0) & (ew >= lo) & (ew <= hi)
    if good.sum() < 10:
        return None
    edges = np.linspace(lo, hi, nbins + 1)
    ratios = []
    for i in range(nbins):
        m = good & (ew >= edges[i]) & (ew < edges[i + 1])
        if m.sum() >= 3:
            ratios.append(float(np.median(gi[m] / ef[m])))
    if len(ratios) < 3:
        return None
    ratios = np.array(ratios)
    scale = float(np.median(ratios))
    scatter = float(np.median(np.abs(ratios - scale)) * 1.4826 / scale) if scale else float('nan')
    return {'scale': scale, 'scatter': scatter, 'nbins': len(ratios), 'lo': lo, 'hi': hi,
            'ratios': ratios}


def flux_anchor(spec_by_channel, ra, dec, *, radius_arcsec=5.0, source_id=None,
                nbins=24, allow_phot_fallback=True) -> Optional[dict]:
    """End-to-end: find the Gaia source near ``(ra, dec)`` (or use ``source_id``), get a reference
    SED, and compute the anchor scale for the extracted spectra. Prefers the XP flux-calibrated SED;
    when the source has no XP (faint, roughly G > 17.6) and ``allow_phot_fallback`` (default True),
    falls back to a coarse SED built from Gaia G/BP/RP broadband photometry. Returns the
    ``anchor_scale`` dict plus ``'source_id'`` / ``'gaia_wave'`` / ``'gaia_flam'`` / ``'method'``
    (``'xp'`` or ``'gaia_phot'``), or None. Propagates NoXPSpectrumError only if the fallback is
    disabled or the source also lacks >=2 broadband magnitudes."""
    if source_id is None:
        source_id = nearest_gaia_source(ra, dec, radius_arcsec=radius_arcsec)
    if source_id is None:
        logger.warning('no Gaia source within %.1f" for the flux anchor', radius_arcsec)
        return None
    method = 'xp'
    try:
        gw, gf = gaia_xp_sed(source_id)
    except NoXPSpectrumError:
        if not allow_phot_fallback:
            raise
        logger.info('Gaia %s has no XP spectrum; falling back to broadband-photometry anchor',
                    source_id)
        gw, gf = gaia_phot_sed(source_id)            # raises NoXPSpectrumError if <2 bands either
        method = 'gaia_phot'
    res = anchor_scale(spec_by_channel, gw, gf, nbins=nbins)
    if res is None:
        return None
    res.update(source_id=source_id, gaia_wave=gw, gaia_flam=gf, method=method)
    return res
