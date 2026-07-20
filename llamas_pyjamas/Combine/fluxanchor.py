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

Functions
---------
nearest_gaia_source   Gaia source_id nearest a position (TAP cone search)
gaia_xp_sed           Gaia DR3 XP flux-calibrated SED (wave A, FLAM) via gaiaxpy
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
    erg/s/cm^2/A. Needs ``gaiaxpy`` (conda install -c conda-forge gaiaxpy) + network."""
    try:
        import gaiaxpy
    except ImportError as exc:                           # noqa: BLE001
        raise ImportError('gaiaxpy is required for the Gaia XP flux anchor '
                          '(conda install -c conda-forge gaiaxpy)') from exc
    calibrated, sampling = gaiaxpy.calibrate([int(source_id)], save_file=False)
    row = calibrated.iloc[0]
    wave_A = np.asarray(sampling, dtype=float) * 10.0            # nm -> Angstrom
    flam = np.asarray(row['flux'], dtype=float) * 100.0         # W m^-2 nm^-1 -> erg s^-1 cm^-2 A^-1
    return wave_A, flam


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
                nbins=24) -> Optional[dict]:
    """End-to-end: find the Gaia source near ``(ra, dec)`` (or use ``source_id``), fetch its XP SED,
    and compute the anchor scale for the extracted spectra. Returns the ``anchor_scale`` dict plus
    ``'source_id'`` / ``'gaia_wave'`` / ``'gaia_flam'``, or None."""
    if source_id is None:
        source_id = nearest_gaia_source(ra, dec, radius_arcsec=radius_arcsec)
    if source_id is None:
        logger.warning('no Gaia source within %.1f" for the flux anchor', radius_arcsec)
        return None
    gw, gf = gaia_xp_sed(source_id)
    res = anchor_scale(spec_by_channel, gw, gf, nbins=nbins)
    if res is None:
        return None
    res.update(source_id=source_id, gaia_wave=gw, gaia_flam=gf)
    return res
