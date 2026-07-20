"""Tests for optimal point-source extraction and the narrowband line image (Phase 4, step 2).

Synthetic: a Gaussian point source sampled by fibres over two exposures (optimal extraction should
recover the template and beat a single fibre / uniform sum), the seeing estimate, and a cube with a
line on one spaxel (narrowband should recover the line integral and continuum-subtract to zero
off-line). Validated further on real may26 data.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_combine_spectrum`).
"""

import numpy as np
from astropy.wcs import WCS

from llamas_pyjamas.Combine.superRSS import ChannelStack, SuperRSS, ExposureMeta
from llamas_pyjamas.Combine.cube import CoaddCube, narrowband_image
from llamas_pyjamas.Combine.spectrum import (optimal_spectrum, estimate_psf_fwhm, measure_dar,
                                             fit_source_profile)

RA0, DEC0 = 150.0, 20.0
FWHM = 1.5


def _point_source_super(nw=8, C=10.0, fwhm=FWHM, nexp=2):
    sigma = fwhm / 2.35482
    offs = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)]     # 7x7 fibres, 0.75" pitch
    dx = np.array([o[0] * 0.75 for o in offs])
    dy = np.array([o[1] * 0.75 for o in offs])
    r = np.hypot(dx, dy)
    P = np.exp(-0.5 * (r / sigma) ** 2)
    cosd = np.cos(np.deg2rad(DEC0))
    ra = RA0 + dx / cosd / 3600.0
    dec = DEC0 + dy / 3600.0
    template = np.full(nw, C)
    flux1 = P[:, None] * template[None, :]                  # (nfib, nw): source * PSF
    wave = np.tile(np.linspace(5000, 5000 + nw - 1, nw), (len(offs), 1))
    ex = []
    stacks = dict(ra=[], dec=[], wave=[], flux=[], var=[], mask=[], solid=[], exp=[])
    for e in range(nexp):
        stacks['ra'].append(ra); stacks['dec'].append(dec); stacks['wave'].append(wave)
        stacks['flux'].append(flux1.copy()); stacks['var'].append(np.ones_like(flux1))
        stacks['mask'].append(np.zeros_like(flux1, bool))
        stacks['solid'].append(np.full(len(offs), 0.44)); stacks['exp'].append(np.full(len(offs), e))
        ex.append(ExposureMeta(f'e{e}', f'p{e}', 1.0, 1.0, float(e)))
    st = ChannelStack('green', np.concatenate(stacks['ra']), np.concatenate(stacks['dec']),
                      np.concatenate(stacks['wave']), np.concatenate(stacks['flux']),
                      np.concatenate(stacks['var']), np.concatenate(stacks['mask']),
                      np.concatenate(stacks['solid']), np.concatenate(stacks['exp']))
    return SuperRSS('T', 'skysub', 'counts', ex, {'green': st})


def test_optimal_spectrum_fits_profile_and_recovers_template():
    sr = _point_source_super(C=10.0)
    spec, fit = optimal_spectrum(sr, RA0, DEC0, radius_arcsec=3.0)     # fits the profile by default
    assert fit.fitted                                    # a real 2-D Gaussian fit, not assumed
    assert abs(fit.fwhm - FWHM) < 0.4                    # recovers the ~1.5" input width
    assert abs(fit.ra - RA0) < 5e-4 and abs(fit.dec - DEC0) < 5e-4    # refined centroid on-source
    wl, flux, var = spec['green']
    assert np.all(np.isfinite(flux))
    assert np.nanstd(flux) / np.nanmean(flux) < 0.02    # flat template -> flat extracted shape
    assert np.nanmedian(flux / np.sqrt(var)) > 15       # optimal total-flux S/N beats one fibre


def test_profile_fit_converges_at_flux_calibrated_scale():
    # FLAM-scale data (~1e-14) must still be FIT, not silently return the initial guess. The LSQ
    # fitter meets its convergence tolerance immediately on tiny values and leaves sigma at its 0.6"
    # start -> a spurious constant 1.41" FWHM (and a wrong extraction profile). Regression for the
    # rescale-to-O(1) fix.
    sr = _point_source_super(C=3e-14, fwhm=1.8)
    r = fit_source_profile(sr, RA0, DEC0)
    assert r is not None
    _g, fit = r
    assert fit.fitted
    assert abs(fit.fwhm - 1.8) < 0.4                     # recovers the input width...
    assert abs(fit.fwhm - 2.3548 * 0.6) > 0.15          # ...and is NOT the un-converged 0.6" init


def test_estimate_psf_fwhm_recovers_seeing():
    sr = _point_source_super(fwhm=1.5)
    fw = estimate_psf_fwhm(sr, RA0, DEC0, radius_arcsec=3.0)
    assert 1.0 < fw < 2.2                                   # ~1.5" (moment estimate, aperture-biased)


def _abs_super(F0=100.0, sigma=1.0, area=0.44, nexp=1, nw=5):
    """A Gaussian point source of known TOTAL flux F0: fibre flux = F0 * (normalised PSF) * area."""
    offs = [(dx, dy) for dx in range(-5, 6) for dy in range(-5, 6)]
    dx = np.array([o[0] * 0.75 for o in offs])
    dy = np.array([o[1] * 0.75 for o in offs])
    psf = np.exp(-0.5 * (dx ** 2 + dy ** 2) / sigma ** 2) / (2 * np.pi * sigma ** 2)   # /arcsec^2
    fib_flux = F0 * psf * area                            # flux in each fibre
    flux1 = np.repeat(fib_flux[:, None], nw, axis=1)      # flat spectrum at level F0 (total)
    cosd = np.cos(np.deg2rad(DEC0))
    ra = RA0 + dx / cosd / 3600.0
    dec = DEC0 + dy / 3600.0
    wave = np.tile(np.linspace(5000, 5000 + nw - 1, nw), (len(offs), 1))
    exps, s = [], dict(ra=[], dec=[], wave=[], flux=[], var=[], mask=[], solid=[], exp=[])
    for e in range(nexp):
        s['ra'].append(ra); s['dec'].append(dec); s['wave'].append(wave)
        s['flux'].append(flux1.copy()); s['var'].append(np.ones_like(flux1))
        s['mask'].append(np.zeros_like(flux1, bool)); s['solid'].append(np.full(len(offs), area))
        s['exp'].append(np.full(len(offs), e))
        exps.append(ExposureMeta(f'e{e}', f'p{e}', 1.0, 1.0, float(e)))
    st = ChannelStack('green', np.concatenate(s['ra']), np.concatenate(s['dec']),
                      np.concatenate(s['wave']), np.concatenate(s['flux']), np.concatenate(s['var']),
                      np.concatenate(s['mask']), np.concatenate(s['solid']), np.concatenate(s['exp']))
    return SuperRSS('T', 'skysub', 'counts', exps, {'green': st})


def test_optimal_spectrum_recovers_absolute_total_flux():
    # F_hat must equal the source total flux AND be invariant to exposure count (the N-count bug fix)
    for nexp in (1, 4):
        sr = _abs_super(F0=100.0, sigma=1.0, nexp=nexp)
        spec, fit = optimal_spectrum(sr, RA0, DEC0, dar=False)
        f = spec['green'][1]
        assert np.isclose(np.nanmedian(f), 100.0, rtol=0.1), (nexp, np.nanmedian(f))


def _dar_super(nw=40, C=10.0, fwhm=1.5, shift_per_A=0.03):
    """Point source whose centroid WALKS in x with wavelength (differential refraction)."""
    sigma = fwhm / 2.35482
    offs = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)]
    dx = np.array([o[0] * 0.75 for o in offs])
    dy = np.array([o[1] * 0.75 for o in offs])
    wl = 5000.0 + np.arange(nw)
    cx = shift_per_A * (wl - wl.mean())                          # centroid track (arcsec)
    flux = C * np.exp(-0.5 * ((dx[:, None] - cx[None, :]) ** 2 + dy[:, None] ** 2) / sigma ** 2)
    cosd = np.cos(np.deg2rad(DEC0))
    ra = RA0 + dx / cosd / 3600.0
    dec = DEC0 + dy / 3600.0
    wave = np.tile(wl, (len(offs), 1))
    exps, s = [], dict(ra=[], dec=[], wave=[], flux=[], var=[], mask=[], solid=[], exp=[])
    for e in range(2):
        s['ra'].append(ra); s['dec'].append(dec); s['wave'].append(wave)
        s['flux'].append(flux.copy()); s['var'].append(np.ones_like(flux))
        s['mask'].append(np.zeros_like(flux, bool)); s['solid'].append(np.full(len(offs), 0.44))
        s['exp'].append(np.full(len(offs), e))
        exps.append(ExposureMeta(f'e{e}', f'p{e}', 1.0, 1.0, float(e)))
    st = ChannelStack('green', np.concatenate(s['ra']), np.concatenate(s['dec']),
                      np.concatenate(s['wave']), np.concatenate(s['flux']), np.concatenate(s['var']),
                      np.concatenate(s['mask']), np.concatenate(s['solid']), np.concatenate(s['exp']))
    return SuperRSS('T', 'skysub', 'counts', exps, {'green': st})


def test_dar_tracks_centroid_and_flattens_extraction():
    sr = _dar_super(shift_per_A=0.03)                            # ~1.2" walk over 40 A
    track = measure_dar(sr, RA0, DEC0)
    assert track is not None
    assert track.shift(5000.0, 5039.0) > 0.5                     # detects the wavelength walk
    dar, fit = optimal_spectrum(sr, RA0, DEC0, dar=True)
    nod, _ = optimal_spectrum(sr, RA0, DEC0, dar=False)
    fd = dar['green'][1]
    fn = nod['green'][1]
    # DAR tracking keeps the extracted flux flat; fixed-centre loses flux where the source walked
    assert np.nanstd(fd) / np.nanmean(fd) < np.nanstd(fn) / np.nanmean(fn)
    assert np.nanmean(fd) > np.nanmean(fn)                       # recovers more total flux
    assert fit.dar_shift > 0.5


def _line_cube(nw=41, line_k=20, ny=6, nx=6):
    wave = 5000.0 + np.arange(nw)
    data = np.full((nw, ny, nx), 2.0)                      # flat continuum everywhere
    data[line_k, 3, 3] += 30.0                             # an emission line on one spaxel
    var = np.ones((nw, ny, nx))
    w = WCS(naxis=3)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'WAVE']
    w.wcs.crval = [150.0, 20.0, 5000.0]; w.wcs.crpix = [3, 3, 1]
    w.wcs.cdelt = [-0.5 / 3600, 0.5 / 3600, 1.0]; w.wcs.cunit = ['deg', 'deg', 'Angstrom']
    return CoaddCube(data, var, wave, np.full((ny, nx), 3, int), np.full((ny, nx), 2, int),
                     w, 'erg/s/cm2/Angstrom/arcsec2', {'FIELD': 'T', 'PIXSCALE': 0.5}), wave[line_k]


def test_narrowband_recovers_line_and_subtracts_continuum():
    cube, line = _line_cube()
    nb = narrowband_image(cube, line, half_width=1.5)      # 3-pixel line window
    assert nb.bunit == 'erg/s/cm2/arcsec2'                 # integrated over lambda -> SB
    # off-line spaxel: pure continuum -> subtracts to ~0
    assert abs(nb.data[0, 0]) < 1e-9
    # source spaxel: the line integral (30 * dwave=1) stands above ~0 continuum
    assert np.isclose(nb.data[3, 3], 30.0, atol=1.0)


if __name__ == '__main__':
    import sys
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f'PASS {name}')
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f'FAIL {name}: {type(e).__name__}: {e}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
