"""Tests for the co-add engine (Combine/coadd.py).

Synthetic FibreTables with known values/variances, checked against the closed-form kernel-weighted
mean: single-fibre value preserved, inverse-variance combine of co-located fibres, coverage/nexp
counting, min-coverage masking, flux vs surface-brightness units, and the output WCS round-trip.
The full pipeline (build -> collapse -> co-add -> FITS) is validated on real may26 data.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_combine_coadd`).
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from llamas_pyjamas.Combine.superRSS import FibreTable
from llamas_pyjamas.Combine.coadd import coadd_image, make_output_grid

RA0, DEC0 = 200.0, 10.0


def _ft(ra, dec, value, var, area, exp):
    n = len(ra)
    return FibreTable(ra=np.array(ra, float), dec=np.array(dec, float),
                      value=np.array(value, float), var=np.array(var, float),
                      solid_angle=np.full(n, float(area)), exposure=np.array(exp, int),
                      channel=np.array(['green'] * n), npix=np.ones(n, int))


def _pixval(img, ra, dec):
    px, py = img.wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    return img, int(round(float(py))), int(round(float(px)))


def test_single_fibre_value_and_coverage():
    # one isolated fibre; its pixel holds value/area (SB), coverage 1, nexp 1.
    ft = _ft([RA0], [DEC0], [10.0], [4.0], area=2.0, exp=[0])
    img = coadd_image(ft, pixscale=1.0, kernel='tophat', kernel_fwhm=1.0, units='sb')
    _, iy, ix = _pixval(img, RA0, DEC0)
    assert np.isclose(img.data[iy, ix], 10.0 / 2.0)        # SB = value/area
    assert np.isclose(img.var[iy, ix], 4.0 / 2.0 ** 2)     # var_sb, single sample
    assert img.coverage[iy, ix] == 1 and img.nexp[iy, ix] == 1


def test_inverse_variance_combine_of_colocated_fibres():
    # two fibres at the SAME position, ivar-weighted -> classic combine
    ft = _ft([RA0, RA0], [DEC0, DEC0], [10.0, 20.0], [4.0, 4.0], area=1.0, exp=[0, 1])
    img = coadd_image(ft, pixscale=1.0, kernel='tophat', kernel_fwhm=1.0, units='sb',
                      weighting='ivar')
    _, iy, ix = _pixval(img, RA0, DEC0)
    assert np.isclose(img.data[iy, ix], 15.0)              # equal ivar -> mean
    assert np.isclose(img.var[iy, ix], 2.0)               # 1/(1/4+1/4)
    assert img.coverage[iy, ix] == 2 and img.nexp[iy, ix] == 2


def test_uniform_vs_ivar_weighting_differ():
    # unequal variances: uniform mean != ivar mean (ivar favours the low-variance fibre)
    ft = _ft([RA0, RA0], [DEC0, DEC0], [10.0, 20.0], [1.0, 100.0], area=1.0, exp=[0, 1])
    ivar = coadd_image(ft, pixscale=1.0, kernel='tophat', kernel_fwhm=1.0, weighting='ivar')
    unif = coadd_image(ft, pixscale=1.0, kernel='tophat', kernel_fwhm=1.0, weighting='uniform')
    _, iy, ix = _pixval(ivar, RA0, DEC0)
    assert ivar.data[iy, ix] < 11.0                        # pulled toward the var=1 fibre (10)
    assert np.isclose(unif.data[iy, ix], 15.0)             # plain mean


def test_min_coverage_masks_thin_pixels():
    ft = _ft([RA0], [DEC0], [10.0], [4.0], area=1.0, exp=[0])
    img = coadd_image(ft, pixscale=1.0, kernel='tophat', kernel_fwhm=1.0, min_coverage=2)
    assert not np.isfinite(img.data).any()                 # lone fibre -> everything masked


def test_flux_units_use_value_directly():
    ft = _ft([RA0], [DEC0], [10.0], [4.0], area=2.0, exp=[0])
    img = coadd_image(ft, pixscale=1.0, kernel='tophat', kernel_fwhm=1.0, units='flux')
    _, iy, ix = _pixval(img, RA0, DEC0)
    assert np.isclose(img.data[iy, ix], 10.0)              # flux value, not divided by area


def test_output_grid_wcs_roundtrip():
    rng = np.random.default_rng(0)
    ra = RA0 + rng.uniform(-0.005, 0.005, 50)
    dec = DEC0 + rng.uniform(-0.005, 0.005, 50)
    wcs, ny, nx, px, py = make_output_grid(ra, dec, pixscale=0.5, pad_pix=3)
    assert (px > 0).all() and (px < nx).all() and (py > 0).all() and (py < ny).all()
    back = wcs.pixel_to_world(px, py)
    assert np.allclose(back.ra.deg, ra, atol=1e-6) and np.allclose(back.dec.deg, dec, atol=1e-6)


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
