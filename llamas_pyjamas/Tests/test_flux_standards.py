"""Unit tests for the spectrophotometric standards catalogue and crossmatch.

The crossmatch is what tells a standard-star exposure from a science target when the header
cannot, so these pin the behaviour that matters: the two may26 standards match and the three
quasars do not, coordinates round-trip through the header keys, and the radius actually gates.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.Tests.test_flux_standards`).
"""

import numpy as np

from llamas_pyjamas.Flux.fluxStandards import (
    Standard,
    StandardsCatalog,
    load_catalog,
    ra_dec_from_header,
)

# Coordinates as bundled (index.dat), used to build a small in-memory catalogue for the unit
# tests that must not depend on the on-disk bundle.
GD108 = Standard('GD108', 150.197208, -7.558667, 13.56, 'sdB', None)
FEIGE110 = Standard('Feige110', 349.993292, -5.165500, 11.82, 'DOp', None)
_MINI = StandardsCatalog([GD108, FEIGE110])


def test_match_hits_standard_within_radius():
    # GD108 pointing from the may26 header.
    m = _MINI.match(150.19716666, -7.55888888)
    assert m is not None and m.name == 'GD108'
    assert m.separation_arcsec < 5.0


def test_match_misses_when_far():
    # A quasar field ~degrees away from any standard.
    assert _MINI.match(243.259, 8.135) is None


def test_radius_gates():
    # 100 arcsec off GD108: outside the default 30, inside a widened 120.
    off_dec = -7.558667 + 100 / 3600.0
    assert _MINI.match(150.197208, off_dec) is None
    near = _MINI.match(150.197208, off_dec, radius_arcsec=120.0)
    assert near is not None and near.name == 'GD108'


def test_nearest_of_two_is_returned():
    assert _MINI.match(349.99, -5.17).name == 'Feige110'


def test_ra_dec_from_header_decimal():
    assert ra_dec_from_header({'RA': 150.2, 'DEC': -7.5}) == (150.2, -7.5)


def test_ra_dec_from_header_missing_or_bad():
    assert ra_dec_from_header({}) == (None, None)
    assert ra_dec_from_header({'RA': 'None', 'DEC': 'None'}) == (None, None)
    assert ra_dec_from_header({'RA': 1.0, 'DEC': None}) == (None, None)


def test_match_returns_none_on_missing_coords():
    assert _MINI.match(None, None) is None
    assert _MINI.match(np.nan, np.nan) is None


def test_bundled_catalog_covers_may26_standards():
    # Integration with the committed bundle: both standards present, with reference spectra.
    catalog = load_catalog()
    assert len(catalog) >= 60

    gd = catalog.match(150.19716666, -7.55888888)
    feige = catalog.match(349.9933, -5.1655)
    assert gd is not None and gd.name == 'GD108'
    assert feige is not None and feige.name == 'Feige110'
    # Phase II needs the reference spectrum; both may26 standards must carry one.
    assert gd.standard.has_spectrum, 'GD108 must have a bundled flux file'
    assert feige.standard.has_spectrum, 'Feige110 must have a bundled flux file'


def test_bundled_catalog_rejects_quasar_fields():
    # The three may26 science targets must not match any standard.
    catalog = load_catalog()
    for ra, dec in [(148.0, 13.79),    # J0958+1347 approx
                    (243.26, 8.135),   # J1613+0808 approx
                    (327.9, 2.59)]:    # J2151+0235 approx
        assert catalog.match(ra, dec) is None, f'{ra},{dec} should not match a standard'


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
