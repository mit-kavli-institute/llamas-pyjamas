"""Tests for the heliocentric/barycentric wavelength-frame correction (Utils/waveFrame.py).

The correction itself is pypeit's geomotion (deterministic given header + site), so these pin
the LLAMAS policy around it: which frames are corrected, the reversible header record, the
factor<->velocity relation, idempotency, and graceful no-op when inputs are missing. Exact
velocities are not asserted (they depend on the ephemeris) -- only that they are physical.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_wave_frame`).
"""

import math

from astropy.io import fits

from llamas_pyjamas.Utils.waveFrame import stamp_and_factor, velocity_and_factor

_C_KMS = 299792.458


def _sci_header(obstype='SCIENCE'):
    """A science-like primary header with everything the correction needs."""
    h = fits.Header()
    h['OBS TYPE'] = obstype
    h['MJD-OBS'] = 61177.0918          # 2026-05-17T02:12 UTC
    h['RA'] = 243.2592                 # J1613+0808
    h['DEC'] = 8.1349
    h['SEXPTIME'] = 2200.0
    return h


def test_disabled_frame_is_identity_no_stamp():
    for frame in ('none', 'None', '', None, 'off', 'observed'):
        h = _sci_header()
        assert stamp_and_factor(h, frame) == (0.0, 1.0)
        assert 'VELFRAME' not in h, f'{frame!r} must not stamp the header'


def test_calibration_frame_not_corrected():
    for obstype in ('COMP', 'FLAT', 'BIAS', 'ARC'):
        h = _sci_header(obstype=obstype)
        assert stamp_and_factor(h, 'heliocentric') == (0.0, 1.0)
        assert 'VELFRAME' not in h


def test_unknown_frame_is_noop():
    h = _sci_header()
    assert stamp_and_factor(h, 'galactic') == (0.0, 1.0)
    assert 'VELFRAME' not in h


def test_science_frame_corrected_stamped_and_reversible():
    h = _sci_header()
    vel, factor = stamp_and_factor(h, 'heliocentric')
    # physical: LCO helio velocities are well within +/-100 km/s, factor within ~3e-4 of 1
    assert abs(vel) < 100.0
    assert abs(factor - 1.0) < 3e-4 and factor > 0
    # factor is exactly the relativistic Doppler factor for that velocity
    expected = math.sqrt((1.0 + vel / _C_KMS) / (1.0 - vel / _C_KMS))
    assert abs(factor - expected) < 1e-9
    # the record is present and reversible
    assert h['VELFRAME'] == 'heliocentric'
    assert abs(h['HELIOVEL'] - round(vel, 4)) < 1e-6      # km/s, as requested
    assert abs(h['VELCORR'] - factor) < 1e-12


def test_idempotent_does_not_rescale():
    h = _sci_header()
    vel1, factor1 = stamp_and_factor(h, 'heliocentric')
    assert factor1 != 1.0
    vel2, factor2 = stamp_and_factor(h, 'heliocentric')
    assert factor2 == 1.0, 'second application must not scale WAVE again'
    assert abs(vel2 - round(vel1, 4)) < 1e-6, 'reports the already-applied velocity'


def test_missing_pointing_or_time_is_noop():
    # no pointing
    h = _sci_header()
    for k in ('RA', 'DEC'):
        del h[k]
    assert stamp_and_factor(h, 'heliocentric') == (0.0, 1.0)
    assert 'VELFRAME' not in h
    # no time
    h2 = _sci_header()
    del h2['MJD-OBS']
    assert velocity_and_factor(h2, 'heliocentric') is None


def test_tel_radec_fallback():
    # decimal RA/DEC absent -> sexagesimal HIERARCH TEL RA/TEL DEC are used
    h = _sci_header()
    del h['RA']
    del h['DEC']
    h['TEL RA'] = '16:13:00.0'
    h['TEL DEC'] = '+08:08:00'
    vel, factor = stamp_and_factor(h, 'heliocentric')
    assert factor != 1.0 and abs(vel) < 100.0


def test_barycentric_close_to_heliocentric():
    vh, _ = stamp_and_factor(_sci_header(), 'heliocentric')
    vb, _ = stamp_and_factor(_sci_header(), 'barycentric')
    assert abs(vh - vb) < 1.0, 'helio and bary differ by << 1 km/s'


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
