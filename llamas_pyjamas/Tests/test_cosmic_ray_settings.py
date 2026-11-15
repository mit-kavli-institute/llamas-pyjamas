"""Tests for the per-channel cosmic-ray settings (config, extraction stamps, resume, GUI).

Runnable with pytest or as a plain script
(`python -m llamas_pyjamas.Tests.test_cosmic_ray_settings`).
"""

import os
import subprocess
import sys
import types

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cloudpickle
import numpy as np
import pytest
from astropy.io import fits
from PyQt6 import QtWidgets

import llamas_pyjamas.GUI.guiExtract as ge
import llamas_pyjamas.Masking.cosmicLlamas as cosmic
from llamas_pyjamas.Bias.biasFirstResume import stale_reason
from llamas_pyjamas.constants import LACOSMIC_COLOR_OVERRIDES, LACOSMIC_DEFAULTS
from llamas_pyjamas.Utils import reduxSetupGUI as gui

TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'example_config.txt')


def settings(**config):
    s, errors, warnings = cosmic.cosmic_ray_settings(config)
    assert not errors, errors
    return s


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def test_defaults_match_the_constants_and_the_legacy_id():
    d = cosmic.default_settings()
    assert d['maxiter'] == LACOSMIC_DEFAULTS['maxiter']
    for c in ('green', 'blue'):
        assert d[c] == {'enabled': True, 'cr_threshold': LACOSMIC_DEFAULTS['cr_threshold'],
                        'contrast': LACOSMIC_DEFAULTS['contrast'],
                        'neighbor_threshold': LACOSMIC_DEFAULTS['neighbor_threshold']}
    red = LACOSMIC_COLOR_OVERRIDES['red']
    assert d['red']['enabled'] is False and d['red']['cr_threshold'] == red['cr_threshold']
    # LEGACY_CRID is the old defaults with flagged pixels replaced; masking them as missing
    # data is a different identity, so resumed runs redo those frames.
    assert cosmic.cosmic_ray_id(d, handling='replace') == cosmic.LEGACY_CRID
    assert cosmic.cosmic_ray_id(d) != cosmic.LEGACY_CRID
    assert cosmic.CR_HANDLING == 'mask'
    assert settings() == d


@pytest.mark.parametrize('value, on', [
    (['red', 'green', 'blue'], {'red', 'green', 'blue'}),
    ('red', {'red'}),
    ('none', set()),
    ('Blue, green,', {'blue', 'green'}),            # quoted/trailing forms from the parser
])
def test_channel_forms(value, on):
    s = settings(cosmic_ray_channels=value)
    assert {c for c in cosmic.CHANNELS if s[c]['enabled']} == on


def test_thresholds_are_coerced_from_strings():
    s = settings(cosmic_ray_channels=['red', 'blue'], cosmic_ray_threshold_red='8',
                 cosmic_ray_contrast_red='7.5', cosmic_ray_neighbor_threshold_blue='0.5',
                 cosmic_ray_maxiter='3')
    assert s['red']['cr_threshold'] == 8.0 and s['red']['contrast'] == 7.5
    assert s['blue']['neighbor_threshold'] == 0.5 and s['maxiter'] == 3


@pytest.mark.parametrize('config, text', [
    ({'cosmic_ray_threshhold_red': '5'}, "unknown cosmic-ray setting"),
    ({'cosmic_ray_channels': 'red, purple'}, 'unknown channel'),
    ({'cosmic_ray_channels': ['none', 'red']}, "'none' cannot be combined"),
    ({'cosmic_ray_threshold_blue': '-1'}, 'must be positive'),
    ({'cosmic_ray_contrast_green': 'high'}, 'must be a number'),
    ({'cosmic_ray_maxiter': '2.5'}, 'whole number'),
    ({'cosmic_ray_maxiter': '0'}, 'whole number'),
    ({'remove_cosmic_rays': 'no'}, 'true or false'),
])
def test_errors(config, text):
    _s, errors, _w = cosmic.cosmic_ray_settings(config)
    assert any(text in e for e in errors), errors


def test_warnings():
    _s, errors, warnings = cosmic.cosmic_ray_settings(
        {'cosmic_ray_threshold_red': '12', 'cosmic_ray_neighbor_threshold_blue': '9'})
    assert not errors
    assert any('red is not in cosmic_ray_channels' in w for w in warnings)
    assert any('blue: the neighbour threshold' in w for w in warnings)


def test_id_follows_only_what_is_applied():
    base = cosmic.cosmic_ray_id(settings())
    assert cosmic.cosmic_ray_id(settings(cosmic_ray_threshold_red='12')) == base   # red off
    assert cosmic.cosmic_ray_id(settings(cosmic_ray_threshold_blue='7')) != base
    assert cosmic.cosmic_ray_id(settings(cosmic_ray_maxiter='3')) != base
    assert cosmic.cosmic_ray_id(settings(cosmic_ray_channels='red, green, blue')) != base
    assert cosmic.cosmic_ray_id(settings(cosmic_ray_channels='none')) == 'off'
    assert cosmic.cosmic_ray_id(settings(), master_on=False) == 'off'


def test_id_is_the_same_in_another_process():
    code = ('import llamas_pyjamas.Masking.cosmicLlamas as c;'
            'print(c.cosmic_ray_id(c.cosmic_ray_settings({"cosmic_ray_channels": "red"})[0]))')
    other = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                           check=True).stdout.strip().splitlines()[-1]
    assert other == cosmic.cosmic_ray_id(settings(cosmic_ray_channels='red'))


def test_stamp_and_copy():
    h, g = fits.Header(), fits.Header()
    cosmic.stamp_cosmic_rays(h, settings(cosmic_ray_channels='red, blue'))
    assert h['CRCHAN'] == 'red,blue' and h['CRID'] != cosmic.LEGACY_CRID
    assert h['CRMODE'] == 'mask'
    cosmic.copy_cosmic_ray_stamps(h, g)
    assert g['CRID'] == h['CRID'] and g['CRCHAN'] == 'red,blue' and g['CRMODE'] == 'mask'
    cosmic.copy_cosmic_ray_stamps(None, g)                     # older pickles: no header
    cosmic.stamp_cosmic_rays(h, settings(), master_on=False)
    assert h['CRID'] == 'off' and h['CRCHAN'] == 'none' and h['CRMODE'] == 'off'


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

@pytest.fixture
def captured(monkeypatch):
    seen = []

    def fake(data, **kw):
        seen.append(kw)
        return data, np.zeros(data.shape, bool)
    monkeypatch.setattr(cosmic, 'remove_cosmics', fake)
    return seen


def test_clean_uses_each_channels_settings(captured):
    s = settings(cosmic_ray_channels='red, blue', cosmic_ray_threshold_red='8',
                 cosmic_ray_contrast_red='7', cosmic_ray_maxiter='3')
    data = np.zeros((10, 10))
    cosmic.clean_cosmic_rays(data, color='red', settings=s)
    assert (captured[-1]['cr_threshold'], captured[-1]['contrast'],
            captured[-1]['neighbor_threshold'], captured[-1]['maxiter']) == (8.0, 7.0, 0.7, 3)
    cosmic.clean_cosmic_rays(data, color='blue', settings=s)
    assert captured[-1]['cr_threshold'] == 6.5
    n = len(captured)
    cleaned, mask = cosmic.clean_cosmic_rays(data, color='green', settings=s)   # off
    assert len(captured) == n and not mask.any()


def test_clean_without_settings_behaves_as_before(captured):
    data = np.zeros((10, 10))
    cosmic.clean_cosmic_rays(data, color='red')
    assert captured == []                                            # red skipped
    cosmic.clean_cosmic_rays(data, color='green')
    assert captured[-1]['cr_threshold'] == LACOSMIC_DEFAULTS['cr_threshold']
    assert captured[-1]['contrast'] == LACOSMIC_DEFAULTS['contrast']


def _trace_file(tmp_path):
    ny, nx = 60, 40
    trace = types.SimpleNamespace(bench='1', side='A', channel='red', fitsfile='x.fits',
                                  nfibers=1, naxis1=nx, naxis2=ny,
                                  traces=np.full((1, nx), 30.0),
                                  fiberimg=-np.ones((ny, nx), int))
    path = tmp_path / 'LLAMAS_red_1_A_traces.pkl'
    with open(path, 'wb') as f:
        cloudpickle.dump(trace, f)
    return str(path), (ny, nx)


def _header(color='red'):
    h = fits.Header()
    h['COLOR'], h['BENCH'], h['SIDE'] = color.upper(), 1, 'A'
    h['BIASSUB'] = True                      # bias-first already applied
    h['EDGEDC'], h['EDGESRC'] = 0.0, 'stripes'
    return h


def test_process_trace_stamps(tmp_path, captured, monkeypatch):
    monkeypatch.setenv('LLAMAS_EXTRACT_METHOD', 'boxcar')
    path, shape = _trace_file(tmp_path)
    run = ge.process_trace._function
    on = settings(cosmic_ray_channels='red', cosmic_ray_threshold_red='8')
    out = run(np.zeros(shape), _header(), path, 1, method='boxcar', cosmic_rays=on, recipe='science')
    h = out['extraction'].hdr
    assert h['CRCLEAN'] is True and h['CRTHRESH'] == 8.0 and h['CRCONTR'] == 9.0
    out = run(np.zeros(shape), _header(), path, 1, method='boxcar', cosmic_rays=settings(),
              recipe='science')
    h = out['extraction'].hdr
    assert h['CRCLEAN'] is False and h['CRWHY'] == 'channel off' and out['cr_mask'] is None
    out = run(np.zeros(shape), _header(), path, 1, method='boxcar', remove_cosmic_rays=False,
              recipe='science')
    assert out['extraction'].hdr['CRWHY'] == 'remove_cosmic_rays false'


def test_process_trace_extracts_cosmic_ray_pixels_as_missing(tmp_path, monkeypatch):
    """L.A.Cosmic's replacement value never reaches the counts: the flagged sample is NaN,
    flagged MASK_CR_MISSING, and counted in CRNMISS."""
    from llamas_pyjamas.File.llamasRSS import MASK_CR_MISSING
    monkeypatch.setenv('LLAMAS_EXTRACT_METHOD', 'boxcar')
    path, shape = _trace_file(tmp_path)

    def fake(data, **kw):                    # one cosmic ray in the aperture, at column 5
        mask = np.zeros(data.shape, bool)
        mask[30, 5] = True
        cleaned = data.copy()
        cleaned[30, 5] = -999.0              # a replacement value that must not be used
        return cleaned, mask
    monkeypatch.setattr(cosmic, 'remove_cosmics', fake)
    out = ge.process_trace._function(np.full(shape, 10.0), _header(), path, 1, method='boxcar',
                                     cosmic_rays=settings(cosmic_ray_channels='red'),
                                     recipe='science')
    ext = out['extraction']
    assert np.isnan(ext.counts[0, 5]) and ext.dq[0, 5] & MASK_CR_MISSING
    assert np.all(np.isfinite(ext.counts[0, 6:]))
    assert ext.hdr['CRNMISS'] == 1 and ext.hdr['CRNPART'] == 0


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

BIAS = {'BFVER': 2, 'BIASID': 'b1', 'DARKID': 'none', 'DARKSCL': 0.0}


def _write(path, **cards):
    h = fits.Header()
    for k, v in cards.items():
        h[k] = v
    fits.PrimaryHDU(np.zeros(2), header=h).writeto(path, overwrite=True)
    return str(path)


def test_resume_compares_crid_on_the_rss_only(tmp_path):
    bc = _write(tmp_path / 'X_SCI24_mef_bias_corrected.fits', **BIAS)
    rss_new = _write(tmp_path / 'X_RSS_red.fits', **BIAS, CRID='aaa')
    rss_old = _write(tmp_path / 'Y_RSS_red.fits', **BIAS)                       # pre-CRID
    red_on = cosmic.cosmic_ray_id(settings(cosmic_ray_channels='red, green, blue'))
    assert stale_reason(dict(BIAS, CRID='aaa'), [bc, rss_new]) == ''
    assert stale_reason(dict(BIAS, CRID='bbb'), [bc, rss_new]) == 'cosmic-ray settings changed'
    assert stale_reason(dict(BIAS, CRID=cosmic.LEGACY_CRID), [bc, rss_old]) == ''
    assert stale_reason(dict(BIAS, CRID=red_on), [bc, rss_old]) == 'cosmic-ray settings changed'
    assert stale_reason(dict(BIAS, CRID='bbb'), [bc]) == ''          # no RSS yet: nothing to check
    # frames cleaned before cosmic-ray pixels became missing data are redone
    defaults = settings()
    replaced = _write(tmp_path / 'Z_RSS_red.fits', **BIAS,
                      CRID=cosmic.cosmic_ray_id(defaults, handling='replace'))
    now = dict(BIAS, CRID=cosmic.cosmic_ray_id(defaults))
    assert stale_reason(now, [bc, replaced]) == 'cosmic-ray settings changed'
    assert stale_reason(now, [bc, rss_old]) == 'cosmic-ray settings changed'
    # frames without cleaning are not
    off = _write(tmp_path / 'W_RSS_red.fits', **BIAS, CRID='off')
    assert stale_reason(dict(BIAS, CRID=cosmic.cosmic_ray_id(defaults, master_on=False)),
                        [bc, off]) == ''
    assert stale_reason(dict(BIAS), [bc, rss_new]) == ''             # calibration: no CRID expected


def test_resume_prefers_the_bias_reason(tmp_path):
    rss = _write(tmp_path / 'X_RSS_red.fits', **BIAS, CRID='aaa')
    old_bc = _write(tmp_path / 'X_SCI24_mef_bias_corrected.fits', BFVER=1)
    assert stale_reason(dict(BIAS, CRID='bbb'), [rss, old_bc]) == 'made by the older bias step'


# ---------------------------------------------------------------------------
# Setup GUI
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def qt():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


def test_gui_catalog_matches_the_pipeline_defaults():
    entries = {e[0]: e for e in dict(gui.CONFIG_CATALOG)['Bias / Cosmic rays']}
    d = cosmic.default_settings()
    assert entries['cosmic_ray_channels'][2] == 'green,blue'
    for c in cosmic.CHANNELS:
        assert float(entries[f'cosmic_ray_threshold_{c}'][2]) == d[c]['cr_threshold']
        assert float(entries[f'cosmic_ray_contrast_{c}'][2]) == d[c]['contrast']
    assert set(cosmic.CONFIG_KEYS) <= set(entries)
    keys = [e[0] for e in dict(gui.CONFIG_CATALOG)['Bias / Cosmic rays']]
    assert keys.index('cosmic_ray_channels') == keys.index('remove_cosmic_rays') + 1


def test_gui_no_channel_writes_none_and_survives_reopening(qt):
    dlg = gui.ConfigOptionsDialog(current={'cosmic_ray_channels': 'blue, green'})
    assert dlg.changed_values() == {}                      # seeded spelling isn't a change
    _kind, w, _c = dlg._widgets['cosmic_ray_channels']
    for cb in w.findChildren(QtWidgets.QCheckBox):
        cb.setChecked(False)
    assert dlg.changed_values() == {'cosmic_ray_channels': 'none'}
    again = gui.ConfigOptionsDialog(current={'cosmic_ray_channels': 'none'})
    assert again.values()['cosmic_ray_channels'] == 'none' and again.changed_values() == {}
    assert dlg._widgets['cosmic_ray_threshold_red'][1].minimum() == pytest.approx(0.1)


def test_gui_writes_the_settings_in_place(tmp_path):
    overrides = {'cosmic_ray_channels': 'red,green,blue', 'cosmic_ray_threshold_red': '8'}
    text = gui.generate_config({'science': ['/x_mef.fits']}, output_dir='/tmp/o',
                               template_path=TEMPLATE, overrides=overrides)
    assert 'cosmic_ray_channels = red,green,blue\n' in text
    assert 'PROCESSING OPTIONS' not in text.split('cosmic_ray_channels')[0]   # not appended
    cfg = tmp_path / 'cfg.txt'
    cfg.write_text(text)
    parsed, _bad = gui.parse_config(str(cfg))
    s = settings(**{k: v for k, v in parsed.items() if k.startswith('cosmic_ray_')})
    assert s['red']['enabled'] and s['red']['cr_threshold'] == 8.0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
