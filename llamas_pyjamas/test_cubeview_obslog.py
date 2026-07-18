"""Tests for the obslog file pickers (scan_rss_exposures + ObslogDialog).

The scan is pure and does the real work -- grouping the three colour planes of an exposure into
one row and reading the header summary -- so most tests hit it directly with tiny on-disk FITS.
The dialog is also constructed in both modes to catch import/wiring regressions and to prove the
selection -> result mapping (representative file for open; all planes flattened for apply). Qt
runs under the offscreen platform.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_cubeview_obslog`).
"""

import os
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from astropy.io import fits
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from llamas_pyjamas.CubeViewer.cubeViewObslog import ObslogDialog, scan_rss_exposures

_app = QApplication.instance() or QApplication([])


def _write_rss(directory, base, colour, obj='TARGET', exptime=120.0, notes='a note'):
    hdr = fits.Header()
    hdr['OBJECT'] = obj
    hdr['SEXPTIME'] = exptime
    if notes is not None:
        hdr['OBS-CMNT'] = notes
    path = os.path.join(directory, f'{base}_RSS_{colour}.fits')
    fits.HDUList([fits.PrimaryHDU(np.zeros((2, 2)), hdr)]).writeto(path, overwrite=True)
    return path


def _populate(directory):
    """Two exposures (3 planes each) + noise files that must be ignored."""
    for colour in ('blue', 'green', 'red'):
        _write_rss(directory, 'EXP_A', colour, obj='Feige110', exptime=60.0, notes='on centre')
        _write_rss(directory, 'EXP_B', colour, obj='J1613', exptime=2200.0, notes='')
    # decoys: a white-light image and a non-RSS fits, neither should appear
    fits.PrimaryHDU(np.zeros((2, 2))).writeto(os.path.join(directory, 'EXP_A_whitelight.fits'))
    fits.PrimaryHDU(np.zeros((2, 2))).writeto(os.path.join(directory, 'random.fits'))


def test_scan_groups_planes_into_one_row_per_exposure():
    with tempfile.TemporaryDirectory() as d:
        _populate(d)
        rows = scan_rss_exposures(d)
        assert len(rows) == 2, 'three colour planes of an exposure collapse to one row'
        by_obj = {r['object']: r for r in rows}
        assert set(by_obj) == {'Feige110', 'J1613'}
        a = by_obj['Feige110']
        assert a['channels'] == ['blue', 'green', 'red']
        assert len(a['paths']) == 3
        assert a['representative'].endswith('_RSS_green.fits'), 'green is preferred to open'
        assert a['exptime'] == 60.0 and a['notes'] == 'on centre'


def test_scan_skips_whitelight_and_non_rss():
    with tempfile.TemporaryDirectory() as d:
        _populate(d)
        files = [os.path.basename(p) for r in scan_rss_exposures(d) for p in r['paths']]
        assert not any('white' in f for f in files)
        assert 'random.fits' not in files


def test_scan_missing_dir_is_empty():
    assert scan_rss_exposures('/no/such/dir') == []
    assert scan_rss_exposures('') == []


def test_scan_records_missing_notes_as_blank():
    with tempfile.TemporaryDirectory() as d:
        _write_rss(d, 'EXP_C', 'green', obj='NoNotes', notes=None)
        row = scan_rss_exposures(d)[0]
        assert row['object'] == 'NoNotes' and row['notes'] == ''


def test_open_dialog_returns_representative():
    with tempfile.TemporaryDirectory() as d:
        _populate(d)
        dialog = ObslogDialog(d, title='Open')
        assert dialog.table.rowCount() == 2
        assert dialog.accept_button.text() == 'Open'
        # nothing selected -> accept disabled
        assert not dialog.accept_button.isEnabled()
        dialog.table.selectRow(0)
        assert dialog.accept_button.isEnabled()
        dialog._accept()
        assert dialog.chosen_path.endswith('_RSS_green.fits')


def test_apply_dialog_needs_sensfunc_and_flattens_planes():
    with tempfile.TemporaryDirectory() as d:
        _populate(d)
        dialog = ObslogDialog(d, multi=True, with_sensfunc=True, title='Apply')
        assert dialog.accept_button.text() == 'Apply'
        dialog.table.selectAll()
        # rows selected but no sensfunc yet -> still disabled
        assert not dialog.accept_button.isEnabled()
        dialog.sensfunc_path = os.path.join(d, 'sens.fits')
        dialog._update_accept()
        assert dialog.accept_button.isEnabled()
        dialog._accept()
        # both exposures, all three planes each
        assert len(dialog.chosen_files) == 6
        assert all(p.endswith('.fits') for p in dialog.chosen_files)


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
