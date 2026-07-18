"""Unit tests for the pipeline output-consolidation switches.

Two lean-output behaviours, pinned without running the pipeline:
* save_extraction_pkl (default off) -> the large extraction pkls are removed once the RSS is
  written (cleanup_extraction_pkls).
* keep_intermediate_rss (default off) -> the extract/_FF/_FF_SKYSUB RSS files are collapsed to
  one per (frame, channel), the most-corrected renamed to a clean name (consolidate_rss_files).

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_reduce_pkl_cleanup`).
"""

import os
import tempfile

from llamas_pyjamas.reduce import cleanup_extraction_pkls, consolidate_rss_files


_BASE = 'LLAMAS_2026-05-17_02-49-56.7_SCI22_mef_bias_corrected_flat_corrected_extract'


def _touch(path, text='x'):
    with open(path, 'w') as fh:
        fh.write(text)


def test_consolidate_collapses_stages_keeping_most_corrected():
    with tempfile.TemporaryDirectory() as d:
        for color in ('blue', 'green', 'red'):
            _touch(os.path.join(d, f'{_BASE}_RSS_{color}.fits'), 'extract')
            _touch(os.path.join(d, f'{_BASE}_RSS_{color}_FF.fits'), 'ff')
            _touch(os.path.join(d, f'{_BASE}_RSS_{color}_FF_SKYSUB.fits'), 'skysub')
        consolidate_rss_files(d, keep_intermediate=False)
        survivors = sorted(f for f in os.listdir(d) if f.endswith('.fits'))
        assert survivors == [f'{_BASE}_RSS_{c}.fits' for c in ('blue', 'green', 'red')]
        # the survivor must carry the most-corrected (_FF_SKYSUB) content, under the clean name
        with open(os.path.join(d, f'{_BASE}_RSS_green.fits')) as fh:
            assert fh.read() == 'skysub'


def test_consolidate_keeps_ff_when_no_skysub():
    with tempfile.TemporaryDirectory() as d:
        _touch(os.path.join(d, f'{_BASE}_RSS_green.fits'), 'extract')
        _touch(os.path.join(d, f'{_BASE}_RSS_green_FF.fits'), 'ff')
        consolidate_rss_files(d, keep_intermediate=False)
        survivors = [f for f in os.listdir(d) if f.endswith('.fits')]
        assert survivors == [f'{_BASE}_RSS_green.fits']
        with open(os.path.join(d, survivors[0])) as fh:
            assert fh.read() == 'ff'


def test_consolidate_noop_for_single_stage():
    with tempfile.TemporaryDirectory() as d:
        _touch(os.path.join(d, f'{_BASE}_RSS_green.fits'), 'extract')
        consolidate_rss_files(d, keep_intermediate=False)
        assert os.listdir(d) == [f'{_BASE}_RSS_green.fits']


def test_consolidate_leaves_subdirs_alone():
    with tempfile.TemporaryDirectory() as d:
        _touch(os.path.join(d, f'{_BASE}_RSS_green.fits'))
        _touch(os.path.join(d, f'{_BASE}_RSS_green_FF.fits'))
        os.makedirs(os.path.join(d, 'flat'))
        _touch(os.path.join(d, 'flat', 'twi_RSS_green_FF.fits'))
        consolidate_rss_files(d, keep_intermediate=False)
        assert os.path.exists(os.path.join(d, 'flat', 'twi_RSS_green_FF.fits'))


def test_consolidate_keep_intermediate_is_noop():
    with tempfile.TemporaryDirectory() as d:
        for suf in ('', '_FF', '_FF_SKYSUB'):
            _touch(os.path.join(d, f'{_BASE}_RSS_green{suf}.fits'))
        assert consolidate_rss_files(d, keep_intermediate=True) == []
        assert len([f for f in os.listdir(d) if f.endswith('.fits')]) == 3


def _make(paths):
    for p in paths:
        with open(p, 'w') as fh:
            fh.write('x')


def test_removes_pkls_by_default():
    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, 'f_extract.pkl')
        sky = os.path.join(d, 'f_corrected_sky1d_extractions.pkl')
        _make([raw, sky])
        removed = cleanup_extraction_pkls([raw, sky], save_pkl=False)
        assert set(removed) == {raw, sky}
        assert not os.path.exists(raw) and not os.path.exists(sky)


def test_keeps_pkls_when_save_set():
    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, 'f_extract.pkl')
        sky = os.path.join(d, 'f_corrected_sky1d_extractions.pkl')
        _make([raw, sky])
        removed = cleanup_extraction_pkls([raw, sky], save_pkl=True)
        assert removed == []
        assert os.path.exists(raw) and os.path.exists(sky)


def test_robust_to_none_missing_and_duplicates():
    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, 'f_extract.pkl')
        _make([raw])
        # None, a duplicate, and a non-existent path must not raise; only the real file goes
        removed = cleanup_extraction_pkls([None, raw, raw, os.path.join(d, 'nope.pkl')],
                                          save_pkl=False)
        assert removed == [raw]
        assert not os.path.exists(raw)


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
