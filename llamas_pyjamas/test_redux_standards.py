"""Tests for flux-standard wiring in the reduction setup GUI.

These cover the non-Qt seams: the header matcher used during a directory scan, and the config
round-trip (generate -> parse) that carries the flux_standard_files selection between sessions.
The crossmatch itself is tested in test_flux_standards; here we check it is wired in correctly.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_redux_standards`).
"""

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from llamas_pyjamas.Utils.reduxSetupGUI import (
    _match_standard,
    generate_config,
    parse_config,
)

TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'llamas_pyjamas', 'example_config.txt')


def _write_and_parse(assignments, tmp_path):
    text = generate_config(assignments, output_dir='/tmp/out')
    cfg_path = os.path.join(tmp_path, 'cfg.txt')
    with open(cfg_path, 'w') as fh:
        fh.write(text)
    config, _bad = parse_config(cfg_path)
    return config


def test_match_standard_on_science_pointing():
    # GD108 pointing (may26), tagged as a science exposure in the header.
    header = {'OBS TYPE': 'SCIENCE', 'RA': 150.19716666, 'DEC': -7.55888888}
    result = _match_standard(header)
    assert result is not None
    name, sep = result
    assert name == 'GD108' and sep < 5.0


def test_match_standard_none_for_quasar():
    header = {'OBS TYPE': 'SCIENCE', 'RA': 243.259, 'DEC': 8.135}
    assert _match_standard(header) is None


def test_generate_config_emits_flux_standard_files(tmp_path):
    config = _write_and_parse(
        {'science': ['/data/sci1.fits'], 'flux_standard': ['/data/gd108.fits']}, str(tmp_path))
    assert config.get('flux_standard_files') == '/data/gd108.fits'
    assert config.get('science_files') == '/data/sci1.fits'


def test_flux_standard_files_omitted_when_none(tmp_path):
    # With no standards assigned, the key stays commented out (pipeline default: none).
    text = generate_config({'science': ['/data/sci1.fits']}, output_dir='/tmp/out')
    active = [l for l in text.splitlines()
              if l.strip().startswith('flux_standard_files') and not l.startswith('#')]
    assert not active


def test_config_roundtrip_multiple_standards(tmp_path):
    stds = ['/data/gd108_a.fits', '/data/gd108_b.fits', '/data/feige110.fits']
    config = _write_and_parse({'science': ['/data/s.fits'], 'flux_standard': stds}, str(tmp_path))
    parsed = [p.strip() for p in config['flux_standard_files'].split(',')]
    assert parsed == stds


if __name__ == '__main__':
    import sys
    import tempfile
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            if 'tmp_path' in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as td:
                    fn(td)
            else:
                fn()
            print(f'PASS {name}')
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f'FAIL {name}: {type(e).__name__}: {e}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
