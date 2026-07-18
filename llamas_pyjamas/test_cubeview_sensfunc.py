"""Tests for the interactive sensfunc panel's headless model (SensFuncModel).

The dialog is a thin Qt wrapper; the logic worth testing lives in the model — region
composition, per-channel fitting, and that user edits actually change the result. Qt runs
under the offscreen platform because importing the module pulls in the matplotlib Qt backend.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_cubeview_sensfunc`).
"""

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np

from llamas_pyjamas.CubeViewer.cubeViewSensFunc import SensFuncModel
from llamas_pyjamas.Flux.sensFunc import SensFunc


def _model(**kw):
    wave = np.linspace(4600.0, 7000.0, 600)
    flux = np.full_like(wave, 400.0)
    ref_wave = np.linspace(4000.0, 7500.0, 400)
    ref_flux = np.full_like(ref_wave, 1e-15)
    return SensFuncModel(spectra={'green': (wave, flux)}, exptime=30.0,
                         ref_wave=ref_wave, ref_flux=ref_flux, standard_name='TEST', **kw)


def test_regions_compose_defaults_and_added():
    m = _model()
    n_default = len(m.regions())
    assert n_default > 5, 'defaults should include tellurics + Balmer'
    m.added_regions.append((5200.0, 5300.0))
    assert len(m.regions()) == n_default + 1
    m.use_default_masks = False
    assert m.regions() == [(5200.0, 5300.0)], 'defaults off leaves only the added region'


def test_fit_channel_ok_and_too_few():
    m = _model()
    result = m.fit_channel('green')
    assert result is not None
    wave, raw, fit, good = result
    assert good.sum() > 100 and np.isfinite(fit).any()

    # a channel with almost no valid points cannot be fit
    m.spectra['green'] = (np.linspace(4600, 4610, 5), np.full(5, 400.0))
    assert m.fit_channel('green') is None


def test_build_returns_sensfunc():
    sf = _model().build()
    assert isinstance(sf, SensFunc)
    assert 'green' in sf.channels
    assert sf.meta.get('standard') == 'TEST'


def test_added_mask_changes_the_fit():
    # Put a dip in the observed flux (=> a spike in S); masking that region must change the
    # fitted value there. Proves the interactive mask actually feeds the fit.
    wave = np.linspace(4600.0, 7000.0, 600)
    flux = np.full_like(wave, 400.0)
    dip = (wave >= 5500) & (wave <= 5600)
    flux[dip] *= 0.3
    ref_wave = np.linspace(4000.0, 7500.0, 400)
    ref_flux = np.full_like(ref_wave, 1e-15)
    # Weighting off so the low-count spike genuinely drags the unmasked fit — this test is
    # about the mask mechanism, not the (separately tested) S/N weighting that would otherwise
    # down-weight the spike.
    m = SensFuncModel(spectra={'green': (wave, flux)}, exptime=30.0,
                      ref_wave=ref_wave, ref_flux=ref_flux, standard_name='T',
                      use_default_masks=False, bkspace=150.0, weighted=False,
                      throughput_floor=0.0)

    unmasked = m.build().value(np.array([5550.0]), 'green')[0]
    m.added_regions.append((5480.0, 5620.0))
    masked = m.build().value(np.array([5550.0]), 'green')[0]
    # unmasked fit is pulled up by the S-spike; masking should lower it toward the continuum
    assert masked < unmasked, 'masking the spike must change (lower) the fitted S there'


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
