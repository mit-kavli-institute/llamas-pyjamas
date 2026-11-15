"""Unit tests for llamas_pyjamas.Flat.lampThroughput.

Validates the lamp high-frequency throughput C_j(λ):
- a single channel-wide reference spans all 8 bench-sides,
- the smooth/bench-level (low-frequency) component is removed,
- the sharp per-fibre scatter is preserved,
- CORRECTION_VAR is populated (non-zero) and scales ~1/counts.

Run with:  pytest Test/test_lamp_throughput.py
Requires the full llamas_pyjamas environment (pypeit etc.).
"""

import os
import sys
import tempfile

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llamas_pyjamas.Utils.recipes import stamp_recipe
from llamas_pyjamas.Flat.lampThroughput import compute_lamp_throughput

CHANNEL = 'red'
BENCHSIDES = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B']
N_FIB = 30
N_WAVE = 200


def _write_smooth_models(path, tilt_amp=0.10, bright_fiber=('2A', 5),
                         spike_fiber=('3B', 10), seed=0):
    """Synthetic flat_smooth_models.fits with a known HF/LF structure.

    smooth_j(λ) = base(λ) · tilt_b · throughput_j, where:
      - base(λ) is a common smooth continuum (cancels in the ratio to R),
      - tilt_b is a smooth LOW-frequency bench-level gradient (must NOT enter C),
      - throughput_j is the per-fibre HIGH-frequency factor (must enter C);
        one fibre carries a sharp +30% spike.
    """
    rng = np.random.default_rng(seed)
    wave = np.linspace(6000.0, 7000.0, N_WAVE)
    base = 1000.0 + 200.0 * np.sin(np.linspace(0, np.pi, N_WAVE))

    primary = fits.PrimaryHDU()
    stamp_recipe(primary.header, 'lamp_flat')
    hdul = fits.HDUList([primary])
    for b, bs in enumerate(BENCHSIDES):
        bench, side = bs[0], bs[1]
        tilt = 1.0 + tilt_amp * (b - 3.5) / 3.5          # smooth across bench-sides
        fiber_ids = np.arange(N_FIB, dtype=np.int32)
        smooth = np.zeros((N_FIB, N_WAVE))
        wave_arr = np.tile(wave, (N_FIB, 1))
        for i in range(N_FIB):
            throughput = 1.0 + 0.02 * rng.standard_normal()   # mild per-fibre HF
            if (bs, i) == spike_fiber:
                throughput = 1.30                              # sharp spike
            bright = 10.0 if (bs, i) == bright_fiber else 1.0  # high-S/N fibre
            smooth[i] = base * tilt * throughput * bright
        cols = [
            fits.Column(name='FIBER_ID', format='J', array=fiber_ids),
            fits.Column(name='SMOOTH', format=f'{N_WAVE}D', array=smooth),
            fits.Column(name='WAVE', format=f'{N_WAVE}D', array=wave_arr),
        ]
        tbl = fits.BinTableHDU.from_columns(cols)
        tbl.header['EXTNAME'] = f'{CHANNEL}_{bench}_{side}'
        tbl.header['CHANNEL'] = CHANNEL
        tbl.header['BENCH'] = bench
        tbl.header['SIDE'] = side
        hdul.append(tbl)
    hdul.writeto(path, overwrite=True)


def _load_corr(path):
    out = {}
    with fits.open(path) as hdul:
        for ext in hdul[1:]:
            if ext.header.get('EXTNAME', '').endswith('_REF') or ext.data is None:
                continue
            if 'CORRECTION' not in ext.data.names:
                continue
            out[ext.header['EXTNAME']] = {
                'fiber_ids': ext.data['FIBER_ID'].copy(),
                'correction': ext.data['CORRECTION'].copy(),
                'correction_var': ext.data['CORRECTION_VAR'].copy(),
            }
    return out


def _run():
    tmp = tempfile.mkdtemp()
    sm = os.path.join(tmp, 'flat_smooth_models.fits')
    _write_smooth_models(sm)
    out = compute_lamp_throughput(sm, tmp, smooth_scale_fibres=15)
    return _load_corr(out)


def test_all_eight_benchsides_present():
    corr = _run()
    benchsides = {f"{c['correction'].shape[0]}" for c in corr.values()}  # noqa
    assert len(corr) == 8, f"expected 8 bench-sides, got {len(corr)}"


def test_correction_near_unity():
    corr = _run()
    allC = np.concatenate([c['correction'].ravel() for c in corr.values()])
    med = np.nanmedian(allC)
    assert abs(med - 1.0) < 0.05, f"median C = {med:.4f}, expected ~1.0"


def test_low_frequency_tilt_removed():
    """Per-bench-side mean of C must be ~equal: the bench-level tilt is NOT in C."""
    corr = _run()
    bs_means = [np.nanmedian(c['correction']) for c in corr.values()]
    spread = (max(bs_means) - min(bs_means))
    assert spread < 0.03, (
        f"bench-side spread in C = {spread:.4f}; the low-frequency tilt leaked "
        f"into the throughput")


def test_high_frequency_spike_preserved():
    """A fibre with an injected +30% sharp throughput shows up in C."""
    corr = _run()
    spike = corr['red_3_B']
    j = np.where(spike['fiber_ids'] == 10)[0][0]
    c_spike = np.nanmedian(spike['correction'][j])
    assert c_spike > 1.15, f"spike fibre C = {c_spike:.3f}, expected >1.15"


def test_correction_var_populated_and_scales():
    corr = _run()
    allVar = np.concatenate([c['correction_var'].ravel() for c in corr.values()])
    assert np.any(allVar > 0), "CORRECTION_VAR is all zero (placeholder not filled)"
    # RELATIVE variance Var[C]/C^2 = 1/counts_e, so the high-S/N (10x counts)
    # fibre must have a markedly lower relative variance than a normal fibre.
    bright = corr['red_2_A']
    jb = np.where(bright['fiber_ids'] == 5)[0][0]
    relvar = bright['correction_var'] / np.clip(bright['correction'] ** 2, 1e-12, None)
    rel_bright = np.nanmedian(relvar[jb])
    rel_normal = np.nanmedian(np.delete(relvar, jb, axis=0))
    assert rel_bright < 0.5 * rel_normal, (
        f"bright-fibre relative var {rel_bright:.2e} not << normal {rel_normal:.2e} "
        f"(relative variance should scale ~1/counts)")


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All lamp-throughput tests passed.")
