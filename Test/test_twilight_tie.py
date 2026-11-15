"""Unit tests for llamas_pyjamas.Flat.twilightTie.

Covers the two-term bench tie L_j = b_b · S(x,y):
- the smooth surface recovers a low-order (x,y) gradient,
- an injected per-bench-side STEP is removed (boundary step -> 0): the test a
  polynomial-only model fails, guarding the explicit b_b offset term,
- the QA gate rejects a low-S/N twilight,
- the archive save/load round-trips (newest epoch wins).

Run with:  pytest Test/test_twilight_tie.py
"""

import os
import sys
import tempfile

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llamas_pyjamas.Flat.twilightTie import (
    _fit_2d_surface, fit_bench_tie, twilight_qa_gate,
    save_archived_tie, load_latest_tie, compute_twilight_tie,
    combine_twilight_frames,
)
from llamas_pyjamas.Utils.recipes import stamp_recipe
from llamas_pyjamas.Flat.lampThroughput import compute_lamp_throughput

BENCHSIDES = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B']
CHANNEL = 'red'
N_FIB = 30
N_WAVE = 120
_WAVE = np.linspace(6000.0, 7000.0, N_WAVE)


def _synthetic_ifu(n_per_bs=60):
    """Build (benchsides, x, y) for 8 contiguous y-stripe bench-sides."""
    rng = np.random.default_rng(1)
    bs, x, y = [], [], []
    for b, name in enumerate(BENCHSIDES):
        for _ in range(n_per_bs):
            bs.append(name)
            x.append(rng.uniform(0.5, 46.0))
            y.append(b * 5.2 + rng.uniform(0.0, 5.0))
    return np.array(bs), np.array(x), np.array(y)


def test_fit_2d_surface_recovers_gradient():
    bs, x, y = _synthetic_ifu()
    truth = 1.0 + 0.01 * x - 0.005 * y
    model, coeffs = _fit_2d_surface(x, y, truth, order=2)
    assert coeffs is not None
    assert np.max(np.abs(model - truth)) < 1e-3, "surface failed to recover gradient"


def test_bench_step_removed():
    """Inject a per-bench-side multiplicative STEP; the tie must zero the boundary step."""
    bs, x, y = _synthetic_ifu()
    offsets = dict(zip(BENCHSIDES, [1.3, 0.7, 1.1, 0.9, 1.2, 0.8, 1.05, 0.95]))
    smooth_illum = 1.0 + 0.002 * x              # mild smooth gradient
    observed = np.array([offsets[b] for b in bs]) * smooth_illum

    L, b, rms = fit_bench_tie(observed, bs, x, y, order=2)
    corrected = observed / L

    before = max(np.median(observed[bs == a]) for a in BENCHSIDES) \
        - min(np.median(observed[bs == a]) for a in BENCHSIDES)
    meds = [np.median(corrected[bs == a]) for a in BENCHSIDES]
    after = max(meds) - min(meds)
    assert before > 0.5, "test setup: expected a large pre-correction bench step"
    # Two-term tie collapses the bench step to well below the 3% QA threshold
    # (residual ~1% reflects per-bench sampling of the smooth gradient, not the
    # bench offset, which b_b removes exactly).
    assert after < 0.02, f"bench step not removed: {after:.2e} (before {before:.3f})"
    assert after < 0.05 * before, f"insufficient reduction: {after:.3f} vs {before:.3f}"


def test_qa_gate_rejects_low_sn():
    class _Ext:
        def __init__(self, counts):
            self.counts = counts

    def make(snr):
        rng = np.random.default_rng(0)
        exts, meta = [], []
        for name in BENCHSIDES:
            level = 5000.0
            noise = level / snr
            counts = level + rng.normal(0, noise, (20, 300))
            exts.append(_Ext(counts))
            meta.append({'channel': 'red', 'bench': name[0], 'side': name[1]})
        return {'extractions': exts, 'metadata': meta, 'n_frames': 3}

    good = twilight_qa_gate(make(snr=100.0))
    bad = twilight_qa_gate(make(snr=2.0))
    assert good['passed'].get('red') is True, f"high-S/N twilight rejected: {good['reason']}"
    assert bad['passed'].get('red') is False, f"low-S/N twilight passed: {bad['reason']}"


def test_archive_roundtrip():
    tmp = tempfile.mkdtemp()
    # Write two dummy tie FITS files with different epochs
    paths = {}
    for epoch in ('2026-06-01', '2026-06-03'):
        p = os.path.join(tmp, f'tie_{epoch}.fits')
        hdu = fits.PrimaryHDU()
        hdu.header['EPOCH'] = epoch
        fits.HDUList([hdu]).writeto(p, overwrite=True)
        paths[epoch] = p

    assert load_latest_tie('green', archive_root=tmp) is None  # empty for this channel
    save_archived_tie(paths['2026-06-01'], 'red', '2026-06-01', archive_root=tmp)
    save_archived_tie(paths['2026-06-03'], 'red', '2026-06-03', archive_root=tmp)

    latest = load_latest_tie('red', archive_root=tmp)
    assert latest is not None
    with fits.open(latest) as hdul:
        assert hdul[0].header['EPOCH'] == '2026-06-03', "newest epoch not selected"


class _FakeExt:
    """Minimal ExtractLlamas stand-in: per-fibre counts + wave on a shared grid."""
    def __init__(self, counts, wave):
        self.counts = counts
        self.wave = wave
        self.dead_fibers = []


def _write_flat_smooth_models(path):
    """Uniform-throughput smooth models (so lamp C≈1, isolating the twilight tie)."""
    base = 1000.0 + 200.0 * np.sin(np.linspace(0, np.pi, N_WAVE))
    hdul = fits.HDUList([fits.PrimaryHDU()])
    stamp_recipe(hdul[0].header, 'lamp_flat')
    for bs in BENCHSIDES:
        smooth = np.tile(base, (N_FIB, 1))
        cols = [
            fits.Column(name='FIBER_ID', format='J', array=np.arange(N_FIB, dtype=np.int32)),
            fits.Column(name='SMOOTH', format=f'{N_WAVE}D', array=smooth),
            fits.Column(name='WAVE', format=f'{N_WAVE}D', array=np.tile(_WAVE, (N_FIB, 1))),
        ]
        tbl = fits.BinTableHDU.from_columns(cols)
        tbl.header['EXTNAME'] = f'{CHANNEL}_{bs[0]}_{bs[1]}'
        tbl.header['CHANNEL'] = CHANNEL
        tbl.header['BENCH'] = bs[0]
        tbl.header['SIDE'] = bs[1]
        hdul.append(tbl)
    hdul.writeto(path, overwrite=True)


def test_compute_twilight_tie_end_to_end():
    """Full path: build twilight extractions with a bench step, recover it as L."""
    tmp = tempfile.mkdtemp()
    sm = os.path.join(tmp, 'flat_smooth_models.fits')
    _write_flat_smooth_models(sm)
    lamp = compute_lamp_throughput(sm, tmp)        # C ≈ 1 everywhere

    offsets = dict(zip(BENCHSIDES, [1.3, 0.7, 1.1, 0.9, 1.2, 0.8, 1.05, 0.95]))
    base = 1000.0 + 200.0 * np.sin(np.linspace(0, np.pi, N_WAVE))
    exts, meta = [], []
    for bs in BENCHSIDES:
        counts = np.tile(base * offsets[bs], (N_FIB, 1))
        exts.append(_FakeExt(counts, np.tile(_WAVE, (N_FIB, 1))))
        meta.append({'channel': CHANNEL, 'bench': bs[0], 'side': bs[1]})
    twi = {'extractions': exts, 'metadata': meta, 'n_frames': 3}

    tie = compute_twilight_tie(twi, lamp, sm, tmp, n_wave_bins=8)

    with fits.open(tie) as hdul:
        ext_tbls = [e for e in hdul[1:]
                    if not e.header.get('EXTNAME', '').endswith('_REF')]
        assert len(ext_tbls) == 8, f"expected 8 bench-sides, got {len(ext_tbls)}"
        per_bs = {}
        for e in ext_tbls:
            bs = f"{e.header['BENCH']}{e.header['SIDE']}"
            per_bs[bs] = float(np.nanmedian(e.data['CORRECTION']))
        # L per bench-side should track the injected offsets (up to a global scale)
        names = list(per_bs.keys())
        L = np.array([per_bs[b] for b in names])
        truth = np.array([offsets[b] for b in names])
        L /= np.median(L)
        truth /= np.median(truth)
        assert np.corrcoef(L, truth)[0, 1] > 0.95, (
            f"recovered tie does not track injected bench offsets: "
            f"corr={np.corrcoef(L, truth)[0, 1]:.3f}")


def test_combine_twilight_frames_nonmutating():
    """Combining cached frames must NOT mutate the inputs (reused across channels)."""
    def frame(level):
        exts = [_FakeExt(np.full((4, 10), level), np.tile(np.arange(10), (4, 1)))]
        meta = [{'channel': 'red', 'bench': '1', 'side': 'A'}]
        return {'extractions': exts, 'metadata': meta}

    f1, f2 = frame(100.0), frame(200.0)
    orig1 = f1['extractions'][0].counts.copy()
    combined = combine_twilight_frames([f1, f2])
    # median of 100 and 200 = 150
    assert np.allclose(combined['extractions'][0].counts, 150.0)
    assert combined['n_frames'] == 2
    # input frame's counts untouched (so the cache can be reused for another channel)
    assert np.allclose(f1['extractions'][0].counts, orig1), "combine mutated its input!"
    # single-frame combine returns the frame as-is
    one = combine_twilight_frames([frame(50.0)])
    assert np.allclose(one['extractions'][0].counts, 50.0) and one['n_frames'] == 1


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All twilight-tie tests passed.")
