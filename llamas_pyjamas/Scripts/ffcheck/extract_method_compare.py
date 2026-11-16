"""Compare Horne and boxcar extraction camera by camera on one reduced night (read-only).

Usage:
    python extract_method_compare.py <night_config> <output_dir> [--workers 4] [--out FILE]

For each colour, re-extracts in memory, with the night's own traces (``<output_dir>/traces``)
and bias-corrected frames (``<output_dir>/bias_corrected``), the configured lamp flat,
twilight and arc, and the first science frame, once with each method. Cosmic-ray cleaning is
off, so only the method differs. Nothing is written into the night folder.

Per camera and frame it prints:

- ``ratio``: median over fibres of the per-fibre Horne / boxcar flux (median over the central
  80% of columns), and ``scat``: the robust scatter of that per-fibre ratio, %;
- ``out``: fibres more than 5 robust sigma from the camera's median ratio;
- ``bad H/B``: samples that are non-finite or negative, Horne / boxcar (bright frames only);
- ``blend r``: Spearman correlation between the per-fibre ratio and the neighbours' brightness
  relative to the fibre's own, ``(B[i-1] + B[i+1]) / 2B[i]`` — a profile model that fails
  where fibres blend shows up here;
- ``t H/B``: extraction time per method, s.

A camera clearly worse than the other cameras of its colour (scatter, outliers or blending
correlation) is flagged with ``<--``.
"""

import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import spearmanr

from llamas_pyjamas.File.llamasIO import process_fits_by_color
import llamas_pyjamas.GUI.guiExtract as ge
from llamas_pyjamas.Bias import camera_of_header

COLOURS = ('red', 'green', 'blue')
FRAMES = (('lamp flat', '{c}_flat_file', 'lamp_flat'),
          ('twilight', '{c}_twilight_flat', 'twilight'),
          ('arc', '{c}_arc_file', 'arc'))


def read_config(path):
    cfg = {}
    with open(path) as fh:
        for line in fh:
            line = line.split('#', 1)[0].strip()
            if '=' in line:
                k, v = line.split('=', 1)
                cfg[k.strip()] = v.strip()
    return cfg


def bias_corrected(output_dir, raw):
    stem = os.path.splitext(os.path.basename(raw))[0]
    hits = glob.glob(os.path.join(output_dir, 'bias_corrected', f'{stem}_bias_corrected.fits'))
    return hits[0] if hits else None


def _per_fibre(counts):
    """Per-fibre median over the central 80% of columns."""
    nx = counts.shape[1]
    sl = slice(int(0.1 * nx), int(0.9 * nx))
    with np.errstate(all='ignore'):
        return np.nanmedian(counts[:, sl], axis=1)


def compare_camera(args):
    data, header, trace_file, idx, recipe, half = args
    os.environ['LLAMAS_BOXCAR_HALFWIDTH'] = str(half)
    out = {}
    counts = {}
    for method in ('horne', 'boxcar'):
        os.environ['LLAMAS_EXTRACT_METHOD'] = method
        t0 = time.perf_counter()
        res = ge.process_trace._function(data, header.copy(), trace_file, idx, method=method,
                                         remove_cosmic_rays=False, recipe=recipe)
        out[f't_{method}'] = time.perf_counter() - t0
        ext = res['extraction']
        if ext is None:
            return None
        counts[method] = np.asarray(ext.counts, dtype=float)
    h, b = counts['horne'], counts['boxcar']
    fh, fb = _per_fibre(h), _per_fibre(b)
    good = np.isfinite(fh) & np.isfinite(fb) & (fb > 0)
    ratio = np.full(fh.shape, np.nan)
    ratio[good] = fh[good] / fb[good]
    r = ratio[good]
    med = float(np.median(r)) if r.size else np.nan
    mad = 1.4826 * float(np.median(np.abs(r - med))) if r.size else np.nan
    out['ratio'] = med
    out['scat'] = 100.0 * mad / med if r.size and med else np.nan
    out['out'] = int(np.sum(np.abs(r - med) > 5 * mad)) if mad > 0 else 0
    out['nfib'] = int(good.sum())
    bright = np.nanmedian(fb) > 50
    out['bad_h'] = int(np.sum(~np.isfinite(h) | (h < 0))) if bright else -1
    out['bad_b'] = int(np.sum(~np.isfinite(b) | (b < 0))) if bright else -1
    nb = np.full(fb.shape, np.nan)
    nb[1:-1] = (fb[:-2] + fb[2:]) / (2 * fb[1:-1])
    ok = good & np.isfinite(nb)
    out['blend'] = float(spearmanr(ratio[ok], nb[ok]).correlation) if ok.sum() > 10 else np.nan
    return out


def jobs_for(path, colours, recipe, trace_dir, half):
    hdu, _ = process_fits_by_color(path)
    jobs = []
    for idx in range(1, len(hdu)):
        bench, side, colour = camera_of_header(hdu[idx].header)
        if colour not in colours or hdu[idx].data is None:
            continue
        trace_file = ge.get_trace_file(colour, bench, side, trace_dir)
        jobs.append(((colour, f'{bench}{side}'),
                     (np.asarray(hdu[idx].data, dtype=np.float32), hdu[idx].header, trace_file,
                      idx, recipe, half)))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('config')
    ap.add_argument('output_dir')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default=None, help='also write the table to this file')
    a = ap.parse_args()
    cfg = read_config(a.config)
    half = float(cfg.get('boxcar_halfwidth', 2.5))
    trace_dir = os.path.join(a.output_dir, 'traces')

    frames = []                                     # (label, path, colours, recipe)
    for label, key, recipe in FRAMES:
        by_file = {}
        for c in COLOURS:
            raw = cfg.get(key.format(c=c))
            if raw:
                by_file.setdefault(raw, []).append(c)
        for raw, cols in by_file.items():
            frames.append((label, bias_corrected(a.output_dir, raw), cols, recipe))
    sci = [s.strip() for s in cfg.get('science_files', '').split(',') if s.strip()]
    if sci:
        flat = glob.glob(os.path.join(
            a.output_dir, os.path.splitext(os.path.basename(sci[0]))[0]
            + '_bias_corrected_flat_corrected.fits'))
        frames.append(('science', flat[0] if flat else bias_corrected(a.output_dir, sci[0]),
                       list(COLOURS), 'science'))

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for label, path, cols, recipe in frames:
            if not path:
                print(f'{label}: no bias-corrected frame found; skipped')
                continue
            jobs = jobs_for(path, cols, recipe, trace_dir, half)
            for (cam, _), res in zip(jobs, pool.map(compare_camera, [j for _, j in jobs])):
                if res is not None:
                    rows.append((label, os.path.basename(path), cam, res))

    lines = [f"Horne vs boxcar (half-width {half} px), cosmic-ray cleaning off",
             f"{'frame':10s} {'cam':9s} {'ratio':>6s} {'scat%':>6s} {'out':>4s} "
             f"{'bad H/B':>13s} {'blend r':>8s} {'t H/B s':>11s}"]
    for label in [f[0] for f in FRAMES] + ['science']:
        sub = [r for r in rows if r[0] == label]
        if not sub:
            continue
        for colour in COLOURS:
            cs = [r for r in sub if r[2][0] == colour]
            if not cs:
                continue
            scat = np.array([r[3]['scat'] for r in cs])
            blend = np.array([abs(r[3]['blend']) for r in cs])
            outs = np.array([r[3]['out'] for r in cs])
            for _, _, cam, res in sorted(cs, key=lambda r: r[2][1]):
                flag = ''
                if (res['scat'] > 2 * np.nanmedian(scat) and res['scat'] > 1.0) \
                        or (abs(res['blend']) > 0.3 and abs(res['blend']) > 2 * np.nanmedian(blend)) \
                        or res['out'] > max(5, 3 * np.median(outs)):
                    flag = '  <--'
                bad = (f"{res['bad_h']}/{res['bad_b']}" if res['bad_h'] >= 0 else '-')
                lines.append(f"{label:10s} {cam[0]:5s} {cam[1]:3s} {res['ratio']:6.3f} "
                             f"{res['scat']:6.2f} {res['out']:4d} {bad:>13s} "
                             f"{res['blend']:8.2f} {res['t_horne']:5.1f}/{res['t_boxcar']:<5.1f}{flag}")
    text = '\n'.join(lines)
    print(text)
    if a.out:
        with open(a.out, 'w') as fh:
            fh.write(text + '\n')


if __name__ == '__main__':
    main()
