"""Acceptance numbers for the fibre flat of one reduced night (read-only).

Usage:
    python fibreflat_acceptance.py <output_dir>        # e.g. <night>/reduced_fffix

Prints, per colour and benchside, the measurements the 2026-11-14 fibre-flat fixes were
checked against:

- sky scatter within each benchside: raw, RSS throughput only, after the flat
  (FFSB/FFST/FFSA, median over frames);
- fibres not flat-fielded (FFNSKIP) and the fibre-flat version (FFVER);
- whether the corrections' FIBER_IDs are the RSS's (physical ids, every benchside);
- narrow features deeper than 5% in C_i (same method as the 9,398 found in green before
  the fix: C_i over its 25-px running median, central 80% of each fibre's range);
- WINRMS / WIDIPTW from the corrections, and the pixel flat's clip-floor share (FCLIPLO).
"""

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
from astropy.io import fits
from scipy.ndimage import label, median_filter

BENCHSIDES = [f"{b}{s}" for b in '1234' for s in 'AB']
COLOURS = ('red', 'green', 'blue')


def narrow_features(corr, wave, depth=0.05, window=25):
    """Count of contiguous runs deeper or higher than ``depth`` in C_i / running median."""
    n = 0
    for c, w in zip(np.asarray(corr, dtype=float), np.asarray(wave, dtype=float)):
        ok = np.isfinite(c) & (c > 0) & np.isfinite(w) & (w > 0)
        if ok.sum() < 3 * window:
            continue
        lo, hi = w[ok].min(), w[ok].max()
        band = ok & (w > lo + 0.1 * (hi - lo)) & (w < hi - 0.1 * (hi - lo))
        r = np.full(c.size, np.nan)
        r[ok] = c[ok] / median_filter(c[ok], window, mode='nearest') - 1
        n += label(band & (np.abs(np.nan_to_num(r)) > depth))[1]
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('output_dir')
    args = parser.parse_args()
    ext_dir = os.path.join(args.output_dir, 'extractions')
    flat_dir = os.path.join(ext_dir, 'flat')

    corr = {}
    with fits.open(os.path.join(flat_dir, 'fibre_flat_corrections.fits')) as h:
        for e in h[1:]:
            if e.name.endswith('_REF'):
                continue
            ch, b, s = e.name.lower().split('_')
            corr[(ch, b + s.upper())] = (e.header, np.asarray(e.data['FIBER_ID']),
                                         narrow_features(e.data['CORRECTION'], e.data['WAVE']))
    clip = {}
    with fits.open(os.path.join(flat_dir, 'pixel_maps.fits')) as h:
        for e in h[1:]:
            key = (str(e.header.get('CHANNEL', '')).lower(), f"{e.header['BENCH']}{e.header['SIDE']}")
            clip[key] = e.header.get('FCLIPLO', np.nan)

    scat = defaultdict(lambda: defaultdict(list))
    skips, ffver, id_ok = defaultdict(list), set(), defaultdict(lambda: True)
    for path in sorted(glob.glob(os.path.join(ext_dir, '*_RSS_*.fits'))):
        with fits.open(path) as h:
            p = h[0].header
            # skip anything that is not a fibre-flatted RSS (e.g. a *_RSS_*_sensfunc.fits,
            # whose header records FIBRFLAT of its standard)
            if not p.get('FIBRFLAT') or 'FIBERMAP' not in h:
                continue
            ch = str(p.get('CHANNEL', '')).lower()
            ffver.add(p.get('FFVER', 1))
            skips[ch].append(int(p.get('FFNSKIP', 0)))
            fm = h['FIBERMAP'].data
            bs_col = np.char.strip(np.asarray(fm['BENCHSIDE']).astype(str))
            for bs in BENCHSIDES:
                for q in 'BTA':
                    v = p.get(f'FFS{q}{bs}')
                    if v is not None:
                        scat[(ch, bs)][q].append(v)
                ids = set(np.asarray(fm['FIBER_ID'])[bs_col == bs].tolist())
                if (ch, bs) in corr and ids != set(corr[(ch, bs)][1].tolist()):
                    id_ok[(ch, bs)] = False

    print(f"FFVER of the _FF products: {sorted(ffver)}; "
          f"FFNSKIP per channel: {dict((c, max(v)) for c, v in skips.items())}")
    print(f"{'':6}{'bs':4}{'raw%':>7}{'tp%':>7}{'after%':>8}{'ids':>5}{'C_i>5%':>8}"
          f"{'WINRMS':>8}{'WIDIPTW':>8}{'clip%':>7}")
    for ch in COLOURS:
        total = 0
        for bs in BENCHSIDES:
            s = scat.get((ch, bs), {})
            med = {q: (np.median(s[q]) if s.get(q) else np.nan) for q in 'BTA'}
            hdr, _, nfeat = corr.get((ch, bs), ({}, None, 0))
            total += nfeat
            print(f"{ch:6}{bs:4}{med['B']:7.2f}{med['T']:7.2f}{med['A']:8.2f}"
                  f"{'ok' if id_ok[(ch, bs)] else 'BAD':>5}{nfeat:8d}"
                  f"{hdr.get('WINRMS', np.nan):8.2f}{hdr.get('WIDIPTW', np.nan):8.2f}"
                  f"{clip.get((ch, bs), np.nan):7.3f}")
        print(f"{ch:6}total narrow C_i features > 5%: {total}")


if __name__ == '__main__':
    main()
