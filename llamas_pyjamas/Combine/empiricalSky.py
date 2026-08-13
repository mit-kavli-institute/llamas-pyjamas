"""
llamas_pyjamas.Combine.empiricalSky
===================================
Empirical multi-dither sky subtraction — a per-fibre nod-and-shuffle generalization.

Instead of modelling the sky with a b-spline, build the sky for each PHYSICAL fibre from the dithers
in which that fibre lands on BLANK sky, and subtract it. Because it is the same fibre in every dither,
the empirical sky carries that fibre's exact throughput / LSF / detector-column signature (a true
A-B match), so the CONTINUUM subtracts very cleanly. Sky LINES are NOT perfectly removed (per-frame
wavelength shifts + OH time-variability) — those residuals are accepted for now.

Why no WCS/registration: "same physical fibre across dithers" == same FIBER_ID (the fibre->detector
layout is fixed; the telescope dithers). The per-frame _FF RSS files of one field share an identical
FIBER_ID ordering, so their COUNTS planes stack directly by row. Blank-vs-source is decided per fibre
by self-reference (the dithers where that fibre's own continuum is faintest = on sky).

Variable sky brightness: the sky level changes between dithers, so the empirical sky is built as a
per-fibre SHAPE (normalized to a reference level) and applied to each target frame scaled by that
frame's OVERALL sky level S_t (emp_sky = shape x S_t). This is the overall per-frame scaling.

Domain: operates on the RSS COUNTS plane (pre-fibre-flat, pre-flux-cal), writing SKY (= empirical sky)
and SKYSUB (= COUNTS - empirical sky). First cut is NON-DESTRUCTIVE: writes *_EMPSKY.fits alongside the
input so the empirical result can be compared against the pipeline's b-spline SKYSUB.
"""
import logging
import os
import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

MIN_BLANK = 3          # min blank dithers to build an empirical sky for a fibre (else fall back)
BLANK_FRAC = 0.6       # fraction of dithers (faintest, per fibre) treated as blank sky
CHANNELS = ('red', 'green', 'blue')


def _fibre_continuum(counts, mask=None):
    """Per-fibre continuum level (nfib,) = median over wavelength, robust to OH lines and masks."""
    c = np.asarray(counts, float)
    if mask is not None:
        c = np.where(np.asarray(mask) != 0, np.nan, c)
    return np.nanmedian(c, axis=1)


def frame_sky_level(counts, mask=None):
    """Overall sky brightness of a frame: median over fibres of the per-fibre continuum. Robust to
    a source-minority IFU (assumes the field is mostly blank sky)."""
    b = _fibre_continuum(counts, mask)
    b = b[np.isfinite(b)]
    return float(np.nanmedian(b)) if b.size else np.nan


def build_empirical_sky(stack, masks, min_blank=MIN_BLANK, blank_frac=BLANK_FRAC):
    """Build the per-(dither,fibre) empirical sky from a stacked field.

    Parameters
    ----------
    stack : (ndith, nfib, nwave) COUNTS, aligned by FIBER_ID across dithers.
    masks : (ndith, nfib, nwave) or None.

    Returns
    -------
    emp_sky : (ndith, nfib, nwave) empirical sky (NaN rows where no empirical estimate — caller
              falls back to the b-spline SKY there).
    nblank  : (nfib,) number of blank dithers used per fibre.
    S       : (ndith,) per-frame overall sky level.
    """
    stack = np.asarray(stack, float)
    ndith, nfib, nwave = stack.shape
    S = np.array([frame_sky_level(stack[d], None if masks is None else masks[d]) for d in range(ndith)])
    Sref = np.nanmedian(S[np.isfinite(S)]) if np.isfinite(S).any() else 1.0
    S_safe = np.where(np.isfinite(S) & (S > 0), S, np.nan)

    # per-fibre continuum per dither -> rank to pick the faintest (blank) dithers
    b = np.full((ndith, nfib), np.nan)
    for d in range(ndith):
        b[d] = _fibre_continuum(stack[d], None if masks is None else masks[d])

    n_blank_target = max(int(min_blank), int(round(blank_frac * ndith)))
    n_blank_target = min(n_blank_target, ndith)

    emp_sky = np.full_like(stack, np.nan)
    nblank = np.zeros(nfib, dtype=int)
    for i in range(nfib):
        bi = b[:, i]
        order = np.argsort(np.where(np.isfinite(bi), bi, np.inf))   # faintest first
        blank = order[:n_blank_target]
        blank = [d for d in blank if np.isfinite(bi[d]) and np.isfinite(S_safe[d])]
        if len(blank) < min_blank:
            continue                                                # fallback (leave NaN)
        nblank[i] = len(blank)
        blank_set = set(blank)
        # normalized sky samples per blank dither (C scaled to the reference level)
        norm = {d: stack[d, i] * (Sref / S_safe[d]) for d in blank}
        full_shape = np.nanmedian(np.stack(list(norm.values())), axis=0)   # (nwave,) @ Sref
        for d in range(ndith):
            scale = (S_safe[d] / Sref) if np.isfinite(S_safe[d]) else 1.0
            # LEAVE-ONE-OUT: never build a frame's sky from itself (would self-subtract its noise).
            if d in blank_set and len(blank) > min_blank:
                loo = np.stack([norm[dd] for dd in blank if dd != d])
                shape = np.nanmedian(loo, axis=0)
            else:
                shape = full_shape                                  # on-source frames aren't in the set
            emp_sky[d, i] = shape * scale
    logger.info("empiricalSky: built for %d/%d fibres (>=%d blank dithers of %d)",
                int((nblank >= min_blank).sum()), nfib, min_blank, ndith)
    return emp_sky, nblank, S


def _load_field(rss_files):
    """Load a field's per-dither COUNTS/MASK aligned by FIBER_ID. Returns (files, stack, masks, fids)."""
    counts, masks, fids0 = [], [], None
    used = []
    for f in rss_files:
        with fits.open(f) as h:
            if 'COUNTS' not in h or 'FIBERMAP' not in h:
                logger.warning("empiricalSky: %s missing COUNTS/FIBERMAP; skipped", os.path.basename(f))
                continue
            fid = np.asarray(h['FIBERMAP'].data['FIBER_ID'])
            if fids0 is None:
                fids0 = fid
            elif not np.array_equal(fid, fids0):
                logger.warning("empiricalSky: %s FIBER_ID mismatch; skipped", os.path.basename(f))
                continue
            counts.append(np.asarray(h['COUNTS'].data, float))
            masks.append(np.asarray(h['MASK'].data) if 'MASK' in h else np.zeros_like(counts[-1]))
            used.append(f)
    if len(used) < 2:
        raise ValueError(f"empiricalSky: need >=2 aligned dithers, got {len(used)}")
    return used, np.stack(counts), np.stack(masks), fids0


def subtract_empirical_sky(rss_files, out_suffix='_EMPSKY', min_blank=MIN_BLANK, blank_frac=BLANK_FRAC):
    """Build + subtract the empirical sky for one field/channel. rss_files are a field's per-dither
    single-channel _FF RSS. Writes <in>_EMPSKY.fits per dither (SKY=empirical, SKYSUB=COUNTS-SKY;
    fibres without enough blank dithers keep the input SKY). Returns the written paths."""
    used, stack, masks, _ = _load_field(rss_files)
    emp_sky, nblank, S = build_empirical_sky(stack, masks, min_blank, blank_frac)
    written = []
    for d, f in enumerate(used):
        with fits.open(f) as h:
            counts = np.asarray(h['COUNTS'].data, float)
            base_sky = np.asarray(h['SKY'].data, float) if 'SKY' in h else np.zeros_like(counts)
            sky = emp_sky[d].copy()
            fallback = ~np.isfinite(sky).any(axis=1)               # fibres with no empirical estimate
            sky[fallback] = base_sky[fallback]                     # keep the b-spline sky there
            sky = np.where(np.isfinite(sky), sky, base_sky)
            skysub = counts - sky
            if 'SKY' in h:
                h['SKY'].data = sky.astype(h['SKY'].data.dtype)
            else:
                h.append(fits.ImageHDU(sky.astype(np.float32), name='SKY'))
            if 'SKYSUB' in h:
                h['SKYSUB'].data = skysub.astype(h['SKYSUB'].data.dtype)
            else:
                h.append(fits.ImageHDU(skysub.astype(np.float32), name='SKYSUB'))
            h[0].header['EMPSKY'] = (True, 'empirical multi-dither sky subtracted')
            h[0].header['EMPSKYNB'] = (int(np.median(nblank[nblank > 0])) if np.any(nblank > 0) else 0,
                                       'median blank dithers per fibre')
            h[0].header['EMPSKYFB'] = (int(fallback.sum()), 'fibres fell back to b-spline sky')
            out = f.replace('.fits', f'{out_suffix}.fits')
            h.writeto(out, overwrite=True)
            written.append(out)
    logger.info("empiricalSky: wrote %d files (frame sky levels S=%s)",
                len(written), np.round(S, 1).tolist())
    return written


def _group_by_channel(rss_files):
    from collections import defaultdict
    g = defaultdict(list)
    for f in rss_files:
        b = os.path.basename(f)
        ch = next((c for c in CHANNELS if f'_RSS_{c}' in b or f'_{c}.fits' in b), None)
        if ch:
            g[ch].append(f)
    return g


def main(argv=None):
    import argparse
    import glob
    p = argparse.ArgumentParser(description='Empirical multi-dither sky subtraction for one field.')
    p.add_argument('rss', nargs='+', help='A field\'s per-dither _FF RSS files (globs OK); grouped by channel')
    p.add_argument('--min-blank', type=int, default=MIN_BLANK)
    p.add_argument('--blank-frac', type=float, default=BLANK_FRAC)
    p.add_argument('--suffix', default='_EMPSKY')
    a = p.parse_args(argv)
    files = []
    for pat in a.rss:
        files.extend(sorted(glob.glob(pat)) if any(ch in pat for ch in '*?[') else [pat])
    for ch, group in _group_by_channel(files).items():
        print(f"[{ch}] {len(group)} dithers")
        try:
            out = subtract_empirical_sky(group, a.suffix, a.min_blank, a.blank_frac)
            print(f"[{ch}] wrote {len(out)} _EMPSKY files")
        except Exception as exc:                                    # noqa: BLE001
            print(f"[{ch}] FAILED: {exc}")


if __name__ == '__main__':
    main()
