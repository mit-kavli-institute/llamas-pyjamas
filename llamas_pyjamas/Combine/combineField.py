"""
CLI: co-add a field's dithers into an image (Phase 4).

Discovers a field's registered RSS files (or takes them explicitly), builds the super-RSS, and
writes a DS9-ready co-add image with its depth maps. A quick way to eyeball a stack before the
photometric-scaling / cube machinery lands.

Examples
--------
Broadband green co-add of J2151, auto-discovered in a reduction directory::

    python -m llamas_pyjamas.Combine.combineField --dir /path/reduced/extractions \\
        --object J2151 --channels green -o j2151_green.fits --png

A narrowband window across all channels, surface brightness, from explicit files::

    python -m llamas_pyjamas.Combine.combineField a_RSS_green.fits b_RSS_green.fits \\
        --band 6560 6620 -o field_nb.fits

Open the result in DS9 (SCI is the primary; VAR/SNR/COVERAGE/NEXP are extensions).
"""

import argparse
import glob
import logging
import os
import sys
from typing import List, Optional

import numpy as np
from astropy.io import fits

from llamas_pyjamas.Combine.superRSS import build_super_rss, CHANNELS
from llamas_pyjamas.Combine.coadd import combine_image

logger = logging.getLogger(__name__)


def discover_field(directory: str, obj: str) -> List[str]:
    """Green RSS files in `directory` whose OBJECT header starts with `obj` (one per exposure)."""
    hits = []
    for f in sorted(glob.glob(os.path.join(directory, '*_RSS_green.fits'))):
        try:
            name = str(fits.getheader(f, 0).get('OBJECT', ''))
        except Exception:                              # noqa: BLE001
            continue
        if name.startswith(obj):
            hits.append(f)
    return hits


def _band(super_rss, args) -> tuple:
    """Resolve the wavelength window: explicit --band, else the full range of the chosen channels."""
    if args.band is not None:
        return float(args.band[0]), float(args.band[1])
    lo, hi = np.inf, -np.inf
    for c in (args.channels or list(super_rss.channels)):
        st = super_rss.channels.get(c)
        if st is None:
            continue
        w = st.wave[np.isfinite(st.wave)]
        if w.size:
            lo, hi = min(lo, float(w.min())), max(hi, float(w.max()))
    return lo, hi


def _write_png(img, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    snr = img.snr()
    for a, (arr, ttl, zscale) in zip(ax, [(img.data, 'co-add', True),
                                          (img.nexp, 'depth: N exposures', False),
                                          (snr, 'S/N', True)]):
        d = arr.astype(float)
        fin = np.isfinite(d)
        if zscale and fin.any():
            vmin, vmax = ZScaleInterval().get_limits(d[fin])
        else:
            vmin, vmax = 0, (np.nanmax(d) if fin.any() else 1)
        im = a.imshow(d, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        a.set_title(f"{img.meta.get('FIELD', '')} {ttl}")
        plt.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description='Co-add a field\'s dithers into a DS9 image.')
    p.add_argument('rss', nargs='*', help='RSS files (any channel; siblings auto-found). '
                                          'Or use --dir/--object to discover.')
    p.add_argument('--dir', help='reduction directory to search for --object')
    p.add_argument('--object', help='OBJECT prefix to select a field (e.g. J2151)')
    p.add_argument('--band', nargs=2, type=float, metavar=('LO', 'HI'),
                   help='wavelength window (A); default = full range of the chosen channels')
    p.add_argument('--channels', nargs='+', choices=CHANNELS, help='channels (default: all)')
    p.add_argument('--units', choices=('sb', 'flux'), default='sb')
    p.add_argument('--weight', choices=('ivar', 'uniform', 'exptime'), default='ivar')
    p.add_argument('--kernel', choices=('gaussian', 'tophat'), default='gaussian')
    p.add_argument('--fwhm', type=float, default=0.9, help='kernel FWHM in arcsec')
    p.add_argument('--pixscale', type=float, default=0.5, help='output pixel scale, arcsec')
    p.add_argument('--min-coverage', type=int, default=1, dest='min_coverage')
    p.add_argument('--plane', choices=('auto', 'flam', 'skysub'), default='auto')
    p.add_argument('-o', '--out', help='output FITS (default: <field>_coadd.fits)')
    p.add_argument('--png', action='store_true', help='also write a quick preview PNG')
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    paths = list(args.rss)
    if args.dir and args.object:
        paths += discover_field(args.dir, args.object)
    if not paths:
        p.error('give RSS files, or --dir and --object')
    logger.info('combining %d exposure file(s)', len(paths))

    sr = build_super_rss(paths, plane=args.plane, channels=args.channels)
    logger.info(sr.summary())
    lo, hi = _band(sr, args)
    logger.info('window %.1f-%.1f A, channels=%s, units=%s, weight=%s',
                lo, hi, args.channels or 'all', args.units, args.weight)

    img = combine_image(sr, lo, hi, channels=args.channels, units=args.units,
                        weighting=args.weight, kernel=args.kernel, kernel_fwhm=args.fwhm,
                        pixscale=args.pixscale, min_coverage=args.min_coverage)

    out = args.out or f'{sr.field or "field"}_coadd.fits'
    img.write(out)
    cov = img.coverage
    logger.info('wrote %s  (%dx%d, max depth %d/%d exposures, max coverage %d fibres)',
                out, img.data.shape[1], img.data.shape[0], int(img.nexp.max()),
                img.meta['NEXPTOT'], int(cov.max()))
    print(f'co-add: {out}')
    if args.png:
        png = os.path.splitext(out)[0] + '.png'
        _write_png(img, png)
        print(f'preview: {png}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
