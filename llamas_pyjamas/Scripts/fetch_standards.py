#!/usr/bin/env python3
"""Fetch the ESO/Oke spectrophotometric standard-star catalogue into LUT/standards/.

This bundles the reference data the flux-calibration path needs, so the pipeline never depends
on network access at run time. Run it once to (re)generate the bundle; the output is committed.

Two things are fetched and written under ``LUT/standards/``:

* ``index.dat`` — one row per standard: name, RA/Dec (decimal degrees, J2000), V, spectral type,
  and the flux file that holds its reference spectrum (blank if none is bundled). Parsed from the
  ESO "RA-ordered list of spectrophotometric standards" (stanlis.html).
* ``flux/f<name>.dat`` — the Oke (1990) optical reference spectra, from the STScI CALOBS set
  mirrored at ESO/STECF. Four columns: wavelength (A), flux (1e-16 erg/cm^2/s/A), flux (mJy),
  bin width (A).

The coordinate list (stanlis) and the flux archive (okestan) are separate ESO products with
overlapping but not identical membership, so a star can appear in ``index.dat`` with no flux
file. Both may26 standards (Feige110, GD108) are in Oke, so this bundle covers that night.

Usage::

    python -m llamas_pyjamas.Scripts.fetch_standards [--dest DIR] [--dry-run]
"""

import argparse
import io
import os
import re
import sys
import urllib.request

from llamas_pyjamas.config import LUT_DIR

STANLIS_URL = 'https://www.eso.org/sci/observing/tools/standards/spectra/stanlis.html'
OKE_DIR_URL = 'https://ftp.eso.org/pub/stecf/standards/okestan/'
OKE_README = 'aaareadme.oke'
XSHOOTER_DIR_URL = 'https://ftp.eso.org/pub/stecf/standards/Xshooter/'
XSHOOTER_README = 'aaareadme.Xshooterspec'

# Oke 'f' files list flux as (erg/s/cm^2/A) x 1e-16; XShooter 'f' files list it directly.
OKE_SCALE = 1e-16
XSHOOTER_SCALE = 1.0
# XShooter spectra are sampled at 0.1 A over 3000-25000 A (~3 MB each); bin to this spacing to
# keep the bundle small — far finer than the LLAMAS pixel scale, so no fidelity is lost.
XSHOOTER_BIN_ANGSTROM = 1.0

DEST_DEFAULT = os.path.join(LUT_DIR, 'standards')
TIMEOUT = 60


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return response.read()


def _bin_spectrum(raw: bytes, bin_ang: float) -> str:
    """Bin a two-column (wavelength, flux) ASCII spectrum to ~`bin_ang` spacing.

    Block-averages rather than decimates, so narrow spikes are smoothed rather than aliased.
    Returns the binned spectrum as two-column text.
    """
    import numpy as np
    data = np.loadtxt(io.BytesIO(raw))
    wave, flux = data[:, 0], data[:, 1]
    step = np.median(np.diff(wave))
    n = max(1, int(round(bin_ang / step)))
    if n <= 1:
        return '\n'.join(f'{w:.4f} {f:.6e}' for w, f in zip(wave, flux)) + '\n'
    trim = (len(wave) // n) * n
    wb = wave[:trim].reshape(-1, n).mean(axis=1)
    fb = flux[:trim].reshape(-1, n).mean(axis=1)
    return '\n'.join(f'{w:.4f} {f:.6e}' for w, f in zip(wb, fb)) + '\n'


def _norm(name: str) -> str:
    """Collapse a star name to a comparison key: lowercase alphanumerics only.

    Bridges the two naming styles — the coordinate list writes ``CD-34d241`` and ``Feige 110``
    while the flux archive writes ``fcd_34d241.dat`` and ``ffeige110.dat`` — so both reduce to
    ``cd34d241`` / ``feige110`` and match.
    """
    return re.sub(r'[^a-z0-9]', '', name.lower())


def parse_stanlis(html: str):
    """Parse the ESO standards list into rows of (name, ra_deg, dec_deg, vmag, sptype).

    Each table row looks like::

        <a href=".../hr9087.html"> HR9087</a>  00 01 49.42  -03 01 39.0  5.12 sdO ...
    """
    block = re.search(r'<pre>(.*?)</pre>', html, re.DOTALL | re.IGNORECASE)
    if not block:
        raise ValueError('Could not find the standards table in stanlis.html')

    rows = []
    seen = set()
    row_re = re.compile(
        r'<a href="[^"]+">\s*([^<]+?)\s*</a>\s+'          # name
        r'(\d{1,2})\s+(\d{1,2})\s+([\d.]+)\s+'            # RA h m s
        r'([+-]\d{1,2})\s+(\d{1,2})\s+([\d.]+)\s+'        # Dec d m s
        r'([\d.]+)?\s*(\S+)?')                            # V, spectral type
    for line in block.group(1).splitlines():
        m = row_re.search(line)
        if not m:
            continue
        name = m.group(1).strip()
        key = _norm(name)
        if key in seen:                                  # the list repeats a few stars
            continue
        seen.add(key)

        ra_deg = 15.0 * (int(m.group(2)) + int(m.group(3)) / 60 + float(m.group(4)) / 3600)
        sign = -1.0 if m.group(5).startswith('-') else 1.0
        dec_deg = sign * (abs(int(m.group(5))) + int(m.group(6)) / 60 + float(m.group(7)) / 3600)
        vmag = m.group(8) or ''
        sptype = m.group(9) or ''
        rows.append((name, ra_deg, dec_deg, vmag, sptype))
    return rows


def list_flux_files(listing_html: str):
    """Map a normalised star key -> flux filename from an okestan directory listing."""
    files = {}
    for href in re.findall(r'href="([^"]+\.dat)"', listing_html, re.IGNORECASE):
        fname = href.rsplit('/', 1)[-1]
        if not fname.lower().startswith('f'):            # 'm' files are AB mag, not flux
            continue
        files[_norm(fname[1:].rsplit('.', 1)[0])] = fname
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dest', default=DEST_DEFAULT,
                        help=f'Output directory (default: {DEST_DEFAULT})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would be fetched without writing anything')
    args = parser.parse_args(argv)

    print(f'Fetching standards list from {STANLIS_URL}')
    stars = parse_stanlis(_get(STANLIS_URL).decode('latin-1'))
    print(f'  parsed {len(stars)} standard stars')

    print(f'Listing flux archives')
    oke_map = list_flux_files(_get(OKE_DIR_URL).decode('latin-1'))
    xsh_map = list_flux_files(_get(XSHOOTER_DIR_URL).decode('latin-1'))
    print(f'  Oke: {len(oke_map)} files | XShooter: {len(xsh_map)} files')

    # Pair each catalogue star with a reference spectrum, preferring XShooter (3000-25000 A,
    # Moehler+2014) over Oke (3200-9200 A, 1990) for its far wider red coverage. Each entry
    # records the source URL, the local filename, and the flux scale for that format.
    plan = {}   # name -> (source, url, local_name, scale)
    for name, *_ in stars:
        key = _norm(name)
        if key in xsh_map:
            plan[name] = ('xshooter', XSHOOTER_DIR_URL + xsh_map[key],
                          'xsh_' + xsh_map[key], XSHOOTER_SCALE)
        elif key in oke_map:
            plan[name] = ('oke', OKE_DIR_URL + oke_map[key], oke_map[key], OKE_SCALE)

    print(f'  matched {len(plan)}/{len(stars)} stars '
          f'({sum(1 for v in plan.values() if v[0] == "xshooter")} XShooter, '
          f'{sum(1 for v in plan.values() if v[0] == "oke")} Oke)')
    for required in ('Feige110', 'GD108'):
        entry = plan.get(required)
        print(f'    {required:10s} -> {entry[0] + " " + entry[2] if entry else "(none)"}')

    if args.dry_run:
        print('Dry run; nothing written.')
        return 0

    flux_dir = os.path.join(args.dest, 'flux')
    os.makedirs(flux_dir, exist_ok=True)

    for name, (source, url, local_name, _scale) in sorted(plan.items()):
        raw = _get(url)
        out = os.path.join(flux_dir, local_name)
        if source == 'xshooter':
            with open(out, 'w') as fh:              # binned to ~1 A to keep the bundle small
                fh.write(_bin_spectrum(raw, XSHOOTER_BIN_ANGSTROM))
        else:
            with open(out, 'wb') as fh:
                fh.write(raw)
    print(f'  wrote {len(plan)} flux files to {flux_dir}')

    for url, fname in ((OKE_DIR_URL + OKE_README, OKE_README),
                       (XSHOOTER_DIR_URL + XSHOOTER_README, XSHOOTER_README)):
        try:
            with open(os.path.join(flux_dir, fname), 'w') as fh:
                fh.write(_get(url).decode('latin-1'))
        except Exception as exc:                          # noqa: BLE001
            print(f'  (could not fetch {fname}: {exc})')

    index_path = os.path.join(args.dest, 'index.dat')
    with open(index_path, 'w') as fh:
        fh.write('# ESO spectrophotometric standard stars\n')
        fh.write(f'# Coordinates: {STANLIS_URL}\n')
        fh.write(f'# Flux spectra: XShooter (Moehler+2014, {XSHOOTER_DIR_URL}) preferred, '
                 f'else Oke 1990 ({OKE_DIR_URL})\n')
        fh.write('# flux_scale multiplies column 2 of flux_file to give erg/s/cm2/A.\n')
        fh.write('# Generated by llamas_pyjamas.Scripts.fetch_standards\n')
        fh.write('#\n')
        fh.write('# {:<14s} {:>11s} {:>11s} {:>6s} {:>8s} {:>10s}  {}\n'.format(
            'name', 'ra_deg', 'dec_deg', 'vmag', 'sptype', 'flux_scale', 'flux_file'))
        for name, ra, dec, vmag, sptype in sorted(stars, key=lambda r: r[1]):
            entry = plan.get(name)
            local_name = entry[2] if entry else '-'
            scale = f'{entry[3]:.0e}' if entry else '-'
            fh.write('{:<16s} {:11.6f} {:+11.6f} {:>6s} {:>8s} {:>10s}  {}\n'.format(
                name.replace(' ', ''), ra, dec, vmag or '-', sptype or '-',
                scale, local_name))
    print(f'  wrote {index_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
