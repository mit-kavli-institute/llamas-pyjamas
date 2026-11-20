"""
llamas_pyjamas.Bias.biasFirstResume
===================================
Which frames of a resumed run were bias-corrected out of date, and moving their
products aside so the normal resume rebuilds them.

``reduce.py`` resumes by file name at every stage (bias-corrected, flat-corrected,
RSS, _FF, _FF_SKYSUB, cubes). A frame corrected by an older bias step, against a
different master bias, or without the master dark this run would now subtract,
would therefore be reused silently. Bias-first stamps its provenance in the
primary header (``BFVER``, ``BIASID``, ``DARKID``, ``DARKSCL``), and every later
product copies that header, so each product of a frame can be checked.

A stale frame's products are moved (never deleted) into
``{output_dir}/stale/<timestamp>/``. What has to be redone depends on the frame:

- a science or flux-standard frame: that exposure (all science frames when the
  empirical 2D sky is on, since its fields borrow donor frames);
- a sky frame: itself, plus every science exposure when the sky model uses it;
- a flat, arc or twilight: everything, including the calibration products built
  from them (traces, flat products, night arcs, pedestal templates, QA tables).
"""

import logging
import os
import re
import shutil
import time

from astropy.io import fits

from llamas_pyjamas.Bias.biasFirst import provenance_of
from llamas_pyjamas.config import BIAS_DIR, CALIB_DIR, LUT_DIR
from llamas_pyjamas.File.llamasRSS import exposure_id
from llamas_pyjamas.Masking.cosmicLlamas import LEGACY_CRID
from llamas_pyjamas.Utils.recipes import EXTRACTION_STAMPS, extraction_method_problem

logger = logging.getLogger(__name__)

CHAIN_MARKER = '_bias_corrected'
PACKAGED = (CALIB_DIR, LUT_DIR, BIAS_DIR)   # packaged calibrations: never moved
CALIBRATION_ROLES = ('flat', 'arc', 'twilight')


def is_product_of(name, exp_id):
    """True if a file name is a pipeline product of exposure ``exp_id``.

    The name must carry the exposure id followed by '_' or '.', and either the
    bias-corrected chain marker or one of the clean consolidated names. Raw frames
    and the telescope quick look (``{raw}_whitelight.fits``) never match.
    """
    if not re.search(re.escape(exp_id) + r'[_.]', name):
        return False
    if CHAIN_MARKER in name:
        return True
    return (name.startswith(f'{exp_id}_RSS_') and name.endswith('.fits')) or name in (
        f'{exp_id}_whitelight_fullpipeline.fits', f'{exp_id}_empsky2d.pkl')


def _is_checked(name):
    """Products whose primary header carries the bias-first provenance: the frame's
    bias-corrected and flat-corrected copies and its RSS files (not side products
    such as cosmic-ray masks)."""
    if not name.endswith('.fits') or name.startswith('cosmic_ray_mask_'):
        return False
    return (name.endswith(CHAIN_MARKER + '.fits') or name.endswith('_flat_corrected.fits')
            or '_RSS' in name)


def _inside(path, root):
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == \
            os.path.abspath(root)
    except ValueError:
        return False


class ProductScan:
    """The files a resumed run may move, found once.

    ``dirs`` are searched non-recursively (``masks/`` subfolders are added by the
    caller). ``protected`` paths (every configured input) are never returned,
    nor is anything under ``stale_root``, in the packaged calibration folders
    (mastercalib, LUT, Bias), or modified at or after ``since`` (made by this run).
    """

    def __init__(self, dirs, protected, stale_root, since):
        self.protected = {os.path.abspath(p) for p in protected}
        self.stale_root = stale_root
        self.since = since
        self.files = []
        seen = set()
        for d in dirs:
            if not d or not os.path.isdir(d):
                continue
            for name in os.listdir(d):
                path = os.path.abspath(os.path.join(d, name))
                if path in seen or not os.path.isfile(path) or not self.movable(path):
                    continue
                seen.add(path)
                self.files.append(path)

    def movable(self, path):
        path = os.path.abspath(path)
        if path in self.protected or _inside(path, self.stale_root):
            return False
        if any(_inside(path, d) for d in PACKAGED):
            return False
        try:
            return os.path.getmtime(path) < self.since
        except OSError:
            return False

    def products(self, raw_path):
        exp_id = exposure_id(raw_path)
        return [p for p in self.files if is_product_of(os.path.basename(p), exp_id)]


def stale_reason(expected, products):
    """Why a frame's existing products are out of date, or '' if they are current.

    The bias-first stamps are compared on every checked product. ``CRID`` (the
    cosmic-ray cleaning, stamped at extraction) is compared on RSS files only, and
    only when ``expected`` has it; an RSS without it was made with the cleaning
    that predates the setting (``LEGACY_CRID``). Likewise ``EXTMETH``/``EXTHW`` (the
    extraction method): an RSS without them was extracted with ``LEGACY_EXTMETH``.
    """
    bias_expected = {k: v for k, v in expected.items()
                     if k != 'CRID' and k not in EXTRACTION_STAMPS}
    cosmic_rays_changed = False
    method_changed = False
    for path in products:
        name = os.path.basename(path)
        if not _is_checked(name):
            continue
        try:
            header = fits.getheader(path, 0)
        except Exception as exc:                                  # noqa: BLE001
            return f'{name} unreadable ({exc})'
        found = provenance_of(header)
        if found == bias_expected:
            if 'CRID' in expected and '_RSS' in name \
                    and header.get('CRID', LEGACY_CRID) != expected['CRID']:
                cosmic_rays_changed = True
            if 'EXTMETH' in expected and '_RSS' in name and extraction_method_problem(
                    header, (expected['EXTMETH'], expected.get('EXTHW'))):
                method_changed = True
            continue
        if found['BFVER'] is None or found['BFVER'] < expected['BFVER']:
            return 'made by the older bias step'
        if found['BIASID'] != expected['BIASID']:
            return 'master bias changed'
        if found['DARKID'] in (None, 'none') and expected['DARKID'] != 'none':
            return 'dark now available'
        if expected['DARKID'] == 'none':
            return 'dark no longer matched'
        if found['DARKID'] != bias_expected['DARKID']:
            return 'master dark changed'
        return 'dark scale changed'
    if cosmic_rays_changed:
        return 'cosmic-ray settings changed'
    return 'extraction method changed' if method_changed else ''


def calibration_products(trace_dir, extraction_dir, flat_dir, output_dir, qa_dir, scan):
    """Calibration-derived products a full redo must move (files and whole folders)."""
    paths = []
    if trace_dir and os.path.isdir(trace_dir):
        paths += [os.path.join(trace_dir, n) for n in os.listdir(trace_dir)
                  if n.endswith('_traces.pkl')]
    for d in (flat_dir, os.path.join(extraction_dir, 'twilight'), os.path.join(output_dir, 'arcs')):
        if d and os.path.isdir(d):
            paths.append(d)
    if extraction_dir and os.path.isdir(extraction_dir):
        paths += [os.path.join(extraction_dir, n) for n in os.listdir(extraction_dir)
                  if n.startswith('floor_template_cps_') or n == 'edge_bias_levels.csv']
    if qa_dir and os.path.exists(os.path.join(qa_dir, 'wavelengthQA_summary.csv')):
        paths.append(os.path.join(qa_dir, 'wavelengthQA_summary.csv'))
    keep = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            if not (any(_inside(p, d) for d in PACKAGED) or _inside(p, scan.stale_root)):
                keep.append(p)
        elif scan.movable(p):
            keep.append(p)
    return keep


def move_aside(paths, output_dir, stale_dir):
    """Move ``paths`` into ``stale_dir``, keeping their place relative to ``output_dir``.

    Later products go first and bias-corrected frames last, so an interruption
    never leaves a fresh-looking bias-corrected frame over old products. Returns
    the number moved.
    """
    def order(p):
        return (os.path.basename(p).endswith(CHAIN_MARKER + '.fits'), p)

    moved = 0
    for path in sorted(set(paths), key=order):
        if not os.path.exists(path):
            continue
        if _inside(path, output_dir):
            rel = os.path.relpath(path, output_dir)
        else:
            rel = os.path.join(os.path.basename(os.path.dirname(os.path.abspath(path))) or 'other',
                               os.path.basename(path))
        dest = os.path.join(stale_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(path, dest)
        moved += 1
    return moved


def redo_stale_frames(frames, dirs, calib_dirs, output_dir, protected, since,
                      sky_used=False, empirical_sky=False):
    """Find out-of-date frames and move their products aside; returns a summary dict.

    ``frames`` maps each raw path that goes through bias-first to
    ``(role, expected)``: role is 'science', 'sky' or one of CALIBRATION_ROLES,
    expected the provenance dict bias-first will stamp. ``dirs`` are the folders
    holding per-exposure products; ``calib_dirs`` is a dict with trace_dir,
    extraction_dir, flat_dir and qa_dir.
    """
    stale_root = os.path.join(output_dir, 'stale')
    scan = ProductScan(dirs, protected, stale_root, since)
    stale = {}
    for raw, (role, expected) in frames.items():
        reason = stale_reason(expected, scan.products(raw))
        if reason:
            stale[raw] = (role, reason)

    summary = {'stale': stale, 'scope': None, 'moved': 0, 'stale_dir': None}
    if not stale:
        return summary

    roles = {role for role, _ in stale.values()}
    science = [r for r, (role, _) in frames.items() if role == 'science']
    redo = set(stale)
    to_move = []
    if roles & set(CALIBRATION_ROLES):
        summary['scope'] = 'all'
        redo = set(frames)
        to_move += calibration_products(calib_dirs.get('trace_dir'), calib_dirs.get('extraction_dir'),
                                        calib_dirs.get('flat_dir'), output_dir,
                                        calib_dirs.get('qa_dir'), scan)
    else:
        if 'sky' in roles and sky_used:
            redo |= set(science)
        if 'science' in roles and empirical_sky:
            redo |= set(science)
        summary['scope'] = 'frames'
    for raw in redo:
        to_move += scan.products(raw)

    stale_dir = os.path.join(stale_root, time.strftime('%Y%m%d_%H%M%S'))
    summary['moved'] = move_aside(to_move, output_dir, stale_dir)
    summary['stale_dir'] = stale_dir
    summary['redo'] = sorted(redo)
    return summary
