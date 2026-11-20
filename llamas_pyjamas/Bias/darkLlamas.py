"""
llamas_pyjamas.Bias.darkLlamas
==============================
Master darks for the bias-first step.

A master dark is the combination of raw dark frames (PRODCATG ``CAL.R-DRK``) that
share a read mode and requested exposure time, minus the same master-bias
extension that bias-first subtracts. Bias-first then removes
``scale x master dark`` from each science, flux-standard and sky frame with the
same read mode and a requested exposure within a tolerance (10% by default), where
``scale`` is the ratio of ``DEXPTIME``, the total dark accumulation time.

Because the dark frames carry the bias, building the master dark against the
exact bias array bias-first subtracts means a matched dark (scale ~1) also
removes any difference between that master bias and the night's true bias.

Every extension is matched to its camera by header and re-confirmed before use,
through the same helpers as the master bias (``Bias.match_camera_hdu`` and
``Bias.verify_camera_hdu``). A dark is a property of the physical detector, so
the detector serial ``CAMSN`` must also agree where both headers carry it.

Combining (per camera, in pipeline orientation):
- 3 or more frames: median (as the reference notebook does);
- 2 frames: mean, except where they differ by more than 5 sigma, where the lower
  value is used (rejects cosmic rays, which a median of two cannot);
- 1 frame: 5x5 median filter, so the frame's own cosmic rays are not printed
  into the science as negative holes. Hot pixels are then not corrected.

The master dark is written in raw orientation, one extension per camera with the
raw dark's camera keywords, so it is read like a master bias.
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.ndimage import median_filter

from llamas_pyjamas.Bias import (BiasNotFoundError, BiasReadModeError, camera_of_header,
                                 match_camera_hdu, verify_camera_hdu)
from llamas_pyjamas.File.llamasIO import datasec_slices, orient_by_color, trim_and_orient
from llamas_pyjamas.Trace.traceLlamasMaster import _grab_bias_hdu
from llamas_pyjamas.Utils.detectorProps import props_for_header

logger = logging.getLogger(__name__)

DARK_PRODCATG = 'CAL.R-DRK'
ALGORITHM = 1             # bump when the combination changes, so old master darks are rebuilt
DEFAULT_TOLERANCE = 0.10  # dark REXPTIME within this fraction of the frame's
SCALE_WARN = 0.05         # |scale - 1| above this: check the exposure (aborted?)
SPREAD_WARN = 5.0         # DN; dark frames whose levels differ by more than this disagree
MIN_DARKS = 3
PAIR_SIGMA = 5.0
SINGLE_FILTER = 5
HOT_SIGMA = 5.0           # hot pixels for the L.A.Cosmic mask
CAMERAS = [(b, s, c) for b in '1234' for s in 'AB' for c in ('red', 'green', 'blue')]


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------

def read_mode(header):
    """READ-MDE upper case; a frame without one counts as SLOW, as for the bias."""
    return (str(header.get('READ-MDE') or '').strip().upper()) or 'SLOW'


def _positive(header, key):
    try:
        value = float(header.get(key))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def requested_exptime(header):
    """Requested exposure (REXPTIME), else the shutter time (SEXPTIME, EXPTIME)."""
    for key in ('REXPTIME', 'SEXPTIME', 'EXPTIME'):
        value = _positive(header, key)
        if value is not None:
            return value
    return None


def dark_time(header):
    """DEXPTIME, the total dark accumulation time, or None."""
    return _positive(header, 'DEXPTIME')


def is_placeholder(hdu):
    """A missing or dead camera: the validation COMMENT marker, or a constant imaging area.

    Only the DATASEC region is tested, so a dead camera whose metadata row still
    varies is caught too. Covers both placeholder conventions (all 0.0 and all 1.0).
    """
    for comment in hdu.header.get('COMMENT', []):
        if 'Placeholder extension created for missing camera' in str(comment):
            return True
    data = hdu.data
    if data is None or data.size == 0:
        return True
    area = data[datasec_slices(hdu.header, data.shape)]
    return area.size == 0 or float(np.min(area)) == float(np.max(area))


def camera_label(bench, side, color):
    return f"{bench}{side} {color}"


def _file_id(path):
    st = os.stat(path)
    return f"{os.path.basename(path)}|{st.st_size}|{int(st.st_mtime)}"


def bias_id(path):
    """Identity of a master-bias file: its name, size and mtime (not its full path)."""
    if not path or not os.path.exists(path):
        return 'none'
    return hashlib.sha1(_file_id(path).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Grouping and matching
# ---------------------------------------------------------------------------

@dataclass
class DarkGroup:
    """Darks with one read mode and requested exposure time."""
    mode: str
    rexptime: float
    files: List[str] = field(default_factory=list)
    dexptimes: List[Optional[float]] = field(default_factory=list)

    @property
    def dexptime(self):
        """Median DEXPTIME of the darks, or None if any lacks it."""
        if not self.dexptimes or any(d is None for d in self.dexptimes):
            return None
        return float(np.median(self.dexptimes))

    @property
    def label(self):
        return f"{self.mode} {self.rexptime:g} s"


def check_dark_files(files):
    """(errors, warnings) for the listed dark files' primary headers."""
    errors, warnings = [], []
    for path in files:
        try:
            hdr = fits.getheader(path, 0)
        except Exception as exc:                                  # noqa: BLE001
            errors.append(f"dark file unreadable: {path} ({exc})")
            continue
        cat = str(hdr.get('PRODCATG') or '').strip()
        if not cat:
            warnings.append(f"dark file has no PRODCATG, assumed to be a dark: "
                            f"{os.path.basename(path)}")
        elif not cat.startswith(DARK_PRODCATG):
            errors.append(f"dark_files lists a frame that is not a dark "
                          f"(PRODCATG = {cat}): {os.path.basename(path)}")
        if requested_exptime(hdr) is None:
            errors.append(f"dark file has no exposure time (REXPTIME): "
                          f"{os.path.basename(path)}")
    return errors, warnings


def group_darks(files):
    """Group dark files by (read mode, requested exposure time)."""
    return group_darks_from_headers((path, fits.getheader(path, 0)) for path in files)


def group_darks_from_headers(items):
    """``group_darks`` from ``(path, primary header)`` pairs already read (header: any mapping)."""
    groups = {}
    for path, hdr in sorted(items, key=lambda item: item[0]):
        texp = requested_exptime(hdr)
        if texp is None:
            continue
        key = (read_mode(hdr), round(texp, 3))
        group = groups.setdefault(key, DarkGroup(mode=key[0], rexptime=key[1]))
        group.files.append(path)
        group.dexptimes.append(dark_time(hdr))
    return [groups[k] for k in sorted(groups)]


def match_dark(header, groups, tolerance=DEFAULT_TOLERANCE):
    """The dark group for a frame, and the scale to apply.

    Returns ``(group, scale, note)``. ``group`` is None when nothing matches, and
    ``note`` then says why. The group must have the frame's read mode and a
    requested exposure within ``tolerance`` of the frame's; the nearest wins.
    ``scale`` is DEXPTIME(frame) / median DEXPTIME(darks), or the ratio of
    requested exposures when either lacks DEXPTIME.
    """
    mode = read_mode(header)
    texp = requested_exptime(header)
    if texp is None:
        return None, None, 'no exposure time in the header'
    near = [g for g in groups
            if g.mode == mode and abs(g.rexptime - texp) <= tolerance * texp]
    if not near:
        return None, None, f'no {mode} {texp:g} s dark'
    group = min(near, key=lambda g: (abs(g.rexptime - texp), -len(g.files)))
    scale = dark_scale(header, group.dexptime, group.rexptime)
    note = ''
    if abs(scale - 1.0) > SCALE_WARN:
        note = f'scale {scale:.3f}: check this exposure, it may have been cut short'
    return group, scale, note


def dark_scale(header, dark_dexptime, dark_rexptime):
    """DEXPTIME(frame) / DEXPTIME(darks); the ratio of requested exposures if either lacks it."""
    frame_dt = dark_time(header)
    if frame_dt is not None and dark_dexptime is not None:
        return float(frame_dt / dark_dexptime)
    return float(requested_exptime(header) / dark_rexptime)


# ---------------------------------------------------------------------------
# Master bias and dark extensions
# ---------------------------------------------------------------------------

def master_bias_hdu(bias_path, bench, side, color, frame_mode=None):
    """The master-bias extension for one camera, matched and verified; None if unusable.

    The same lookup (``_grab_bias_hdu``) and confirmation (``verify_camera_hdu``)
    as ``select_bias_for_extension``, without its frame-derived fallback, so the
    master dark is built against exactly the extension bias-first subtracts.
    The data are in pipeline orientation (the bias cache trims and flips them).
    """
    if not bias_path:
        return None
    try:
        bias = _grab_bias_hdu(bench=bench, side=side, color=color,
                              dir=os.fspath(bias_path), required_readmode=frame_mode)
    except (BiasNotFoundError, BiasReadModeError, ValueError) as exc:
        logger.warning(f"Master bias unusable for {camera_label(bench, side, color)}: {exc}")
        return None
    if not verify_camera_hdu(bias.header, bench, side, color):
        got = camera_of_header(bias.header)
        logger.error(f"Bias BSC mismatch: expected {camera_label(bench, side, color)}, "
                     f"got {camera_label(*got)}")
        return None
    return bias


def camera_extension(hdul, bench, side, color, source, camsn=None):
    """``(hdu, reason)``: one camera's extension, matched and confirmed like the bias.

    ``reason`` is '' on success. ``camsn``, when given, must equal the extension's
    CAMSN where it has one.
    """
    try:
        hdu = match_camera_hdu(hdul, bench, side, color, source=source)
    except ValueError as exc:
        return None, f'extension lookup failed ({exc})'
    if hdu is None:
        return None, 'no extension for this camera'
    if not verify_camera_hdu(hdu.header, bench, side, color):
        got = camera_of_header(hdu.header)
        return None, f'extension is labelled {camera_label(*got)}'
    if camsn is not None and hdu.header.get('CAMSN') is not None \
            and str(hdu.header['CAMSN']).strip() != str(camsn).strip():
        return None, (f"detector serial differs (CAMSN {str(hdu.header['CAMSN']).strip()} "
                      f"vs {str(camsn).strip()})")
    return hdu, ''


def combine_frames(frames, gain, readnoise_e, bias):
    """Combine one camera's dark frames (pipeline orientation, bias not removed).

    Returns ``(combined, method)``. See the module docstring for the methods.
    """
    stack = np.stack(frames)
    n = len(frames)
    if n >= 3:
        return np.median(stack, axis=0).astype(np.float32), 'median'
    if n == 2:
        mean = stack.mean(axis=0)
        signal = np.clip(mean - bias, 0.0, None)
        sigma_pair = np.sqrt(2.0 * ((readnoise_e / gain) ** 2 + signal / gain))
        rejected = np.abs(stack[0] - stack[1]) > PAIR_SIGMA * sigma_pair
        combined = np.where(rejected, stack.min(axis=0), mean)
        return combined.astype(np.float32), 'pair_min'
    return median_filter(stack[0], size=SINGLE_FILTER).astype(np.float32), f'filter{SINGLE_FILTER}'


def noise_factor(ncombine, method):
    """Variance of the master dark per unit single-frame variance, per aperture pixel (c/N).

    A median of N >= 3 frames has (pi/2)/N; a mean of N frames 1/N. A filtered
    single frame's pixels are correlated across the aperture, so summed over a
    ~5 px aperture its variance is that of (pi/2)/5 per pixel, not (pi/2)/25.
    """
    n = max(int(ncombine or 1), 1)
    if str(method).startswith('median') and n >= 3:
        return (np.pi / 2.0) / n
    if str(method).startswith('filter'):
        return (np.pi / 2.0) / SINGLE_FILTER
    return 1.0 / n


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def hdu_name(header):
    return header.get('EXTNAME') or None


def dark_id(group, bias_path):
    """Identity of a master dark: its frames (name, size, mtime), the bias and the algorithm."""
    parts = sorted(_file_id(f) for f in group.files) + [bias_id(bias_path), f'alg{ALGORITHM}']
    return hashlib.sha1('|'.join(parts).encode()).hexdigest()[:12]


def master_dark_path(out_dir, group, darkid):
    return os.path.join(out_dir, f"master_dark_{group.mode}_{group.rexptime:g}s_{darkid}.fits")


@dataclass
class MasterDark:
    """A master dark on disk, as bias-first and extraction need it."""
    path: str
    darkid: str
    mode: str
    rexptime: float
    dexptime: Optional[float]
    ncombine: int
    cameras: dict = field(default_factory=dict)   # label -> report row

    @property
    def label(self):
        return f"{self.mode} {self.rexptime:g} s"


def _report_from_file(path):
    rows = {}
    with fits.open(path, memmap=True) as hdul:
        prim = hdul[0].header
        for hdu in hdul[1:]:
            b, s, c = camera_of_header(hdu.header)
            rows[camera_label(b, s, c)] = {
                'level': hdu.header.get('DARKLVL'), 'spread': hdu.header.get('DARKSPRD'),
                'ncombine': hdu.header.get('NCOMBINE'), 'method': hdu.header.get('DARKMETH'),
                'note': ''}
        for value in prim.get('HISTORY', []):
            text = str(value)
            if text.startswith('skipped '):
                label, _, reason = text[len('skipped '):].partition(': ')
                rows[label] = {'level': None, 'spread': None, 'ncombine': 0,
                               'method': None, 'note': reason}
        info = MasterDark(path=path, darkid=prim.get('DARKID'), mode=read_mode(prim),
                          rexptime=float(prim.get('REXPTIME')), dexptime=dark_time(prim),
                          ncombine=int(prim.get('NCOMBINE', 0)), cameras=rows)
    return info


def build_master_dark(group, bias_path, out_dir, qa_dir=None):
    """Build (or reuse) the master dark for a group; returns a MasterDark.

    Reused when a file with the same DARKID already exists. Each camera's dark
    frames are matched and confirmed by header, placeholders are left out, and the
    master bias extension is the one bias-first subtracts. A camera with no usable
    frames, no master bias, a constant master bias (a fill, not a measurement) or
    disagreeing detector serials is left out of the master dark and recorded.
    """
    darkid = dark_id(group, bias_path)
    path = master_dark_path(out_dir, group, darkid)
    if os.path.exists(path):
        try:
            if fits.getheader(path, 0).get('DARKID') == darkid:
                print(f"Master dark {group.label}: reusing {os.path.basename(path)}")
                return _report_from_file(path)
        except Exception as exc:                                  # noqa: BLE001
            logger.warning(f"Unreadable master dark {path} ({exc}); rebuilding")

    os.makedirs(out_dir, exist_ok=True)
    n = len(group.files)
    print(f"Building master dark {group.label} from {n} dark frame(s) "
          f"against {os.path.basename(str(bias_path))}")
    if n < MIN_DARKS:
        logger.warning(f"Master dark {group.label}: only {n} dark frame(s); "
                       + ("cosmic rays are rejected pairwise" if n == 2 else
                          f"the frame is {SINGLE_FILTER}x{SINGLE_FILTER} median-filtered, "
                          f"so hot pixels are not corrected"))

    primary = fits.getheader(group.files[0], 0).copy()
    out = [fits.PrimaryHDU(header=primary)]
    rows, skipped = {}, []
    for bench, side, color in CAMERAS:
        label = camera_label(bench, side, color)
        frames, headers, notes, camsn = [], [], [], None
        for path_in in group.files:
            with fits.open(path_in, memmap=False) as hdul:
                hdu, reason = camera_extension(hdul, bench, side, color, path_in, camsn=camsn)
                if hdu is None:
                    if reason != 'no extension for this camera':
                        notes.append(f"{os.path.basename(path_in)}: {reason}")
                    continue
                if is_placeholder(hdu):
                    continue
                if camsn is None and hdu.header.get('CAMSN') is not None:
                    camsn = hdu.header['CAMSN']
                frames.append(np.asarray(trim_and_orient(hdu.data, hdu.header), dtype=np.float32))
                headers.append(hdu.header.copy())
        if notes:
            logger.error(f"Master dark {group.label} {label}: frames left out: {'; '.join(notes)}")
        if not frames:
            skipped.append((label, 'no usable dark frames for this camera'))
            continue
        bias = master_bias_hdu(bias_path, bench, side, color, frame_mode=None)
        if bias is None:
            skipped.append((label, 'no master bias for this camera'))
            continue
        bias_data = np.asarray(bias.data, dtype=np.float32)
        if float(bias_data.min()) == float(bias_data.max()):
            skipped.append((label, 'the master bias is a constant fill for this camera'))
            continue
        if bias_data.shape != frames[0].shape or any(f.shape != frames[0].shape for f in frames):
            skipped.append((label, f'shape mismatch (bias {bias_data.shape}, '
                                   f'darks {frames[0].shape})'))
            continue
        gain, rn, _src = props_for_header(headers[0], 1.0, 2.5)
        combined, method = combine_frames(frames, gain, rn, bias_data)
        master = combined - bias_data
        level = float(np.median(master))
        spread = float(np.ptp([np.median(f - bias_data) for f in frames]))
        if spread > SPREAD_WARN:
            logger.warning(f"Master dark {group.label} {label}: dark frames disagree "
                           f"(levels spread {spread:.1f} DN)")

        hdr = headers[0]
        for key in ('BZERO', 'BSCALE', 'INFOSEC'):
            hdr.remove(key, ignore_missing=True, remove_all=True)
        hdr['NCOMBINE'] = (len(frames), 'Dark frames combined for this camera')
        hdr['DARKMETH'] = (method, 'How the dark frames were combined')
        hdr['DARKLVL'] = (level, 'Median of the master dark (DN)')
        hdr['DARKSPRD'] = (spread, 'Spread of per-frame dark levels (DN)')
        hdr['DARKID'] = (darkid, 'Master dark identity')
        out.append(fits.ImageHDU(data=orient_by_color(master, color).astype(np.float32),
                                 header=hdr, name=hdu_name(hdr)))
        rows[label] = {'level': level, 'spread': spread, 'ncombine': len(frames),
                       'method': method, 'note': ''}

    for label, reason in skipped:
        logger.warning(f"Master dark {group.label} {label}: not built ({reason}); "
                       f"frames get no dark for this camera")
        rows[label] = {'level': None, 'spread': None, 'ncombine': 0, 'method': None,
                       'note': reason}

    prim = out[0].header
    prim['NCOMBINE'] = (n, 'Number of dark frames in the group')
    prim['REXPTIME'] = (group.rexptime, 'Requested exposure of the darks (s)')
    if group.dexptime is not None:
        prim['DEXPTIME'] = (group.dexptime, 'Median dark accumulation time of the darks (s)')
    else:
        prim.remove('DEXPTIME', ignore_missing=True)
    prim['READ-MDE'] = group.mode
    prim['BIASFILE'] = (os.path.basename(str(bias_path)), 'Master bias subtracted from the darks')
    prim['BIASID'] = (bias_id(bias_path), 'Master bias identity (name, size, mtime)')
    prim['DARKID'] = (darkid, 'Master dark identity')
    prim['DARKALG'] = (ALGORITHM, 'Master dark algorithm version')
    for path_in in group.files:
        prim.add_history(f"dark: {os.path.basename(path_in)}")
    for label, reason in skipped:
        prim.add_history(f"skipped {label}: {reason}")

    tmp = path + '.tmp'
    fits.HDUList(out).writeto(tmp, overwrite=True)
    os.replace(tmp, path)
    print(f"Master dark {group.label}: {len(out) - 1} cameras -> {os.path.basename(path)}")

    if qa_dir:
        try:
            plot_master_dark(path, os.path.join(
                qa_dir, os.path.basename(path).replace('.fits', '.png')))
        except Exception as exc:                                  # noqa: BLE001
            logger.warning(f"Master dark QA plot failed: {exc}")

    return MasterDark(path=path, darkid=darkid, mode=group.mode, rexptime=group.rexptime,
                      dexptime=group.dexptime, ncombine=n, cameras=rows)


def plot_master_dark(path, png):
    """Grid of every camera's master dark (raw orientation), like the notebook's."""
    os.makedirs(os.path.dirname(png) or '.', exist_ok=True)
    fig = Figure(figsize=(15, 10))
    FigureCanvasAgg(fig)
    axes = fig.subplots(4, 6).ravel()
    for ax in axes:
        ax.axis('off')
    with fits.open(path, memmap=True) as hdul:
        imgs = [(camera_label(*camera_of_header(h.header)), np.asarray(h.data[::4, ::4]))
                for h in hdul[1:]]
        title = f"Master dark {read_mode(hdul[0].header)} {hdul[0].header.get('REXPTIME'):g} s"
    if not imgs:
        return
    vmax = np.percentile(np.concatenate([d.ravel() for _l, d in imgs]), 99.5)
    for ax, (label, img) in zip(axes, sorted(imgs)):
        im = ax.imshow(img, origin='lower', cmap='Greys', vmin=0, vmax=vmax)
        ax.set_title(f"{label}: {np.median(img):.0f} DN", fontsize=9)
    fig.colorbar(im, ax=list(axes), shrink=0.6, label='Dark signal (DN)')
    fig.suptitle(title)
    fig.savefig(png, dpi=80)


# ---------------------------------------------------------------------------
# Loading one camera (bias-first and extraction)
# ---------------------------------------------------------------------------

def load_master_dark_camera(path, header, darkid=None, mode=None):
    """One camera's master dark in pipeline orientation, matched and confirmed.

    ``header`` is the frame extension's header; its camera (COLOR/BENCH/SIDE or
    CAM_NAME) and CAMSN select and check the dark extension, exactly as the master
    bias is selected and checked. Returns ``(array, info)``; on any failure the
    array is None and ``info['reason']`` says why.
    """
    bench, side, color = camera_of_header(header)
    if not (bench and side and color):
        return None, {'reason': 'the frame extension does not name its camera'}
    if not path or not os.path.exists(path):
        return None, {'reason': f'master dark not found: {path}'}
    try:
        with fits.open(path, memmap=True) as hdul:
            prim = hdul[0].header
            if darkid and prim.get('DARKID') != darkid:
                return None, {'reason': f"master dark identity changed "
                                        f"({prim.get('DARKID')} vs {darkid})"}
            if mode and read_mode(prim) != mode:
                return None, {'reason': f'master dark is {read_mode(prim)}, frame is {mode}'}
            hdu, reason = camera_extension(hdul, bench, side, color, path,
                                           camsn=header.get('CAMSN'))
            if hdu is None:
                return None, {'reason': reason}
            data = np.array(trim_and_orient(hdu.data, hdu.header), dtype=np.float32)
            info = {'reason': '', 'ncombine': int(hdu.header.get('NCOMBINE', 1)),
                    'method': hdu.header.get('DARKMETH', 'median'),
                    'level': hdu.header.get('DARKLVL')}
    except Exception as exc:                                      # noqa: BLE001
        return None, {'reason': f'master dark unreadable ({exc})'}
    return data, info


def cosmic_ray_inputs(scaled_dark):
    """(smooth background, hot-pixel mask) for L.A.Cosmic from a scaled master dark.

    L.A.Cosmic adds ``background`` back to the image before it looks for cosmic
    rays, so it gets a smoothed dark (only its noise model needs the level). Hot
    pixels, where the dark rises well above its smoothed version, are masked so
    their residual is neither flagged nor used to replace neighbours.
    """
    smooth = median_filter(scaled_dark, size=SINGLE_FILTER)
    excess = scaled_dark - smooth
    mad = np.median(np.abs(excess - np.median(excess)))
    sigma = 1.4826 * mad if mad > 0 else np.std(excess)
    hot = excess > HOT_SIGMA * sigma if sigma > 0 else np.zeros(scaled_dark.shape, bool)
    return smooth, hot
