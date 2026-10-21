import cloudpickle as pickle
import os
from astropy.io import fits
import numpy as np
import re
from astropy.coordinates import SkyCoord
from astropy import units as u
import logging
from datetime import datetime

from llamas_pyjamas.Utils.deadfibers import live_fibre_ids
from llamas_pyjamas.Utils.wcsLlamas import FIBRE_AREA_ARCSEC2

logger = logging.getLogger(__name__)


def _fibre_sky_table(fiber_ids, benchsides, primary_hdr):
    """Plate-solved per-fibre sky coords from the (rough header) WCS.

    Returns ``(ras, decs, xs, ys, prov)`` (all arrays length len(fiber_ids); prov a dict). Fibre-
    map (x,y) come from ``FiberMap_LUT`` (NaN on a LUT miss / dead fibre); RA/DEC from a fibre-map
    celestial WCS (``celestial_wcs`` at 0.75"/fibre-unit, CRPIX at the fibre-field centre,
    rotation from the header). If the header has no pointing, RA/DEC are all NaN and prov.method
    is 'none' -- x/y are still returned so a later step can re-solve. This is the ROUGH solution;
    Phase 3 registration overwrites RA/DEC and bumps the provenance in place.
    """
    from llamas_pyjamas.Image.WhiteLightModule import FiberMap_LUT, FIELD_XMAX, FIELD_YMAX
    from llamas_pyjamas.Utils.wcsLlamas import (celestial_wcs, pointing_from_header,
                                                ARCSEC_PER_FIBRE, IFU_PA_OFFSET)

    n = len(fiber_ids)
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    for i, (bench, fid) in enumerate(zip(benchsides, fiber_ids)):
        x, y = FiberMap_LUT(str(bench).strip(), int(fid))
        if not (x < 0 and y < 0):                       # (-1,-1) is the LUT miss sentinel
            xs[i], ys[i] = float(x), float(y)

    ras = np.full(n, np.nan)
    decs = np.full(n, np.nan)
    prov = {'method': 'none', 'tier': 'none', 'refined': False,
            'pa_offset': float('nan'), 'catalog': '', 'rms': float('nan'), 'nstars': 0}

    ra, dec, pa = pointing_from_header(primary_hdr)
    if ra is not None and dec is not None:
        cx, cy = FIELD_XMAX / 2.0, FIELD_YMAX / 2.0     # fibre-field centre = boresight (rough)
        wcs = celestial_wcs(ra, dec, crpix=(cx + 1.0, cy + 1.0),
                            arcsec_per_pixel=ARCSEC_PER_FIBRE, pa_deg=pa)
        finite = np.isfinite(xs) & np.isfinite(ys)
        if finite.any():
            sky = wcs.pixel_to_world(xs[finite], ys[finite])  # fibre-map (x,y) as pixel coords
            ras[finite] = sky.ra.deg
            decs[finite] = sky.dec.deg
        prov = {'method': 'rough-header', 'tier': 'header', 'refined': False,
                'pa_offset': float(IFU_PA_OFFSET), 'catalog': '', 'rms': float('nan'), 'nstars': 0}
    return ras, decs, xs, ys, prov


def _fiberwcs_hdu(fiber_ids, benchsides, ras, decs, xs, ys, prov):
    """Build the FIBERWCS binary-table HDU (per-fibre x/y + RA/DEC + astrometry provenance)."""
    def _num(v):                                        # FITS headers can't hold NaN
        return float(v) if np.isfinite(v) else -1.0
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='FIBER_ID', format='J', array=np.array(fiber_ids)),
        fits.Column(name='BENCHSIDE', format='10A', array=np.array(benchsides)),
        fits.Column(name='X_FIBERMAP', format='D', array=np.asarray(xs, dtype=float)),
        fits.Column(name='Y_FIBERMAP', format='D', array=np.asarray(ys, dtype=float)),
        fits.Column(name='RA', format='D', array=np.asarray(ras, dtype=float)),
        fits.Column(name='DEC', format='D', array=np.asarray(decs, dtype=float)),
    ])
    h = hdu.header
    h['EXTNAME'] = 'FIBERWCS'
    h['WCSMETH'] = (prov['method'], 'Astrometry method (rough-header/astrometric)')
    h['WCSTIER'] = (prov['tier'], 'Registration tier')
    h['WCSREFIN'] = (bool(prov['refined']), 'True if refined against a catalog')
    h['PAOFFSET'] = (_num(prov['pa_offset']), 'PA offset applied to TEL_ROT [deg]')
    h['WCSCAT'] = (prov['catalog'], 'Reference catalog (blank if none)')
    h['WCSRMS'] = (_num(prov['rms']), 'Astrometric residual RMS [arcsec], -1=n/a')
    h['WCSNSTAR'] = (int(prov['nstars']), 'Stars used in the solution')
    h['FIBAREA'] = (FIBRE_AREA_ARCSEC2, 'Lenslet solid angle [arcsec2] (flux<->SB)')
    return hdu


def apply_fibre_astrometry(hdul, ras=None, decs=None, xs=None, ys=None, prov=None):
    """Write per-fibre astrometry onto an open RSS HDUList: fill FIBERMAP RA/DEC, (re)build the
    FIBERWCS extension, and set FIBAREA. Reusable by generate_rss (rough, at build), the patch
    below, and Phase-3 registration (pass a refined ras/decs/prov to overwrite in place).

    If ``ras``/``decs`` are not supplied, the rough solution is computed from the primary header
    pointing + the FIBERMAP fibre ids. Returns the provenance dict written.
    """
    fmap = hdul['FIBERMAP'].data
    fiber_ids = list(fmap['FIBER_ID'])
    benchsides = list(fmap['BENCHSIDE'])
    if ras is None or decs is None:
        ras, decs, xs, ys, prov = _fibre_sky_table(fiber_ids, benchsides, hdul[0].header)
    hdul['FIBERMAP'].data['RA'][:] = np.asarray(ras, dtype=float)
    hdul['FIBERMAP'].data['DEC'][:] = np.asarray(decs, dtype=float)
    new = _fiberwcs_hdu(fiber_ids, benchsides, ras, decs, xs, ys, prov)
    names = [h.name for h in hdul]
    if 'FIBERWCS' in names and _same_fibre_rows(hdul['FIBERWCS'], fiber_ids, benchsides,
                                                 new.columns.names):
        # Same fibres in the same order: update the table in place, so an update-mode file is
        # not rewritten (FIBERWCS sits mid-file in fibre-flat products).
        old = hdul['FIBERWCS']
        for col in new.columns.names:
            old.data[col][:] = new.data[col]
        for key in ('WCSMETH', 'WCSTIER', 'WCSREFIN', 'PAOFFSET', 'WCSCAT', 'WCSRMS', 'WCSNSTAR',
                    'FIBAREA'):
            old.header[key] = (new.header[key], new.header.comments[key])
    else:
        # Missing, or rows that do not match FIBERMAP row for row (equal shape is not enough:
        # a permuted table would hand each fibre another fibre's position): rebuild it.
        if 'FIBERWCS' in names:
            del hdul[names.index('FIBERWCS')]
        hdul.append(new)
    hdul[0].header['FIBAREA'] = (FIBRE_AREA_ARCSEC2, 'Lenslet solid angle [arcsec2] (flux<->SB)')
    return prov


def _same_fibre_rows(hdu, fiber_ids, benchsides, columns):
    """True when a FIBERWCS table has exactly ``columns`` and FIBER_ID/BENCHSIDE equal to the
    given (FIBERMAP) values row for row."""
    data = hdu.data
    if data is None or len(data) != len(fiber_ids) or list(data.columns.names) != list(columns):
        return False
    strip = lambda a: np.char.strip(np.asarray(a).astype(str))   # noqa: E731
    return (np.array_equal(np.asarray(data['FIBER_ID']), np.asarray(fiber_ids))
            and np.array_equal(strip(data['BENCHSIDE']), strip(benchsides)))


def patch_rss_astrometry(path):
    """Patch an existing RSS file in place: fill FIBERMAP RA/DEC + (re)build FIBERWCS + FIBAREA
    from the rough header WCS. Idempotent (re-run overwrites). Returns the provenance dict."""
    with fits.open(path, mode='update') as hdul:
        if 'FIBERMAP' not in [h.name for h in hdul]:
            logger.warning('patch_rss_astrometry: %s has no FIBERMAP; skipped', path)
            return None
        prov = apply_fibre_astrometry(hdul)
        hdul.flush()
    return prov


#: RSS MASK bit registry. INFORMATIONAL bits record how a pixel was processed and leave it usable;
#: every other bit means the value must not be used. 32 and 64 are reserved for the per-fibre
#: sky_quality codes (channel-global fallback / no sky), which are not written to MASK today.
MASK_FLAT_LAMP_ONLY = np.int16(1)         # informational: T_i not from this fibre's twilight
                                          # (lamp offset for the benchside, or benchside median)
MASK_UNMAPPED_FIBRE_INTERP = np.int16(2)  # informational: fibre absent from FiberMap_LUT; its
                                          # twilight gradient was interpolated from peers
MASK_NOT_FLAT_FIELDED = np.int16(4)       # bad: the fibre flat had no correction for this fibre
MASK_DEAD_FIBRE = np.int16(8)             # bad: reserved (legacy Flat/fibre_flat.py)
MASK_NO_COUNTS = np.int16(16)             # bad: extraction wrote no data here (COUNTS zero-fill)
MASK_INFO_BITS = np.int16(MASK_FLAT_LAMP_ONLY | MASK_UNMAPPED_FIBRE_INTERP)


def skysub_extname(hdul):
    """Name of the sky-subtracted science plane in an RSS HDUList.

    Returns ``'SKYSUB'`` when present, else ``'FLUX'``. This bridges the ``FLUX -> SKYSUB``
    rename: files written before the rename still hold ``FLUX``, files after hold ``SKYSUB``,
    and every reader goes through here so both load. ``FLUX`` was a misleading name for the
    sky-subtracted *instrumental* spectrum — the flux-calibrated spectrum is ``FLAM``.
    """
    names = {hdu.name for hdu in hdul}
    if 'SKYSUB' in names:
        return 'SKYSUB'
    if 'FLUX' in names:
        return 'FLUX'
    raise KeyError('RSS has neither a SKYSUB nor a FLUX extension')


def bad_pixel_mask_arrays(mask=None, counts=None, shape=None):
    """Boolean "do not use" mask from MASK and COUNTS arrays (see :func:`bad_pixel_mask`)."""
    if shape is None:
        shape = np.shape(mask if mask is not None else counts)
    bad = np.zeros(shape, dtype=bool)
    if mask is not None:
        bad |= (np.asarray(mask).astype(np.int64) & ~int(MASK_INFO_BITS)) != 0
    if counts is not None:
        c = np.asarray(counts)
        bad |= ~np.isfinite(c) | (c == 0)
    return bad


def bad_pixel_mask(hdul):
    """The one rule every RSS reader applies: True where a pixel must not be used.

    A pixel is bad when MASK carries any non-informational bit (bits 1 and 2 only record how the
    fibre flat was derived), or when COUNTS is exactly 0 or non-finite -- the extraction's
    zero-fill, where SKYSUB = -SKY is junk. The COUNTS test makes legacy files, whose MASK is all
    zero, agree with new ones. A missing MASK or COUNTS adds nothing; the shape comes from the
    science plane.
    """
    names = {h.name for h in hdul}
    mask = np.asarray(hdul['MASK'].data) if 'MASK' in names else None
    counts = np.asarray(hdul['COUNTS'].data) if 'COUNTS' in names else None
    shape = None
    if mask is None and counts is None:
        shape = np.shape(hdul[skysub_extname(hdul)].data)
    return bad_pixel_mask_arrays(mask, counts, shape)


# ---------------------------------------------------------------------------------------------
# Processing-stage contract
#
# One exposure/channel can exist on disk at up to three correction stages: base (extract + sky),
# _FF (twilight fibre flat) and _FF_SKYSUB (fibre flat + residual-sky framework). After
# consolidation the surviving file is renamed {exposure}_RSS_{channel}.fits and its name no longer
# says which stage it is. Every post-processing tool resolves "which file of this exposure do I
# use" through the helpers below, so there is one rule rather than one per tool, and anything
# ambiguous raises instead of silently picking a file.
# ---------------------------------------------------------------------------------------------

#: Correction stages, least to most corrected.
RSS_STAGES = ('base', 'ff', 'ff_skysub')
#: Filename stage suffix -> rank (the consolidator keeps the highest).
RSS_STAGE_RANK = {'': 0, '_FF': 1, '_FF_SKYSUB': 2}
#: An RSS plane at any stage: <base>_RSS_<channel>[_FF[_SKYSUB]].fits. Anchored, so derived files
#: (``_EMPSKY``, ``_sensfunc``, QA products) never count as a stage of the exposure.
RSS_STAGE_RE = re.compile(r'^(.*)_RSS_(blue|green|red)(_FF(?:_SKYSUB)?)?\.fits$')
RSS_STAGE_RE_NOCASE = re.compile(RSS_STAGE_RE.pattern, re.IGNORECASE)
#: Matches the channel token in e.g. ``..._extract_RSS_green_FF_SKYSUB.fits``.
CHANNEL_RE = re.compile(r'(_RSS_)(blue|green|red)(?=[_.])')
RSS_CHANNELS = ('blue', 'green', 'red')

_SUFFIX_STAGE = {'_FF': 'ff', '_FF_SKYSUB': 'ff_skysub'}
_STAGE_RANK = {s: i for i, s in enumerate(RSS_STAGES)}

#: Primary-header keys that legitimately differ between the stages of one exposure.
STAGE_HEADER_ALLOWLIST = frozenset({
    'FIBRFLAT', 'FFLATFIL', 'FFMETHD', 'RSSSTAGE', 'HISTORY', 'COMMENT', '',
    'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND', 'PCOUNT', 'GCOUNT',
    'CHECKSUM', 'DATASUM',
})
#: Keys that record a change to the DATA (pedestal, heliocentric WAVE scaling, line refinement).
#: A difference here cannot be repaired by copying header cards: the stages hold different data.
STAGE_HEADER_DERIVED = frozenset({
    'SKYPED', 'SKYPEDSC', 'VELFRAME', 'HELIOVEL', 'VELCORR', 'SKYLR', 'SKYLRTMP', 'CHANNEL',
})


class RSSStageError(ValueError):
    """The processing stage of an RSS file, or of the files of one exposure, is not usable."""


class DuplicateStageError(RSSStageError):
    """Two files hold the same stage of one exposure/channel (e.g. an old consolidated file next
    to a newer long-named one). Never resolved by heuristic: ``promote_final_rss
    --keep-duplicates`` moves the unwanted ones aside deliberately."""


class MixedStageError(RSSStageError):
    """The channels (or exposures) being combined sit at different processing stages."""


class RegistrationMismatchError(RSSStageError):
    """Another stage of the exposure holds a registered (refined) astrometric solution that the
    file being used lacks. Combining would place its fibres at the rough header positions."""


class StageHeaderMismatchError(RSSStageError):
    """The primary headers of an exposure's stages disagree outside the stage keywords, i.e. one
    stage was corrected after reduction and the others were not."""


def exposure_id(name):
    """The bare exposure identifier ``LLAMAS_<date>_<time>`` for a frame.

    Everything from the exposure-type marker (``_SCI##`` / ``_CAL##``) onward — including the
    whole processing chain (``_mef_bias_corrected_flat_corrected_extract`` …) — is dropped, so
    the same id is recovered from a raw file, any intermediate product, or the consolidated
    RSS. This is what the consolidated RSS is named after and what resume detection keys on;
    the exposure timestamp is unique, so it identifies the frame on its own. Falls back to the
    old stem (extension + trailing ``_mef`` stripped) when no type marker is present.
    """
    stem = os.path.basename(str(name))
    m = re.match(r'^(.*?)_(?:SCI|CAL)\d', stem)
    if m:
        return m.group(1)
    # Strip only a real FITS extension -- NOT os.path.splitext, which would eat the fractional
    # second of a timestamp (e.g. '..._23-21-07.7' -> '..._23-21-07'). That decimal must be kept
    # so it matches the raw frames and the consolidated name is stable across re-runs.
    for _ext in ('.fits.gz', '.fits', '.fit'):
        if stem.lower().endswith(_ext):
            stem = stem[:-len(_ext)]
            break
    if stem.endswith('_mef'):
        stem = stem[:-len('_mef')]
    return stem


def exposure_key(path):
    """Exposure identity of an RSS path, the same for its long per-stage names and its
    consolidated name (``exposure_id`` of the part before ``_RSS_``)."""
    return exposure_id(os.path.basename(str(path)).split('_RSS_')[0])


def channel_siblings(path):
    """Find every channel RSS belonging to the same exposure and processing stage.

    Substitutes the channel token in the filename and keeps the files that exist, so
    ``..._RSS_green_FF.fits`` finds ``..._RSS_blue_FF.fits`` and ``..._RSS_red_FF.fits`` but
    never the un-flat-fielded or ``_SKYSUB`` variants — the suffix is part of the stem and is
    preserved. Use :func:`final_siblings` to resolve the stage instead.

    Returns ``{channel: path}``; empty when the name does not follow the convention.
    """
    directory, name = os.path.split(os.path.abspath(path))
    match = CHANNEL_RE.search(name)
    if match is None:
        logger.warning("Filename %r does not match the _RSS_{channel} convention; "
                       "loading it alone", name)
        return {}

    found = {}
    for channel in RSS_CHANNELS:
        sibling = os.path.join(directory, CHANNEL_RE.sub(rf'\g<1>{channel}', name))
        if os.path.exists(sibling):
            found[channel] = sibling
    return found


def _hdul_label(hdul):
    try:
        return os.path.basename(hdul.filename() or '') or '<in-memory RSS>'
    except Exception:                                   # noqa: BLE001 - label only
        return '<in-memory RSS>'


def rss_stage(rss):
    """Processing stage of an RSS (``'base'``, ``'ff'`` or ``'ff_skysub'``) from its headers.

    The structural markers decide: a SKYRESID extension or ``SKYSUB2`` on the sky-subtracted
    plane means the residual-sky framework ran; primary ``FIBRFLAT`` means the fibre flat ran.
    ``RSSSTAGE`` is only a cross-check — primary headers are copied forward between stages (and
    into derived files such as ``_EMPSKY``), so an inherited value must never decide. A
    disagreement is logged. ``rss`` is a path or an open HDUList; a path reads headers only.
    """
    if not isinstance(rss, fits.HDUList):
        with fits.open(rss) as hdul:
            return rss_stage(hdul)
    hdul = rss
    names = {h.name for h in hdul}
    primary = hdul[0].header
    residual = 'SKYRESID' in names
    if not residual:
        try:
            residual = bool(hdul[skysub_extname(hdul)].header.get('SKYSUB2', False))
        except KeyError:                                # no science plane (header-only file)
            residual = False
    flat = bool(primary.get('FIBRFLAT', False))
    if residual and not flat:
        raise RSSStageError(f'{_hdul_label(hdul)}: residual-sky framework products without a '
                            'fibre flat (FIBRFLAT unset) -- not a pipeline stage')
    stage = 'ff_skysub' if residual else ('ff' if flat else 'base')
    declared = primary.get('RSSSTAGE')
    if declared is not None and str(declared).strip().lower() != stage:
        logger.warning('%s: RSSSTAGE=%r disagrees with its structural markers; using %r',
                       _hdul_label(hdul), declared, stage)
    return stage


def _stage_candidates(path, channel=None):
    """``[(abs_path, suffix_stage_or_None)]`` for the files of ``path``'s exposure and channel.

    Only top-level files matching the anchored :data:`RSS_STAGE_RE` are considered. A suffixed
    name carries its stage; a suffix-less one is ``None`` (base, or a consolidated file of any
    stage) and needs its headers read.
    """
    directory = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(path)
    m = RSS_STAGE_RE.match(name)
    if m is None:
        return []
    key = exposure_key(name)
    chan = channel or m.group(2)
    out = []
    for fname in sorted(os.listdir(directory)):
        fm = RSS_STAGE_RE.match(fname)
        if fm is None or fm.group(2) != chan or exposure_key(fname) != key:
            continue
        full = os.path.join(directory, fname)
        if os.path.isfile(full):
            out.append((full, _SUFFIX_STAGE.get(fm.group(3) or '')))
    return out


def stage_files(path, channel=None):
    """``{stage: path}`` for every stage on disk of ``path``'s exposure and channel.

    Raises :class:`DuplicateStageError` when two files hold the same stage. A path that does not
    follow the naming convention maps to just itself.
    """
    cands = _stage_candidates(path, channel)
    if not cands:
        p = os.path.abspath(path)
        return {rss_stage(p): p}
    found, dups = {}, {}
    for full, stage in cands:
        stage = stage or rss_stage(full)
        if stage in found:
            dups.setdefault(stage, [found[stage]]).append(full)
        else:
            found[stage] = full
    if dups:
        detail = '; '.join(f'{s}: ' + ', '.join(os.path.basename(f) for f in fs)
                           for s, fs in dups.items())
        raise DuplicateStageError(
            f'{exposure_key(path)}: more than one file at the same stage ({detail}). Move the '
            f'unwanted ones aside with promote_final_rss --keep-duplicates.')
    return found


def final_rss(path, stage='auto'):
    """The file to use for ``path``'s exposure and channel.

    ``stage='auto'`` returns the most-corrected stage on disk; a named stage returns that file or
    raises ``FileNotFoundError``. A lone candidate under ``'auto'`` is returned without reading any
    header. A non-conforming name resolves to itself.
    """
    if stage not in ('auto',) + RSS_STAGES:
        raise ValueError(f"stage must be 'auto' or one of {RSS_STAGES}, not {stage!r}")
    cands = _stage_candidates(path)
    if not cands:
        p = os.path.abspath(path)
        if stage != 'auto' and rss_stage(p) != stage:
            raise FileNotFoundError(f'{p} is not at stage {stage!r}')
        return p
    if stage == 'auto' and len(cands) == 1:
        return cands[0][0]
    files = stage_files(path)
    if stage == 'auto':
        return files[max(files, key=_STAGE_RANK.get)]
    if stage not in files:
        raise FileNotFoundError(f'{exposure_key(path)} {RSS_STAGE_RE.match(os.path.basename(path)).group(2)}: '
                                f'no {stage!r} file in {os.path.dirname(os.path.abspath(path))} '
                                f'(have {sorted(files, key=_STAGE_RANK.get)})')
    return files[stage]


def final_siblings(path, stage='auto', allow_mixed=False):
    """``{channel: path}`` for ``path``'s exposure, each channel resolved by :func:`final_rss`.

    Raises :class:`MixedStageError` when the channels land on different stages (e.g. one channel
    never reached the sky framework), unless ``allow_mixed``. Empty for a non-conforming name.
    """
    m = RSS_STAGE_RE.match(os.path.basename(path))
    if m is None:
        return {}
    out = {}
    for chan in RSS_CHANNELS:
        cands = _stage_candidates(path, channel=chan)
        if cands:
            out[chan] = final_rss(cands[0][0], stage)
    if len(out) > 1 and not allow_mixed:
        stages = {c: rss_stage(p) for c, p in out.items()}
        if len(set(stages.values())) > 1:
            raise MixedStageError(f'{exposure_key(path)}: channels are at different stages '
                                  f'{stages}')
    return out


def stage_of(path):
    """Stage of a resolved file, from its filename suffix when it has one, else its headers."""
    m = RSS_STAGE_RE.match(os.path.basename(path))
    if m and m.group(3):
        return _SUFFIX_STAGE[m.group(3)]
    return rss_stage(path)


def wcs_quality(rss):
    """``(rank, label)`` of the astrometry an RSS carries.

    2 — a refined solution (``FIBERWCS`` with ``WCSREFIN=True``); 1 — rough positions only
    (``rough-header``, any unrefined method such as a failed ``relative-source`` pin, or no
    FIBERWCS but finite FIBERMAP RA/DEC as in legacy fibre-flat files); 0 — no positions.
    """
    if not isinstance(rss, fits.HDUList):
        with fits.open(rss) as hdul:
            return wcs_quality(hdul)
    hdul = rss
    names = {h.name for h in hdul}
    if 'FIBERWCS' in names:
        hdr = hdul['FIBERWCS'].header
        meth = str(hdr.get('WCSMETH', '') or '').strip() or 'unknown'
        if bool(hdr.get('WCSREFIN', False)):
            return 2, meth
        ra = np.asarray(hdul['FIBERWCS'].data['RA'], dtype=float)
        if meth != 'none' and np.isfinite(ra).any():
            return 1, f'{meth} (unrefined)'
        return 0, meth
    if 'FIBERMAP' in names:
        ra = np.asarray(hdul['FIBERMAP'].data['RA'], dtype=float)
        if (np.isfinite(ra) & (ra >= 0)).any():
            return 1, 'fibermap only (no FIBERWCS)'
    return 0, 'none'


def _fibre_positions(hdul):
    """``{(benchside, fiber_id): (ra, dec)}`` from FIBERWCS, else FIBERMAP."""
    ext = 'FIBERWCS' if 'FIBERWCS' in hdul else 'FIBERMAP'
    tab = hdul[ext].data
    return {(str(b).strip(), int(f)): (float(r), float(d))
            for b, f, r, d in zip(tab['BENCHSIDE'], tab['FIBER_ID'], tab['RA'], tab['DEC'])}


def _header_value(v):
    return v.strip() if isinstance(v, str) else v


def primary_header_diff(hdr_a, hdr_b):
    """Keys whose primary-header values differ between two stages of one exposure, ignoring
    :data:`STAGE_HEADER_ALLOWLIST`. Returns ``{key: (value_a, value_b)}`` (``None`` = absent)."""
    keys = (set(hdr_a.keys()) | set(hdr_b.keys())) - STAGE_HEADER_ALLOWLIST
    diff = {}
    for k in sorted(keys):
        a = _header_value(hdr_a.get(k)) if k in hdr_a else None
        b = _header_value(hdr_b.get(k)) if k in hdr_b else None
        if a != b:
            diff[k] = (a, b)
    return diff


class StageReport:
    """Result of :func:`check_stage_consistency` for one file."""

    def __init__(self, path, stage, files):
        self.path = path
        self.stage = stage
        self.files = files                    # {stage: path}
        self.wcs = {}                         # {stage: (rank, label)}
        self.better_wcs_stage = None          # stage holding a refined solution `path` lacks
        self.header_diffs = {}                # {stage: {key: (value_here, value_there)}}
        self.warnings = []

    @property
    def derived_keys(self):
        return sorted({k for d in self.header_diffs.values() for k in d
                       if k in STAGE_HEADER_DERIVED})

    def __repr__(self):
        return (f'StageReport({os.path.basename(self.path)!r}, stage={self.stage!r}, '
                f'wcs={self.wcs}, header_diffs={ {s: sorted(d) for s, d in self.header_diffs.items()} })')


def check_stage_consistency(path, *, allow_header_mismatch=False, raise_errors=True,
                            check_registration=True, sep_tol_arcsec=0.01):
    """Compare ``path`` with the other stages of its exposure/channel before it is used.

    * **Registration** — another stage refined (rank 2) while ``path`` is not raises
      :class:`RegistrationMismatchError`. Two different refined solutions (WCSMETH differs, or any
      fibre more than ``sep_tol_arcsec`` apart) and rough-only positions everywhere are warnings.
    * **Primary header** — any key that differs outside the stage keywords raises
      :class:`StageHeaderMismatchError` unless ``allow_header_mismatch``; keys in
      :data:`STAGE_HEADER_DERIVED` are flagged as needing a re-reduction.

    Also raises :class:`DuplicateStageError` (via :func:`stage_files`). With
    ``raise_errors=False`` nothing raises and the findings are returned in the report;
    ``check_registration=False`` skips only the registration error (flux calibration does not
    depend on fibre positions).
    """
    path = os.path.abspath(path)
    try:
        files = stage_files(path)
    except DuplicateStageError:
        if raise_errors:
            raise
        files = {}
    stage = next((s for s, f in files.items() if os.path.samefile(f, path)), None)
    if stage is None:
        stage = rss_stage(path)
    report = StageReport(path, stage, files)

    with fits.open(path) as here:
        report.wcs[stage] = wcs_quality(here)
        here_hdr = here[0].header.copy()
        here_pos = _fibre_positions(here) if ('FIBERWCS' in here or 'FIBERMAP' in here) else {}
        here_meth = (str(here['FIBERWCS'].header.get('WCSMETH', '')).strip()
                     if 'FIBERWCS' in here else '')
    for other_stage, other in files.items():
        if other_stage == stage:
            continue
        with fits.open(other) as there:
            report.wcs[other_stage] = q = wcs_quality(there)
            diff = primary_header_diff(here_hdr, there[0].header)
            if diff:
                report.header_diffs[other_stage] = diff
            if q[0] == 2 and report.wcs[stage][0] == 2:
                there_meth = str(there['FIBERWCS'].header.get('WCSMETH', '')).strip()
                there_pos = _fibre_positions(there)
                common = set(here_pos) & set(there_pos)
                sep = 0.0
                for k in common:
                    (r1, d1), (r2, d2) = here_pos[k], there_pos[k]
                    if np.all(np.isfinite([r1, d1, r2, d2])):
                        dra = (r1 - r2) * np.cos(np.deg2rad(0.5 * (d1 + d2)))
                        sep = max(sep, float(np.hypot(dra, d1 - d2)) * 3600.0)
                if there_meth != here_meth or sep > sep_tol_arcsec:
                    report.warnings.append(
                        f'{os.path.basename(path)}: {other_stage} holds a different refined '
                        f'solution ({there_meth}, max fibre offset {sep:.3f}") than {stage} '
                        f'({here_meth}); using {stage}.')
    best = max(report.wcs.values(), key=lambda q: q[0])
    if best[0] == 2 and report.wcs[stage][0] < 2:
        report.better_wcs_stage = next(s for s, q in report.wcs.items() if q[0] == 2)
    elif best[0] < 2:
        report.warnings.append(f'{os.path.basename(path)}: no registered astrometry in any stage '
                               f'({report.wcs[stage][1]}); fibres sit at rough positions.')
    for w in report.warnings:
        logger.warning(w)

    if raise_errors and check_registration and report.better_wcs_stage is not None:
        raise RegistrationMismatchError(
            f'{os.path.basename(path)} ({stage}) has only {report.wcs[stage][1]} astrometry but '
            f'its {report.better_wcs_stage} stage ({os.path.basename(files[report.better_wcs_stage])}) '
            f'is registered ({report.wcs[report.better_wcs_stage][1]}). Copy the solution across '
            f'with promote_final_rss, or choose that stage explicitly.')
    if raise_errors and report.header_diffs and not allow_header_mismatch:
        detail = '; '.join(f'vs {s}: ' + ', '.join(sorted(d)) for s, d in report.header_diffs.items())
        derived = report.derived_keys
        hint = (f' Keys {derived} record data changes and need a re-reduction.' if derived else
                ' Repair with promote_final_rss --sync-header-from STAGE, or override with '
                '--allow-header-mismatch.')
        raise StageHeaderMismatchError(f'{os.path.basename(path)} ({stage}): primary header differs '
                                       f'from its other stages ({detail}).{hint}')
    return report


def write_rss_manifest(extraction_dir, run_record=None, filename='rss_manifest.json'):
    """Record, per exposure and channel, which RSS file is final and what is on disk.

    ``run_record`` (optional) is ``{'written': set(basenames), 'notes': {basename: [str]}}`` from
    the reduction that just ran, so the manifest says which files that run regenerated and which
    it kept as they were. :func:`final_rss` stays the authority; this is the persisted record for
    external tools. Written atomically. Returns the manifest path (``None`` if nothing to record).
    """
    import json
    if not os.path.isdir(extraction_dir):
        return None
    run_record = run_record or {}
    written = {os.path.basename(f) for f in run_record.get('written', ())}
    notes = run_record.get('notes', {})
    groups = {}
    for fname in sorted(os.listdir(extraction_dir)):
        m = RSS_STAGE_RE.match(fname)
        if m and os.path.isfile(os.path.join(extraction_dir, fname)):
            groups.setdefault((exposure_key(fname), m.group(2)), fname)
    entries = {}
    for (key, chan), fname in sorted(groups.items()):
        rec = {}
        try:
            files = stage_files(os.path.join(extraction_dir, fname))
            final = files[max(files, key=_STAGE_RANK.get)]
            rec['final'] = os.path.basename(final)
            rec['stage'] = next(s for s, f in files.items() if f == final)
            rec['stages'] = {s: os.path.basename(f) for s, f in files.items()}
            rec['regenerated'] = {s: os.path.basename(f) in written for s, f in files.items()}
            rec['notes'] = sorted({n for f in files.values()
                                   for n in notes.get(os.path.basename(f), ())})
        except RSSStageError as exc:
            rec['error'] = str(exc)
        entries.setdefault(key, {})[chan] = rec
    if not entries:
        return None
    out = os.path.join(extraction_dir, filename)
    tmp = out + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump({'created': datetime.now().isoformat(timespec='seconds'),
                   'rule': 'final = most-corrected stage on disk (base < ff < ff_skysub)',
                   'exposures': entries}, fh, indent=2, sort_keys=True)
    os.replace(tmp, out)
    return out


class RSSgeneration:
    def __init__(self, logger=None):
        """
        Initialize the RSSgeneration class.
        
        Parameters:
            logger (Logger, optional): Logger instance. If None, creates a new one.
        """
        # Set up logging
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
            
        self.logger.info("RSSgeneration initialized successfully")
        return


    def generate_rss(self, extraction_file, output_file, subtract_sky=True, noflat_file=None,
                     wave_frame='heliocentric'):
        """
        Generate a row-stacked spectra (RSS) FITS file with the following structure:
        Extension 0 - PRIMARY: primary header only, no data
        Extension 1 - FLUX: sky-subtracted flux (counts - sky) [NFIBER x NWAVE]
                      If subtract_sky=False or no sky model, same as COUNTS.
        Extension 2 - NOFLAT (optional): like FLUX but without 2D pixel flat, only throughput-corrected.
                      Present only when noflat_file is provided.
        Extension 3 - COUNTS: raw extracted counts prior to sky subtraction [NFIBER x NWAVE]
        Extension 4 - SKY: sky model for each fiber [NFIBER x NWAVE]
        Extension 5 - MASK: the pixel mask array for each fiber [NFIBER x NWAVE]
        Extension 6 - WAVE: the wavelength array for each fiber [NFIBER x NWAVE]
        Extension 7 - FWHM: the full width half max array for each fiber [NFIBER x NWAVE]
        Extension 8 - FIBERMAP: the complete fibermap [BINARY FITS TABLE]

        Note: Each row in the data extensions represents one fiber's full spectrum across all wavelengths.
              Access fiber N's data as: data[N, :] (row N, all columns)

        Parameters:
        -----------
        extraction_file : str
            Path to the extraction pickle file
        output_file : str
            Path to save the output RSS FITS file
        subtract_sky : bool, optional
            If True (default), subtract the sky model from counts in the FLUX extension.
            The sky model is always saved in the SKY extension regardless of this setting.
            If the extraction has no sky model, FLUX is unchanged and SKY is all zeros.
        noflat_file : str, optional
            Path to a second extraction pickle produced from the un-flat-corrected science
            FITS.  When provided a NOFLAT extension is inserted between FLUX and COUNTS
            containing (noflat_counts - sky) / throughput — i.e. sky-subtracted and
            throughput-corrected but without the 2D pixel flat divided through.
        """
        self.logger.info(f"Generating RSS file from {extraction_file}")

        self.new_rss_filenames = []

        # Optionally load a parallel extraction from the un-flat-corrected FITS for NOFLAT ext.
        noflat_objects = None
        noflat_meta = None
        if noflat_file is not None:
            try:
                with open(noflat_file, 'rb') as f:
                    _nf = pickle.load(f)
                noflat_objects = _nf['extractions']
                noflat_meta    = _nf['metadata']
                self.logger.info(f"Loaded NOFLAT extraction from {noflat_file} ({len(noflat_objects)} objects)")
            except Exception as e:
                self.logger.warning(f"Could not load noflat_file {noflat_file}: {e}. NOFLAT extension will be skipped.")
                noflat_objects = None

        try:
            with open(extraction_file, 'rb') as f:
                _data = pickle.load(f)
            primary_hdr = _data.get('primary_header')
            extraction_objects = _data['extractions']
            _metadata = _data['metadata']

            self.logger.info(f"Loaded extraction file with {len(extraction_objects)} extraction objects")

            # Heliocentric/barycentric wavelength-frame correction, applied here (after sky
            # subtraction) so OH sky lines stay in the observed frame. Per-exposure and uniform
            # across all fibres/channels; stamps VELFRAME/HELIOVEL/VELCORR into the primary
            # header once. Returns 1.0 for calibrations or when disabled -> WAVE untouched.
            from llamas_pyjamas.Utils.waveFrame import stamp_and_factor
            _helio_vel, _helio_factor = stamp_and_factor(primary_hdr, wave_frame)
            if _helio_factor != 1.0:
                self.logger.info(f"Wavelength frame: {primary_hdr.get('VELFRAME')} "
                                 f"v={_helio_vel:.4f} km/s, WAVE *= {_helio_factor:.8f}")

            # Group by channel
            channel_groups = {}
            meta_groups = {}
            for obj, meta in zip(extraction_objects, _metadata):
                channel = meta.get('channel', 'UNKNOWN')
                if channel not in channel_groups:
                    channel_groups[channel] = []
                    meta_groups[channel] = []
                channel_groups[channel].append(obj)
                meta_groups[channel].append(meta)

            self.logger.info(f"Found channels: {list(channel_groups.keys())}")

            # Group noflat objects by channel (if provided)
            noflat_channel_groups = {}
            if noflat_objects is not None and noflat_meta is not None:
                for obj, meta in zip(noflat_objects, noflat_meta):
                    ch = meta.get('channel', 'UNKNOWN')
                    if ch not in noflat_channel_groups:
                        noflat_channel_groups[ch] = []
                    noflat_channel_groups[ch].append(obj)

            # Process each channel separately - we'll create one RSS file per channel
            for channel, obj_list in channel_groups.items():
                meta_list = meta_groups[channel]
                self.logger.info(f"Processing channel {channel} with {len(obj_list)} extraction objects")
                
                channel_output_file = output_file
                if len(channel_groups) > 1:
                    # If multiple channels, append channel name to output file
                    base, ext = os.path.splitext(output_file)
                    channel_output_file = f"{base}_{channel}{ext}"
                    self.new_rss_filenames.append(channel_output_file)
                    self.logger.info(f"Creating channel-specific output file: {channel_output_file}")
                
                # Create HDU list with primary header (Extension 0 - PRIMARY)
                hdul = fits.HDUList()
                primary_hdu = fits.PrimaryHDU(header=primary_hdr)
                primary_hdu.header['CHANNEL'] = channel
                primary_hdu.header['FIBAREA'] = (FIBRE_AREA_ARCSEC2,
                                                 'Lenslet solid angle [arcsec2] (flux<->SB)')
                primary_hdu.header['RSSSTAGE'] = ('base', 'RSS processing stage (see rss_stage)')
                hdul.append(primary_hdu)
                
                # Collect data for all fibers in this channel
                all_flux = []
                all_errors = []
                all_waves = []
                all_dq = []
                all_fwhm = []
                all_sky = []
                fiber_ids = []
                benchsides = []
                # F3: tally per-fibre sky provenance (1=channel-global fallback,
                # 2=no sky available) for SKY-extension header keywords.
                n_sky_fallback_total = 0
                n_sky_none_total = 0
                fiber_types = []
                fiber_ras = []
                fiber_decs = []
                
                # Process each extraction object
                for i, (obj, meta) in enumerate(zip(obj_list, meta_list)):
                    # Get the counts (flux)
                    counts = obj.counts
                    n_fibers = counts.shape[0]
                    original_n_fibers = n_fibers  # Track original count before any removals

                    # Get relative throughput (1D, one scalar per fiber); fall back to 1.0
                    # The following per-object fallbacks are all expected for
                    # several product types (flats/twilights carry no per-fibre
                    # errors/DQ/FWHM), handled by creating defaults, and fire once
                    # per camera — so they are INFO, not WARNING.
                    throughput = getattr(obj, 'relative_throughput', None)
                    if throughput is None or len(throughput) != n_fibers:
                        self.logger.info(f"Object {i}: missing or mismatched relative_throughput — no throughput correction applied")
                        throughput = np.ones(n_fibers, dtype=np.float32)
                    else:
                        throughput = np.array(throughput, dtype=np.float32)
                        bad_tp = ~np.isfinite(throughput) | (throughput <= 0)
                        if np.any(bad_tp):
                            self.logger.info(f"Object {i}: {bad_tp.sum()} fibers have invalid throughput values — setting those to 1.0")
                            throughput[bad_tp] = 1.0

                    # Get or create error arrays
                    errors = getattr(obj, 'errors', None)
                    if errors is None or errors.shape != counts.shape:
                        self.logger.info(f"No valid error data for object {i}. Creating zero array.")
                        errors = np.zeros_like(counts, dtype=np.float32)

                    # Get or create data quality (mask) arrays
                    dq = getattr(obj, 'dq', None)
                    if dq is None or dq.shape != counts.shape:
                        self.logger.info(f"No valid DQ data for object {i}. Creating zero array.")
                        dq = np.zeros_like(counts, dtype=np.int16)
                    
                    # Get wavelength arrays with improved shape handling
                    waves = getattr(obj, 'wave', None)
                    if waves is None:
                        self.logger.warning(f"No wavelength attribute for object {i}. Using NaN arrays.")
                        waves = np.full(counts.shape, np.nan, dtype=np.float32)
                    elif waves.shape != counts.shape:
                        # Shape mismatch is usually just missing dead-fibre rows and
                        # is reconciled below; info here, warn only if it cannot be
                        # reconciled (see below).
                        self.logger.info(f"Object {i} wavelength shape {waves.shape} != counts shape {counts.shape}")

                        # Check if wavelength array is smaller (missing dead fiber rows)
                        if waves.shape[0] < counts.shape[0] and waves.shape[1] == counts.shape[1]:
                            # Wavelength array is missing dead fiber rows - need to insert them
                            n_missing = counts.shape[0] - waves.shape[0]
                            self.logger.info(f"Wavelength array missing {n_missing} rows. This may be due to dead fiber insertion.")

                            # Get dead fibers that should have been inserted
                            dead_fibers_list = getattr(obj, 'dead_fibers', None)
                            if dead_fibers_list and len(dead_fibers_list) == n_missing:
                                # Insert NaN rows at dead fiber positions
                                self.logger.info(f"Inserting NaN rows at dead fiber positions: {dead_fibers_list}")
                                for dead_idx in sorted(dead_fibers_list):
                                    nan_row = np.full((1, waves.shape[1]), np.nan, dtype=np.float32)
                                    waves = np.insert(waves, dead_idx, nan_row, axis=0)
                                self.logger.info(f"After insertion, wavelength shape: {waves.shape}")
                            else:
                                self.logger.warning(f"Cannot reconcile shape mismatch. Using NaN arrays.")
                                waves = np.full(counts.shape, np.nan, dtype=np.float32)
                        else:
                            # Other shape mismatch - use NaN arrays
                            self.logger.warning(f"Incompatible wavelength shape. Using NaN arrays.")
                            waves = np.full(counts.shape, np.nan, dtype=np.float32)

                    # Final validation
                    if waves.shape != counts.shape:
                        self.logger.error(f"Object {i} wavelength shape {waves.shape} still != counts shape {counts.shape}. Using NaN arrays.")
                        waves = np.full(counts.shape, np.nan, dtype=np.float32)
                    
                    # Get or create FWHM arrays (may not exist in all extractions)
                    fwhm = getattr(obj, 'fwhm', None)
                    if (fwhm is None or fwhm.shape != counts.shape):
                        self.logger.info(f"No valid FWHM data for object {i}. Creating default array.")
                        fwhm = np.full(counts.shape, 2.5, dtype=np.float32)  # Default FWHM of 2.5 pixels

                    # Get sky model array (populated by skyModel_1d; zeros if not available)
                    sky = getattr(obj, 'sky', None)
                    if sky is None:
                        sky = np.zeros_like(counts, dtype=np.float32)
                    elif sky.shape != counts.shape:
                        # Sky may be missing dead-fiber rows inserted into counts during extraction
                        if sky.shape[0] < counts.shape[0] and sky.shape[1] == counts.shape[1]:
                            n_missing = counts.shape[0] - sky.shape[0]
                            dead_fibers_list = getattr(obj, 'dead_fibers', None)
                            if dead_fibers_list and len(dead_fibers_list) == n_missing:
                                self.logger.info(f"Inserting {n_missing} zero rows into sky at dead fiber positions: {dead_fibers_list}")
                                for dead_idx in sorted(dead_fibers_list):
                                    zero_row = np.zeros((1, sky.shape[1]), dtype=np.float32)
                                    sky = np.insert(sky, dead_idx, zero_row, axis=0)
                            else:
                                self.logger.warning(f"Sky shape {sky.shape} != counts shape {counts.shape} for object {i}, cannot reconcile. Using zeros.")
                                sky = np.zeros_like(counts, dtype=np.float32)
                        else:
                            self.logger.warning(f"Sky shape {sky.shape} != counts shape {counts.shape} for object {i}. Using zeros.")
                            sky = np.zeros_like(counts, dtype=np.float32)

                    # Dead fibres are NOT removed here: the extraction now keeps
                    # every per-fibre array LIVE-indexed (dead fibres already
                    # absent, all arrays mutually aligned). The dead_fibers list
                    # holds the fibremap positions and is used only to assign the
                    # physical fibre id of each live row (below). Removing rows
                    # here would (a) desync counts from the fibre-id list and
                    # (b) wrongly drop a faint live fibre whose row reads as zero.
                    dead_fibers = getattr(obj, 'dead_fibers', None)
                    removed_dead_fibers = []
                    if dead_fibers:
                        self.logger.debug(f"Object {i} dead fibres (fibremap positions, live-indexed): {dead_fibers}")

                    # Enhanced wavelength validation logging
                    if waves is not None:
                        nan_count = np.sum(np.isnan(waves))
                        zero_count = np.sum(waves == 0)
                        valid_mask = ~np.isnan(waves) & (waves != 0)
                        valid_count = np.sum(valid_mask)

                        nan_percentage = (nan_count / waves.size) * 100
                        valid_percentage = (valid_count / waves.size) * 100

                        self.logger.info(f"Wavelength array for object {i}:")
                        self.logger.info(f"  Shape: {waves.shape}")
                        self.logger.info(f"  NaN count: {nan_count} ({nan_percentage:.2f}%)")
                        self.logger.info(f"  Zero count: {zero_count} ({100*zero_count/waves.size:.2f}%)")
                        self.logger.info(f"  Valid count: {valid_count} ({valid_percentage:.2f}%)")

                        if valid_count > 0:
                            self.logger.info(f"  Valid range: {waves[valid_mask].min():.2f}-{waves[valid_mask].max():.2f} Å")

                            # Check for wavelength inversions per fiber
                            n_inversions = 0
                            for fiber_idx in range(waves.shape[0]):
                                fiber_wave = waves[fiber_idx]
                                diffs = np.diff(fiber_wave)
                                n_backwards = np.sum(diffs < 0)
                                if n_backwards > 10:  # More than 10 backwards steps is problematic
                                    n_inversions += 1
                                    self.logger.warning(f"  Fiber {fiber_idx} has {n_backwards} wavelength inversions!")

                            if n_inversions > 0:
                                self.logger.warning(f"  Total fibers with inversions: {n_inversions}/{waves.shape[0]}")
                            else:
                                self.logger.info(f"  No wavelength inversions detected")
                        else:
                            self.logger.error(f"  ERROR: No valid wavelength data!")
                    
                    # Apply relative throughput correction — divides out fiber-to-fiber
                    # sensitivity variations so all extensions are on the same scale.
                    # Broadcast (n_fibers,) → (n_fibers, n_pixels) with [:, np.newaxis].
                    tp = throughput[:, np.newaxis]
                    counts = counts / tp
                    sky    = sky    / tp
                    errors = errors / tp

                    # Add data for each fiber
                    all_flux.append(counts)
                    all_errors.append(errors)
                    all_waves.append(waves)
                    all_dq.append(dq)
                    all_fwhm.append(fwhm)
                    all_sky.append(sky)

                    # F3: accumulate sky provenance flags for this extension.
                    _sq = getattr(obj, 'sky_quality', None)
                    if _sq is not None:
                        _sq = np.asarray(_sq)
                        n_sky_fallback_total += int(np.sum(_sq == 1))
                        n_sky_none_total += int(np.sum(_sq == 2))
                    
                    # Add metadata for fiber map
                    benchside_str = f"{meta.get('bench', '')}{meta.get('side', '')}"
                    fiber_type = meta.get('fiber_type', ['UNKNOWN'] * n_fibers)
                    fiber_ra = meta.get('fiber_ra', [np.nan] * n_fibers)
                    fiber_dec = meta.get('fiber_dec', [np.nan] * n_fibers)

                    benchsides.extend([benchside_str] * n_fibers)

                    # Physical fibre id of each (live-indexed) row: the fibremap
                    # positions skipping the camera's dead fibres. Derived from
                    # the canonical dead_fibers list so it is correct whether or
                    # not the (legacy, backward-compat) removal path above fired —
                    # arrays are live-indexed, so removed_dead_fibers is normally
                    # empty and sequential IDs would be wrong.
                    _dead_for_ids = dead_fibers if dead_fibers else removed_dead_fibers
                    if _dead_for_ids and len(_dead_for_ids) > 0:
                        fibers = live_fibre_ids(n_fibers, _dead_for_ids)
                        fiber_ids.extend(fibers)
                        self.logger.info(f"Object {i}: fibre IDs skip dead fibres {sorted(_dead_for_ids)}")
                    else:
                        fiber_ids.extend(np.arange(n_fibers))

                    fiber_types.extend(fiber_type if len(fiber_type) == n_fibers else ['UNKNOWN'] * n_fibers)
                    fiber_ras.extend(fiber_ra if len(fiber_ra) == n_fibers else [np.nan] * n_fibers)
                    fiber_decs.extend(fiber_dec if len(fiber_dec) == n_fibers else [np.nan] * n_fibers)
                
                # Stack all arrays
                counts_stack = np.vstack(all_flux)   # raw counts, always preserved
                error_stack = np.vstack(all_errors)
                wave_stack = np.vstack(all_waves)
                if _helio_factor != 1.0:               # shift into the heliocentric/bary frame
                    wave_stack = wave_stack * _helio_factor
                dq_stack = np.vstack(all_dq).astype(np.int16)
                # Zero-fill from extraction (no profile pixels / off the detector) carries no
                # data: SKYSUB there is -SKY. Flag it so no reader mistakes it for a value.
                _no_counts = ~np.isfinite(counts_stack) | (counts_stack == 0)
                dq_stack[_no_counts] |= MASK_NO_COUNTS
                self.logger.info(f"Channel {channel}: {int(_no_counts.sum())} zero-count pixels "
                                 f"({100.0 * _no_counts.mean():.1f}%) flagged MASK_NO_COUNTS")
                fwhm_stack = np.vstack(all_fwhm)
                sky_stack = np.vstack(all_sky)

                # FLUX = counts - sky if subtract_sky=True and sky model is non-zero
                has_sky = np.any(sky_stack != 0)
                if subtract_sky and has_sky:
                    self.logger.info(f"Subtracting sky model from FLUX for channel {channel}")
                    flux_stack = counts_stack - sky_stack
                else:
                    flux_stack = counts_stack
                    if subtract_sky and not has_sky:
                        self.logger.warning(f"subtract_sky=True but sky model is all zeros for channel {channel}. FLUX unchanged.")
                
                # Get array dimensions
                n_total_fibers, n_pixels = flux_stack.shape
                self.logger.info(f"Channel {channel} stacked data shape: {flux_stack.shape} ({n_total_fibers} fibers, {n_pixels} pixels)")
                
                # Log some statistics
                self.logger.info(f"Counts stack min/max: {np.nanmin(counts_stack):.4f} / {np.nanmax(counts_stack):.4f}")
                self.logger.info(f"Wavelength stack min/max: {np.nanmin(wave_stack):.4f} / {np.nanmax(wave_stack):.4f}")
                self.logger.info(f"Error stack min/max: {np.nanmin(error_stack):.4f} / {np.nanmax(error_stack):.4f}")
                
                # Create common header for data extensions
                common_header = fits.Header()
                common_header['CHANNEL'] = channel
                common_header['NFIBERS'] = n_total_fibers
                common_header['NWAVE'] = n_pixels
                
                # Try to determine wavelength range if available
                try:
                    valid_waves = wave_stack[~np.isnan(wave_stack)]
                    valid_count = len(valid_waves)
                    total_count = wave_stack.size
                    valid_percentage = (valid_count / total_count) * 100
                    
                    self.logger.info(f"Channel {channel} has {valid_count} valid wavelength points " +
                                    f"out of {total_count} total points ({valid_percentage:.2f}%)")
                    
                    if valid_count > 0:
                        min_wave = np.min(valid_waves)
                        max_wave = np.max(valid_waves)
                        common_header['WAVEMIN'] = min_wave
                        common_header['WAVEMAX'] = max_wave
                        common_header['COMMENT'] = f'Wavelength range: {min_wave:.2f}-{max_wave:.2f} Angstroms'
                        self.logger.info(f"Channel {channel} wavelength range: {min_wave:.2f}-{max_wave:.2f} Angstroms")
                except Exception as e:
                    self.logger.error(f"Could not determine wavelength range: {str(e)}")
                
                # Extension 1 - SKYSUB: sky-subtracted instrumental spectrum [NFIBER x NWAVE].
                # = COUNTS - SKY when subtract_sky=True and a sky model is present. Named SKYSUB
                # (was FLUX) so it is not confused with the flux-calibrated FLAM extension; the
                # units are instrumental, not physical flux, despite the BUNIT label.
                flux_hdu = fits.ImageHDU(flux_stack, header=common_header)
                flux_hdu.header['EXTNAME'] = 'SKYSUB'
                flux_hdu.header['BUNIT'] = '10^(-17) erg/s/cm2/Ang/fiber'
                flux_hdu.header['SKYSUB'] = subtract_sky and has_sky
                hdul.append(flux_hdu)
                self.logger.info(f"Added SKYSUB extension with shape {flux_stack.shape}")

                # Extension - ERROR: 1-sigma uncertainty per fiber [NFIBER x NWAVE].
                # error_stack is already computed above; emit it so the fibre-flat stage
                # (apply_fibre_flat_to_rss reads hdul['ERROR']) and downstream variance
                # handling have it. The sky-framework RSS refactor had dropped this.
                error_hdu = fits.ImageHDU(error_stack, header=common_header)
                error_hdu.header['EXTNAME'] = 'ERROR'
                error_hdu.header['BUNIT'] = '10^(-17) erg/s/cm2/Ang/fiber'
                hdul.append(error_hdu)
                self.logger.info(f"Added ERROR extension with shape {error_stack.shape}")

                # Extension 2 (optional) - NOFLAT: sky-subtracted, throughput-corrected,
                # but extracted from the un-flat-corrected FITS (no 2D pixel flat applied).
                if noflat_channel_groups.get(channel):
                    nf_obj_list = noflat_channel_groups[channel]
                    all_nf = []
                    for nf_obj in nf_obj_list:
                        nf_counts = nf_obj.counts.copy().astype(np.float32)
                        n_nf = nf_counts.shape[0]
                        nf_tp = getattr(nf_obj, 'relative_throughput', None)
                        if nf_tp is None or len(nf_tp) != n_nf:
                            nf_tp = np.ones(n_nf, dtype=np.float32)
                        else:
                            nf_tp = np.array(nf_tp, dtype=np.float32)
                            bad = ~np.isfinite(nf_tp) | (nf_tp <= 0)
                            nf_tp[bad] = 1.0
                        nf_sky = getattr(nf_obj, 'sky', None)
                        if nf_sky is None or nf_sky.shape != nf_counts.shape:
                            nf_sky = np.zeros_like(nf_counts)

                        # Dead fibres are NOT removed here, for the same reason as in
                        # the main path above: the extraction is LIVE-indexed, so the
                        # dead fibres are already absent and the row layout already
                        # matches FLUX. The old block treated dead_fibers (which holds
                        # FIBREMAP positions) as row indices and dropped whichever row
                        # happened to read as zero -- which, while extractLlamas was
                        # blanking live row 270/49, silently desynced NOFLAT from
                        # SKYSUB by one row per affected camera.

                        nf_tp2d = nf_tp[:, np.newaxis]
                        nf_flux = (nf_counts - nf_sky) / nf_tp2d
                        all_nf.append(nf_flux)
                    if all_nf:
                        noflat_stack = np.vstack(all_nf)
                        # Trim/pad rows to match flux_stack if fiber counts differ
                        n_main = flux_stack.shape[0]
                        if noflat_stack.shape[0] > n_main:
                            noflat_stack = noflat_stack[:n_main, :]
                        elif noflat_stack.shape[0] < n_main:
                            pad = np.full((n_main - noflat_stack.shape[0], noflat_stack.shape[1]),
                                         np.nan, dtype=np.float32)
                            noflat_stack = np.vstack([noflat_stack, pad])
                        noflat_hdu = fits.ImageHDU(noflat_stack, header=common_header)
                        noflat_hdu.header['EXTNAME'] = 'NOFLAT'
                        noflat_hdu.header['BUNIT'] = '10^(-17) erg/s/cm2/Ang/fiber'
                        noflat_hdu.header['SKYSUB'] = subtract_sky and has_sky
                        noflat_hdu.header['COMMENT'] = 'Sky-subtracted, throughput-corrected; NO 2D pixel flat applied'
                        hdul.append(noflat_hdu)
                        self.logger.info(f"Added NOFLAT extension with shape {noflat_stack.shape}")

                # Extension 3 - COUNTS: raw extracted counts before sky subtraction [NFIBER x NWAVE]
                counts_hdu = fits.ImageHDU(counts_stack, header=common_header)
                counts_hdu.header['EXTNAME'] = 'COUNTS'
                counts_hdu.header['BUNIT'] = '10^(-17) erg/s/cm2/Ang/fiber'
                counts_hdu.header['COMMENT'] = 'Raw extracted counts prior to sky subtraction'
                hdul.append(counts_hdu)
                self.logger.info(f"Added COUNTS extension with shape {counts_hdu.data.shape}")

                # Extension 3 - SKY: the sky model for each fiber [NFIBER x NWAVE]
                sky_hdu = fits.ImageHDU(sky_stack, header=common_header)
                sky_hdu.header['EXTNAME'] = 'SKY'
                sky_hdu.header['BUNIT'] = '10^(-17) erg/s/cm2/Ang/fiber'
                sky_hdu.header['SKYSUB'] = subtract_sky
                # F3: sky provenance — fibres whose sky came from a channel-global
                # fallback (NSKYFALL) or that have no sky model at all (NSKYNONE).
                sky_hdu.header['NSKYFALL'] = (n_sky_fallback_total,
                                              'n fibres using channel-global fallback sky')
                sky_hdu.header['NSKYNONE'] = (n_sky_none_total,
                                              'n fibres with no sky model (excl. from science)')
                sky_hdu.header['COMMENT'] = 'Sky model per fiber. FLUX = COUNTS - SKY if SKYSUB=True.'
                hdul.append(sky_hdu)
                self.logger.info(f"Added SKY extension with shape {sky_stack.shape} (subtract_sky={subtract_sky}, has_sky={has_sky})")

                # Extension 4 - MASK: the pixel mask array [NFIBER x NWAVE]
                mask_hdu = fits.ImageHDU(dq_stack, header=common_header)
                mask_hdu.header['EXTNAME'] = 'MASK'
                mask_hdu.header['COMMENT'] = 'Pixel mask: 0=good, >0=various issues'
                hdul.append(mask_hdu)
                self.logger.info(f"Added MASK extension with shape {dq_stack.shape}")

                # Extension 5 - WAVE: the wavelength array for each fiber [NFIBER x NWAVE]
                wave_hdu = fits.ImageHDU(wave_stack, header=common_header)
                wave_hdu.header['EXTNAME'] = 'WAVE'
                wave_hdu.header['BUNIT'] = 'Angstrom'
                if valid_count > 0:
                    wave_hdu.header['WAVESTAT'] = 'GOOD'
                    valid_percentage = (valid_count / total_count) * 100
                    wave_hdu.header['COMMENT'] = f'Wavelength data: {valid_percentage:.2f}% valid points'
                else:
                    wave_hdu.header['WAVESTAT'] = 'POOR'
                    wave_hdu.header['COMMENT'] = 'Warning: Limited or no valid wavelength data'
                hdul.append(wave_hdu)
                self.logger.info(f"Added WAVE extension with shape {wave_stack.shape}")

                # Extension 6 - FWHM: the full width half max array [NFIBER x NWAVE]
                fwhm_hdu = fits.ImageHDU(fwhm_stack, header=common_header)
                fwhm_hdu.header['EXTNAME'] = 'FWHM'
                fwhm_hdu.header['BUNIT'] = 'Pixels'
                hdul.append(fwhm_hdu)
                self.logger.info(f"Added FWHM extension with shape {fwhm_stack.shape}")

                # Plate-solved per-fibre sky coordinates from the (rough header) WCS: fibre-map
                # (x,y) via FiberMap_LUT -> RA/DEC via the fibre-map celestial WCS. Fills FIBERMAP
                # RA/DEC (LVM/other-pipeline compat) and drives the dedicated FIBERWCS extension.
                fiber_ras, fiber_decs, fiber_x, fiber_y, _sky_prov = _fibre_sky_table(
                    fiber_ids, benchsides, primary_hdr)

                # Extension 7 - FIBERMAP: binary table with fiber information
                fibermap_cols = [
                    fits.Column(name='FIBER_ID', format='J', array=np.array(fiber_ids)),
                    fits.Column(name='BENCHSIDE', format='10A', array=np.array(benchsides)),
                    fits.Column(name='FIBER_TYPE', format='10A', array=np.array(fiber_types)),
                    fits.Column(name='RA', format='D', array=np.array(fiber_ras)),
                    fits.Column(name='DEC', format='D', array=np.array(fiber_decs))
                ]

                fibermap_hdu = fits.BinTableHDU.from_columns(fibermap_cols)
                fibermap_hdu.header['EXTNAME'] = 'FIBERMAP'
                hdul.append(fibermap_hdu)
                self.logger.info(f"Added FIBERMAP binary table with {len(fiber_ids)} entries")

                # Extension - FIBERWCS: fibre-map (x,y) + plate-solved RA/DEC + astrometry
                # provenance. Kept separate from FIBERMAP so the astrometry can be re-solved and
                # overwritten in place (Phase 3) without disturbing the base fibre map.
                hdul.append(_fiberwcs_hdu(fiber_ids, benchsides, fiber_ras, fiber_decs,
                                          fiber_x, fiber_y, _sky_prov))
                self.logger.info(f"Added FIBERWCS ({_sky_prov['method']}, "
                                 f"{int(np.isfinite(fiber_ras).sum())} fibres with RA/DEC)")
                
                # Write to file
                hdul.writeto(channel_output_file, overwrite=True)
                self.logger.info(f"RSS file written to: {channel_output_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating RSS file: {str(e)}", exc_info=True)
            raise
        
        return self.new_rss_filenames




#############
def update_ra_dec_in_fits(fits_file, logger=None):
    """
    Update RA and DEC values in a FITS file header using HIERARCH TEL values if needed.
    
    Parameters:
    -----------
    fits_file : str
        Path to the FITS file to update
    logger : Logger, optional
        Logger instance. If None, creates a new one.
    """
    # Set up logging if not provided
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Updating RA/DEC in FITS file: {fits_file}")
    
    try:
        with fits.open(fits_file, mode='update') as hdul:
            primary_hdr = hdul[0].header

            ra = primary_hdr.get('RA')
            dec = primary_hdr.get('DEC')

            missing_ra = (ra is None or str(ra).strip() == "")
            missing_dec = (dec is None or str(dec).strip() == "")

            if missing_ra or missing_dec:
                logger.info(f"Missing RA={missing_ra} or DEC={missing_dec}, attempting to use HIERARCH TEL values")
                tel_ra = primary_hdr.get('HIERARCH TEL RA')
                tel_dec = primary_hdr.get('HIERARCH TEL DEC')

                if tel_ra is None or tel_dec is None:
                    logger.warning("HIERARCH TEL RA and/or HIERARCH TEL DEC not found in primary header; "
                                   "leaving RA/DEC unchanged")
                    hdul.flush()
                    return

                # Convert telescope coordinates from sexagesimal to decimal.
                # TEL RA is in HH:MM:SS.s (hourangle), TEL DEC is in DD:MM:SS.s (degrees).
                c = SkyCoord(ra=str(tel_ra), dec=str(tel_dec), unit=(u.hourangle, u.deg))
                
                logger.info(f"Converted HIERARCH TEL coordinates: RA={c.ra.deg}, DEC={c.dec.deg}")

                if missing_ra:
                    primary_hdr['RA'] = c.ra.deg
                    logger.info(f"Updated RA value to {c.ra.deg}")
                if missing_dec:
                    primary_hdr['DEC'] = c.dec.deg
                    logger.info(f"Updated DEC value to {c.dec.deg}")

            hdul.flush()
            logger.info(f"Primary header updated in {fits_file}")
    except Exception as e:
        logger.error(f"Error updating RA/DEC in FITS file: {str(e)}", exc_info=True)
        raise
