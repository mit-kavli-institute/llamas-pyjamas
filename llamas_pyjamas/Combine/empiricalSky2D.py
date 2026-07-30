"""
llamas_pyjamas.Combine.empiricalSky2D
=====================================
Empirical multi-dither sky subtraction on the BIAS+FLAT-CORRECTED 2D DETECTOR FRAMES (pre-extraction).

Rationale (RS): the post-extraction b-spline model cannot see the sky's true 2D optical footprint —
per-fibre LSF/optics aberrations (worst in the red) and inter-fibre scattered light. Subtracting the
sky on the raw 2D frame, from the dithers where each fibre lands on BLANK sky, matches that footprint
pixel-for-pixel (same detector pixel = same optics), then the frame is re-extracted. OH airglow varies
between dithers, so a residual remains after the overall per-frame sky-level scaling — that residual is
small vs the total sky and is cleaned afterward by the existing b-spline/PCA sky stage on the
re-extracted RSS.

Assumptions:
  * flexure negligible (instrument flexure-compensation) -> the per-dither 2D frames of a field align
    pixel-for-pixel, and a fixed trace/fiberimg maps pixels -> fibres in every dither.
  * blank-vs-source per fibre is decided by self-reference from a light trace-collapse of the 2D frame
    (median of the fibre's own pixels); the faintest dithers = on blank sky.

Per detector (channel,bench,side): flip into trace space, blank-select per fibre, build the per-pixel
leave-one-out median of the blank dithers (scaled to each target frame's overall sky level; inter-fibre
GAP pixels are sky in every dither), subtract, un-flip, write a sky-subtracted 2D MEF for re-extraction.
"""
import logging
import os
import pickle
import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

MIN_BLANK = 3
BLANK_FRAC = 0.6
MIN_FIELD_FRAMES = 5       # below this, borrow nearest-in-time neighbours as extra blank donors
FIELD_SEP_ARCSEC = 180.0   # max pointing offset (from the OBJECT-group median) still in the field
# NB: the pipeline extracts on the RAW HDU orientation with the matching trace (guiExtract builds
# ExtractLlamas(trace, hdu.data) with no flip), so build_field_empirical_sky works in RAW orientation
# using trace.fiberimg/traces directly. FLIP is retained only for the legacy red quick-look path.
FLIP = {'green': 'lr', 'blue': 'udlr', 'red': 'none'}


def _apply_flip(a, color):
    f = FLIP.get(str(color).lower(), 'none')
    if f == 'lr':
        return np.fliplr(a)
    if f == 'udlr':
        return np.flipud(np.fliplr(a))
    return a


def _load_trace(trace_dir, color, bench, side):
    p = os.path.join(trace_dir, f"LLAMAS_{color}_{bench}_{side}_traces.pkl")
    if not os.path.exists(p):
        return None
    return pickle.load(open(p, "rb"))


def build_camera_2d_sky(stack, fiberimg, min_blank=MIN_BLANK, blank_frac=BLANK_FRAC):
    """Per-pixel empirical 2D sky for one detector across a field's dithers (all in trace space).

    stack : (ndith, ny, nx) bias+flat-corrected 2D frames (flexure-aligned).
    fiberimg : (ny, nx) fibre id per pixel (>=0), -1 in inter-fibre gaps.

    Returns emp_sky (ndith, ny, nx) (leave-one-out per target frame), nblank (nfib,), S (ndith,).
    """
    stack = np.asarray(stack, float)
    ndith, ny, nx = stack.shape
    nfib = int(fiberimg.max()) + 1

    # per-(dither,fibre) continuum from a trace-collapse (blank selection self-reference)
    cont = np.full((ndith, nfib), np.nan)
    for f in range(nfib):
        ys, xs = np.where(fiberimg == f)
        if xs.size == 0:
            continue
        vals = stack[:, ys, xs]                             # (ndith, npix_f)
        cont[:, f] = np.nanmedian(vals, axis=1)
    # per-frame overall sky level (median over fibres); robust to a source-minority IFU
    S = np.nanmedian(cont, axis=1)
    S_safe = np.where(np.isfinite(S) & (S > 0), S, np.nan)

    # blank dithers per fibre = faintest n_blank_target
    n_blank_target = min(max(int(min_blank), int(round(blank_frac * ndith))), ndith)
    blank_of = {}                                          # fibre -> list of blank dither indices
    nblank = np.zeros(nfib, dtype=int)
    for f in range(nfib):
        bi = cont[:, f]
        order = np.argsort(np.where(np.isfinite(bi), bi, np.inf))
        blank = [d for d in order[:n_blank_target] if np.isfinite(bi[d]) and np.isfinite(S_safe[d])]
        if len(blank) >= min_blank:
            blank_of[f] = blank
            nblank[f] = len(blank)

    # per-dither blank map projected to pixels (gaps = sky in every dither)
    all_d = [d for d in range(ndith) if np.isfinite(S_safe[d])]
    blankimg = np.zeros((ndith, ny, nx), bool)
    gap = fiberimg < 0
    fid = np.where(gap, 0, fiberimg)
    for d in range(ndith):
        # fibre blank in dither d?
        fibre_blank = np.zeros(nfib, bool)
        for f, bl in blank_of.items():
            fibre_blank[f] = d in bl
        bm = fibre_blank[fid]
        bm[gap] = np.isfinite(S_safe[d])                   # gaps: always sky (when frame usable)
        blankimg[d] = bm

    # DUAL per-frame scaling (RS): the empirical sky is right up to a per-frame amplitude, but that
    # amplitude differs for the OH lines (airglow ~low-dimensional in intensity, ~one brightness scale)
    # vs the less-variable inter-line continuum. So build a leave-one-out sky SHAPE (native-level median
    # of the blank dithers) and fit TWO amplitudes to the target frame's own blank pixels: a_oh on OH
    # pixels, a_cont on continuum pixels. Collapses the OH-variability residual the single scale left.
    OH_FACTOR = 3.0
    emp_sky = np.full_like(stack, np.nan)
    amps = {}
    acc = np.empty((ndith, ny, nx), np.float32)

    def _amp(shape, target, px):
        x = shape[px]; y = target[px]
        m = np.isfinite(x) & np.isfinite(y) & (x != 0)
        if m.sum() < 50:
            return np.nan
        a = float(np.nansum(x[m] * y[m]) / np.nansum(x[m] ** 2))
        return float(np.clip(a, 0.0, 5.0))

    for t in range(ndith):
        if not np.isfinite(S_safe[t]):
            continue
        acc[:] = np.nan
        for d in all_d:
            if d == t:
                continue
            acc[d] = np.where(blankimg[d], stack[d], np.nan)     # native level (no pre-scaling)
        shape = np.nanmedian(acc, axis=0)                         # LOO sky shape @ ~median blank level
        bt = blankimg[t] & np.isfinite(shape) & np.isfinite(stack[t])
        if bt.sum() < 500:                                        # too few blank px: single-scale fallback
            lvl = np.nanmedian(shape[bt]) if bt.any() else np.nan
            emp_sky[t] = shape * (S_safe[t] / lvl) if (lvl and lvl > 0) else shape
            continue
        cont_level = np.nanmedian(shape[bt])
        oh = shape > OH_FACTOR * cont_level                       # OH-line vs continuum pixels
        a_cont = _amp(shape, stack[t], bt & ~oh)
        a_oh = _amp(shape, stack[t], bt & oh)
        if not np.isfinite(a_cont):
            a_cont = 1.0
        if not np.isfinite(a_oh):
            a_oh = a_cont
        amps[t] = (round(a_cont, 3), round(a_oh, 3))
        emp_sky[t] = shape * np.where(oh, a_oh, a_cont)
    logger.info("empiricalSky2D: %d/%d fibres >=%d blank of %d; dual-scale amps (a_cont,a_oh)=%s",
                int((nblank >= min_blank).sum()), nfib, min_blank, ndith, amps)
    return emp_sky, nblank, S


def subtract_field_2d(frame_files, trace_dir, out_dir=None, suffix='_EMPSKY2D',
                      channels=('red', 'green', 'blue'), min_blank=MIN_BLANK, blank_frac=BLANK_FRAC):
    """Build + subtract the 2D empirical sky for a field. frame_files are the field's per-dither
    bias+flat-corrected 2D MEFs (same detector layout). Writes one sky-subtracted MEF per dither
    (SCI = 2D_frame - empirical_sky) for re-extraction. Non-destructive (new files). Returns paths."""
    out_dir = out_dir or os.path.dirname(frame_files[0])
    hduls = [fits.open(f) for f in frame_files]
    ndith = len(hduls)
    # output copies (start from the inputs; we overwrite each extension's data)
    outs = [h[0].header.copy() for h in hduls]
    # accumulate per-extension sky-subtracted data
    sub = [{} for _ in range(ndith)]                       # per dither: extindex -> 2D array
    n_ext = len(hduls[0])
    for i in range(1, n_ext):
        eh = hduls[0][i].header
        color = str(eh.get('COLOR', '')).strip().lower()
        bench = eh.get('BENCH'); side = str(eh.get('SIDE', '')).strip()
        if color not in channels:
            for d in range(ndith):
                sub[d][i] = np.asarray(hduls[d][i].data, float)
            continue
        trace = _load_trace(trace_dir, color, bench, side)
        if trace is None or trace.fiberimg is None:
            logger.warning("empiricalSky2D: no trace for %s_%s_%s; passthrough", color, bench, side)
            for d in range(ndith):
                sub[d][i] = np.asarray(hduls[d][i].data, float)
            continue
        # into trace space (fiberimg space)
        stack = np.stack([_apply_flip(np.asarray(hduls[d][i].data, float), color) for d in range(ndith)])
        emp, nblank, S = build_camera_2d_sky(stack, trace.fiberimg, min_blank, blank_frac)
        for d in range(ndith):
            sky = emp[d]
            skysub = np.where(np.isfinite(sky), stack[d] - sky, stack[d])   # fallback: unchanged where no sky
            sub[d][i] = _apply_flip(skysub, color)          # back to raw space
    written = []
    for d, f in enumerate(frame_files):
        hd = [fits.PrimaryHDU(header=outs[d])]
        for i in range(1, n_ext):
            src = hduls[d][i]
            hdu = fits.ImageHDU(sub[d][i].astype(np.float32), header=src.header.copy())
            hdu.header['EMPSKY2D'] = (True, 'empirical multi-dither 2D sky subtracted')
            hd.append(hdu)
        out = os.path.join(out_dir, os.path.basename(f).replace('.fits', f'{suffix}.fits'))
        fits.HDUList(hd).writeto(out, overwrite=True)
        written.append(out)
    for h in hduls:
        h.close()
    logger.info("empiricalSky2D: wrote %d sky-subtracted 2D frames", len(written))
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline path: field grouping + per-fibre empirical sky seeded into the pkl .sky
# ─────────────────────────────────────────────────────────────────────────────

def _object_of(header):
    """Field name from OBJECT (drop the '_rankNNN' suffix used per pointing)."""
    return str(header.get('OBJECT', '')).split('_rank')[0].strip()


def _pointing_of(header):
    """(ra_deg, dec_deg) from RA/DEC (decimal) with a HIERARCH TEL RA/DEC sexagesimal fallback.
    Mirrors reduce._pointing_from_header without importing the pipeline module."""
    try:
        ra = float(header.get('RA'))
        dec = float(header.get('DEC'))
        if np.isfinite(ra) and np.isfinite(dec):
            return ra, dec
    except (TypeError, ValueError):
        pass
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c = SkyCoord(str(header.get('TEL RA')), str(header.get('TEL DEC')),
                     unit=(u.hourangle, u.deg))
        return float(c.ra.deg), float(c.dec.deg)
    except Exception:                                          # noqa: BLE001
        return np.nan, np.nan


def _mjd_of(header):
    """Observation MJD for time-ordering (UTC MJD-OBS preferred; NaN if absent)."""
    try:
        return float(header.get('MJD-OBS'))
    except (TypeError, ValueError):
        return np.nan


def _sep_arcsec(ra1, dec1, ra2, dec2):
    if not all(np.isfinite(v) for v in (ra1, dec1, ra2, dec2)):
        return np.inf
    d2r = np.pi / 180.0
    dra = (ra1 - ra2) * np.cos(0.5 * (dec1 + dec2) * d2r)
    return float(np.hypot(dra, dec1 - dec2) * 3600.0)


def group_fields(frame_files, min_frames=MIN_FIELD_FRAMES, sep_arcsec=FIELD_SEP_ARCSEC):
    """Group science frames into fields for the empirical multi-dither sky.

    (1) group by OBJECT; (2) within an OBJECT group split off any frame whose pointing is
    > ``sep_arcsec`` from the group's median RA/DEC (distinct pointing reusing the name);
    (3) if a field has < ``min_frames`` members, borrow the nearest-in-time frames from the rest
    of the run as extra blank DONORS (used only to enrich the sky estimate; each donor is still
    subtracted in its own field). Returns a list of dicts {object, members:[...], donors:[...]}.
    """
    info = {}
    for f in frame_files:
        try:
            h = fits.getheader(f, 0)
        except Exception:                                      # noqa: BLE001
            continue
        info[f] = dict(obj=_object_of(h), ra=_pointing_of(h)[0], dec=_pointing_of(h)[1],
                       mjd=_mjd_of(h))
    files = [f for f in frame_files if f in info]

    # (1)+(2): OBJECT, then pointing clusters
    fields = []
    by_obj = {}
    for f in files:
        by_obj.setdefault(info[f]['obj'], []).append(f)
    for obj, group in by_obj.items():
        remaining = list(group)
        while remaining:
            seed = remaining[0]
            ra0, dec0 = info[seed]['ra'], info[seed]['dec']
            members = [g for g in remaining
                       if _sep_arcsec(info[g]['ra'], info[g]['dec'], ra0, dec0) <= sep_arcsec]
            if not members:                                    # no usable pointing -> keep whole group
                members = list(remaining)
            fields.append(dict(object=obj, members=members))
            remaining = [g for g in remaining if g not in members]

    # (3): borrow nearest-in-time neighbours as donors for thin fields
    order = sorted(files, key=lambda f: (info[f]['mjd'] if np.isfinite(info[f]['mjd']) else 1e18))
    for fld in fields:
        donors = []
        need = min_frames - len(fld['members'])
        if need > 0:
            pool = [f for f in order if f not in fld['members']]
            ref = np.nanmedian([info[m]['mjd'] for m in fld['members']])
            pool.sort(key=lambda f: abs((info[f]['mjd'] if np.isfinite(info[f]['mjd']) else 1e18)
                                        - (ref if np.isfinite(ref) else 0.0)))
            donors = pool[:need]
        fld['donors'] = donors
    logger.info("empiricalSky2D.group_fields: %d field(s): %s", len(fields),
                [(f['object'], len(f['members']), len(f['donors'])) for f in fields])
    return fields


def build_field_empirical_sky(member_files, donor_files, trace_dir, method='boxcar',
                              channels=('red', 'green', 'blue'),
                              min_blank=MIN_BLANK, blank_frac=BLANK_FRAC):
    """Per-member empirical sky, EXTRACTED with the pipeline aperture.

    For each camera, stack the field's (member+donor) bias+flat-corrected 2D frames in RAW
    orientation, build the dual-scaled per-pixel empirical 2D sky (build_camera_2d_sky), then for
    each member frame extract that sky with an ExtractLlamas built on the SAME trace and extraction
    ``method`` — so the result is on the member's native-pixel grid, aperture-for-aperture identical
    to its science COUNTS (keeps SKYSUB = COUNTS - SKY exact once seeded).

    Returns {member_file: {(channel, bench_str, side): sky_counts (nfibers, naxis1)}}.
    """
    from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
    seen = []
    for f in list(member_files) + list(donor_files):
        if f not in seen:
            seen.append(f)
    all_files = seen
    hduls = [fits.open(f) for f in all_files]
    member_pos = [all_files.index(m) for m in member_files]
    result = {m: {} for m in member_files}
    try:
        n_ext = len(hduls[0])
        for i in range(1, n_ext):
            eh = hduls[0][i].header
            color = str(eh.get('COLOR', '')).strip().lower()
            if color not in channels:
                continue
            bench = eh.get('BENCH'); side = str(eh.get('SIDE', '')).strip()
            trace = _load_trace(trace_dir, color, bench, side)
            if trace is None or getattr(trace, 'fiberimg', None) is None:
                logger.warning("empiricalSky2D: no trace for %s_%s_%s; field-sky skipped for it",
                               color, bench, side)
                continue
            stack = np.stack([np.asarray(hd[i].data, float) for hd in hduls])   # RAW orientation
            emp, nblank, S = build_camera_2d_sky(stack, trace.fiberimg, min_blank, blank_frac)
            for pos, m in zip(member_pos, member_files):
                ex = ExtractLlamas(trace, np.nan_to_num(emp[pos], nan=0.0),
                                   dict(hduls[pos][i].header), method=method)
                result[m][(color, str(bench), side)] = ex.counts.astype(np.float32)
    finally:
        for hd in hduls:
            hd.close()
    return result


def save_field_empirical_sky(sky_by_member, out_dir, stem_fn):
    """Write one sidecar pkl per member frame: {stem}_empsky2d.pkl holding its
    {(channel, bench, side): sky_counts} dict. ``stem_fn`` maps a frame path -> stem."""
    written = {}
    for m, d in sky_by_member.items():
        p = os.path.join(out_dir, f"{stem_fn(m)}_empsky2d.pkl")
        with open(p, 'wb') as fh:
            pickle.dump({'sky': d}, fh)
        written[m] = p
    return written


def seed_empirical_sky(pkl_file, empsky_pkl):
    """Seed each extraction's .sky with the pre-extracted empirical sky, in place on ``pkl_file``.

    Matches by (channel, bench, side); leaves fibres with no empirical estimate at their prior .sky.
    This is the base for skyModel_1d(residual=True): SKY starts at the empirical sky and every
    downstream stage (b-spline residual, OH line refine, framework) ADDS on top. Returns n cameras seeded.
    """
    from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas, save_extractions
    with open(empsky_pkl, 'rb') as fh:
        emp = pickle.load(fh)['sky']
    d = ExtractLlamas.loadExtraction(pkl_file)
    exts = d['extractions']; meta = d['metadata']
    n = 0
    for i, e in enumerate(exts):
        key = (str(meta[i]['channel']).strip().lower(), str(meta[i]['bench']),
               str(meta[i]['side']).strip())
        arr = emp.get(key)
        if arr is None or getattr(e, 'sky', None) is None:
            continue
        if arr.shape != e.sky.shape:
            logger.warning("empiricalSky2D.seed: shape %s vs .sky %s for %s; skipped",
                           arr.shape, e.sky.shape, key)
            continue
        e.sky = np.asarray(arr, float)
        e.empirical_sky2d = True                              # provenance
        n += 1
    save_extractions(exts, primary_header=d['primary_header'], savefile=pkl_file)
    logger.info("empiricalSky2D.seed: seeded empirical sky into %d/%d cameras of %s",
                n, len(exts), os.path.basename(pkl_file))
    return n
