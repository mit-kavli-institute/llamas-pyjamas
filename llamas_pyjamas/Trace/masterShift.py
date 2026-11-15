"""Place a mastercalib trace on a night's flat: the fallback when a camera fails to trace.

The fibres move between epochs. From the commissioning to the 2026 masters, red and green
shifted by 0-4.4 px; blue shifted by up to 35 px, with a 0.3-0.5% slit compression,
tilts of up to 16 px across x, and a smooth distortion change that a shift + stretch +
tilt plane misses by 0.2-0.4 px. A bicubic surface in (x, y) fits the motion of every
camera to ~0.03 px. An unshifted master on another night therefore extracts neighbouring
fibres, so a camera's master is moved onto that night's flat instead:

1. **Offset within one pitch.** On a grid of columns x slit segments, the tracer's own
   valley-subtracted comb is summed at the master positions shifted by delta, for delta
   across one pitch. The template is the master *positions*, not its stored flat: the
   flat carries an odd/even throughput pattern that would favour two-fibre aliases.
2. **Plane fit.** ``dy = a + b (y - y0) + c (x - x0)`` is fitted to those offsets modulo
   the pitch, with no unwrapping, so one bad grid point cannot break it. The plane is only
   good to a fraction of a pixel, which is all the next step needs.
3. **Whole-fibre part**, from evidence that does not repeat every pitch:
   - the slit ends: every predicted trace must sit on a fibre, and the first live slot
     beyond each end must be dark (``slit_end_votes``);
   - the throughput fingerprint, which confirms the slit-end answer. By default it must
     confirm it clearly; ``require_fingerprint=False`` only rejects a confident mismatch,
     as the tracer does.
4. **Surface.** With the whole-fibre part fixed, every fibre's offset is measured at each
   column (``offset_table``) and a bicubic surface is fitted to them (``fit_surface``).
5. **Gates**: every fibre must sit on its shifted trace, the profile model is rebuilt on
   the night's flat
   (``profileFit``), the comb check, and ``audit_trace_registration`` against that flat.

``traceLlamasMaster.py`` is not modified, so re-traces still match the code that built
the 2026 masters.
"""
import logging
import os

import numpy as np
from scipy.ndimage import median_filter

from llamas_pyjamas.DataModel.validate import is_placeholder_extension
from llamas_pyjamas.File.llamasIO import process_fits_by_color
from llamas_pyjamas.Trace.traceLlamasMaster import (
    TraceLlamas, audit_trace_registration, measure_trace_heights, normalise_trace_heights,
    validate_trace_comb)

logger = logging.getLogger(__name__)

SWITCH_HINT = ("set shifted_master_require_fingerprint = false in the config to accept "
               "the slit-end match alone")

N_COLUMNS = 15           # measurement columns across the detector
N_SEGMENTS = 6           # slit segments per column
STEP = 0.05              # offset search step (px)
MAX_STRETCH = 0.006      # |b|: blue compresses the slit by up to 0.005
MAX_TILT = 0.025         # |c|: blue tilts by up to 0.01
K_RANGE = range(-10, 11)  # whole-fibre offsets tried (about +-64 px)
MIN_INLIERS = 0.8        # fraction of grid points / fibre offsets a fit must keep
SURFACE_DEGREE = 3       # bicubic: blue's epoch distortion needs the cubic terms
PROFILE_SIGMA = 1.3      # Gaussian template for per-fibre offsets (px)
FIBRE_SEARCH = 1.5       # per-fibre offset search about the plane prediction (px)
MIN_PROMINENCE = 0.2     # 2A dead-299 leak <= 0.12, faintest live fibre >= 0.36
R_MIN = 0.5              # fingerprint correlation needed to confirm or to contradict
STRICT_MARGIN = 0.15     # lag-2 aliases correlate at up to 0.77 x the true r
RELAXED_MARGIN = 0.1     # the tracer's own confident-mismatch margin
EDGE_MARGIN = 3.0        # every trace this far inside the detector (px)
FIBRE_RESID = 0.3        # per-fibre median |offset from its trace| (the tracer's maxdev)
CAMERA_RESID = 0.15      # camera median of those
MIN_VALID = 0.9          # fraction of fibre offsets that must be measurable


class MasterShiftError(RuntimeError):
    """The master could not be placed on the night's flat with confidence.

    ``fingerprint_only`` is True when only the strict fingerprint rule refused it, so
    relaxing ``shifted_master_require_fingerprint`` would let it through.
    """

    def __init__(self, message, fingerprint_only=False):
        super().__init__(message)
        self.fingerprint_only = fingerprint_only


def read_flat_cameras(flat_path, channel=None) -> dict:
    """Read a flat's cameras exactly as the tracer sees them.

    ``process_fits_by_color`` applies the tracer's DATASEC trim and colour flips; it
    writes nothing to disk.

    Returns:
        dict: ``{(channel, bench, side): (data, header, is_placeholder)}``.

    Raises:
        MasterShiftError: the flat cannot be read.
    """
    hdul, _ = process_fits_by_color(flat_path)
    if hdul is None:
        raise MasterShiftError(f"cannot read flat {os.path.basename(str(flat_path))}")

    cameras = {}
    for hdu in hdul[1:]:
        if hdu.data is None:
            continue
        hdr = hdu.header
        if 'COLOR' in hdr:
            cam = (hdr['COLOR'].lower(), str(hdr['BENCH']), str(hdr['SIDE']))
        elif 'CAM_NAME' in hdr:
            name = hdr['CAM_NAME']
            cam = (name.split('_')[1].lower(), name[0], name[1])
        else:
            continue
        if channel is not None and cam[0] != channel.lower():
            continue
        data = np.asarray(hdu.data, dtype=float)
        placeholder = is_placeholder_extension(hdu) or np.nanmin(data) == np.nanmax(data)
        cameras[cam] = (data, dict(hdr), bool(placeholder))
    return cameras


def comb_reader(data, channel):
    """``comb(col)``: the tracer's valley-subtracted cross-section of ``data``, or None."""
    tl = TraceLlamas('comb')
    tl.data = np.asarray(data, dtype=float)
    tl.naxis2, tl.naxis1 = tl.data.shape
    tl.channel = channel
    cache = {}

    def comb(col):
        if col not in cache:
            try:
                cache[col] = np.asarray(tl.find_comb(rownum=col), dtype=float)
            except Exception as e:                                   # noqa: BLE001
                logger.debug("no comb at column %d: %s", col, e)
                cache[col] = None
        return cache[col]
    return comb


def measurement_columns(naxis1, n=N_COLUMNS):
    """Columns spanning the tracer's range (200 to naxis1-100 on a full detector)."""
    lo, hi = min(200, naxis1 // 4), naxis1 - min(100, naxis1 // 4)
    return np.unique(np.linspace(lo, hi, n).round().astype(int))


def best_offset(comb, rows, lo, hi):
    """Offset in [lo, hi) maximising the comb summed at ``rows + offset``, to sub-pixel."""
    deltas = np.arange(lo, hi, STEP)
    y = np.arange(comb.size)
    score = np.interp(rows[None, :] + deltas[:, None], y, comb, left=0, right=0).sum(axis=1)
    i = int(np.argmax(score))
    if 0 < i < deltas.size - 1:
        s0, s1, s2 = score[i - 1], score[i], score[i + 1]
        den = s0 - 2 * s1 + s2
        if den < 0:
            return deltas[i] + 0.5 * STEP * (s0 - s2) / den
    return deltas[i]


def row_nanmedian(a):
    """Median of each row ignoring NaN; NaN for an all-NaN row (without a warning)."""
    a = np.asarray(a, dtype=float)
    out = np.full(a.shape[0], np.nan)
    has = np.isfinite(a).any(axis=1)
    out[has] = np.nanmedian(a[has], axis=1)
    return out


def wrap(v, period):
    """Wrap to [-period/2, period/2)."""
    return (np.asarray(v) + period / 2) % period - period / 2


def segment_grid(comb, master, columns, pitch, n_seg=N_SEGMENTS):
    """Offset of each (column, slit segment) from the master positions, modulo the pitch.

    Returns:
        (x, y, d): column, master row of the segment centre, offset in [-p/2, p/2).
    """
    xs, ys, ds = [], [], []
    for col in columns:
        c = comb(int(col))
        if c is None:
            continue
        m = master[:, col]
        on = (m > pitch) & (m < c.size - 1 - pitch)
        for seg in np.array_split(np.flatnonzero(on), n_seg):
            if seg.size < 5:
                continue
            xs.append(col)
            ys.append(float(np.median(m[seg])))
            ds.append(best_offset(c, m[seg], -pitch / 2 - 1, pitch / 2 + 1))
    return np.asarray(xs, float), np.asarray(ys, float), wrap(ds, pitch)


def offset_table(comb, predicted, columns, pitch):
    """Each fibre's offset from ``predicted``, per measurement column.

    Each offset maximises the comb's correlation with a Gaussian profile within
    FIBRE_SEARCH px of the prediction. The template is symmetric, so unlike a windowed
    centroid it is not pulled by where the fibre falls within a pixel.

    Returns:
        array (n_fibres, n_columns); NaN where the fibre could not be measured.
    """
    deltas = np.arange(-FIBRE_SEARCH, FIBRE_SEARCH + STEP / 2, STEP)
    taps = np.arange(-5, 6)
    table = np.full((predicted.shape[0], len(columns)), np.nan)
    for j, col in enumerate(columns):
        c = comb(int(col))
        if c is None:
            continue
        p = predicted[:, col]
        on = np.flatnonzero((p > pitch) & (p < c.size - 1 - pitch))
        rows = np.rint(p[on]).astype(int)[:, None] + taps[None, :]   # (n, taps)
        values = c[np.clip(rows, 0, c.size - 1)]
        # score[n, k] = sum_t comb(row_t) * G(row_t - p - delta_k)
        offset = rows[:, None, :] - (p[on][:, None, None] + deltas[None, :, None])
        score = np.einsum('nt,nkt->nk', values, np.exp(-0.5 * (offset / PROFILE_SIGMA) ** 2))
        best = np.argmax(score, axis=1)
        top = score[np.arange(on.size), best]
        good = (best > 0) & (best < deltas.size - 1) & (top > 0)
        i = np.flatnonzero(good)
        s0, s1, s2 = score[i, best[i] - 1], top[i], score[i, best[i] + 1]
        den = s0 - 2 * s1 + s2
        delta = deltas[best[i]].astype(float)
        curved = den < 0
        delta[curved] += 0.5 * STEP * (s0 - s2)[curved] / den[curved]
        table[on[i], j] = delta
    return table


def fibre_offsets(comb, master, predicted, columns, pitch):
    """Every fibre's offset from the master, measured near ``predicted``, per column.

    Returns:
        (x, y, d): column, master row, measured offset (one entry per fibre and column).
    """
    table = offset_table(comb, predicted, columns, pitch)
    cols = np.asarray(columns, dtype=int)
    ok = np.isfinite(table)
    x = np.broadcast_to(cols[None, :], table.shape)[ok].astype(float)
    y = master[:, cols][ok]
    d = (predicted[:, cols] - master[:, cols] + np.nan_to_num(table))[ok]
    return x, y, d


def surface_terms(x, y, x0, y0, degree=SURFACE_DEGREE):
    """Design matrix of x^i y^j (i, j <= degree), in units of 1000 px about (x0, y0)."""
    u, v = (np.asarray(x, float) - x0) / 1000.0, (np.asarray(y, float) - y0) / 1000.0
    return np.stack([u ** i * v ** j for i in range(degree + 1) for j in range(degree + 1)],
                    axis=-1)


def fit_surface(x, y, d, x0, y0, degree=SURFACE_DEGREE):
    """3-sigma-clipped least-squares bicubic surface ``d(x, y)``.

    Returns:
        (coef, inliers, rms)
    """
    design = surface_terms(x, y, x0, y0, degree)
    keep = np.ones(d.size, dtype=bool)
    for _ in range(5):
        coef, *_ = np.linalg.lstsq(design[keep], d[keep], rcond=None)
        resid = d - design @ coef
        sigma = max(1.4826 * float(np.median(np.abs(resid[keep]))), 0.02)
        new = np.abs(resid) < 3 * sigma
        if np.array_equal(new, keep) or new.sum() < design.shape[1]:
            break
        keep = new
    return coef, keep, float(np.sqrt(np.mean(resid[keep] ** 2)))


def apply_surface(master, coef, x0, y0, degree=SURFACE_DEGREE):
    """Master positions moved by the surface."""
    x = np.broadcast_to(np.arange(master.shape[1])[None, :], master.shape)
    return master + surface_terms(x, master, x0, y0, degree) @ coef


def fit_plane(x, y, d, x0, y0, pitch=None):
    """Least-squares ``d = a + b (y - y0) + c (x - x0)`` with 3-sigma clipping.

    With ``pitch``, ``d`` is only known modulo the pitch: (b, c) are found by a grid
    search maximising the circular mean resultant length, and the offsets are then
    unwrapped about that solution before the least-squares refinement.

    Returns:
        (a, b, c, inliers)
    """
    dx, dy = x - x0, y - y0
    if pitch is not None:
        z = np.exp(2j * np.pi * d / pitch)
        bs = np.linspace(-MAX_STRETCH, MAX_STRETCH, 121)
        cs = np.linspace(-MAX_TILT, MAX_TILT, 251)
        eb = np.exp(-2j * np.pi * np.outer(bs, dy) / pitch)
        ec = np.exp(-2j * np.pi * np.outer(cs, dx) / pitch)
        resultant = np.abs((eb * z) @ ec.T) / d.size
        ib, ic = np.unravel_index(int(np.argmax(resultant)), resultant.shape)
        b, c = bs[ib], cs[ic]
        a = float(np.angle(np.mean(z * np.exp(-2j * np.pi * (b * dy + c * dx) / pitch)))
                  * pitch / (2 * np.pi))
        model = a + b * dy + c * dx
        d = model + wrap(d - model, pitch)

    design = np.column_stack([np.ones_like(dx), dy, dx])
    keep = np.ones(d.size, dtype=bool)
    for _ in range(5):
        coef, *_ = np.linalg.lstsq(design[keep], d[keep], rcond=None)
        resid = d - design @ coef
        sigma = max(1.4826 * float(np.median(np.abs(resid[keep]))), 0.05)
        new = np.abs(resid) < 3 * sigma
        if np.array_equal(new, keep) or new.sum() < 3:
            break
        keep = new
    return float(coef[0]), float(coef[1]), float(coef[2]), keep


def apply_plane(master, a, b, c, x0, y0):
    """Master positions moved by the plane."""
    x = np.arange(master.shape[1])[None, :]
    return master + a + b * (master - y0) + c * (x - x0)


def prominence(comb, rows, pitch):
    """Peak at ``rows`` minus the mean of the valleys half a pitch either side."""
    y = np.arange(comb.size)

    def sample(r, offsets, reduce):
        return reduce(np.interp(r[:, None] + offsets[None, :], y, comb), axis=1)

    near = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    peak = sample(rows, near, np.max)
    below = sample(rows - pitch / 2, near, np.min)
    above = sample(rows + pitch / 2, near, np.min)
    return peak - 0.5 * (below + above)


def beyond_slots(n_live, dead_fibers):
    """Slots stepped outward from each slit end to the first live (or off-slit) slot."""
    dead = {int(s) for s in dead_fibers}
    live = [s for s in range(n_live + len(dead)) if s not in dead]
    lo = next(s for s in range(live[0] - 1, -len(dead) - 2, -1) if s not in dead)
    hi = next(s for s in range(live[-1] + 1, live[-1] + len(dead) + 2) if s not in dead)
    return live[0] - lo, hi - live[-1]


def slit_end_votes(comb, columns, traces, pitch, ny, steps_out):
    """Whether this placement of the comb agrees with the slit's ends.

    Every on-chip trace must sit on a fibre (relative prominence >= MIN_PROMINENCE),
    and the first live slot beyond each end must be dark. An end within a pitch of the
    detector edge has no vote.

    Returns:
        (ok, detail): ok is False on any violation; detail says which.
    """
    n = traces.shape[0]
    prom = np.full((n, len(columns)), np.nan)
    beyond = {'low': [], 'high': []}
    for j, col in enumerate(columns):
        c = comb(int(col))
        if c is None:
            continue
        rows = traces[:, col]
        on = (rows > pitch) & (rows < ny - 1 - pitch)
        prom[on, j] = prominence(c, rows[on], pitch)
        for end, i, sign, steps in (('low', 0, -1, steps_out[0]),
                                    ('high', n - 1, 1, steps_out[1])):
            end_pitch = float(np.median(np.abs(np.diff(rows[:6] if end == 'low' else rows[-6:]))))
            r = rows[i] + sign * steps * end_pitch
            if pitch < r < ny - 1 - pitch:
                beyond[end].append(float(prominence(c, np.array([r]), pitch)[0]))

    per_fibre = row_nanmedian(prom)
    scored = np.isfinite(per_fibre)
    if scored.sum() < 0.5 * n:
        return False, "most traces fall off the detector"
    # Reference: a running median, mirrored at the ends ('nearest' would pad a dark end
    # with itself and make it its own reference), floored at a fraction of the slit
    # median so a long dark run at an end cannot become its own reference either.
    filled = np.where(scored, per_fibre, np.nanmedian(per_fibre))
    local = np.maximum(median_filter(filled, size=31, mode='mirror'),
                       0.3 * float(np.median(filled)))
    rel = per_fibre / np.where(local > 0, local, np.nan)
    empty = np.flatnonzero(scored & ~(rel >= MIN_PROMINENCE))
    if empty.size:
        return False, f"{empty.size} trace(s) on no fibre (first at trace {int(empty[0])})"

    for end, idx in (('low', slice(0, 15)), ('high', slice(-15, None))):
        if len(beyond[end]) < len(columns) // 2:
            continue                                     # too close to the edge to vote
        ref = float(np.nanmedian(per_fibre[idx]))
        level = float(np.median(beyond[end])) / ref if ref > 0 else np.inf
        if level >= MIN_PROMINENCE:
            return False, f"an unclaimed fibre beyond the {end} end ({level:.2f} of its neighbours)"
    return True, ""


def fingerprint_r(data, traces, fingerprint):
    """Correlation of the fibre throughput at ``traces`` with the reference pattern."""
    if fingerprint is None or len(fingerprint) != traces.shape[0]:
        return np.nan
    mid = traces.shape[1] // 2
    order = np.argsort(traces[:, mid])
    heights = measure_trace_heights(data, traces[order, mid], mid)
    if np.isfinite(heights).sum() < 0.5 * heights.size:
        return np.nan
    pattern = normalise_trace_heights(heights)
    return float(np.corrcoef(pattern, np.asarray(fingerprint, float))[0, 1])


def decide_whole_fibre(votes, rs, require_fingerprint):
    """Pick the whole-fibre offset: the slit ends decide, the fingerprint confirms.

    Args:
        votes: ``{k: (ok, detail)}`` from ``slit_end_votes``.
        rs: ``{k: fingerprint r}`` (NaN when unavailable).
        require_fingerprint: strict rule (confirmation needed) or the tracer's rule.

    Returns:
        (k, r_k, r_next, k_next)

    Raises:
        MasterShiftError
    """
    passing = [k for k, (ok, _) in votes.items() if ok]
    if len(passing) != 1:
        if not passing:
            near = votes.get(min(votes, key=lambda k: abs(k)))
            raise MasterShiftError(
                f"no placement agrees with the slit ends (nearest: {near[1]})")
        raise MasterShiftError(
            f"the slit ends allow {len(passing)} placements ({passing} fibres); "
            f"both ends are too close to the detector edge to decide")
    k = passing[0]
    r_k = rs.get(k, np.nan)
    others = {j: r for j, r in rs.items() if j != k and np.isfinite(r)}
    k_next = max(others, key=others.get) if others else None
    r_next = others[k_next] if others else np.nan

    contradicted = (np.isfinite(r_next) and r_next >= R_MIN and
                    r_next - (r_k if np.isfinite(r_k) else -1.0) >= RELAXED_MARGIN)
    if contradicted:
        raise MasterShiftError(
            f"the fingerprint contradicts the slit ends: r={r_next:.2f} at {k_next - k:+d} "
            f"fibres vs {r_k:.2f} at the slit-end placement")
    if require_fingerprint:
        confirmed = (np.isfinite(r_k) and r_k >= R_MIN and
                     (not np.isfinite(r_next) or r_k - r_next >= STRICT_MARGIN))
        if not confirmed:
            detail = ("no fingerprint for this camera" if not np.isfinite(r_k) else
                      f"r={r_k:.2f} at the slit-end placement, next best {r_next:.2f} "
                      f"at {k_next - k:+d} fibres")
            raise MasterShiftError(
                f"the fingerprint does not confirm the slit-end match ({detail}); "
                f"{SWITCH_HINT}", fingerprint_only=True)
    return k, r_k, r_next, k_next


def measure_master_shift(master, data, channel, dead_fibers=(), fingerprint=None,
                         require_fingerprint=True, k_candidates=K_RANGE):
    """Move ``master`` traces onto the fibres of ``data`` (plane, whole-fibre offset, surface).

    Args:
        master: master trace positions, (n_fibres, naxis1), in detector order.
        data: the night's flat for this camera, oriented as the tracer sees it.
        channel: colour, for the tracer's comb.
        dead_fibers: this benchside's dead fibremap slots.
        fingerprint: reference throughput pattern (one value per live slot), or None.
        require_fingerprint: strict confirmation (default) or the tracer's rule.
        k_candidates: whole-fibre offsets to consider.

    Returns:
        (traces, record): shifted traces and a dict describing the fit.

    Raises:
        MasterShiftError
    """
    master = np.asarray(master, dtype=float)
    data = np.asarray(data, dtype=float)
    ny, nx = data.shape
    if master.shape[1] != nx:
        raise MasterShiftError(f"the flat has {nx} columns, the master traces {master.shape[1]}")
    master = master[np.argsort(master[:, nx // 2])]
    pitch = float(np.median(np.diff(master[:, nx // 2])))
    x0, y0 = nx / 2.0, ny / 2.0
    columns = measurement_columns(nx)
    comb = comb_reader(data, channel)

    # 1-2. Offset within one pitch, then the plane, modulo the pitch.
    x, y, d = segment_grid(comb, master, columns, pitch)
    n_grid = len(columns) * N_SEGMENTS
    if d.size < MIN_INLIERS * n_grid:
        raise MasterShiftError(f"only {d.size} of {n_grid} grid points measurable")
    a, b, c, keep = fit_plane(x, y, d, x0, y0, pitch=pitch)
    if keep.sum() < MIN_INLIERS * n_grid:
        raise MasterShiftError(
            f"the comb offset does not follow a shift + stretch + tilt "
            f"({keep.sum()} of {n_grid} grid points fit)")

    # 3. Whole-fibre part.
    steps_out = beyond_slots(master.shape[0], dead_fibers)
    votes, rs = {}, {}
    for k in k_candidates:
        traces_k = apply_plane(master, a + k * pitch, b, c, x0, y0)
        votes[k] = slit_end_votes(comb, columns, traces_k, pitch, ny, steps_out)
        rs[k] = fingerprint_r(data, traces_k, fingerprint)
    k, r_k, r_next, k_next = decide_whole_fibre(votes, rs, require_fingerprint)
    a += k * pitch

    # 4. Every fibre's offset about the chosen placement, and a bicubic surface.
    predicted = apply_plane(master, a, b, c, x0, y0)
    for _ in range(2):
        x, y, d = fibre_offsets(comb, master, predicted, columns, pitch)
        n_meas = len(columns) * master.shape[0]
        if d.size < MIN_INLIERS * n_meas:
            raise MasterShiftError(f"only {d.size} of {n_meas} fibre offsets measurable")
        coef, keep, rms = fit_surface(x, y, d, x0, y0)
        if keep.sum() < MIN_INLIERS * d.size:
            raise MasterShiftError(
                f"the fibre offsets do not follow a smooth surface ({keep.sum()} of "
                f"{d.size} fit)")
        predicted = apply_surface(master, coef, x0, y0)
    traces = predicted

    def offset_at(xx, yy):
        return float(surface_terms(xx, yy, x0, y0) @ coef)

    bottom, top = float(master[:, nx // 2].min()), float(master[:, nx // 2].max())
    record = {
        'shift': offset_at(x0, y0),
        'stretch_px': offset_at(x0, top) - offset_at(x0, bottom),
        'tilt_px': offset_at(float(columns[-1]), y0) - offset_at(float(columns[0]), y0),
        'surface': coef.tolist(), 'x0': x0, 'y0': y0, 'degree': SURFACE_DEGREE,
        'plane': {'shift': a, 'stretch': b, 'tilt': c},
        'pitch': pitch, 'whole_fibres': int(k), 'fibre_offsets': int(d.size),
        'surface_rms': rms, 'surface_inliers': float(keep.mean()),
        'fingerprint_r': None if not np.isfinite(r_k) else float(r_k),
        'fingerprint_next': None if not np.isfinite(r_next) else float(r_next),
        'fingerprint_next_offset': None if k_next is None else int(k_next - k),
        'require_fingerprint': bool(require_fingerprint),
    }
    gate_record = check_shifted_traces(comb, columns, traces, pitch, ny)
    record.update(gate_record)
    return traces, record


def check_shifted_traces(comb, columns, traces, pitch, ny):
    """Gate the shifted traces against the night's fibres.

    Checked over the columns the tracer itself traces; beyond them the traces are
    extrapolated, as they are in every trace pickle.

    Raises:
        MasterShiftError: a trace is off the detector, or the fibres do not sit on the
            traces (per-fibre offsets measured as in ``offset_table``).
    """
    used = traces[:, int(columns[0]):int(columns[-1]) + 1]
    if used.min() < EDGE_MARGIN or used.max() > ny - 1 - EDGE_MARGIN:
        raise MasterShiftError(
            f"shifted traces leave the detector between columns {columns[0]} and "
            f"{columns[-1]} (rows {used.min():.1f} to {used.max():.1f} of {ny})")
    resid = offset_table(comb, traces, columns, pitch)
    frac_valid = float(np.isfinite(resid).mean())
    per_fibre = row_nanmedian(np.abs(resid))
    worst = float(np.nanmax(per_fibre)) if np.isfinite(per_fibre).any() else np.inf
    camera = float(np.nanmedian(per_fibre)) if np.isfinite(per_fibre).any() else np.inf
    if frac_valid < MIN_VALID:
        raise MasterShiftError(f"only {frac_valid:.0%} of fibres measurable at the shifted "
                               f"positions")
    if worst > FIBRE_RESID or camera > CAMERA_RESID:
        worst_fibre = int(np.nanargmax(per_fibre))
        raise MasterShiftError(
            f"fibres sit off the shifted traces (worst: fibre {worst_fibre}, {worst:.2f} px; "
            f"median {camera:.2f} px; limits {FIBRE_RESID} and {CAMERA_RESID})")
    return {'fibre_median': camera, 'fibre_worst': worst, 'fibre_valid': frac_valid}


def build_shifted_trace(master, data, hdr, channel, bench, side, dead_fibers=(),
                        fingerprint=None, require_fingerprint=True, flat_name='',
                        master_path='', expected_count=None):
    """A current ``TraceLlamas`` with the master moved onto the night's flat.

    The profile model is rebuilt on the night's flat (``profileFit``), and the result
    must pass the comb check and ``audit_trace_registration`` against that flat.

    Returns:
        TraceLlamas, with ``master_shift`` recording the fit.

    Raises:
        MasterShiftError
    """
    channel = channel.lower()
    benchside = f"{bench}{side}"
    traces, record = measure_master_shift(master, data, channel, dead_fibers, fingerprint,
                                          require_fingerprint)
    data = np.asarray(data, dtype=float)
    hdr = dict(hdr or {})

    tl = TraceLlamas(os.path.basename(flat_name) or 'shifted_master')
    tl.fitsfile = os.path.basename(flat_name) or 'shifted_master'
    tl.hdr = hdr
    tl.data = data
    tl.channel, tl.bench, tl.side, tl.benchside = channel, str(bench), str(side), benchside
    tl.naxis2, tl.naxis1 = data.shape
    tl.traces = traces
    tl.nfibers = traces.shape[0]
    tl.bspline_ssets = []
    tl.bias_pending = False
    tl.bias_source = hdr.get('BIASSRC', 'master_bias' if hdr.get('BIASSUB') else 'none')
    tl.bias_level = hdr.get('BIASLVL')
    n_tracefit = max(int((tl.naxis1 - 100 - tl.xmin) // tl.window), 0)
    xtrace = tl.xmin + tl.window * np.arange(n_tracefit)
    tl.tracearr = traces[:, xtrace]
    tl.xtracefit = np.outer(np.ones(tl.nfibers), xtrace)

    if traces.min() < 22:
        logger.warning("shifted master %s %s: a trace reaches row %.0f, inside the rows "
                       "4-20 profileFit uses for its background", benchside, channel,
                       traces.min())
    try:
        tl.profileFit()
    except Exception as e:                                           # noqa: BLE001
        raise MasterShiftError(f"the profile model cannot be fitted on the flat: {e}") from e

    mid = tl.naxis1 // 2
    expected = traces.shape[0] if expected_count is None else int(expected_count)
    tl.comb_ok = validate_trace_comb(traces[:, mid], benchside, dead_fibers=list(dead_fibers),
                                     expected_count=expected)
    if not tl.comb_ok:
        raise MasterShiftError("the shifted comb fails the comb check")
    tl.fingerprint_r = record['fingerprint_r']
    record.update(master=os.path.basename(master_path), flat=os.path.basename(flat_name))
    tl.master_shift = record
    reason = audit_trace_registration(tl, dead_fibers=list(dead_fibers),
                                      fingerprint=fingerprint, expected_count=expected)
    if reason is not None:
        raise MasterShiftError(f"the shifted trace fails the registration check: {reason}")
    return tl


def describe_shift(record) -> str:
    """One line describing a ``master_shift`` record, for the terminal and the log."""
    r = record.get('fingerprint_r')
    fp = 'no fingerprint' if r is None else f"fingerprint r {r:.2f}"
    return (f"{record['shift']:+.2f} px, stretch {record['stretch_px']:+.1f} px, "
            f"tilt {record['tilt_px']:+.1f} px; fibres within {record['fibre_median']:.2f} px; "
            f"{fp}")
