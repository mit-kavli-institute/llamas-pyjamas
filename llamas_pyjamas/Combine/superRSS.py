"""
Super-RSS: the substrate for combining a field's exposures (Phase 4).

Combining LLAMAS dithers is done through a single in-memory table that holds EVERY fibre from EVERY
exposure of a field, tagged with its plate-solved sky position. The deliverables (co-added images,
cubes, aperture spectra) are then *views* of this table, so the combine logic lives in one place
and each product just chooses a weighting, a resampling and a unit.

Two design rules shape it:

* **Keep each fibre's NATIVE wavelength solution.** Every fibre has its own WAVE array; we do NOT
  pre-resample onto a common grid here. Resampling (and its flux-conservation cost) is confined to
  the view that needs it and happens ONCE, at the end. Broadband/narrowband images never resample
  spectrally at all -- they sum each fibre over the window in its own frame.
* **Put everything on one photometric system up front.** A per-exposure multiplicative scale
  (variable transparency/weather; see the scaling modules) is applied to flux AND variance
  (var *= scale^2) at build time, so a downstream inverse-variance co-add both rescales and
  down-weights a hazy exposure in one step. ``scales=None`` leaves everything at 1.0.

The table is built ON THE FLY when you combine; it is not written to disk.

Facility note: this is deliberately general. `plane` (calibrated FLAM vs instrumental SKYSUB),
`channels`, and the per-exposure `scales` are all caller choices; the combine engine that reads a
:class:`SuperRSS` picks weighting/units/resampling per product.

Classes
-------
ExposureMeta   Per-exposure provenance (id, exptime, airmass, MJD, applied scale)
ChannelStack   One channel's fibres stacked across all exposures (native spectra)
FibreTable     A collapsed per-fibre scalar table (one value/var per fibre) -- the image view input
SuperRSS       The whole substrate: exposures + per-channel stacks, with band-collapse

Functions
---------
exposure_prefix   RSS path -> the exposure id (stem before ``_RSS_``)
load_exposure     Read one exposure's channel siblings into native per-fibre arrays
build_super_rss   Assemble a field's exposures into a SuperRSS
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from astropy.io import fits

from llamas_pyjamas.Utils.maskedSums import MIN_VALID_FRAC, band_integral, common_coverage

logger = logging.getLogger(__name__)

CHANNELS = ('blue', 'green', 'red')

#: Plane pairs: name -> (flux_ext, err_ext). 'flam' = flux-calibrated, 'skysub' = instrumental
#: (the sky-subtracted plane is SKYSUB, or FLUX in files written before the rename).
_PLANES = {'flam': ('FLAM', 'FLAM_ERR'), 'skysub': ('SKYSUB', 'ERROR')}
PLANES = ('auto',) + tuple(_PLANES)


class MissingFluxCalibrationError(ValueError):
    """``plane='auto'``/``'flam'`` but an exposure has no FLAM/FLAM_ERR in a loaded channel.

    There is no silent fallback to the instrumental SKYSUB plane: an uncalibrated exposure would
    otherwise be combined in counts/s next to (or instead of) calibrated ones. Flux-calibrate it
    (apply_fluxcal / promote_final_rss), or ask for ``plane='skysub'`` explicitly."""


def exposure_prefix(path: str) -> str:
    """The exposure id shared by a frame's channel siblings: the basename before ``_RSS_``."""
    return os.path.basename(path).split('_RSS_')[0]


def combined_dir(rss_paths, create=False):
    """The standard directory for combined products: ``<reduced>/combined``. Exposures live in
    ``<reduced>/extractions``, so combined output is a sibling ``combined/`` beside it (else a
    ``combined/`` subdir next to the inputs). ``create=True`` makes it."""
    d = os.path.dirname(os.path.abspath(rss_paths[0]))
    parent, base = os.path.split(d)
    out = os.path.join(parent, 'combined') if base == 'extractions' else os.path.join(d, 'combined')
    if create:
        os.makedirs(out, exist_ok=True)
    return out


@dataclass
class ExposureMeta:
    """Provenance for one exposure (one dither); its channel planes share this."""
    exposure_id: str
    path: str                     #: the RSS path this was loaded from (detection channel)
    exptime: float
    airmass: float
    mjd: float
    scale: float = 1.0            #: photometric scale applied to flux (var *= scale^2)
    stage: str = ''               #: RSS processing stage loaded (base / ff / ff_skysub)


@dataclass
class ChannelStack:
    """One channel's fibres stacked across every exposure of the field. Native per-fibre WAVE;
    nothing is resampled here. All arrays are row-aligned; row -> a single fibre in one exposure."""
    channel: str
    ra: np.ndarray                # (N,)     plate-solved, deg
    dec: np.ndarray               # (N,)     deg
    wave: np.ndarray              # (N, nw)  native, Angstrom
    flux: np.ndarray              # (N, nw)  scaled
    var: np.ndarray               # (N, nw)  scaled (by scale^2)
    mask: np.ndarray              # (N, nw)  bool, True = reject
    solid_angle: np.ndarray       # (N,)     arcsec^2 per fibre
    exposure: np.ndarray          # (N,)     int index into SuperRSS.exposures

    @property
    def n_fibres(self) -> int:
        return self.ra.shape[0]


@dataclass
class FibreTable:
    """A per-fibre SCALAR table: each fibre collapsed to one value over a wavelength window. This is
    the input to the image co-add (position + value + variance + weight, no spectral axis)."""
    ra: np.ndarray                # (M,)
    dec: np.ndarray               # (M,)
    value: np.ndarray             # (M,)  summed flux in the window
    var: np.ndarray               # (M,)  variance of the sum
    solid_angle: np.ndarray       # (M,)  arcsec^2
    exposure: np.ndarray          # (M,)  exposure index
    channel: np.ndarray           # (M,)  channel name per fibre
    npix: np.ndarray              # (M,)  unmasked pixels that went into the sum

    def __len__(self) -> int:
        return self.ra.shape[0]

    def surface_brightness(self):
        """(value, var) converted to per-arcsec^2 surface brightness (divide by fibre area)."""
        a = self.solid_angle
        return self.value / a, self.var / a ** 2


@dataclass
class SuperRSS:
    """All fibres from all exposures of one field, tagged with sky position, on one photometric
    system. Views (image/cube/spectrum) read this. Built on the fly; not archived."""
    field: str
    plane: str                                     #: 'flam' | 'skysub'
    bunit: str
    exposures: List[ExposureMeta] = field(default_factory=list)
    channels: Dict[str, ChannelStack] = field(default_factory=dict)
    stage: str = ''                                #: common RSS stage, or 'mixed'
    overrides: Dict[str, bool] = field(default_factory=dict)  #: safety checks overridden

    @property
    def n_exposures(self) -> int:
        return len(self.exposures)

    def provenance(self) -> Dict[str, object]:
        """Header cards recording what a product was built from: the data plane, the RSS stage
        and any safety check that was overridden (``OVR_*``), so a cube says what it is."""
        meta = {'PLANE': self.plane, 'RSSSTAGE': self.stage or 'unknown'}
        names = {'header': 'OVR_HDR', 'mixed': 'OVR_MIX', 'nocheck': 'OVR_NCHK'}
        for key, flag in self.overrides.items():
            if flag and key in names:
                meta[names[key]] = True
        return meta

    def n_fibres(self, channels: Optional[Sequence[str]] = None) -> int:
        want = self._resolve_channels(channels)
        return int(sum(self.channels[c].n_fibres for c in want))

    def _resolve_channels(self, channels: Optional[Sequence[str]]) -> List[str]:
        if channels is None:
            return [c for c in CHANNELS if c in self.channels]
        return [c for c in channels if c in self.channels]

    def collapse_band(self, wave_min: float, wave_max: float,
                      channels: Optional[Sequence[str]] = None, *,
                      min_frac: float = MIN_VALID_FRAC) -> FibreTable:
        """Collapse every fibre to its integrated flux over ``[wave_min, wave_max]`` in its OWN
        frame (no cross-fibre resampling), for the image co-add.

        The integral is ``mean(valid) * N`` over the window clipped to the channel's common
        coverage (:func:`~llamas_pyjamas.Utils.maskedSums.band_integral`): a masked pixel never
        counts as zero flux, so a fibre with masked pixels in the window is not biased low (and,
        under ivar weighting, not up-weighted for it). A fibre with less than ``min_frac`` of the
        window valid, or whose coverage misses it, is dropped. Variance is
        ``(N/n)^2 * sum(var_valid)``; ``npix`` is the valid count n."""
        want = self._resolve_channels(channels)
        ras, decs, vals, vars, oms, exps, chs, npix = ([] for _ in range(8))
        for c in want:
            st = self.channels[c]
            common = common_coverage(st.wave, ~st.mask & np.isfinite(st.flux))
            bi = band_integral(st.flux, st.wave, st.mask, wave_min, wave_max, var=st.var,
                               common=common, min_frac=min_frac)
            keep = np.isfinite(bi.value)
            if not keep.any():
                continue
            n = bi.n
            ras.append(st.ra[keep]); decs.append(st.dec[keep])
            vals.append(bi.value[keep]); vars.append(bi.var[keep])
            oms.append(st.solid_angle[keep]); exps.append(st.exposure[keep])
            npix.append(n[keep])
            chs.append(np.full(int(keep.sum()), c))
        if not ras:
            empty = np.array([])
            return FibreTable(empty, empty, empty, empty, empty, empty.astype(int),
                              np.array([], dtype='<U5'), empty.astype(int))
        return FibreTable(
            np.concatenate(ras), np.concatenate(decs), np.concatenate(vals),
            np.concatenate(vars), np.concatenate(oms), np.concatenate(exps).astype(int),
            np.concatenate(chs), np.concatenate(npix).astype(int))

    def mask_bad_fibres(self, *, neg_nsigma=5.0) -> Dict[str, int]:
        """Mask whole fibres that are strong NEGATIVE outliers (broken / badly over-subtracted) as
        no-data (mask=True, var=inf), so they drop out of every view and the coverage map simply
        goes shallower there -- other dithers cover the same sky and fill in. Sky-subtracted flux
        is >=0 + noise, so a strongly-negative fibre is an artifact, not signal. Judged per channel
        on each fibre's robust (median) flux vs the field. Returns {channel: n_masked}."""
        out: Dict[str, int] = {}
        for c, st in self.channels.items():
            good = ~st.mask
            with np.errstate(invalid='ignore', divide='ignore'):
                rob = np.nanmedian(np.where(good, st.flux, np.nan), axis=1)
            finite = np.isfinite(rob)
            if finite.sum() < 10:
                out[c] = 0
                continue
            med = float(np.median(rob[finite]))
            mad = float(np.median(np.abs(rob[finite] - med))) * 1.4826 or 1.0
            bad = finite & (rob < med - neg_nsigma * mad) & (rob < 0)
            st.mask[bad, :] = True
            st.var[bad, :] = np.inf
            out[c] = int(bad.sum())
        if any(out.values()):
            logger.info('masked bad fibres (no-data): %s',
                        ', '.join(f'{c}:{n}' for c, n in out.items() if n))
        return out

    def apply_scales(self, scales: Dict[str, float]) -> None:
        """Apply per-exposure photometric scales in place: flux *= s, var *= s^2, for the scale
        RATIO needed to reach the target (so it composes with any scale applied at build). Records
        the new scale on each :class:`ExposureMeta`. ``scales`` maps exposure_id -> target scale;
        exposures absent from the map are left unchanged."""
        target = np.array([float(scales.get(e.exposure_id, e.scale)) for e in self.exposures])
        current = np.array([e.scale for e in self.exposures])
        factor = np.where(current > 0, target / current, 1.0)
        for st in self.channels.values():
            f = factor[st.exposure]
            st.flux *= f[:, None].astype(st.flux.dtype)
            st.var *= (f ** 2)[:, None]
        for i, e in enumerate(self.exposures):
            e.scale = float(target[i])

    def summary(self) -> str:
        chans = ', '.join(f'{c}:{self.channels[c].n_fibres}' for c in self._resolve_channels(None))
        scales = ', '.join(f'{e.scale:.3f}' for e in self.exposures)
        return (f'SuperRSS[{self.field}] {self.n_exposures} exposure(s), plane={self.plane}, '
                f'stage={self.stage or "?"}, fibres per channel: {chans}; scales: {scales}')


def _read_channel(hdul, flux_ext, err_ext, sky_penalty=0.0):
    """Extract (wave, flux, var, mask, ra, dec, solid_angle) for one channel's fibres.

    RA/DEC come from FIBERWCS (the registered, plate-solved coords) when present, else FIBERMAP.
    A fibre with no valid position (ra<0 / NaN) is fully masked so it never enters a view. Bad or
    non-finite flux/err pixels are masked; variance = err^2.

    ``sky_penalty`` (alpha, 0 disables) inflates the variance where the sky-subtraction residual
    exceeds the formal error: ``var *= 1 + (alpha * SKYRESID/ERROR)^2``. The formal ERROR carries
    sky SHOT noise but not the sky-subtraction SYSTEMATIC (measured ~6x larger at red sky lines),
    so without this an ivar co-add cannot down-weight sky-line wavelengths and their residuals
    stack. The SKYRESID/ERROR ratio is plane-independent (the sensfunc cancels), so this is correct
    for both FLAM and SKYSUB planes."""
    from llamas_pyjamas.File.llamasRSS import bad_pixel_mask
    flux = np.asarray(hdul[flux_ext].data, dtype=np.float32)
    err = np.asarray(hdul[err_ext].data, dtype=np.float32)
    wave = np.asarray(hdul['WAVE'].data, dtype=np.float64)
    # The shared rule (bad MASK bits, zero-count pixels); the fibre-flat informational bits
    # leave a pixel usable.
    mask = bad_pixel_mask(hdul)
    mask |= ~np.isfinite(flux) | ~np.isfinite(err) | (err <= 0) | ~np.isfinite(wave)
    var = np.where(mask, np.inf, err.astype(np.float64) ** 2)

    if sky_penalty and 'SKYRESID' in hdul and 'ERROR' in hdul:
        resid = np.abs(np.asarray(hdul['SKYRESID'].data, dtype=np.float64))
        ierr = np.asarray(hdul['ERROR'].data, dtype=np.float64)
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = np.where((ierr > 0) & np.isfinite(resid), resid / ierr, 0.0)
        var = var * (1.0 + (sky_penalty * ratio) ** 2)    # inf stays inf (masked)

    tab = hdul['FIBERWCS'].data if 'FIBERWCS' in hdul else hdul['FIBERMAP'].data
    ra = np.asarray(tab['RA'], dtype=np.float64)
    dec = np.asarray(tab['DEC'], dtype=np.float64)
    unplaced = ~np.isfinite(ra) | ~np.isfinite(dec) | (ra < 0)
    mask[unplaced, :] = True                          # drop unregistered fibres from every view
    area = float(hdul[0].header.get('FIBAREA', np.pi * (0.75 / 2) ** 2))
    solid = np.full(flux.shape[0], area, dtype=np.float64)
    return wave, flux, var, mask, ra, dec, solid


def _hdr_float(*values, default=np.nan):
    """First of ``values`` that casts to float, else ``default``.

    Pre-TCS/old frames can carry an EMPTY header card (key present, value ``None``); then
    ``hdr.get(key, fallback)`` returns that ``None`` rather than the fallback, and ``float(None)``
    raises. This skips ``None``/blank/non-numeric entries so a missing MJD-OBS/AIRMASS/EXPTIME
    degrades to ``default`` (NaN) instead of crashing the whole combine."""
    for v in values:
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return float(default)


def load_exposure(rss_path, *, plane='auto', channels=None, sky_penalty=0.0, stage=None,
                  check_stages=True, allow_header_mismatch=False, allow_mixed_stage=False):
    """Read one exposure (its channel siblings) into native per-fibre arrays.

    Returns ``(meta, plane, data)``: ``meta`` is an :class:`ExposureMeta` (scale left at 1.0,
    ``stage`` set), ``plane`` the resolved plane name and ``data`` maps channel ->
    dict(wave, flux, var, mask, ra, dec, solid_angle).

    ``plane='auto'`` uses FLAM, and requires FLAM and FLAM_ERR in EVERY loaded channel:
    otherwise :class:`MissingFluxCalibrationError` (no silent fallback to SKYSUB). ``'flam'``
    is the same requirement; ``'skysub'`` is the explicit opt-in to the instrumental plane.

    ``stage`` picks which stage of the exposure to read, per channel
    (:func:`~llamas_pyjamas.File.llamasRSS.final_siblings`): ``'auto'`` the most-corrected one on
    disk, or ``'base'`` / ``'ff'`` / ``'ff_skysub'``; ``None`` reads ``rss_path`` and its
    same-stage channel siblings exactly as given (a cube rebuild from its recorded files).

    ``check_stages`` runs :func:`~llamas_pyjamas.File.llamasRSS.check_stage_consistency` on each
    loaded channel file (registration held by another stage, primary headers that differ between
    stages, duplicate stage files raise). Channels at different stages raise
    :class:`~llamas_pyjamas.File.llamasRSS.MixedStageError` unless ``allow_mixed_stage``.
    ``sky_penalty`` -> see :func:`_read_channel`."""
    from llamas_pyjamas.File.llamasRSS import (MixedStageError, channel_siblings,
                                               check_stage_consistency, final_siblings,
                                               skysub_extname, stage_of)
    if plane not in PLANES:
        raise ValueError(f"plane must be one of {PLANES}, not {plane!r}")
    if stage is None:
        siblings = channel_siblings(rss_path) or {'': rss_path}
    else:
        siblings = (final_siblings(rss_path, stage, allow_mixed=allow_mixed_stage)
                    or {'': os.path.abspath(rss_path)})
    want = [c for c in CHANNELS if c in siblings] if channels is None \
        else [c for c in channels if c in siblings]
    if not want:
        want = list(siblings)

    det = siblings.get('green') or siblings[want[0]]
    with fits.open(det) as hd:
        hdr = hd[0].header
        from llamas_pyjamas.Utils.utils import exposure_time
        exptime = _hdr_float(exposure_time(hdr, default=None))
        airmass = _hdr_float(hdr.get('AIRMASS'), hdr.get('TEL AIRMASS'))
        mjd = _hdr_float(hdr.get('MJD-OBS'))
        if not np.isfinite(mjd):                          # pre-TCS: rebuild MJD from TEL DATE-OBS+UTC
            try:
                from llamas_pyjamas.Utils.waveFrame import _obs_time
                _t = _obs_time(hdr)
                if _t is not None:
                    mjd = float(_t.mjd)
            except Exception:                             # noqa: BLE001 - ordering nicety, never fatal
                pass
    resolved_plane = 'flam' if plane == 'auto' else plane

    stages = {}
    if resolved_plane == 'flam':
        missing = []
        for c in want:
            with fits.open(siblings[c]) as hd:
                if 'FLAM' not in hd or 'FLAM_ERR' not in hd:
                    missing.append(c or os.path.basename(siblings[c]))
        if missing:
            raise MissingFluxCalibrationError(
                f'{exposure_prefix(det)}: no FLAM/FLAM_ERR in channel(s) {missing} of '
                f'{os.path.basename(det)}. Flux-calibrate it (apply_fluxcal, or promote_final_rss '
                f'for a reduction calibrated at another stage), or combine the instrumental plane '
                f'explicitly with plane="skysub" / --plane skysub.')

    # SKYSUB is raw counts (NOT exposure-time normalised); FLAM is already per-second (the sensfunc
    # step divides by exptime). To honour this module's "one photometric system up front" invariant,
    # put SKYSUB on a counts/SECOND system too, so exposures of different exposure time co-add
    # coherently and the per-exposure `scale` is left to carry ONLY transparency (not exptime).
    to_per_second = (resolved_plane == 'skysub') and np.isfinite(exptime) and (exptime > 0)
    if resolved_plane == 'skysub' and not to_per_second:
        logger.warning('%s: SKYSUB plane but no usable exposure time (%.3g); combining in raw counts '
                       '(exposures of different exptime will NOT match)', exposure_prefix(det), exptime)

    data = {}
    for c in want:
        if check_stages:
            check_stage_consistency(siblings[c], allow_header_mismatch=allow_header_mismatch)
        stages[c] = stage_of(siblings[c])
        with fits.open(siblings[c]) as hd:
            if resolved_plane == 'flam':
                flux_ext, err_ext = _PLANES['flam']
            else:
                try:
                    flux_ext, err_ext = skysub_extname(hd), 'ERROR'
                except KeyError:
                    flux_ext, err_ext = 'SKYSUB', 'ERROR'
            if flux_ext not in hd or err_ext not in hd:
                logger.warning('%s %s: no %s/%s plane; channel skipped',
                               exposure_prefix(rss_path), c, flux_ext, err_ext)
                continue
            w, f, v, m, ra, dec, om = _read_channel(hd, flux_ext, err_ext, sky_penalty=sky_penalty)
        if to_per_second:
            f = f / exptime
            v = v / (exptime * exptime)                 # variance of counts/s
        data[c] = dict(wave=w, flux=f, var=v, mask=m, ra=ra, dec=dec, solid_angle=om)

    distinct = sorted(set(stages.values()))
    if len(distinct) > 1 and not allow_mixed_stage:
        raise MixedStageError(f'{exposure_prefix(det)}: channels are at different stages {stages}; '
                              f'combining them mixes flat-fielded and un-flat-fielded data. '
                              f'Override with allow_mixed_stage / --allow-mixed-stage.')
    stage = distinct[0] if len(distinct) == 1 else 'mixed'
    meta = ExposureMeta(exposure_id=exposure_prefix(det), path=det, exptime=exptime,
                        airmass=airmass, mjd=mjd, stage=stage)
    return meta, resolved_plane, data


def build_super_rss(rss_paths, *, plane='auto', channels=None, scales=None, reject_bad_fibres=True,
                    sky_penalty=0.0, stage='auto', check_stages=True, allow_header_mismatch=False,
                    allow_mixed_stage=False):
    """Assemble a field's exposures into a :class:`SuperRSS`.

    ``rss_paths`` may list any channel -- and any stage -- of each exposure and may repeat an
    exposure; exposures are de-duplicated by exposure (long per-stage names and a consolidated
    name are the same exposure) and ordered by MJD. ``stage='auto'`` (default) reads each
    exposure's most-corrected stage on disk, so whichever file was listed, the combine uses the
    final one; a named stage forces it; ``None`` reads the given files exactly (see
    :func:`load_exposure`).
    ``scales`` is an optional ``{exposure_id: scale}`` map (variable-transparency correction);
    missing/None -> 1.0. The scale multiplies flux and scales variance by scale^2, so the substrate
    is on one photometric system and inverse-variance weighting down-weights hazy exposures.

    ``reject_bad_fibres`` (default True) masks strongly-negative (broken / over-subtracted) fibres
    as no-data via :meth:`mask_bad_fibres`, so they leave a shallower coverage rather than negative
    holes in the co-add.

    ``plane``, ``check_stages``, ``allow_header_mismatch`` and ``allow_mixed_stage`` -> see
    :func:`load_exposure`. Exposures at different stages raise
    :class:`~llamas_pyjamas.File.llamasRSS.MixedStageError` unless ``allow_mixed_stage`` (then
    ``SuperRSS.stage == 'mixed'``). Overrides used are recorded in ``SuperRSS.overrides``.

    Returns a SuperRSS holding native per-fibre spectra per channel.
    """
    from llamas_pyjamas.File.llamasRSS import exposure_key
    prefixes, seen = [], set()
    for p in rss_paths:                              # de-dup exposures, keep first path seen
        k = exposure_key(p) if stage is not None else exposure_prefix(p)
        if k not in seen:
            seen.add(k)
            prefixes.append(p)

    loaded = []
    for p in prefixes:
        meta, resolved_plane, data = load_exposure(
            p, plane=plane, channels=channels, sky_penalty=sky_penalty, stage=stage,
            check_stages=check_stages, allow_header_mismatch=allow_header_mismatch,
            allow_mixed_stage=allow_mixed_stage)
        if not data:
            logger.warning('%s: no usable channels; exposure skipped', meta.exposure_id)
            continue
        loaded.append((meta, resolved_plane, data))
    if not loaded:
        raise ValueError('No usable exposures to combine')

    # Name any exposure whose pointing is far (>600") from the block median: mislabelled OBJECT,
    # garbage RA/DEC, or a bogus no-Gaia WCS solve. make_output_grid excludes it from the grid so
    # it can't OOM the co-add, but flag WHICH frame is rogue rather than silently dropping it.
    def _expo_center(data):
        for d in data.values():
            ra = np.asarray(d['ra'], float); dec = np.asarray(d['dec'], float)
            g = np.isfinite(ra) & np.isfinite(dec)
            if g.any():
                return float(np.nanmedian(ra[g])), float(np.nanmedian(dec[g]))
        return np.nan, np.nan
    _centers = np.array([_expo_center(d) for _, _, d in loaded])
    if len(_centers) >= 2 and np.isfinite(_centers).all():
        _rac, _decc = np.nanmedian(_centers[:, 0]), np.nanmedian(_centers[:, 1])
        _cosd = np.cos(np.deg2rad(_decc))
        _sep = np.hypot((_centers[:, 0] - _rac) * _cosd, _centers[:, 1] - _decc) * 3600.0
        for (meta, _, _), s in zip(loaded, _sep):
            if s > 600.0:
                logger.warning('exposure %s pointing is %.0f" (%.2f deg) from the field centre — '
                               'likely mislabelled OBJECT or bad WCS; excluded from the co-add grid',
                               meta.exposure_id, s, s / 3600.0)

    loaded.sort(key=lambda t: (np.nan_to_num(t[0].mjd, nan=np.inf), t[0].exposure_id))
    planes = sorted({pl for _, pl, _ in loaded})
    if len(planes) != 1:                             # cannot happen with the rules above; guard it
        raise ValueError(f'exposures resolved to different planes {planes}; refusing to co-add '
                         f'flux-calibrated and instrumental data')
    plane_resolved = planes[0]
    from llamas_pyjamas.File.llamasRSS import MixedStageError
    stages = sorted({m.stage for m, _, _ in loaded})
    if len(stages) > 1 and not allow_mixed_stage:
        detail = ', '.join(f'{m.exposure_id}:{m.stage}' for m, _, _ in loaded)
        raise MixedStageError(f'exposures are at different stages ({detail}); a co-add of '
                              f'flat-fielded and un-flat-fielded exposures is biased by the flat '
                              f'factors. Override with allow_mixed_stage / --allow-mixed-stage.')
    stage = stages[0] if len(stages) == 1 else 'mixed'
    if stage == 'mixed':
        logger.warning('combining exposures at mixed stages: %s', stages)
    field_name = ''
    with fits.open(loaded[0][0].path) as hd:
        field_name = str(hd[0].header.get('OBJECT', '')).split('_rank')[0]
    bunit = 'erg/s/cm2/Angstrom' if plane_resolved == 'flam' else 'counts/s'

    exposures: List[ExposureMeta] = []
    acc: Dict[str, dict] = {}                        # channel -> lists of per-exposure arrays
    for idx, (meta, _pl, data) in enumerate(loaded):
        sc = 1.0 if scales is None else float(scales.get(meta.exposure_id, 1.0))
        meta.scale = sc
        exposures.append(meta)
        for c, d in data.items():
            a = acc.setdefault(c, dict(ra=[], dec=[], wave=[], flux=[], var=[], mask=[],
                                       solid=[], exp=[]))
            n = d['flux'].shape[0]
            a['ra'].append(d['ra']); a['dec'].append(d['dec']); a['wave'].append(d['wave'])
            a['flux'].append((d['flux'] * sc).astype(np.float32))
            a['var'].append((d['var'] * sc ** 2))
            a['mask'].append(d['mask']); a['solid'].append(d['solid_angle'])
            a['exp'].append(np.full(n, idx, dtype=np.int32))

    channels_out: Dict[str, ChannelStack] = {}
    for c, a in acc.items():
        channels_out[c] = ChannelStack(
            channel=c,
            ra=np.concatenate(a['ra']), dec=np.concatenate(a['dec']),
            wave=np.concatenate(a['wave'], axis=0), flux=np.concatenate(a['flux'], axis=0),
            var=np.concatenate(a['var'], axis=0), mask=np.concatenate(a['mask'], axis=0),
            solid_angle=np.concatenate(a['solid']), exposure=np.concatenate(a['exp']))

    overrides = {k: True for k, v in (('header', allow_header_mismatch),
                                       ('mixed', allow_mixed_stage), ('nocheck', not check_stages))
                 if v}
    sr = SuperRSS(field=field_name, plane=plane_resolved, bunit=bunit,
                  exposures=exposures, channels=channels_out, stage=stage, overrides=overrides)
    if reject_bad_fibres:
        sr.mask_bad_fibres()
    logger.info(sr.summary())
    return sr
