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

logger = logging.getLogger(__name__)

CHANNELS = ('blue', 'green', 'red')

#: Plane pairs: name -> (flux_ext, err_ext). 'flam' = flux-calibrated, 'skysub' = instrumental.
_PLANES = {'flam': ('FLAM', 'FLAM_ERR'), 'skysub': ('SKYSUB', 'ERROR')}


def exposure_prefix(path: str) -> str:
    """The exposure id shared by a frame's channel siblings: the basename before ``_RSS_``."""
    return os.path.basename(path).split('_RSS_')[0]


@dataclass
class ExposureMeta:
    """Provenance for one exposure (one dither); its channel planes share this."""
    exposure_id: str
    path: str                     #: the RSS path this was loaded from (detection channel)
    exptime: float
    airmass: float
    mjd: float
    scale: float = 1.0            #: photometric scale applied to flux (var *= scale^2)


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

    @property
    def n_exposures(self) -> int:
        return len(self.exposures)

    def n_fibres(self, channels: Optional[Sequence[str]] = None) -> int:
        want = self._resolve_channels(channels)
        return int(sum(self.channels[c].n_fibres for c in want))

    def _resolve_channels(self, channels: Optional[Sequence[str]]) -> List[str]:
        if channels is None:
            return [c for c in CHANNELS if c in self.channels]
        return [c for c in channels if c in self.channels]

    def collapse_band(self, wave_min: float, wave_max: float,
                      channels: Optional[Sequence[str]] = None) -> FibreTable:
        """Collapse every fibre to its summed flux over ``[wave_min, wave_max]`` in its OWN frame
        (no cross-fibre resampling), for the image co-add. A fibre whose channel/coverage misses
        the window contributes nothing (dropped). Variance adds in quadrature over the window."""
        want = self._resolve_channels(channels)
        ras, decs, vals, vars, oms, exps, chs, npix = ([] for _ in range(8))
        for c in want:
            st = self.channels[c]
            inside = (st.wave >= wave_min) & (st.wave <= wave_max) & (~st.mask)
            n = inside.sum(axis=1)
            keep = n > 0                              # fibres with any good pixel in the window
            if not keep.any():
                continue
            fsum = np.where(inside, st.flux, 0.0).sum(axis=1)
            vsum = np.where(inside, st.var, 0.0).sum(axis=1)
            ras.append(st.ra[keep]); decs.append(st.dec[keep])
            vals.append(fsum[keep]); vars.append(vsum[keep])
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
                f'fibres per channel: {chans}; scales: {scales}')


def _read_channel(hdul, flux_ext, err_ext):
    """Extract (wave, flux, var, mask, ra, dec, solid_angle) for one channel's fibres.

    RA/DEC come from FIBERWCS (the registered, plate-solved coords) when present, else FIBERMAP.
    A fibre with no valid position (ra<0 / NaN) is fully masked so it never enters a view. Bad or
    non-finite flux/err pixels are masked; variance = err^2."""
    flux = np.asarray(hdul[flux_ext].data, dtype=np.float32)
    err = np.asarray(hdul[err_ext].data, dtype=np.float32)
    wave = np.asarray(hdul['WAVE'].data, dtype=np.float64)
    mask = np.zeros(flux.shape, dtype=bool)
    if 'MASK' in hdul:
        mask |= np.asarray(hdul['MASK'].data) != 0
    mask |= ~np.isfinite(flux) | ~np.isfinite(err) | (err <= 0) | ~np.isfinite(wave)
    var = np.where(mask, np.inf, err.astype(np.float64) ** 2)

    tab = hdul['FIBERWCS'].data if 'FIBERWCS' in hdul else hdul['FIBERMAP'].data
    ra = np.asarray(tab['RA'], dtype=np.float64)
    dec = np.asarray(tab['DEC'], dtype=np.float64)
    unplaced = ~np.isfinite(ra) | ~np.isfinite(dec) | (ra < 0)
    mask[unplaced, :] = True                          # drop unregistered fibres from every view
    area = float(hdul[0].header.get('FIBAREA', np.pi * (0.75 / 2) ** 2))
    solid = np.full(flux.shape[0], area, dtype=np.float64)
    return wave, flux, var, mask, ra, dec, solid


def load_exposure(rss_path, *, plane='auto', channels=None):
    """Read one exposure (its channel siblings) into native per-fibre arrays.

    Returns ``(meta, data)`` where ``meta`` is an :class:`ExposureMeta` (scale left at 1.0) and
    ``data`` maps channel -> dict(wave, flux, var, mask, ra, dec, solid_angle). ``plane='auto'``
    uses FLAM if the detection channel carries it, else SKYSUB."""
    from llamas_pyjamas.CubeViewer.cubeViewRSS import channel_siblings
    siblings = channel_siblings(rss_path) or {'': rss_path}
    want = [c for c in CHANNELS if c in siblings] if channels is None \
        else [c for c in channels if c in siblings]
    if not want:
        want = list(siblings)

    det = siblings.get('green') or siblings[want[0]]
    with fits.open(det) as hd:
        hdr = hd[0].header
        if plane == 'auto':
            resolved_plane = 'flam' if 'FLAM' in hd else 'skysub'
        else:
            resolved_plane = plane
        exptime = float(hdr.get('SEXPTIME', hdr.get('REXPTIME', hdr.get('EXPTIME', np.nan))))
        airmass = float(hdr.get('AIRMASS', hdr.get('TEL AIRMASS', np.nan)))
        mjd = float(hdr.get('MJD-OBS', np.nan))
    flux_ext, err_ext = _PLANES[resolved_plane]

    data = {}
    for c in want:
        with fits.open(siblings[c]) as hd:
            if flux_ext not in hd or err_ext not in hd:
                logger.warning('%s %s: no %s/%s plane; channel skipped',
                               exposure_prefix(rss_path), c, flux_ext, err_ext)
                continue
            w, f, v, m, ra, dec, om = _read_channel(hd, flux_ext, err_ext)
        data[c] = dict(wave=w, flux=f, var=v, mask=m, ra=ra, dec=dec, solid_angle=om)

    meta = ExposureMeta(exposure_id=exposure_prefix(det), path=det, exptime=exptime,
                        airmass=airmass, mjd=mjd)
    return meta, resolved_plane, data


def build_super_rss(rss_paths, *, plane='auto', channels=None, scales=None, reject_bad_fibres=True):
    """Assemble a field's exposures into a :class:`SuperRSS`.

    ``rss_paths`` may list any channel of each exposure (siblings are found automatically) and may
    repeat an exposure; exposures are de-duplicated by :func:`exposure_prefix` and ordered by MJD.
    ``scales`` is an optional ``{exposure_id: scale}`` map (variable-transparency correction);
    missing/None -> 1.0. The scale multiplies flux and scales variance by scale^2, so the substrate
    is on one photometric system and inverse-variance weighting down-weights hazy exposures.

    ``reject_bad_fibres`` (default True) masks strongly-negative (broken / over-subtracted) fibres
    as no-data via :meth:`mask_bad_fibres`, so they leave a shallower coverage rather than negative
    holes in the co-add.

    Returns a SuperRSS holding native per-fibre spectra per channel.
    """
    prefixes, seen = [], set()
    for p in rss_paths:                              # de-dup exposures, keep first path seen
        k = exposure_prefix(p)
        if k not in seen:
            seen.add(k)
            prefixes.append(p)

    loaded = []
    for p in prefixes:
        meta, resolved_plane, data = load_exposure(p, plane=plane, channels=channels)
        if not data:
            logger.warning('%s: no usable channels; exposure skipped', meta.exposure_id)
            continue
        loaded.append((meta, resolved_plane, data))
    if not loaded:
        raise ValueError('No usable exposures to combine')

    loaded.sort(key=lambda t: (np.nan_to_num(t[0].mjd, nan=np.inf), t[0].exposure_id))
    plane_resolved = loaded[0][1]
    field_name = ''
    with fits.open(loaded[0][0].path) as hd:
        field_name = str(hd[0].header.get('OBJECT', '')).split('_rank')[0]
    bunit = 'erg/s/cm2/Angstrom' if plane_resolved == 'flam' else 'counts'

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

    sr = SuperRSS(field=field_name, plane=plane_resolved, bunit=bunit,
                  exposures=exposures, channels=channels_out)
    if reject_bad_fibres:
        sr.mask_bad_fibres()
    logger.info(sr.summary())
    return sr
