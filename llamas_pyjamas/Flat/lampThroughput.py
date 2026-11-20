"""Lamp-derived high-frequency fibre-to-fibre throughput for LLAMAS.

This module produces the per-fibre throughput correction ``C_j(λ)`` from the
continuum-lamp smooth models (``flat_smooth_models.fits``).  It implements the
*frequency split* that is the core of the IFU-standard flat-fielding rework:

1. Build **one** channel-wide reference ``R(λ)`` from the median over *all*
   fibres of *all 8 bench-sides* (never a per-bench-side reference — that would
   divide out the very cross-bench signal we want to preserve).
2. Form the per-fibre ratio ``ratio_j(λ) = response_j(λ) / R(λ)``.
3. Split ``ratio_j`` **along the spatial/fibre axis** into a smooth large-scale
   component (illumination + bench offsets) and a sharp per-fibre component.
   The lamp keeps **only the sharp per-fibre scatter** → ``C_j(λ)``.  Its smooth
   / bench-level component is screen-illumination-contaminated and is discarded;
   the twilight tie (:mod:`llamas_pyjamas.Flat.twilightTie`) supplies it.

The output FITS reuses the schema written by
:func:`llamas_pyjamas.Flat.fibreFlat._write_corrections_fits`
(``CORRECTION / WAVE / FLAGS / CORRECTION_VAR``) so it is consumed unchanged by
:func:`llamas_pyjamas.Flat.fibreFlat.apply_fibre_flat_to_rss`.  ``CORRECTION_VAR``
is populated from lamp Poisson statistics (no longer a zero placeholder).
"""

import os
import logging

import numpy as np
from scipy.ndimage import median_filter

from llamas_pyjamas.constants import gain_noise_lookup
from llamas_pyjamas.Image.WhiteLightModule import FiberMap_LUT
from llamas_pyjamas.Flat.fibreFlat import (
    _load_smooth_models,
    _write_corrections_fits,
    CORRECTION_CLAMP,
    MIN_REF_FLUX,
)

logger = logging.getLogger(__name__)

# Spatial smoothing window (in fibres) used to isolate the LOW-frequency
# illumination/bench component that the lamp must NOT contribute.  Biased
# deliberately LARGE: over-smoothing leaks lamp illumination into C (applied
# every night), whereas under-smoothing is harmlessly recovered by the twilight
# tie.  Tunable via the ``lamp_smooth_scale`` config key.
DEFAULT_SMOOTH_SCALE_FIBRES = 15

# Quality flag (matches fibreFlat.MASK_UNMAPPED_FIBRE_INTERP semantics): fibre
# absent from FiberMap_LUT, so it cannot be spatially placed → C set to 1.0.
MASK_LAMP_NO_THROUGHPUT = np.int16(2)

# Floor on lamp counts (electrons) when forming the Poisson variance, to avoid
# divide-by-zero in low-signal regions.
_MIN_COUNTS_E = 1.0


def compute_lamp_throughput(smooth_models_file, output_dir, common_wave_grid=None,
                            smooth_scale_fibres=DEFAULT_SMOOTH_SCALE_FIBRES,
                            output_name='lamp_throughput_corrections.fits'):
    """Compute per-fibre high-frequency throughput ``C_j(λ)`` + variance.

    Parameters
    ----------
    smooth_models_file : str
        Path to ``flat_smooth_models.fits`` (absolute per-fibre lamp envelopes;
        verified to retain absolute levels, so the ratio to ``R`` measures
        throughput).
    output_dir : str
        Directory for the output corrections file.
    common_wave_grid : ndarray, optional
        Pre-built common wavelength grid per channel.  If ``None`` (default),
        one is built per channel from the union of fibre wavelength ranges.
    smooth_scale_fibres : int
        Spatial smoothing window (fibres) for the HF/LF split.
    output_name : str
        Basename of the output FITS file.

    Returns
    -------
    corrections_file : str
        Path to the written ``lamp_throughput_corrections.fits``.
    """
    models = _load_smooth_models(smooth_models_file)

    # Group benchside extensions by channel
    channels = {}
    for ext_name, data in models.items():
        channels.setdefault(data['channel'].lower(), []).append(ext_name)

    corrections = {}
    for channel in sorted(channels.keys()):
        ext_names = channels[channel]
        ch_corr = _compute_channel_throughput(
            channel, ext_names, models, smooth_scale_fibres, common_wave_grid)
        corrections.update(ch_corr)

    output_path = os.path.join(output_dir, output_name)
    _write_corrections_fits(
        corrections, output_path, method='lamp_hf',
        header_extra={
            'SMTHFILE': os.path.basename(smooth_models_file),
            'SMSCALE': (int(smooth_scale_fibres),
                        'Spatial smoothing window (fibres)'),
        })
    return output_path


def _build_common_grid(models, ext_names):
    """Channel-wide common wavelength grid from the union of all fibre ranges."""
    all_wave = []
    steps = []
    for ext_name in ext_names:
        wave_arr = models[ext_name]['wave']
        for i in range(wave_arr.shape[0]):
            w = wave_arr[i]
            good = np.isfinite(w) & (w > 0)
            if np.sum(good) > 10:
                all_wave.append(w[good])
                dw = np.median(np.abs(np.diff(w[good])))
                if dw > 0:
                    steps.append(dw)
    if not all_wave or not steps:
        return np.array([])
    w_min = min(w.min() for w in all_wave)
    w_max = max(w.max() for w in all_wave)
    dw_common = np.median(steps)
    n_common = int(np.ceil((w_max - w_min) / dw_common)) + 1
    return np.linspace(w_min, w_max, n_common)


def _spatial_lowfreq(ratio_sorted, window):
    """Low-frequency component of the ratio along the (spatially ordered) fibre axis.

    NaNs are filled with the per-wavelength median before filtering so they do
    not propagate, then a median filter of ``window`` fibres is applied along
    the fibre axis only (one column = one wavelength).
    """
    col_med = np.nanmedian(ratio_sorted, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 1.0)
    filled = np.where(np.isfinite(ratio_sorted), ratio_sorted, col_med[None, :])
    lf = median_filter(filled, size=(int(window), 1), mode='nearest')
    return np.where(np.isfinite(lf) & (lf > 0), lf, 1.0)


def _compute_channel_throughput(channel, ext_names, models, smooth_scale_fibres,
                                common_wave_grid):
    """Compute ``C_j(λ)`` + variance for all bench-sides of one channel.

    Returns a ``{ext_name: corrections_dict}`` mapping ready for
    ``_write_corrections_fits``.
    """
    common_wave = (common_wave_grid if common_wave_grid is not None
                   else _build_common_grid(models, ext_names))
    if len(common_wave) == 0:
        logger.warning(f"Channel {channel}: no valid wavelengths, skipping")
        return {}
    n_common = len(common_wave)

    # ── Gather every fibre across all bench-sides of the channel ──
    records = []        # one dict per fibre: ext_name, local_i, x, y, mapped
    resampled_rows = []  # smooth model resampled onto the common grid
    for ext_name in sorted(ext_names):
        data = models[ext_name]
        bs = f"{data['bench']}{data['side']}"
        smooth_arr = data['smooth']
        wave_arr = data['wave']
        fiber_ids = data['fiber_ids']
        for i in range(smooth_arr.shape[0]):
            w = wave_arr[i]
            s = smooth_arr[i]
            good = np.isfinite(w) & np.isfinite(s) & (w > 0) & (s > 0)
            if np.sum(good) < 10:
                row = np.full(n_common, np.nan)
            else:
                row = np.interp(common_wave, w[good], s[good],
                                left=np.nan, right=np.nan)
            x, y = FiberMap_LUT(bs, int(fiber_ids[i]))
            mapped = not (x == -1 and y == -1)
            records.append({'ext_name': ext_name, 'local_i': i,
                            'x': float(x), 'y': float(y), 'mapped': mapped})
            resampled_rows.append(row)

    resampled = np.vstack(resampled_rows)            # (n_fib, n_common)
    n_fib = resampled.shape[0]

    # ── Channel-wide reference R(λ): median over ALL fibres of ALL bench-sides ──
    R = np.nanmedian(resampled, axis=0)
    R_safe = np.where(np.isfinite(R) & (R > MIN_REF_FLUX), R, np.nan)

    # ── ratio_j(λ) = response_j / R ──
    ratio = resampled / R_safe[None, :]              # nan where R unreliable

    # ── Spatial HF/LF split: order by IFU (y, x), median-filter along fibres ──
    mapped = np.array([r['mapped'] for r in records])
    # Unmapped fibres (y=-1) sort to the front as a contiguous block; they are
    # overridden to C=1 below, so their filter position does not matter.
    sort_keys = np.array([(r['y'], r['x']) for r in records],
                         dtype=[('y', float), ('x', float)])
    order = np.argsort(sort_keys, order=('y', 'x'))
    inv_order = np.argsort(order)

    lf_sorted = _spatial_lowfreq(ratio[order], smooth_scale_fibres)
    lf = lf_sorted[inv_order]

    C = np.where(np.isfinite(ratio), ratio / lf, 1.0)
    C = np.clip(C, CORRECTION_CLAMP[0], CORRECTION_CLAMP[1])
    C[~mapped, :] = 1.0                               # cannot place → no correction

    # ── Scatter per-fibre results back onto each bench-side's NATIVE grid ──
    n_alive = int(np.sum(mapped & np.isfinite(ratio).any(axis=1)))
    gain = gain_noise_lookup.get(
        (channel, models[ext_names[0]]['bench'], models[ext_names[0]]['side']),
        {'gain': 1.25})['gain']

    corrections = {}
    for ext_name in sorted(ext_names):
        data = models[ext_name]
        smooth_arr = data['smooth']
        wave_arr = data['wave']
        n_ext, n_pix = smooth_arr.shape
        corr_native = np.ones((n_ext, n_pix), dtype=np.float64)
        var_native = np.zeros((n_ext, n_pix), dtype=np.float64)
        flags = np.zeros(n_ext, dtype=np.int16)
        corrections[ext_name] = {
            'fiber_ids': data['fiber_ids'],
            'correction': corr_native,
            'wave': wave_arr,
            'channel': channel,
            'bench': data['bench'],
            'side': data['side'],
            'n_ref': n_alive,
            'common_wave': common_wave,
            'reference': np.where(np.isfinite(R), R, 0.0),
            'correction_var': var_native,
            'fibre_flags': flags,
        }

    for k, rec in enumerate(records):
        ext_name = rec['ext_name']
        i = rec['local_i']
        out = corrections[ext_name]
        w_native = models[ext_name]['wave'][i]
        s_native = models[ext_name]['smooth'][i]
        good = np.isfinite(w_native) & (w_native > 0)
        if not rec['mapped'] or np.sum(good) < 10:
            out['fibre_flags'][i] |= MASK_LAMP_NO_THROUGHPUT
            continue
        # Interpolate C from the common grid back onto this fibre's native grid
        c_native = np.interp(w_native, common_wave, C[k], left=1.0, right=1.0)
        c_native = np.clip(c_native, CORRECTION_CLAMP[0], CORRECTION_CLAMP[1])
        out['correction'][i] = c_native
        # Lamp Poisson variance: relative var = 1 / counts_e (reference is a
        # high-S/N median → treated as noiseless).  Var[C] = C^2 / counts_e.
        counts_e = np.where(np.isfinite(s_native) & (s_native > 0),
                            s_native * gain, np.nan)
        counts_e = np.maximum(counts_e, _MIN_COUNTS_E)
        var = c_native ** 2 / counts_e
        out['correction_var'][i] = np.where(np.isfinite(var), var, 0.0)

    for ext_name in ext_names:
        corr = corrections[ext_name]['correction']
        logger.info(f"  {ext_name}: C median={np.nanmedian(corr):.4f} "
                    f"std={np.nanstd(corr):.4f} "
                    f"n_alive_channel={n_alive}")

    return corrections
