"""
llamas_pyjamas.Masking.cosmicLlamas
===================================
L.A.Cosmic cosmic-ray cleaning, and the per-channel settings users control from the
config (``remove_cosmic_rays``, ``cosmic_ray_channels``, ``cosmic_ray_threshold_{c}``,
``cosmic_ray_contrast_{c}``, ``cosmic_ray_neighbor_threshold_{c}``,
``cosmic_ray_maxiter``). Defaults come from ``constants.LACOSMIC_DEFAULTS`` and
``LACOSMIC_COLOR_OVERRIDES``. See Masking/README.md.
"""

import hashlib
import logging
import os

import numpy as np
from astropy.io import fits
from lacosmic import remove_cosmics

from llamas_pyjamas.constants import LACOSMIC_DEFAULTS, LACOSMIC_COLOR_OVERRIDES, gain_noise_lookup
from llamas_pyjamas.Utils.detectorProps import props_for_header

logger = logging.getLogger("llamas_pyjamas")

CHANNELS = ('red', 'green', 'blue')
#: config name -> L.A.Cosmic parameter, set per channel as cosmic_ray_{name}_{channel}
CHANNEL_PARAMS = {'threshold': 'cr_threshold', 'contrast': 'contrast',
                  'neighbor_threshold': 'neighbor_threshold'}
CONFIG_KEYS = ({'cosmic_ray_channels', 'cosmic_ray_maxiter'}
               | {f'cosmic_ray_{name}_{c}' for name in CHANNEL_PARAMS for c in CHANNELS})
#: cosmic_ray_id of the behaviour before these settings existed (blue and green at
#: 6.5 / 5 / 0.7, red off, maxiter 2). Products without a CRID stamp were made this way.
LEGACY_CRID = 'ca6a2541d387'


def default_settings():
    """Per-channel L.A.Cosmic settings from the constants ('skip' means off).

    ``{'red': {'enabled', 'cr_threshold', 'contrast', 'neighbor_threshold'},
    'green': {...}, 'blue': {...}, 'maxiter': int}``
    """
    out = {'maxiter': int(LACOSMIC_DEFAULTS['maxiter'])}
    for color in CHANNELS:
        override = LACOSMIC_COLOR_OVERRIDES.get(color, {})
        params = dict(LACOSMIC_DEFAULTS, **override)
        out[color] = {'enabled': not override.get('skip', False),
                      **{p: float(params[p]) for p in CHANNEL_PARAMS.values()}}
    return out


def _items(value):
    if isinstance(value, str):
        value = value.split(',')
    if not isinstance(value, (list, tuple)):
        return None
    return [str(v).strip().lower() for v in value if str(v).strip()]


def cosmic_ray_settings(config):
    """The cosmic-ray settings a reduce config asks for: ``(settings, errors, warnings)``.

    Starts from :func:`default_settings` and applies the config keys. Values are
    coerced to numbers (the config parser leaves them as strings); errors describe
    anything that would otherwise be ignored or fail inside an extraction worker.
    """
    settings = default_settings()
    errors, warnings = [], []

    master = config.get('remove_cosmic_rays', True)
    if not isinstance(master, bool):
        errors.append(f"remove_cosmic_rays must be true or false, got {master!r}")

    for key in sorted(config):
        if key.startswith('cosmic_ray_') and key not in CONFIG_KEYS:
            errors.append(f"unknown cosmic-ray setting '{key}'")

    if 'cosmic_ray_channels' in config:
        items = _items(config['cosmic_ray_channels'])
        if items is None:
            errors.append(f"cosmic_ray_channels must list red, green, blue or none, "
                          f"got {config['cosmic_ray_channels']!r}")
            items = []
        unknown = [i for i in items if i not in CHANNELS + ('none',)]
        if unknown:
            errors.append(f"cosmic_ray_channels: unknown channel(s) {', '.join(unknown)}")
        if 'none' in items and len(items) > 1:
            errors.append("cosmic_ray_channels: 'none' cannot be combined with channels")
        for color in CHANNELS:
            settings[color]['enabled'] = color in items

    for color in CHANNELS:
        for name, param in CHANNEL_PARAMS.items():
            key = f'cosmic_ray_{name}_{color}'
            if key not in config:
                continue
            try:
                value = float(config[key])
            except (TypeError, ValueError):
                errors.append(f"{key} must be a number, got {config[key]!r}")
                continue
            if not value > 0:
                errors.append(f"{key} must be positive, got {value:g}")
                continue
            settings[color][param] = value
            if not settings[color]['enabled']:
                warnings.append(f"{key} is set but {color} is not in cosmic_ray_channels")
        if settings[color]['neighbor_threshold'] > settings[color]['cr_threshold']:
            warnings.append(f"{color}: the neighbour threshold "
                            f"({settings[color]['neighbor_threshold']:g}) is above the "
                            f"cosmic-ray threshold ({settings[color]['cr_threshold']:g})")

    if 'cosmic_ray_maxiter' in config:
        try:
            value = float(config['cosmic_ray_maxiter'])
            if value != int(value) or value < 1:
                raise ValueError
            settings['maxiter'] = int(value)
        except (TypeError, ValueError):
            errors.append(f"cosmic_ray_maxiter must be a whole number of at least 1, "
                          f"got {config['cosmic_ray_maxiter']!r}")
    return settings, errors, warnings


def enabled_channels(settings, master_on=True):
    """The channels that are cleaned, in red/green/blue order."""
    if not master_on:
        return []
    return [c for c in CHANNELS if settings[c]['enabled']]


def cosmic_ray_id(settings, master_on=True):
    """Short, stable identity of the cleaning applied: 'off', or a hash of the
    enabled channels' parameters and maxiter (a disabled channel's values don't count).
    Stamped as CRID so a resumed run can tell when the settings have changed."""
    on = enabled_channels(settings, master_on)
    if not on:
        return 'off'
    text = ';'.join(f"{c}:{settings[c]['cr_threshold']:.6g}/{settings[c]['contrast']:.6g}/"
                    f"{settings[c]['neighbor_threshold']:.6g}" for c in on)
    text += f";maxiter:{int(settings['maxiter'])}"
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def describe_settings(settings, master_on=True):
    """One line for the log, e.g. 'blue 6.5/5/0.7, green 6.5/5/0.7; red off (maxiter 2)'."""
    if not master_on:
        return 'off (remove_cosmic_rays = false)'
    on = enabled_channels(settings)
    off = [c for c in CHANNELS if c not in on]
    text = ', '.join(f"{c} {settings[c]['cr_threshold']:g}/{settings[c]['contrast']:g}/"
                     f"{settings[c]['neighbor_threshold']:g}" for c in on) or 'no channel'
    if off:
        text += f"; {', '.join(off)} off"
    return text + f" (maxiter {int(settings['maxiter'])}; threshold/contrast/neighbour)"


CR_STAMPS = ('CRID', 'CRCHAN')


def stamp_cosmic_rays(header, settings, master_on=True):
    """Record the cleaning applied in a primary header: CRID (compared on resume) and CRCHAN."""
    header['CRID'] = (cosmic_ray_id(settings, master_on), 'Cosmic-ray settings identity')
    header['CRCHAN'] = (','.join(enabled_channels(settings, master_on)) or 'none',
                        'Channels cleaned of cosmic rays')


def copy_cosmic_ray_stamps(source, target):
    """Copy CRID/CRCHAN from one header to another (where the source has them)."""
    for key in CR_STAMPS:
        if source is not None and key in source:
            target[key] = (source[key], source.comments[key])


def cosmic_ray_enabled(color, settings=None):
    """True if ``color`` is cleaned under ``settings`` (the defaults when None)."""
    entry = (settings or default_settings()).get(str(color).lower())
    return bool(isinstance(entry, dict) and entry.get('enabled'))


def clean_cosmic_rays(data, color=None, bench=None, side=None, header=None,
                      background=None, mask=None, settings=None):
    """Clean cosmic rays from a 2D detector image using L.A.Cosmic.

    Parameters
    ----------
    data : numpy.ndarray
        2D bias-subtracted detector image.
    color : str, optional
        Detector colour channel ('red', 'green', 'blue').
    bench : str, optional
        Bench identifier ('1', '2', '3', '4').
    side : str, optional
        Side identifier ('A', 'B').
    header : astropy.io.fits.Header, optional
        Extension header.  When provided, the L.A.Cosmic gain/read-noise are
        resolved from the lab-characterisation CSV keyed on the CAMSN serial
        (via :func:`props_for_header`) — the SAME source the extraction error
        model uses — so CR detection and the ivar share one gain/RN table.
        Falls back to the legacy ``gain_noise_lookup`` (color/bench/side) when
        no header is given.
    background : numpy.ndarray, optional
        Signal already subtracted from ``data`` that still contributes Poisson
        noise (the master dark). L.A.Cosmic adds it back before detection, so pass
        a smooth version (``darkLlamas.cosmic_ray_inputs``).
    mask : numpy.ndarray, optional
        Pixels L.A.Cosmic must ignore (True), e.g. hot pixels of the master dark.
    settings : dict, optional
        Per-channel settings (:func:`cosmic_ray_settings`); None uses the defaults.
        A channel that is off returns the data unchanged with an empty mask.

    Returns
    -------
    cleaned : numpy.ndarray
        Cleaned 2D image with cosmic rays replaced.
    mask : numpy.ndarray
        Boolean mask where True indicates a cosmic ray pixel.
    """
    settings = settings or default_settings()
    params = dict(LACOSMIC_DEFAULTS, maxiter=int(settings['maxiter']))

    if color is not None:
        if not cosmic_ray_enabled(color, settings):
            logger.info(f"Skipping cosmic ray cleaning for {color} channel")
            return data.copy(), np.zeros(data.shape, dtype=bool)
        entry = settings[str(color).lower()]
        params.update({p: entry[p] for p in CHANNEL_PARAMS.values()})

    # Prefer the serial-keyed lab CSV (single source of truth, matches the error
    # model); fall back to the legacy color/bench/side lookup.
    if header is not None:
        gain, readnoise, src = props_for_header(
            header, params['effective_gain'], params['readnoise'])
        params['effective_gain'] = gain
        params['readnoise'] = readnoise
        logger.debug(f"L.A.Cosmic gain/RN from {src}: gain={gain}, RN={readnoise}")
    elif color is not None and bench is not None and side is not None:
        key = (color.lower(), str(bench), str(side).upper())
        detector_props = gain_noise_lookup.get(key)
        if detector_props:
            params['effective_gain'] = detector_props['gain']
            params['readnoise'] = detector_props['noise']

    cleaned, mask = remove_cosmics(
        data,
        contrast=params['contrast'],
        cr_threshold=params['cr_threshold'],
        neighbor_threshold=params['neighbor_threshold'],
        effective_gain=params['effective_gain'],
        readnoise=params['readnoise'],
        maxiter=params['maxiter'],
        background=background,
        mask=mask,
    )

    n_cr = int(mask.sum())
    logger.info(f"L.A.Cosmic cleaned {n_cr} cosmic ray pixels")

    return cleaned, mask


def save_cosmic_ray_masks(masks_dict, primary_header, original_filename, output_dir):
    """Save cosmic ray masks as a multi-extension FITS file.

    Parameters
    ----------
    masks_dict : dict
        Mapping of ``{hdu_index: 2D boolean mask array}``.
    primary_header : astropy.io.fits.Header
        Primary header from the original science file.
    original_filename : str
        Path to the original science FITS file (used for naming).
    output_dir : str
        Base output directory. Masks are written to ``{output_dir}/masks/``.

    Returns
    -------
    str
        Path to the saved mask FITS file.
    """
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    basename = os.path.basename(original_filename)
    outfile = os.path.join(masks_dir, f"cosmic_ray_mask_{basename}")

    hdul = fits.HDUList()

    primary_hdu = fits.PrimaryHDU(header=primary_header.copy())
    primary_hdu.header['CRCLEAN'] = (True, 'Cosmic ray masks from L.A.Cosmic')
    hdul.append(primary_hdu)

    for hdu_index in sorted(masks_dict.keys()):
        mask = masks_dict[hdu_index]
        img_hdu = fits.ImageHDU(data=mask.astype(np.uint8))
        img_hdu.header['EXTVER'] = hdu_index
        img_hdu.header['CRNPIX'] = (int(mask.sum()), 'Number of CR pixels in this extension')
        hdul.append(img_hdu)

    hdul.writeto(outfile, overwrite=True)
    logger.info(f"Saved cosmic ray masks to {outfile}")

    return outfile
