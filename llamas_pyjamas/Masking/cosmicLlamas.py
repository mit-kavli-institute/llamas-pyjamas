import os
import logging

import numpy as np
from astropy.io import fits
from lacosmic import remove_cosmics

from llamas_pyjamas.constants import LACOSMIC_DEFAULTS, LACOSMIC_COLOR_OVERRIDES, gain_noise_lookup

logger = logging.getLogger("llamas_pyjamas")


def clean_cosmic_rays(data, color=None, bench=None, side=None):
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

    Returns
    -------
    cleaned : numpy.ndarray
        Cleaned 2D image with cosmic rays replaced.
    mask : numpy.ndarray
        Boolean mask where True indicates a cosmic ray pixel.
    """
    params = dict(LACOSMIC_DEFAULTS)

    if color is not None:
        color_overrides = LACOSMIC_COLOR_OVERRIDES.get(color.lower(), {})
        if color_overrides.get('skip', False):
            logger.info(f"Skipping cosmic ray cleaning for {color} channel")
            return data.copy(), np.zeros(data.shape, dtype=bool)
        params.update(color_overrides)

    if color is not None and bench is not None and side is not None:
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
