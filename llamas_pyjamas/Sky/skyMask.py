"""
llamas_pyjamas.Sky.skyMask
==========================
Identify sky-dominated fibres in a per-colour RSS so the sky-subtraction
framework can (a) leave object flux out of the PCA eigenbasis and (b) anchor
the per-fibre OH scaling on fibres that are genuinely sky-dominated.

This closes the gap noted in ``skyLlamas.skyModel_1d``'s own docstring: *"the
user must generate a mask to exclude sources from the sky estimation."*

Public API
----------
build_sky_fiber_mask(counts, fibermap, config) -> np.ndarray[bool]
    True where a fibre is treated as sky-dominated.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _decode_column(col):
    """Return a list of stripped python strings from a FITS string column."""
    out = []
    for v in col:
        if isinstance(v, bytes):
            out.append(v.decode().strip())
        else:
            out.append(str(v).strip())
    return out


def white_light(counts: np.ndarray) -> np.ndarray:
    """Per-fibre white-light brightness proxy = nansum over wavelength.

    Parameters
    ----------
    counts : np.ndarray
        ``(n_fiber, n_wave)`` array (the RSS ``COUNTS`` extension is ideal — it
        is throughput-corrected and not sky-subtracted).

    Returns
    -------
    np.ndarray
        ``(n_fiber,)`` white-light flux; non-finite fibres map to ``-inf`` so
        they sort to the faint (rejected) end.
    """
    wl = np.nansum(counts, axis=1)
    wl[~np.isfinite(wl)] = -np.inf
    return wl


def _skymap_mask(wl, finite, fibermap, config, n_fiber):
    """Sky-fibre mask from a user sky map (selection_method='skymap').

    Returns a boolean array, or None if the map / FIBERMAP positions are
    unavailable so the caller can fall back to the white-light cut.
    """
    from llamas_pyjamas.Sky.skySelect import (load_sky_map, fibres_in_sky_region,
                                              select_sky_fibres)

    if not getattr(config, "sky_map_file", None):
        logger.warning("skyMask: selection_method='skymap' but no sky_map_file set")
        return None
    names = getattr(fibermap, "names", []) if fibermap is not None else []
    if fibermap is None or "BENCHSIDE" not in names or "FIBER_ID" not in names:
        logger.warning("skyMask: skymap needs FIBERMAP BENCHSIDE/FIBER_ID columns")
        return None
    try:
        skymap = load_sky_map(config.sky_map_file)
        benchsides = _decode_column(fibermap["BENCHSIDE"])
        fibers = np.asarray(fibermap["FIBER_ID"])
        ra = np.asarray(fibermap["RA"]) if "RA" in names else None
        dec = np.asarray(fibermap["DEC"]) if "DEC" in names else None
        region = fibres_in_sky_region(benchsides, fibers, skymap, ra=ra, dec=dec)
    except Exception as exc:
        logger.error("skyMask: skymap selection failed (%s)", exc, exc_info=True)
        return None
    mask = select_sky_fibres(wl, finite, method="skymap", in_sky_region=region)
    logger.info("skyMask: skymap selected %d/%d fibres", int(mask.sum()), n_fiber)
    return mask


def build_sky_fiber_mask(counts: np.ndarray, fibermap, config) -> np.ndarray:
    """Boolean mask of sky-dominated fibres for one colour.

    Selection logic
    ---------------
    1. If the ``FIBERMAP`` ``FIBER_TYPE`` column explicitly flags ``sky``
       fibres, those are used directly (intersected with the finite/non-dead
       set).
    2. Otherwise fall back to a white-light brightness cut: keep fibres
       fainter than ``sky_fiber_percentile`` (object fibres are the bright
       tail) but brighter than ``bright_reject_percentile`` (drop dead/noisy
       fibres).
    3. ``mask_method == 'none'`` returns every finite fibre.

    Parameters
    ----------
    counts : np.ndarray
        ``(n_fiber, n_wave)`` white-light source (RSS ``COUNTS``).
    fibermap : astropy.io.fits.FITS_rec or None
        The ``FIBERMAP`` binary table; may be ``None``.
    config : SkySubtractConfig

    Returns
    -------
    np.ndarray
        ``(n_fiber,)`` boolean; True => sky-dominated.
    """
    n_fiber = counts.shape[0]
    wl = white_light(counts)
    finite = np.isfinite(wl) & (wl > -np.inf)

    if config.mask_method == "none":
        logger.info("skyMask: mask_method='none' — using all %d finite fibres", finite.sum())
        return finite

    # 1. Explicit sky fibres from FIBER_TYPE, if available and informative.
    if fibermap is not None and "FIBER_TYPE" in getattr(fibermap, "names", []):
        ftypes = np.array(_decode_column(fibermap["FIBER_TYPE"]))
        if ftypes.shape[0] == n_fiber:
            is_sky = np.array([t.lower() == "sky" for t in ftypes])
            if is_sky.any():
                mask = is_sky & finite
                logger.info("skyMask: %d fibres flagged FIBER_TYPE='sky'", mask.sum())
                return mask
            # else: no sky-typed fibres -> fall through to brightness cut

    # 2. Selection method (kept consistent with the base model skyModel_1d).
    #    'middle-third' and 'skymap' map directly; 'dimmest'/'frame'/default
    #    use the faint-population white-light cut below (a broad sky basis the
    #    PCA stage needs — a fixed dimmest-N would be too few for the eigenbasis).
    sm = getattr(config, "selection_method", "dimmest")
    if sm == "middle-third":
        from llamas_pyjamas.Sky.skySelect import select_sky_fibres
        mask = select_sky_fibres(wl, finite, method="middle-third")
        logger.info("skyMask: middle-third selected %d/%d fibres", mask.sum(), n_fiber)
        return mask
    if sm == "skymap":
        mask = _skymap_mask(wl, finite, fibermap, config, n_fiber)
        if mask is not None:
            return mask
        logger.warning("skyMask: skymap unavailable; falling back to white-light cut")
    elif sm == "frame":
        logger.info("skyMask: selection_method='frame' — the dedicated sky frame "
                    "drives the base SKY template; the framework PCA basis still "
                    "masks the science RSS by faint white-light percentile")

    # 3. White-light brightness cut.
    good_wl = wl[finite]
    if good_wl.size == 0:
        logger.warning("skyMask: no finite fibres; returning all-False mask")
        return np.zeros(n_fiber, dtype=bool)

    hi = np.percentile(good_wl, config.sky_fiber_percentile)
    lo = np.percentile(good_wl, config.bright_reject_percentile)
    mask = finite & (wl <= hi) & (wl >= lo)

    if mask.sum() < 3:
        # Degenerate (e.g. nearly flat field) — relax to everything finite.
        logger.warning("skyMask: brightness cut left %d fibres; relaxing to all "
                       "finite (%d)", mask.sum(), finite.sum())
        mask = finite

    logger.info("skyMask: %d/%d fibres selected as sky "
                "(white-light in [%.3g, %.3g])", mask.sum(), n_fiber, lo, hi)
    return mask
