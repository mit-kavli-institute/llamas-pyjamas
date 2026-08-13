"""
llamas_pyjamas.Sky.skyResidual
==============================
ZAP-style PCA residual cleaning (Soto et al. 2016, adapted to the per-colour RSS
domain).  After base sky subtraction and per-fibre OH scaling, correlated OH-line
residuals remain.  Because real sky residuals are *correlated across many
fibres* while astrophysical signal is not, an eigenbasis built from
sky-dominated fibres isolates the residual sky; projecting the leading
components out of every fibre removes it.

To make "correlated across fibres" meaningful, fibres are first resampled onto a
common wavelength grid (each fibre has its own ``WAVE`` solution), the basis is
built and components projected there, and the per-fibre residual model is mapped
back onto each fibre's native wavelength grid.

Public API
----------
clean_residuals(flux, wave, sky_mask, config) -> (residual_model, info)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _resample_to_grid(flux, wave, grid):
    """Interpolate each fibre onto ``grid``; NaN outside coverage.

    Returns ``(R, valid)`` with shapes ``(n_fiber, len(grid))``.
    """
    n_fiber = flux.shape[0]
    R = np.full((n_fiber, grid.size), np.nan, dtype=float)
    for i in range(n_fiber):
        w = wave[i]
        fi = flux[i]
        m = np.isfinite(w) & np.isfinite(fi)
        if m.sum() < 10:
            continue
        order = np.argsort(w[m])
        R[i] = np.interp(grid, w[m][order], fi[m][order],
                         left=np.nan, right=np.nan)
    return R, np.isfinite(R)


def clean_residuals(flux: np.ndarray, wave: np.ndarray,
                    sky_mask: np.ndarray, config):
    """Remove correlated residual-sky structure via PCA.

    Parameters
    ----------
    flux : np.ndarray
        ``(n_fiber, n_wave)`` OH-scaled, base-sky-subtracted flux.
    wave : np.ndarray
        ``(n_fiber, n_wave)`` per-fibre wavelength solution.
    sky_mask : np.ndarray
        ``(n_fiber,)`` boolean; True fibres build the eigenbasis.
    config : SkySubtractConfig

    Returns
    -------
    residual_model : np.ndarray
        ``(n_fiber, n_wave)`` correlated residual to subtract, on each fibre's
        native wavelength grid.  Zero where a fibre has no usable coverage.
    info : dict
        Diagnostics: ``singular_values``, ``ncomp``, ``n_basis``, ``grid``.
    """
    n_fiber, n_wave = flux.shape
    residual_model = np.zeros((n_fiber, n_wave), dtype=np.float32)
    info = {"singular_values": np.array([]), "ncomp": 0, "n_basis": 0,
            "grid": None}

    has_signal = np.array([np.any(np.isfinite(flux[i])) for i in range(n_fiber)])
    basis_rows = np.where(sky_mask & has_signal)[0]
    if basis_rows.size < 3:
        logger.warning("skyResidual: only %d basis fibres — skipping PCA",
                       basis_rows.size)
        return residual_model, info

    # Common wavelength grid from the basis fibres.
    wmin = np.nanmin(wave[basis_rows])
    wmax = np.nanmax(wave[basis_rows])
    if not np.isfinite(wmin) or not np.isfinite(wmax) or wmax <= wmin:
        logger.warning("skyResidual: degenerate wavelength range — skipping PCA")
        return residual_model, info
    grid = np.linspace(wmin, wmax, n_wave)
    info["grid"] = grid

    # Resample basis (and later all fibres) onto the grid.
    Rb, Vb = _resample_to_grid(flux[basis_rows], wave[basis_rows], grid)

    # Optional sub-sampling of the basis for speed.
    if config.pca_max_basis_fibers and basis_rows.size > config.pca_max_basis_fibers:
        sel = np.linspace(0, basis_rows.size - 1,
                          config.pca_max_basis_fibers).astype(int)
        Rb, Vb = Rb[sel], Vb[sel]

    # Mean spectrum across valid basis fibres per wavelength = common residual sky.
    counts = Vb.sum(axis=0)
    meanspec = np.where(counts > 0,
                        np.nansum(np.where(Vb, Rb, 0.0), axis=0) / np.maximum(counts, 1),
                        0.0)

    Xc = np.where(Vb, Rb - meanspec, 0.0)
    n_basis = Xc.shape[0]
    ncomp = int(min(config.pca_ncomp, n_basis - 1, n_wave))
    if ncomp < 1:
        logger.warning("skyResidual: ncomp<1 after clamping — skipping PCA")
        return residual_model, info

    try:
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError as e:
        logger.warning("skyResidual: SVD failed (%s) — skipping PCA", e)
        return residual_model, info

    comps = Vt[:ncomp]            # (ncomp, n_wave), orthonormal rows
    info.update(singular_values=S, ncomp=ncomp, n_basis=n_basis)
    logger.info("skyResidual: PCA basis=%d fibres, removing %d components "
                "(top sv=%.3g)", n_basis, ncomp, float(S[0]) if S.size else 0.0)

    # Resample ALL fibres, build model on the grid, map back to native wave.
    Rall, Vall = _resample_to_grid(flux, wave, grid)
    for i in range(n_fiber):
        if not Vall[i].any():
            continue
        r = np.where(Vall[i], Rall[i] - meanspec, 0.0)
        coeffs = comps @ r        # (ncomp,)
        proj = coeffs @ comps     # (n_wave,)
        model_grid = np.where(Vall[i], meanspec + proj, 0.0)
        # Map back onto fibre i's native wavelength grid.
        residual_model[i] = np.interp(wave[i], grid, model_grid,
                                      left=0.0, right=0.0).astype(np.float32)

    return residual_model, info
