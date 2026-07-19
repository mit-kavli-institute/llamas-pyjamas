"""
SpectralScene — the geometry-agnostic data abstraction behind CubeViewer.

The GUI never learns what a fibre is. It holds a :class:`SpectralScene`, asks it to collapse
a wavelength window into an image, and asks it what spectra live at a pixel. Fibres, spaxels,
hexagons and dithered mosaics are all details of a scene implementation.

The rule that keeps this honest: **every scene returns an** :class:`astropy.wcs.WCS`, always.
Today the RSS scene has no per-fibre astrometry (the RSS ``FIBERMAP`` RA/DEC columns are
written as NaN), so it synthesises a *linear* WCS in fibre-map units. A cube carries a real
``RA---TAN``/``DEC--TAN`` WCS. Both satisfy the same interface, so when per-fibre sky
coordinates are eventually populated the RSS scene swaps its linear WCS for a TAN one and
nothing above it changes. That is what makes registering and combining dithered exposures a
new scene rather than a rewrite.

:meth:`SpectralScene.spectra_at` returns a *list*, never a single spectrum. One fibre today;
several weighted fibres once aperture extraction lands; one spaxel for a cube; a resampled
stack for a mosaic. The plotting panel already handles all of these because it only ever sees
a list.

Classes
-------
Spectrum        One spectrum: wavelength, flux, provenance and weight
SpectralScene   Abstract interface implemented by cubeViewRSS and cubeViewCube

Functions
---------
linear_wcs      Build a linear (non-sky) WCS from CRPIX/CRVAL/CDELT keys
combine         Weighted combination of spectra sharing a channel
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
from astropy.wcs import WCS

logger = logging.getLogger(__name__)

# Channel plotting order and colours, used by the spectrum panel.
CHANNEL_ORDER = ('blue', 'green', 'red')
CHANNEL_COLOURS = {'blue': '#3B6FD4', 'green': '#2E9B57', 'red': '#C0392B'}


@dataclass
class Spectrum:
    """One extracted spectrum.

    Attributes
    ----------
    wave : ndarray
        Wavelength, Angstrom. Native sampling — RSS stores a per-fibre wavelength array, so
        no common grid or resampling is imposed.
    flux : ndarray
        Flux, same length as `wave`.
    channel : str
        'blue', 'green' or 'red'.
    label : str
        Human-readable provenance, e.g. "2B fibre 137".
    weight : float
        Contribution weight, for aperture combination. 1.0 for a single element.
    mask : ndarray, optional
        Boolean, True where the sample is bad. Plotted as a gap.
    flam : ndarray, optional
        Flux-calibrated spectrum (erg/s/cm^2/A) from the RSS FLAM extension, if the file has
        been flux-calibrated. None when uncalibrated. Same length as `wave`.
    """

    wave: np.ndarray
    flux: np.ndarray
    channel: str
    label: str
    weight: float = 1.0
    mask: Optional[np.ndarray] = None
    flam: Optional[np.ndarray] = None
    #: False when there is no instrumental (counts) plane -- e.g. a stacked cube built only from the
    #: calibrated (FLAM) plane, where ``flux`` mirrors ``flam``. Lets the panel grey out "Counts".
    has_counts: bool = True

    @property
    def has_flam(self) -> bool:
        return self.flam is not None

    def good(self, calibrated: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Wavelength and flux with masked and non-finite samples removed.

        With ``calibrated=True`` returns the flux-calibrated ``flam`` plane instead of the
        instrumental ``flux``; raises if this spectrum carries no calibration.
        """
        values = self.flam if calibrated else self.flux
        if values is None:
            raise ValueError('spectrum has no calibrated (FLAM) plane')
        ok = np.isfinite(self.wave) & np.isfinite(values)
        if self.mask is not None:
            ok &= ~self.mask.astype(bool)
        return self.wave[ok], values[ok]


def linear_wcs(crpix: Sequence[float], crval: Sequence[float],
               cdelt: Sequence[float], unit: str = '') -> WCS:
    """Build a 2-D linear (non-celestial) WCS.

    Used for scenes that have no astrometry — the image axes carry instrument coordinates
    (fibre-map units) rather than sky coordinates. ``CTYPE`` is left as ``'LINEAR'`` so that
    nothing downstream mistakes these for celestial coordinates.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['LINEAR', 'LINEAR']
    wcs.wcs.crpix = list(crpix)
    wcs.wcs.crval = list(crval)
    wcs.wcs.cdelt = list(cdelt)
    if unit:
        wcs.wcs.cunit = [unit, unit]
    return wcs


def combine(spectra: Sequence[Spectrum], label: str = 'aperture',
            mode: str = 'sum') -> List[Spectrum]:
    """Combine spectra, per channel, onto the first member's wavelength grid.

    Spectra of different channels are never mixed — the result has one entry per channel.

    Parameters
    ----------
    spectra : sequence of Spectrum
    label : str
        Label for the combined spectra.
    mode : {'sum', 'mean'}
        ``'sum'`` totals the members (weighted by :attr:`Spectrum.weight`), which is what an
        aperture extraction wants: the source's total flux. ``'mean'`` divides by the total
        weight, giving a representative spectrum rather than a total.

    Notes
    -----
    **Combining resamples.** Each LLAMAS fibre carries its own wavelength solution, so members
    are interpolated onto the first member's grid before they can be added at all. Linear
    interpolation is not flux-conserving; over a handful of fibres on a source the error is
    small compared with the gain in signal, but it is an approximation, and a
    flux-conserving resampling would be the honest fix if this is ever used photometrically.
    A single unweighted member is returned untouched, so picking one fibre never resamples.
    """
    if not spectra:
        return []
    if mode not in ('sum', 'mean'):
        raise ValueError(f"mode must be 'sum' or 'mean', not {mode!r}")

    out: List[Spectrum] = []
    for channel in CHANNEL_ORDER:
        members = [s for s in spectra if s.channel == channel]
        if not members:
            continue
        if len(members) == 1 and members[0].weight == 1.0:
            out.append(members[0])          # identity: no resampling for a single fibre
            continue

        reference = members[0].wave
        total_weight = float(sum(m.weight for m in members))
        if total_weight <= 0:
            logger.warning("Zero total weight combining %d %s spectra", len(members), channel)
            continue

        def _stack(getter):
            acc = np.zeros_like(reference, dtype=float)
            for member in members:
                vals = getter(member)
                if member.wave is not reference:
                    vals = np.interp(reference, member.wave, vals, left=np.nan, right=np.nan)
                acc += member.weight * np.nan_to_num(vals, nan=0.0)
            return acc / total_weight if mode == 'mean' else acc

        stack = _stack(lambda m: m.flux)
        # Calibrated flux adds the same way; carry it only if every member has it.
        flam = (_stack(lambda m: m.flam) if all(m.has_flam for m in members) else None)
        out.append(Spectrum(wave=reference, flux=stack, channel=channel, label=label,
                            weight=total_weight, flam=flam))
    return out


class SpectralScene(ABC):
    """A source of collapsed images and spectra, independent of spatial geometry.

    Implementations must be cheap to query repeatedly: :meth:`collapse` runs on every
    wavelength change and :meth:`spectra_at` runs on every crosshair move.
    """

    #: Channels available in this scene, in :data:`CHANNEL_ORDER`.
    channels: Tuple[str, ...] = ()

    @abstractmethod
    def wavelength_range(self, channel: Optional[str] = None) -> Tuple[float, float]:
        """Wavelength coverage in Angstrom, for one channel or across all of them."""

    @abstractmethod
    def collapse(self, wave_min: float, wave_max: float,
                 channels: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, WCS, Dict]:
        """Collapse a wavelength window into an image, in memory.

        Channels whose coverage does not overlap the window are skipped, not zero-filled.

        Returns
        -------
        image : ndarray
            2-D, NaN where the scene has no data.
        wcs : astropy.wcs.WCS
            Maps image pixels to this scene's world coordinates.
        meta : dict
            Extra FITS header keys describing the render, plus a ``'contributions'`` entry
            giving the per-channel sample count, so the GUI can show what actually went in.
        """

    @abstractmethod
    def element_at(self, x_pix: float, y_pix: float) -> Optional[Hashable]:
        """Opaque identity of the spatial element under an image pixel, or None.

        The value is compared for equality and nothing else — a fibre key, a spaxel index, a
        mosaic cell. It exists so the crosshair poller can tell "still the same element" from
        "moved to a new one" without knowing what kind of element a scene has, and so the
        expensive work only runs when the selection actually changes.
        """

    @abstractmethod
    def spectra_at(self, x_pix: float, y_pix: float) -> List[Spectrum]:
        """Spectra of the spatial element under an image pixel.

        Returns an empty list when the pixel maps to nothing — off the field, or a dead
        fibre's empty tile. Callers must handle that; it is a normal outcome, not an error.
        """

    @abstractmethod
    def marker_region(self, x_pix: float, y_pix: float) -> str:
        """A DS9 region string outlining the element under an image pixel.

        Empty string when nothing is selected.
        """

    @abstractmethod
    def elements_within(self, x_pix: float, y_pix: float,
                        radius_pix: float) -> List[Hashable]:
        """Every element whose centre lies within `radius_pix` of an image pixel.

        A radius of zero means the single element under the pixel, so an aperture of zero
        degenerates to a normal pick. Returns an empty list when the aperture catches nothing.
        """

    @abstractmethod
    def spectra_of(self, elements: Sequence[Hashable]) -> List[Spectrum]:
        """The individual spectra of the given elements — one per element per channel.

        Not combined: the caller decides whether to sum, average or weight, so this stays
        useful for aperture extraction and for inspecting members individually.
        """

    @abstractmethod
    def region_for(self, elements: Sequence[Hashable]) -> str:
        """A DS9 region outlining every element in `elements`.

        Empty string for an empty selection.
        """

    def describe_at(self, x_pix: float, y_pix: float) -> str:
        """One-line description of what sits under a pixel, for the status bar."""
        spectra = self.spectra_at(x_pix, y_pix)
        if not spectra:
            return 'no data'
        return spectra[0].label
