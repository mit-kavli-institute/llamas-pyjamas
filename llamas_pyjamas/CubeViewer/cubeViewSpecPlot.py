"""
The spectrum panel — wavelength against flux for the current selection.

An embedded matplotlib canvas with the standard navigation toolbar, so pan and zoom come for
free and behave the way every other matplotlib window does. DS9 owns the image display;
this panel owns the spectrum, and the two never argue about it.

Channels are drawn in their own colours at native sampling. No resampling happens anywhere:
the RSS ``WAVE`` extension is per fibre, so each spectrum is simply plotted against its own
wavelength array. Where the channels overlap in wavelength they are drawn over one another,
which is the honest representation — they are separate measurements, not a mosaic.

The panel imports the ``backend_qtagg`` module rather than the ``backend_qt5agg`` alias used
in ``GUI/obslog.py``; the alias resolves under PyQt6 but is the wrong name for new code.

Note this module must never import ``Bias.biasPlots``, which calls ``matplotlib.use('Agg')``
at import time and would silently kill the interactive canvas.

Classes
-------
SpectrumPanel   QWidget holding the canvas, toolbar and channel toggles
"""

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from llamas_pyjamas.CubeViewer.cubeViewScene import (
    CHANNEL_COLOURS,
    CHANNEL_ORDER,
    Spectrum,
)

logger = logging.getLogger(__name__)


class SpectrumPanel(QWidget):
    """Plots the spectra of the current selection.

    Parameters
    ----------
    parent : QWidget, optional
    hold_limits : bool
        When True the view is kept across selections, so scrubbing fibres compares like with
        like instead of autoscaling under the user on every move. Exposed as a checkbox.
    """

    def __init__(self, parent: Optional[QWidget] = None, hold_limits: bool = False) -> None:
        super().__init__(parent)

        self.figure = Figure(figsize=(10, 3.2), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.axes = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)

        self._visible: Dict[str, QCheckBox] = {}
        controls = QHBoxLayout()
        controls.addWidget(QLabel('Channels:'))
        for channel in CHANNEL_ORDER:
            box = QCheckBox(channel.capitalize())
            box.setChecked(True)
            box.setStyleSheet(f'color: {CHANNEL_COLOURS[channel]}; font-weight: bold;')
            box.stateChanged.connect(self._replot)
            self._visible[channel] = box
            controls.addWidget(box)

        self.hold_box = QCheckBox('Hold zoom')
        self.hold_box.setChecked(bool(hold_limits))
        self.hold_box.setToolTip(
            'Keep the current axis limits when the selection changes, instead of autoscaling'
        )
        controls.addWidget(self.hold_box)

        # Show the flux-calibrated FLAM plane instead of sky-subtracted counts. Enabled only
        # when the selection actually carries calibration (the RSS has a FLAM extension).
        self.flam_box = QCheckBox('Flux-cal')
        self.flam_box.setToolTip('Plot the flux-calibrated spectrum (FLAM, erg/s/cm²/Å) '
                                 'instead of sky-subtracted counts. Available once the RSS '
                                 'has been flux-calibrated.')
        self.flam_box.setEnabled(False)
        self.flam_box.stateChanged.connect(self._replot)
        controls.addWidget(self.flam_box)
        controls.addStretch(1)

        self.title = QLabel('No selection')
        self.title.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.title)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)

        self._spectra: List[Spectrum] = []
        self._wave_range: Optional[tuple] = None
        self._decorate()
        self.canvas.draw_idle()

    @pyqtSlot(float, float)
    def set_wavelength_range(self, wave_min: float, wave_max: float) -> None:
        """Zoom the plot to a wavelength window, and keep it there across selections.

        Called when the white-light window changes, so the spectrum shows the same range the
        image was built from. The y range then follows the samples *inside* this window
        rather than the whole spectrum, which is the point — scaling to 3170-10051 A while
        looking at a 1000 A slice would flatten whatever is in the slice.
        """
        if wave_max > wave_min:
            self._wave_range = (float(wave_min), float(wave_max))
            self._replot()

    def clear_wavelength_range(self) -> None:
        """Return to scaling on the full extent of the data."""
        self._wave_range = None
        self._replot()

    def _decorate(self) -> None:
        self.axes.set_xlabel('Wavelength (Å)')
        self.axes.set_ylabel('Flux')
        self.axes.grid(True, alpha=0.25, linewidth=0.5)

    @pyqtSlot(list)
    def set_spectra(self, spectra: Sequence[Spectrum]) -> None:
        """Show a new selection. An empty sequence clears the plot."""
        self._spectra = list(spectra)
        # Offer the flux-cal view only when the selection carries a FLAM plane.
        have_flam = bool(spectra) and all(s.has_flam for s in spectra)
        self.flam_box.setEnabled(have_flam)
        if not have_flam:
            self.flam_box.setChecked(False)
        self._replot()

    def set_calibrated_default(self, calibrated: bool) -> None:
        """Set the default state of the Flux-cal toggle when a file is loaded.

        Called on load so a flux-calibrated file shows FLAM by default; the checkbox only
        takes visible effect once a selection with a FLAM plane is made. The user can still
        toggle it during the session, and that choice is preserved across selections.
        """
        self.flam_box.setChecked(bool(calibrated))

    def _show_calibrated(self) -> bool:
        return self.flam_box.isEnabled() and self.flam_box.isChecked()

    def _replot(self) -> None:
        limits = (self.axes.get_xlim(), self.axes.get_ylim())
        had_data = bool(self.axes.lines)
        calibrated = self._show_calibrated()

        self.axes.clear()
        self._decorate()
        self.axes.set_ylabel('Flux (erg/s/cm²/Å)' if calibrated else 'Counts (sky-subtracted)')

        if not self._spectra:
            self.title.setText('No selection')
            self.axes.text(0.5, 0.5, 'Click a spaxel in DS9', ha='center', va='center',
                           transform=self.axes.transAxes, alpha=0.4)
            self.canvas.draw_idle()
            return

        plotted = 0
        for spectrum in self._spectra:
            box = self._visible.get(spectrum.channel)
            if box is not None and not box.isChecked():
                continue
            wave, flux = spectrum.good(calibrated=calibrated)
            if wave.size == 0:
                continue
            self.axes.plot(wave, flux, linewidth=0.8,
                           color=CHANNEL_COLOURS.get(spectrum.channel, '#444444'),
                           label=spectrum.channel)
            plotted += 1

        suffix = '  [flux-calibrated]' if calibrated else ''
        self.title.setText(self._spectra[0].label + suffix)
        if plotted:
            self.axes.legend(loc='upper right', fontsize='small', framealpha=0.8)

        if self.hold_box.isChecked() and had_data:
            self.axes.set_xlim(*limits[0])
            self.axes.set_ylim(*limits[1])
        else:
            self._autoscale()
        self.canvas.draw_idle()

    def _autoscale(self) -> None:
        """Scale to the data, ignoring outliers.

        Sky-subtraction residuals and cosmic rays leave spikes orders of magnitude above the
        continuum; a plain autoscale on those flattens the spectrum into a line at zero. A
        high percentile keeps the continuum legible while still showing real lines.
        """
        calibrated = self._show_calibrated()
        values, waves = [], []
        for spectrum in self._spectra:
            box = self._visible.get(spectrum.channel)
            if box is not None and not box.isChecked():
                continue
            wave, flux = spectrum.good(calibrated=calibrated)
            if self._wave_range is not None:
                inside = (wave >= self._wave_range[0]) & (wave <= self._wave_range[1])
                wave, flux = wave[inside], flux[inside]
            if wave.size:
                values.append(flux)
                waves.append(wave)

        if self._wave_range is not None:
            self.axes.set_xlim(*self._wave_range)
        if not values:
            # The window may fall outside every visible channel; keep the requested x range
            # so the user can see that it is empty rather than silently rescaling.
            return

        flux = np.concatenate(values)
        wave = np.concatenate(waves)
        if self._wave_range is None:
            self.axes.set_xlim(float(wave.min()), float(wave.max()))

        low, high = np.percentile(flux, [1.0, 99.5])
        if not np.isfinite(low) or not np.isfinite(high):
            return
        if high <= low:
            # Flat within the percentile range — a near-dead fibre, possibly with a cosmic
            # ray. The limits must still be set explicitly: bailing out here would leave
            # matplotlib's default autoscale in charge, and that *does* include the spike,
            # which is the very thing this method exists to avoid.
            level = float(low)
            span = max(abs(level) * 0.05, 1.0)
            self.axes.set_ylim(level - span, level + span)
            return
        margin = 0.05 * (high - low)
        self.axes.set_ylim(low - margin, high + margin)

    @pyqtSlot(str)
    def set_status(self, text: str) -> None:
        self.title.setText(text)
