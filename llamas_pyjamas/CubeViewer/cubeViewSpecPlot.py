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
        controls.addStretch(1)

        self.title = QLabel('No selection')
        self.title.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self.title)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)

        self._spectra: List[Spectrum] = []
        self._decorate()
        self.canvas.draw_idle()

    def _decorate(self) -> None:
        self.axes.set_xlabel('Wavelength (Å)')
        self.axes.set_ylabel('Flux')
        self.axes.grid(True, alpha=0.25, linewidth=0.5)

    @pyqtSlot(list)
    def set_spectra(self, spectra: Sequence[Spectrum]) -> None:
        """Show a new selection. An empty sequence clears the plot."""
        self._spectra = list(spectra)
        self._replot()

    def _replot(self) -> None:
        limits = (self.axes.get_xlim(), self.axes.get_ylim())
        had_data = bool(self.axes.lines)

        self.axes.clear()
        self._decorate()

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
            wave, flux = spectrum.good()
            if wave.size == 0:
                continue
            self.axes.plot(wave, flux, linewidth=0.8,
                           color=CHANNEL_COLOURS.get(spectrum.channel, '#444444'),
                           label=spectrum.channel)
            plotted += 1

        self.title.setText(self._spectra[0].label)
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
        values, waves = [], []
        for spectrum in self._spectra:
            box = self._visible.get(spectrum.channel)
            if box is not None and not box.isChecked():
                continue
            wave, flux = spectrum.good()
            if wave.size:
                values.append(flux)
                waves.append(wave)
        if not values:
            return

        flux = np.concatenate(values)
        wave = np.concatenate(waves)
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
