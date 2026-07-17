"""
Interactive sensitivity-function fitting for CubeViewer.

When a loaded RSS is a spectrophotometric standard (identified by the Phase I crossmatch on its
header), the user defines an aperture on the star in DS9 the same way as for any spectrum, then
opens this dialog to turn the aperture's summed counts into a sensitivity function.

Two modes, both driven by the same core (:mod:`llamas_pyjamas.Flux.sensFunc`):

* **Auto** — open the dialog and the fit is already there, using the default telluric + stellar
  masks and a span-derived breakpoint spacing. "Let it rip"; just Save.
* **Interactive** — drag on the plot to add a mask region, toggle the default masks, and change
  the breakpoint spacing / spline order; the fit and residuals redraw live before you Save.

The fit follows the instrument, not the star, so the default masks exclude telluric bands and
the broad Balmer/He lines of the hot standards; drawing over a residual the defaults missed is
the manual escape hatch.

Classes
-------
SensFuncModel   Headless fit state — spectra, masks, params -> SensFunc (testable without Qt)
SensFuncDialog  The Qt dialog around the model
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from llamas_pyjamas.CubeViewer.cubeViewScene import CHANNEL_COLOURS, CHANNEL_ORDER
from llamas_pyjamas.Flux.sensFunc import (
    DEFAULT_NORD,
    DEFAULT_SIGMA,
    SensFunc,
    build_good_mask,
    default_masks,
    fit_sensitivity,
    sensitivity_ratio,
)

logger = logging.getLogger(__name__)


@dataclass
class SensFuncModel:
    """Headless sensitivity-fit state, independent of Qt so it can be unit-tested.

    Holds the observed aperture spectra, the reference spectrum, and the editable fit controls
    (masks, breakpoint spacing, order). :meth:`build` produces a :class:`SensFunc` from the
    current state; the dialog just edits the state and redraws.
    """

    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]]     # channel -> (wave, flux)
    exptime: float
    ref_wave: np.ndarray
    ref_flux: np.ndarray
    standard_name: str = ''
    use_default_masks: bool = True
    added_regions: List[Tuple[float, float]] = field(default_factory=list)
    bkspace: Optional[float] = None
    nord: int = DEFAULT_NORD
    sigma: float = DEFAULT_SIGMA

    def regions(self) -> List[Tuple[float, float]]:
        """Effective exclusion regions: defaults (if enabled) plus user-added spans."""
        base = list(default_masks()) if self.use_default_masks else []
        return base + list(self.added_regions)

    def raw(self, channel: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw ratio (wave, sens, valid) for a channel."""
        wave, flux = self.spectra[channel]
        return sensitivity_ratio(wave, flux, self.exptime, self.ref_wave, self.ref_flux)

    def fit_channel(self, channel: str):
        """Fit one channel with the current controls. Returns (wave, raw, fit, good) or None."""
        wave, sens, valid = self.raw(channel)
        good_in = valid & build_good_mask(wave, self.regions())
        if good_in.sum() < max(2 * self.nord, 10):
            return None
        try:
            fit, used = fit_sensitivity(wave, sens, good_in, bkspace=self.bkspace,
                                        nord=self.nord, sigma=self.sigma)
        except ValueError as exc:
            logger.warning('fit failed for %s: %s', channel, exc)
            return None
        return wave, sens, fit, used

    def build(self, meta: Optional[Dict] = None) -> SensFunc:
        """Produce a SensFunc from the current state (same core as the auto path)."""
        from llamas_pyjamas.Flux.sensFunc import build_sensfunc
        full_meta = {'standard': self.standard_name, 'naper': len(self.spectra)}
        full_meta.update(meta or {})
        return build_sensfunc(self.spectra, self.exptime, self.ref_wave, self.ref_flux,
                              regions=self.regions(), bkspace=self.bkspace,
                              nord=self.nord, sigma=self.sigma, meta=full_meta)


class SensFuncDialog(QDialog):
    """Interactive fit dialog around a :class:`SensFuncModel`."""

    def __init__(self, model: SensFuncModel, default_path: str = '',
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.model = model
        self.default_path = default_path
        self.saved_path: Optional[str] = None
        self.setWindowTitle(f'Sensitivity function — {model.standard_name or "standard"}')
        self.resize(950, 640)

        self.figure = Figure(figsize=(9, 6), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax_sens = self.figure.add_subplot(2, 1, 1)
        self.ax_resid = self.figure.add_subplot(2, 1, 2, sharex=self.ax_sens)

        layout = QVBoxLayout(self)
        layout.addLayout(self._build_controls())
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)
        layout.addLayout(self._build_buttons())

        # One span selector per channel axis would overlap; a single selector on the sens axis
        # adds a mask across whatever channels cover that wavelength.
        self._span = SpanSelector(self.ax_sens, self._on_span, 'horizontal', useblit=True,
                                  props=dict(alpha=0.2, facecolor='red'), interactive=False)
        self._refit_and_draw()

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.default_masks_box = QCheckBox('Default masks')
        self.default_masks_box.setChecked(self.model.use_default_masks)
        self.default_masks_box.setToolTip('Exclude telluric bands and the broad Balmer/He '
                                          'lines of the standard from the fit')
        self.default_masks_box.toggled.connect(self._on_default_masks)
        row.addWidget(self.default_masks_box)

        row.addSpacing(12)
        row.addWidget(QLabel('Breakpoint (Å)'))
        self.bkspace_spin = QDoubleSpinBox()
        self.bkspace_spin.setRange(20.0, 2000.0)
        self.bkspace_spin.setSingleStep(25.0)
        self.bkspace_spin.setSpecialValueText('auto')
        self.bkspace_spin.setValue(self.model.bkspace or 20.0)   # 20 == special 'auto'
        self.bkspace_spin.setToolTip('B-spline breakpoint spacing; larger = smoother. '
                                     'Lowest value = auto (span/20).')
        self.bkspace_spin.valueChanged.connect(self._on_params)
        row.addWidget(self.bkspace_spin)

        row.addWidget(QLabel('Order'))
        self.nord_spin = QSpinBox()
        self.nord_spin.setRange(1, 5)
        self.nord_spin.setValue(self.model.nord)
        self.nord_spin.valueChanged.connect(self._on_params)
        row.addWidget(self.nord_spin)

        self.clear_button = QPushButton('Clear added masks')
        self.clear_button.clicked.connect(self._clear_added)
        row.addWidget(self.clear_button)
        row.addStretch(1)
        return row

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.status = QLabel('')
        row.addWidget(self.status, stretch=1)
        save = QPushButton('Save…')
        save.clicked.connect(self._save)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        row.addWidget(save)
        row.addWidget(cancel)
        return row

    # ---- interaction ----
    def _on_span(self, xmin: float, xmax: float) -> None:
        if xmax - xmin < 1.0:            # ignore stray clicks
            return
        self.model.added_regions.append((float(xmin), float(xmax)))
        self._refit_and_draw()

    def _on_default_masks(self, checked: bool) -> None:
        self.model.use_default_masks = checked
        self._refit_and_draw()

    def _on_params(self) -> None:
        v = self.bkspace_spin.value()
        self.model.bkspace = None if v <= self.bkspace_spin.minimum() else v
        self.model.nord = self.nord_spin.value()
        self._refit_and_draw()

    def _clear_added(self) -> None:
        self.model.added_regions.clear()
        self._refit_and_draw()

    # ---- draw ----
    def _refit_and_draw(self) -> None:
        self.ax_sens.clear()
        self.ax_resid.clear()
        fitted = 0
        for channel in CHANNEL_ORDER:
            if channel not in self.model.spectra:
                continue
            colour = CHANNEL_COLOURS.get(channel, '#444')
            result = self.model.fit_channel(channel)
            if result is None:
                continue
            wave, raw, fit, good = result
            self.ax_sens.plot(wave[good], raw[good], '.', ms=2, color=colour, alpha=0.35)
            self.ax_sens.plot(wave, fit, '-', color=colour, lw=1.4, label=channel)
            with np.errstate(invalid='ignore', divide='ignore'):
                resid = (raw - fit) / fit
            self.ax_resid.plot(wave[good], resid[good], '.', ms=2, color=colour, alpha=0.4)
            fitted += 1

        for low, high in self.model.regions():
            self.ax_sens.axvspan(low, high, color='grey', alpha=0.12)

        self.ax_sens.set_ylabel('S = F$_{ref}$ / (counts/s)')
        self.ax_sens.set_yscale('log')
        if fitted:
            self.ax_sens.legend(loc='upper right', fontsize='small')
        self.ax_resid.axhline(0.0, color='k', lw=0.5)
        self.ax_resid.set_ylim(-0.5, 0.5)
        self.ax_resid.set_ylabel('(raw − fit)/fit')
        self.ax_resid.set_xlabel('Wavelength (Å)')
        self.status.setText(
            f'{fitted} channel(s) fitted | {len(self.model.added_regions)} added mask(s)'
            if fitted else 'No channel could be fitted — check the aperture / masks')
        self.canvas.draw_idle()

    def _save(self) -> None:
        try:
            sensfunc = self.model.build()
        except ValueError as exc:
            QMessageBox.warning(self, 'Sensitivity function', f'Could not build: {exc}')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save sensitivity function', self.default_path, 'FITS files (*.fits)')
        if not path:
            return
        sensfunc.save(path)
        self.saved_path = path
        self.accept()
