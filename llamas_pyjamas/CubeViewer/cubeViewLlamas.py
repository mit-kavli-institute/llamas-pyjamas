"""
CubeViewer main window.

Ties the pieces together: open a reduced data product, choose a wavelength window, send the
collapsed image to DS9, and plot the spectrum of whatever you pick there.

The division of labour is deliberate. DS9 owns the image — stretch, colourmap, zoom, pan, and
every other thing it is already better at than a bespoke widget would be. This window owns the
data, the wavelength selection and the spectrum panel. Nothing here duplicates DS9.

DS9 is optional. The window opens, loads data and plots spectra without it; only the image
display and picking need it, and their absence is reported rather than fatal. This matters
because ``GUI/obslog.py`` connects to DS9 in its constructor and therefore cannot start at all
without one.

Run with::

    python -m llamas_pyjamas.CubeViewer

Classes
-------
CubeViewerWindow   The main window

Functions
---------
main               Application entry point
"""

import logging
import os
import sys
from typing import List, Optional

import numpy as np
from astropy.io import fits
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QAction, QDoubleValidator
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from llamas_pyjamas.CubeViewer.cubeViewDS9 import DS9, DS9Error
from llamas_pyjamas.CubeViewer.cubeViewPick import ElementPicker
from llamas_pyjamas.CubeViewer.cubeViewRSS import RSSScene
from llamas_pyjamas.CubeViewer.cubeViewScene import CHANNEL_ORDER, SpectralScene, Spectrum
from llamas_pyjamas.CubeViewer.cubeViewSpecPlot import SpectrumPanel

logger = logging.getLogger(__name__)


class CubeViewerWindow(QMainWindow):
    """Main window: controls along the top, spectrum panel across the bottom."""

    def __init__(self, path: Optional[str] = None, target: str = 'ds9') -> None:
        super().__init__()
        self.setWindowTitle('LLAMAS CubeViewer')
        self.resize(1200, 760)

        self.ds9 = DS9(target=target)
        self.scene: Optional[SpectralScene] = None
        self._path: Optional[str] = None
        self._header = None
        self._standard = None                 # StandardMatch if the loaded file is a standard
        self._current_spectra: List[Spectrum] = []
        self.picker = ElementPicker(self.ds9)
        self.picker.selectionChanged.connect(self._on_selection)
        self.picker.statusChanged.connect(self._on_pick_status)
        self.picker.lost.connect(self._on_ds9_lost)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(self._build_file_row())
        layout.addLayout(self._build_wave_row())

        self.panel = SpectrumPanel()
        layout.addWidget(self.panel, stretch=1)
        self.setCentralWidget(central)

        self._build_menu()
        self.statusBar().showMessage('Open a reduced RSS to begin')
        self._set_enabled(False)

        if path:
            self.load(path)

    # ------------------------------------------------------------------ construction

    def _build_file_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.open_button = QPushButton('Open RSS…')
        self.open_button.clicked.connect(self.choose_file)
        row.addWidget(self.open_button)

        self.file_label = QLabel('(no file)')
        self.file_label.setStyleSheet('color: grey;')
        row.addWidget(self.file_label, stretch=1)
        return row

    def _build_wave_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel('Wavelength'))

        self.wave_min = QLineEdit()
        self.wave_max = QLineEdit()
        for box, tip in ((self.wave_min, 'Minimum wavelength (A)'),
                         (self.wave_max, 'Maximum wavelength (A)')):
            box.setValidator(QDoubleValidator(0.0, 1e6, 2))
            box.setMaximumWidth(90)
            box.setToolTip(tip)
            box.returnPressed.connect(self.display)
        row.addWidget(self.wave_min)
        row.addWidget(QLabel('to'))
        row.addWidget(self.wave_max)
        row.addWidget(QLabel('Å'))

        self.full_button = QPushButton('Full range')
        self.full_button.clicked.connect(self._reset_wave_range)
        row.addWidget(self.full_button)

        row.addSpacing(16)
        row.addWidget(QLabel('Collapse:'))
        self.collapse_boxes = {}
        for channel in CHANNEL_ORDER:
            box = QCheckBox(channel.capitalize())
            box.setChecked(True)
            box.setToolTip(f'Include {channel} in the white-light image')
            self.collapse_boxes[channel] = box
            row.addWidget(box)

        self.hex_box = QCheckBox('Hex tiles')
        self.hex_box.setChecked(True)
        self.hex_box.setToolTip(
            'Exact hexagonal fibre tiles (no interpolation), so a click maps to exactly one '
            'fibre. Unticked uses the interpolated white-light grid, which looks smoother but '
            'makes picking approximate.'
        )
        row.addWidget(self.hex_box)

        row.addSpacing(16)
        self.display_button = QPushButton('Send to DS9')
        self.display_button.clicked.connect(self.display)
        row.addWidget(self.display_button)

        self.pick_box = QCheckBox('Pick')
        self.pick_box.setToolTip('Track the DS9 crosshair and plot the fibre under it')
        self.pick_box.toggled.connect(self._toggle_picking)
        row.addWidget(self.pick_box)

        row.addWidget(QLabel('Aperture'))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.0, 20.0)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.setDecimals(1)
        self.radius_spin.setValue(0.0)
        self.radius_spin.setSuffix(' fib')
        self.radius_spin.setMaximumWidth(80)
        self.radius_spin.setToolTip(
            'Aperture radius in fibre spacings. 0 selects the single fibre under the '
            'crosshair; larger sums every fibre whose centre falls inside.')
        self.radius_spin.valueChanged.connect(self._on_radius_changed)
        row.addWidget(self.radius_spin)

        self.grow_box = QCheckBox('Grow')
        self.grow_box.setToolTip(
            'Click fibres in DS9 to build up an aperture instead of replacing the selection.\n'
            'Click a fibre again to drop it (move away and back — DS9 only reports the '
            'crosshair position, so a second click in place looks identical to the first).\n'
            'Drag to paint. Spectra are summed.')
        self.grow_box.toggled.connect(self._on_grow_toggled)
        row.addWidget(self.grow_box)

        self.clear_button = QPushButton('Clear')
        self.clear_button.setToolTip('Empty the accumulated aperture')
        self.clear_button.clicked.connect(self.picker.clear_aperture)
        self.clear_button.setEnabled(False)
        row.addWidget(self.clear_button)

        self.sensfunc_button = QPushButton('Sensitivity Fn…')
        self.sensfunc_button.setToolTip('Build a sensitivity function from the aperture '
                                        '(enabled when the loaded file is a flux standard)')
        self.sensfunc_button.clicked.connect(self.build_sensfunc)
        self.sensfunc_button.setEnabled(False)
        row.addWidget(self.sensfunc_button)

        row.addStretch(1)
        return row

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu('&File')
        open_action = QAction('&Open RSS…', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.choose_file)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        quit_action = QAction('&Quit', self)
        quit_action.setShortcut('Ctrl+Q')
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (self.wave_min, self.wave_max, self.full_button, self.hex_box,
                       self.display_button, self.pick_box, self.radius_spin,
                       self.grow_box):
            widget.setEnabled(enabled)
        for box in self.collapse_boxes.values():
            box.setEnabled(enabled)

    # ------------------------------------------------------------------ data

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open reduced RSS', '', 'FITS files (*.fits *.fit *.fits.gz);;All files (*)')
        if path:
            self.load(path)

    def load(self, path: str) -> None:
        """Open an RSS and every channel sibling beside it."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            scene = RSSScene.open(path)
        except Exception as exc:                       # noqa: BLE001 - report anything
            QApplication.restoreOverrideCursor()
            logger.exception('Could not open %s', path)
            QMessageBox.critical(self, 'Could not open file', f'{path}\n\n{exc}')
            return
        QApplication.restoreOverrideCursor()

        self.scene = scene
        self._path = path
        self.picker.set_scene(scene)
        self.panel.set_spectra([])
        self._current_spectra = []

        for channel, box in self.collapse_boxes.items():
            present = channel in scene.channels
            box.setEnabled(present)
            box.setChecked(present)

        self.file_label.setText(os.path.basename(path))
        self.file_label.setStyleSheet('')
        self._set_enabled(True)
        self._reset_wave_range()
        self._on_radius_changed(self.radius_spin.value())

        # Is this a standard? Crossmatch the header so the sensfunc action knows the reference.
        self._header, self._standard = self._identify_standard(path)

        low, high = scene.wavelength_range()
        std_note = ''
        if self._standard is not None:
            std_note = f" | STANDARD: {self._standard.name} ({self._standard.separation_arcsec:.1f}\")"
        self._update_sensfunc_button()
        self.statusBar().showMessage(
            f"{len(scene.keys)} fibres | channels: {', '.join(scene.channels)} | "
            f"{low:.0f}-{high:.0f} A{std_note}")

    @staticmethod
    def _identify_standard(path: str):
        """Return (primary_header, StandardMatch|None) for the loaded file."""
        try:
            from astropy.io import fits
            from llamas_pyjamas.Flux.fluxStandards import load_catalog
            header = fits.getheader(path, 0)
            match = load_catalog().match_header(header)
            return header, match
        except Exception as exc:                       # noqa: BLE001
            logger.debug('standard identification skipped: %s', exc)
            return None, None

    def _update_sensfunc_button(self) -> None:
        is_std = self._standard is not None and self._standard.standard.has_spectrum
        self.sensfunc_button.setEnabled(is_std)
        if self._standard is not None and not self._standard.standard.has_spectrum:
            self.sensfunc_button.setToolTip(
                f'{self._standard.name} has no bundled reference spectrum')
        elif is_std:
            self.sensfunc_button.setToolTip(
                f'Build a sensitivity function from the aperture on {self._standard.name}')
        else:
            self.sensfunc_button.setToolTip('Loaded file is not a recognised flux standard')

    def _reset_wave_range(self) -> None:
        if self.scene is None:
            return
        low, high = self.scene.wavelength_range()
        self.wave_min.setText(f'{low:.1f}')
        self.wave_max.setText(f'{high:.1f}')

    def _selected_channels(self) -> List[str]:
        return [c for c, box in self.collapse_boxes.items()
                if box.isChecked() and box.isEnabled()]

    # ------------------------------------------------------------------ display

    def display(self) -> None:
        """Collapse the current window and send the image to DS9."""
        if self.scene is None:
            return
        try:
            wave_min = float(self.wave_min.text())
            wave_max = float(self.wave_max.text())
        except ValueError:
            QMessageBox.warning(self, 'Invalid wavelengths',
                                'Enter numeric minimum and maximum wavelengths.')
            return
        if wave_max <= wave_min:
            QMessageBox.warning(self, 'Invalid wavelengths',
                                'Maximum wavelength must exceed the minimum.')
            return

        channels = self._selected_channels()
        if not channels:
            QMessageBox.warning(self, 'No channels', 'Select at least one channel to collapse.')
            return

        if getattr(self.scene, 'hex_tiles', None) is not None:
            self.scene.hex_tiles = self.hex_box.isChecked()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            image, wcs, meta = self.scene.collapse(wave_min, wave_max, channels)
        except Exception as exc:                       # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, 'Could not build image', str(exc))
            return

        header = fits.Header()
        for key, value in meta.items():
            if key != 'contributions':
                header[key] = value
        hdul = fits.HDUList([fits.PrimaryHDU(np.asarray(image, dtype=np.float32),
                                             header=header)])
        try:
            self.ds9.set_fits(hdul)
            self.ds9.set('zoom to fit')
            self.ds9.set('scale zscale')
        except DS9Error as exc:
            QApplication.restoreOverrideCursor()
            self._report_ds9(exc)
            return
        QApplication.restoreOverrideCursor()

        self.panel.set_wavelength_range(wave_min, wave_max)

        contributions = meta.get('contributions', {})
        summary = ', '.join(f'{c}:{n}' for c, n in contributions.items())
        self.statusBar().showMessage(
            f'{wave_min:.0f}-{wave_max:.0f} A | {image.shape[1]}x{image.shape[0]} px | '
            f'fibres contributing - {summary}')

        if not self.pick_box.isChecked():
            self.pick_box.setChecked(True)          # starts picking

    # ------------------------------------------------------------------ picking

    @pyqtSlot(float)
    def _on_radius_changed(self, value: float) -> None:
        """Convert an aperture radius in fibre spacings to image pixels for the picker.

        The user thinks in fibres, not pixels, and the pixel scale changes with the render
        (hex tiles vs the interpolated grid), so the conversion has to come from the scene.
        """
        if self.scene is None:
            return
        step = getattr(self.scene, '_step', None)
        pitch = getattr(self.scene, 'pitch', None) or 1.0
        if step is None:
            return
        self.picker.set_radius(value * pitch / step())

    @pyqtSlot(bool)
    def _on_grow_toggled(self, enabled: bool) -> None:
        self.picker.set_accumulate(enabled)
        self.clear_button.setEnabled(enabled)
        if enabled and not self.pick_box.isChecked():
            self.pick_box.setChecked(True)     # growing an aperture implies picking

    @pyqtSlot(bool)
    def _toggle_picking(self, enabled: bool) -> None:
        if enabled:
            try:
                self.picker.start()
            except DS9Error as exc:
                self.pick_box.setChecked(False)
                self._report_ds9(exc)
        else:
            self.picker.stop()

    @pyqtSlot(list)
    def _on_selection(self, spectra: List[Spectrum]) -> None:
        self._current_spectra = list(spectra)
        self.panel.set_spectra(spectra)

    @pyqtSlot(str)
    def _on_pick_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    @pyqtSlot(str)
    def _on_ds9_lost(self, message: str) -> None:
        self.pick_box.setChecked(False)
        self.statusBar().showMessage(f'Lost DS9: {message}')

    def build_sensfunc(self) -> None:
        """Open the sensitivity-function dialog for the current aperture on the standard."""
        if self._standard is None or not self._standard.standard.has_spectrum:
            return
        if not self._current_spectra:
            QMessageBox.information(
                self, 'Sensitivity function',
                'Define an aperture on the standard first: enable Pick (and Grow for a '
                'multi-fibre aperture) and select the star in DS9.')
            return

        exptime = None
        if self._header is not None:
            exptime = self._header.get('SEXPTIME', self._header.get('REXPTIME',
                                       self._header.get('EXPTIME')))
        try:
            exptime = float(exptime)
        except (TypeError, ValueError):
            QMessageBox.warning(self, 'Sensitivity function',
                                'Could not read the exposure time from the header.')
            return

        try:
            ref_wave, ref_flux = self._standard.standard.load_spectrum()
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.warning(self, 'Sensitivity function',
                                f'Could not load the reference spectrum: {exc}')
            return

        # The picker already summed the aperture per channel; hand those spectra straight in.
        spectra = {s.channel: s.good() for s in self._current_spectra}

        airmass = None
        if self._header is not None:
            airmass = self._header.get('AIRMASS', self._header.get('TEL AIRMASS'))
            try:
                airmass = float(airmass)
            except (TypeError, ValueError):
                airmass = None

        from llamas_pyjamas.CubeViewer.cubeViewSensFunc import SensFuncDialog, SensFuncModel
        model = SensFuncModel(spectra=spectra, exptime=exptime,
                              ref_wave=ref_wave, ref_flux=ref_flux,
                              standard_name=self._standard.name, airmass=airmass)
        default_path = ''
        if self._path:
            base = os.path.splitext(os.path.basename(self._path))[0]
            default_path = os.path.join(os.path.dirname(self._path), f'{base}_sensfunc.fits')
        dialog = SensFuncDialog(model, default_path=default_path, parent=self)
        if dialog.exec() and dialog.saved_path:
            self.statusBar().showMessage(f'Saved sensitivity function: {dialog.saved_path}')

    def _report_ds9(self, exc: Exception) -> None:
        QMessageBox.warning(
            self, 'DS9 unavailable',
            f'{exc}\n\nStart DS9 and try again. The spectrum panel works without it.')

    def closeEvent(self, event) -> None:            # noqa: N802 - Qt naming
        self.picker.stop()
        super().closeEvent(event)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for ``python -m llamas_pyjamas.CubeViewer``."""
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(name)s: %(message)s')
    argv = list(sys.argv if argv is None else argv)
    path = next((a for a in argv[1:] if not a.startswith('-')), None)

    app = QApplication(argv)
    window = CubeViewerWindow(path=path)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
