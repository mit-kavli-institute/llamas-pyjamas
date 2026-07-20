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
    QGroupBox,
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

        # Wavelength window and aperture selection share the top row, each in its own box.
        top_row = QHBoxLayout()
        top_row.addWidget(self._build_wave_box())
        top_row.addWidget(self._build_aperture_box())
        top_row.addStretch(1)
        layout.addLayout(top_row)

        layout.addWidget(self._build_image_display_box())

        self.panel = SpectrumPanel()
        spectrum_box = QGroupBox('IFU spectrum')
        spectrum_layout = QVBoxLayout(spectrum_box)
        spectrum_layout.setContentsMargins(6, 6, 6, 6)
        spectrum_layout.addWidget(self.panel)
        layout.addWidget(spectrum_box, stretch=1)
        self.setCentralWidget(central)

        self._build_menu()
        self.statusBar().showMessage('Open a reduced RSS to begin')
        self._set_enabled(False)

        if path:
            self.load(path)

    # ------------------------------------------------------------------ construction

    def _build_file_row(self) -> QHBoxLayout:
        # Opening is on the File menu (Ctrl+O); this row just shows the current file.
        row = QHBoxLayout()
        self.file_label = QLabel('(no file — File ▸ Open RSS…)')
        self.file_label.setStyleSheet('color: grey;')
        row.addWidget(self.file_label, stretch=1)
        return row

    def _build_wave_box(self) -> QGroupBox:
        """Wavelength window that drives the white-light collapse."""
        box = QGroupBox('Wavelength')
        row = QHBoxLayout(box)

        self.wave_min = QLineEdit()
        self.wave_max = QLineEdit()
        for label, edit, tip in (('min', self.wave_min, 'Minimum wavelength (Å)'),
                                 ('max', self.wave_max, 'Maximum wavelength (Å)')):
            edit.setValidator(QDoubleValidator(0.0, 1e6, 2))
            edit.setMinimumWidth(90)         # show the full number, e.g. 10051.4
            edit.setMaximumWidth(110)
            edit.setToolTip(tip)
            edit.returnPressed.connect(self.display)
            row.addWidget(QLabel(label))
            row.addWidget(edit)
        row.addWidget(QLabel('Å'))

        self.full_button = QPushButton('Full range')
        self.full_button.clicked.connect(self._reset_wave_range)
        row.addWidget(self.full_button)
        return box

    def _build_image_display_box(self) -> QGroupBox:
        """What goes into the DS9 white-light image: channels, hex tiles, and Send."""
        box = QGroupBox('Image Display')
        row = QHBoxLayout(box)

        self.collapse_boxes = {}
        for channel in CHANNEL_ORDER:
            check = QCheckBox(channel.capitalize())
            check.setChecked(True)
            check.setToolTip(f'Include {channel} in the white-light image')
            self.collapse_boxes[channel] = check
            row.addWidget(check)

        row.addSpacing(16)
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
        self.display_button.setToolTip('Build the white-light image over the wavelength window '
                                       'and display it in DS9')
        self.display_button.clicked.connect(self.display)
        row.addWidget(self.display_button)

        row.addStretch(1)
        return box

    def _build_aperture_box(self) -> QGroupBox:
        """The aperture-selection controls, boxed so they read as one section."""
        box = QGroupBox('Aperture')
        row = QHBoxLayout(box)

        self.pick_box = QCheckBox('Pick')
        self.pick_box.setToolTip('Track the DS9 crosshair and plot the fibre under it')
        self.pick_box.toggled.connect(self._toggle_picking)
        row.addWidget(self.pick_box)

        row.addWidget(QLabel('Radius'))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.0, 20.0)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.setDecimals(1)
        self.radius_spin.setValue(0.0)
        self.radius_spin.setSuffix(' fib')
        self.radius_spin.setMaximumWidth(90)
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

        row.addStretch(1)
        self.aperture_box = box
        return box

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu('&File')
        open_action = QAction('&Open RSS…', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.choose_file)
        file_menu.addAction(open_action)
        obslog_action = QAction('Open from o&bslog…', self)
        obslog_action.setShortcut('Ctrl+B')
        obslog_action.setToolTip('Pick an exposure by object name / notes instead of filename')
        obslog_action.triggered.connect(self.open_from_obslog)
        file_menu.addAction(obslog_action)
        open_cube_action = QAction('Open combined &cube…', self)
        open_cube_action.setShortcut('Ctrl+K')
        open_cube_action.setToolTip('Open a combined cube (loads all channel siblings; '
                                    'built by Combine ▸ Combine field into cube)')
        open_cube_action.triggered.connect(self.open_cube)
        file_menu.addAction(open_cube_action)
        file_menu.addSeparator()
        quit_action = QAction('&Quit', self)
        quit_action.setShortcut('Ctrl+Q')
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Sensitivity-function tasks live in a menu; the action greys out unless the loaded
        # file is a flux standard with a bundled reference spectrum.
        sens_menu = self.menuBar().addMenu('&Sensitivity')
        sens_menu.setToolTipsVisible(True)
        self.sensfunc_action = QAction('&Build Sensitivity Function…', self)
        self.sensfunc_action.triggered.connect(self.build_sensfunc)
        self.sensfunc_action.setEnabled(False)
        sens_menu.addAction(self.sensfunc_action)
        # Applying does not need a standard loaded, so this stays enabled.
        apply_action = QAction('&Apply Sensitivity Function to files…', self)
        apply_action.setToolTip('Flux-calibrate science RSS files with a saved sensitivity '
                                'function (adds FLAM/FLAM_ERR)')
        apply_action.triggered.connect(self.apply_sensfunc_to_files)
        sens_menu.addAction(apply_action)

        # Astrometric WCS: automated Gaia registration, an interactive fallback, QA and a revert.
        wcs_menu = self.menuBar().addMenu('&WCS')
        wcs_menu.setToolTipsVisible(True)
        self.wcs_auto_action = QAction('Auto-register this exposure', self)
        self.wcs_auto_action.setToolTip('Solve the WCS from Gaia for the loaded exposure')
        self.wcs_auto_action.triggered.connect(self.auto_register_exposure)
        wcs_menu.addAction(self.wcs_auto_action)
        # Block registration picks its own files, so it does not need one loaded.
        wcs_block_action = QAction('Auto-register block…', self)
        wcs_block_action.setToolTip('Pick a set of dithers and register them as one block '
                                    '(one shared, held rotation)')
        wcs_block_action.triggered.connect(self.auto_register_block)
        wcs_menu.addAction(wcs_block_action)
        wcs_menu.addSeparator()
        self.wcs_refine_action = QAction('&Refine WCS interactively…', self)
        self.wcs_refine_action.setToolTip('Click stars in DS9 to build the WCS by hand -- the '
                                          'fallback when the automated solve does not work')
        self.wcs_refine_action.triggered.connect(self.refine_wcs_interactive)
        wcs_menu.addAction(self.wcs_refine_action)
        wcs_menu.addSeparator()
        self.wcs_qa_action = QAction('Show registration QA', self)
        self.wcs_qa_action.setToolTip('Write a QA plot: fitted centroids + Gaia over the white '
                                      'light image')
        self.wcs_qa_action.triggered.connect(self.show_registration_qa)
        wcs_menu.addAction(self.wcs_qa_action)
        self.wcs_reset_action = QAction('Reset to rough (header) WCS', self)
        self.wcs_reset_action.setToolTip('Discard the star-tied solution and revert to the TCS '
                                         'header pointing')
        self.wcs_reset_action.triggered.connect(self.reset_rough_wcs)
        wcs_menu.addAction(self.wcs_reset_action)
        # These need a file loaded; block registration above does not.
        self.wcs_file_actions = (self.wcs_auto_action, self.wcs_refine_action,
                                 self.wcs_qa_action, self.wcs_reset_action)
        for action in self.wcs_file_actions:
            action.setEnabled(False)

        # Combine: stack a field's dithers into a cube and inspect it (Phase 4).
        combine_menu = self.menuBar().addMenu('&Combine')
        combine_menu.setToolTipsVisible(True)
        cube_action = QAction('Combine field into &cube…', self)
        cube_action.setToolTip('Stack a field\'s registered dithers into an (RA,DEC,wave) cube '
                               '(green, surface brightness) and open it for spaxel picking')
        cube_action.triggered.connect(self.combine_field_cube)
        combine_menu.addAction(cube_action)
        self.narrowband_action = QAction('&Narrowband line image…', self)
        self.narrowband_action.setToolTip('Continuum-subtracted narrowband image at a chosen line '
                                          'wavelength (e.g. Lya), sent to a new DS9 frame')
        self.narrowband_action.triggered.connect(self.narrowband_image_dialog)
        combine_menu.addAction(self.narrowband_action)

        # Extraction: point-source spectra from a combined cube.
        extract_menu = self.menuBar().addMenu('&Extraction')
        extract_menu.setToolTipsVisible(True)
        self.optspec_action = QAction('Optimal &spectrum at crosshair', self)
        self.optspec_action.setToolTip('Fit a 2-D Gaussian to the source at the DS9 crosshair and '
                                       'PSF/inverse-variance extract its spectrum (deepest QSO '
                                       'spectrum); draws the 1σ/2σ aperture')
        self.optspec_action.triggered.connect(self.optimal_spectrum_here)
        extract_menu.addAction(self.optspec_action)

        # Enabled only when a combined cube is loaded.
        self.cube_actions = (self.optspec_action, self.narrowband_action)
        for action in self.cube_actions:
            action.setEnabled(False)

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

    def _start_dir(self) -> str:
        """Directory to seed the obslog/apply pickers -- the open file's, else the cwd."""
        return os.path.dirname(self._path) if self._path else os.getcwd()

    def open_from_obslog(self) -> None:
        """Pick an exposure from a header-driven table instead of by filename."""
        from llamas_pyjamas.CubeViewer.cubeViewObslog import ObslogDialog
        dialog = ObslogDialog(self._start_dir(), title='Open from obslog', parent=self)
        if dialog.exec() and dialog.chosen_path:
            self.load(dialog.chosen_path)

    def combine_field_cube(self) -> None:
        """Stack a field's registered dithers into a cube and open it for spaxel picking."""
        from llamas_pyjamas.CubeViewer.cubeViewObslog import ObslogDialog
        dialog = ObslogDialog(self._start_dir(), multi=True,
                              title='Combine field into cube (pick the dithers)', parent=self)
        if not dialog.exec():
            return
        paths = getattr(dialog, 'chosen_files', None) or []
        if not paths:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            from llamas_pyjamas.Combine.superRSS import build_super_rss
            from llamas_pyjamas.Combine.cube import combine_field_cubes
            from llamas_pyjamas.Combine.transparency import transparency_scales
            from llamas_pyjamas.CubeViewer.cubeViewCube import CoaddCubeScene
            sr = build_super_rss(paths)             # all channels
            try:                                   # transparency is best-effort (needs a bright src)
                sr.apply_scales(transparency_scales(sr))
            except Exception as exc:               # noqa: BLE001
                logger.warning('transparency scaling skipped: %s', exc)
            cubes = combine_field_cubes(sr, units='sb', weighting='ivar')
            scene = CoaddCubeScene(cubes, super_rss=sr)     # keep super-RSS for optimal extraction
            # Save the built cubes to the standard combined/ directory so they are findable.
            from llamas_pyjamas.Combine.superRSS import combined_dir
            outdir = combined_dir(paths, create=True)
            field = scene.object or 'field'
            written = [cubes[c].write(os.path.join(outdir, f'{field}_cube_{c}.fits'))
                       for c in scene.channels]
        except Exception as exc:                   # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, 'Combine field', f'Could not build the cube:\n{exc}')
            return
        QApplication.restoreOverrideCursor()
        self._path = written[0] if written else None
        self._open_cube_scene(scene, f"{scene.object or 'field'} combined cube (SB)")
        self.statusBar().showMessage(
            f'Combined {len(paths)} exposures -> {", ".join(scene.channels)} cubes in '
            f'{outdir}')
        QMessageBox.information(self, 'Combine field',
                               f'Wrote {len(written)} cube(s) to:\n{outdir}')

    def open_cube(self) -> None:
        """Open a cube FITS previously written by the combine step."""
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open combined cube', self._start_dir(), 'FITS files (*.fits);;All files (*)')
        if not path:
            return
        from llamas_pyjamas.CubeViewer.cubeViewCube import CoaddCubeScene
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            scene = CoaddCubeScene.from_fits(path)
        except Exception as exc:                   # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, 'Open cube', f'Could not open cube:\n{exc}')
            return
        QApplication.restoreOverrideCursor()
        self._path = path
        self._open_cube_scene(scene, os.path.basename(path))

    def _open_cube_scene(self, scene, label) -> None:
        """Activate a cube scene and display it."""
        self._header, self._standard = None, None
        self._update_sensfunc_action()
        self._activate_scene(scene, label, allow_wcs=False, is_cube=True)
        try:
            self.display()
        except Exception:                          # noqa: BLE001
            pass

    def optimal_spectrum_here(self) -> None:
        """Extract the PSF/ivar-weighted point-source spectrum at the DS9 crosshair."""
        from llamas_pyjamas.CubeViewer.cubeViewCube import CoaddCubeScene
        if not isinstance(self.scene, CoaddCubeScene):
            return
        try:
            x, y = self.ds9.crosshair('image')
        except DS9Error as exc:
            self._report_ds9(exc)
            return
        sky = self.scene.cube.wcs.celestial.pixel_to_world(x - 1, y - 1)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            spectra, fit = self.scene.optimal_spectrum(float(sky.ra.deg), float(sky.dec.deg))
        except Exception as exc:                   # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, 'Optimal spectrum', f'Could not extract:\n{exc}')
            return
        QApplication.restoreOverrideCursor()
        self._current_spectra = list(spectra)
        self.panel.set_spectra(spectra)
        try:                                        # show the fitted 1σ/2σ aperture in DS9
            self.ds9.delete_region_group('cubeview-ext')
            self.ds9.set_regions(self.scene.profile_ellipse_region(fit))
        except DS9Error as exc:
            logger.debug('could not draw extraction aperture: %s', exc)
        kind = 'fitted' if fit.fitted else 'assumed'
        self.statusBar().showMessage(
            f'Optimal spectrum at ({fit.ra:.4f}, {fit.dec:+.4f}) [{kind}], FWHM {fit.fwhm:.2f}"')

    def narrowband_image_dialog(self) -> None:
        """Build a continuum-subtracted narrowband image at a chosen line and send it to DS9."""
        from llamas_pyjamas.CubeViewer.cubeViewCube import CoaddCubeScene
        if not isinstance(self.scene, CoaddCubeScene):
            return
        from PyQt6.QtWidgets import QInputDialog
        lo, hi = self.scene.wavelength_range()
        default = 0.5 * (lo + hi)
        line, ok = QInputDialog.getDouble(self, 'Narrowband image', 'Line wavelength (Å):',
                                          default, lo, hi, 1)
        if not ok:
            return
        hw, ok2 = QInputDialog.getDouble(self, 'Narrowband image', 'Half-width (Å):', 8.0, 0.5,
                                         500.0, 1)
        if not ok2:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            nb = self.scene.narrowband(line, half_width=hw)
        except Exception as exc:                   # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, 'Narrowband image', f'Could not build:\n{exc}')
            return
        header = nb.wcs.to_header()
        header['BUNIT'] = nb.bunit
        header['OBJECT'] = f"{self.scene.object} narrowband {line:.0f}A"
        hdul = fits.HDUList([fits.PrimaryHDU(np.asarray(nb.data, dtype=np.float32), header=header)])
        try:
            self.ds9.set('frame new')
            self.ds9.set_fits(hdul)
            self.ds9.set('scale zscale')
        except DS9Error as exc:
            QApplication.restoreOverrideCursor()
            self._report_ds9(exc)
            return
        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f'Narrowband {line:.0f}±{hw:.0f} Å (continuum-subtracted) '
                                     '-> new DS9 frame')

    def _activate_scene(self, scene, label: str, *, allow_wcs: bool = True,
                        is_cube: bool = False) -> None:
        """Wire a scene (RSS or cube) into the window: picker, panel, channel boxes, wavelength
        range and controls. Shared by :meth:`load` and the Combine menu."""
        for action in getattr(self, 'cube_actions', ()):     # cube-only tools
            action.setEnabled(is_cube)
        self.scene = scene
        self.picker.set_scene(scene)
        self.panel.set_spectra([])
        self._current_spectra = []
        # Default to the flux-calibrated view when the scene carries a FLAM/calibrated plane.
        self.panel.set_calibrated_default(getattr(scene, 'has_flam', False))

        for channel, box in self.collapse_boxes.items():
            present = channel in scene.channels
            box.setEnabled(present)
            box.setChecked(present)

        self.file_label.setText(label)
        self.file_label.setStyleSheet('')
        self._set_enabled(True)
        for action in self.wcs_file_actions:     # registration applies to an RSS, not a cube
            action.setEnabled(allow_wcs)
        self._reset_wave_range()
        self._on_radius_changed(self.radius_spin.value())

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

        self._path = path
        self._activate_scene(scene, os.path.basename(path))

        # Is this a standard? Crossmatch the header so the sensfunc action knows the reference.
        self._header, self._standard = self._identify_standard(path)

        low, high = scene.wavelength_range()
        std_note = ''
        if self._standard is not None:
            std_note = f" | STANDARD: {self._standard.name} ({self._standard.separation_arcsec:.1f}\")"
        self._update_sensfunc_action()
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

    def _update_sensfunc_action(self) -> None:
        is_std = self._standard is not None and self._standard.standard.has_spectrum
        self.sensfunc_action.setEnabled(is_std)
        if self._standard is not None and not self._standard.standard.has_spectrum:
            tip = f'{self._standard.name} has no bundled reference spectrum'
        elif is_std:
            tip = f'Build a sensitivity function from the aperture on {self._standard.name}'
        else:
            tip = 'Enabled only when the loaded file is a recognised flux standard'
        self.sensfunc_action.setToolTip(tip)
        self.sensfunc_action.setStatusTip(tip)

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
        # Provenance/meta keys, minus WCS cards -- the authoritative WCS is the returned `wcs`
        # object (previously it was discarded and DS9 only saw meta's linear map).
        _wcs_prefixes = ('CRPIX', 'CRVAL', 'CDELT', 'CTYPE', 'CUNIT', 'CD', 'PC',
                         'CROTA', 'LONPOLE', 'LATPOLE', 'RADESYS', 'WCSAXES', 'EQUINOX')
        for key, value in meta.items():
            if key == 'contributions' or key.upper().startswith(_wcs_prefixes):
                continue
            header[key] = value
        if wcs is not None:
            header.update(wcs.to_header())
        obj = getattr(self.scene, 'object', '')
        if obj:
            header['OBJECT'] = obj                      # so DS9 shows the object name
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

    def apply_sensfunc_to_files(self) -> None:
        """Flux-calibrate one or more science RSS files with a saved sensitivity function.

        Adds FLAM/FLAM_ERR in place (the GUI equivalent of the apply_fluxcal CLI). Independent
        of the currently-loaded file, so it works whether or not a standard is open.
        """
        from llamas_pyjamas.CubeViewer.cubeViewObslog import ObslogDialog
        dialog = ObslogDialog(self._start_dir(), multi=True, with_sensfunc=True,
                              title='Apply sensitivity function to files', parent=self)
        if not dialog.exec():
            return
        sens_path = dialog.sensfunc_path
        rss_paths = dialog.chosen_files
        if not sens_path or not rss_paths:
            return

        reply = QMessageBox.question(
            self, 'Apply sensitivity function',
            f'Add FLAM / FLAM_ERR to {len(rss_paths)} plane(s) '
            f'(all channels of the selected exposures) in place, using\n'
            f'{os.path.basename(sens_path)}?\n\n'
            'Differential atmospheric extinction is applied. The instrumental planes are '
            'left unchanged.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        from llamas_pyjamas.Flux.fluxCalibrate import flux_calibrate_file
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        n_ok, failures = 0, []
        try:
            for path in rss_paths:
                try:
                    flux_calibrate_file(path, sens_path, apply_extinction=True)
                    n_ok += 1
                except Exception as exc:               # noqa: BLE001
                    failures.append(f'{os.path.basename(path)}: {exc}')
        finally:
            QApplication.restoreOverrideCursor()

        # If the open file was among those calibrated, reload so its FLAM shows now.
        if self._path in rss_paths:
            self.load(self._path)

        message = f'Flux-calibrated {n_ok}/{len(rss_paths)} file(s).'
        if failures:
            message += '\n\nFailed:\n' + '\n'.join(failures)
        self.statusBar().showMessage(message.splitlines()[0])
        QMessageBox.information(self, 'Apply sensitivity function', message)

    # ------------------------------------------------------------------ WCS / registration

    def _require_file(self, title: str) -> bool:
        if not self._path or self.scene is None:
            QMessageBox.information(self, title, 'Open an RSS first.')
            return False
        return True

    def _after_registration(self, message: str) -> None:
        """Reload the current file so the freshly-written (refined or rough) WCS is picked up, then
        re-display and report."""
        if self._path:
            self.load(self._path)                 # reload -> scene.refined_wcs from FIBERWCS
            try:
                self.display()
            except Exception:                     # noqa: BLE001 - display is best-effort here
                pass
        self.statusBar().showMessage(message)
        QMessageBox.information(self, 'WCS registration', message)

    def auto_register_exposure(self) -> None:
        """Solve the WCS from Gaia for the loaded exposure and write it in place."""
        if not self._require_file('Auto-register'):
            return
        from llamas_pyjamas.Utils.register import register_exposure
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = register_exposure(self._path)
        except Exception as exc:                  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, 'Auto-register', f'Registration failed:\n{exc}')
            return
        QApplication.restoreOverrideCursor()
        if result.refined:
            msg = (f'Refined ({result.tier}): {result.n_stars} star(s), '
                   f'RMS {result.rms_arcsec:.2f}".')
        else:
            msg = 'No confident Gaia match; kept the rough header WCS.'
        self._after_registration(msg)

    def auto_register_block(self) -> None:
        """Pick a set of dithers and register them as one block (one shared, held rotation)."""
        from llamas_pyjamas.CubeViewer.cubeViewObslog import ObslogDialog
        dialog = ObslogDialog(self._start_dir(), multi=True,
                              title='Auto-register block (pick the dithers)', parent=self)
        if not dialog.exec():
            return
        paths = getattr(dialog, 'chosen_files', None) or []
        if not paths:
            return
        from llamas_pyjamas.Utils.register import register_exposures
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            results = register_exposures(paths)
        except Exception as exc:                  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, 'Auto-register block', f'Failed:\n{exc}')
            return
        QApplication.restoreOverrideCursor()
        n_ref = sum(1 for r in results.values() if r.refined)
        self._after_registration(f'Registered {n_ref}/{len(paths)} frame(s) in the block.')

    def refine_wcs_interactive(self) -> None:
        """Open the interactive star-clicking WCS dialog for the loaded exposure."""
        if not self._require_file('Refine WCS'):
            return
        if self.scene.ra is None:
            QMessageBox.information(self, 'Refine WCS',
                                   'This frame has no header pointing, so there is no initial WCS '
                                   'to refine.')
            return
        # The interactive tool drives the DS9 crosshair; pause the aperture picker to avoid
        # contention while it is open.
        if self.pick_box.isChecked():
            self.pick_box.setChecked(False)
        from llamas_pyjamas.CubeViewer.cubeViewRegister import RegisterDialog
        dialog = RegisterDialog(self.scene, self.ds9, self._path, parent=self)
        if dialog.exec() and dialog.written:
            try:
                self.display()                    # scene.refined_wcs is now set -> new grid
            except Exception:                     # noqa: BLE001
                pass
            self.statusBar().showMessage(f'Wrote the interactive WCS to {len(dialog.written)} '
                                         'file(s).')

    def show_registration_qa(self) -> None:
        """Write the registration QA plot (fitted centroids + Gaia over the white light)."""
        if not self._require_file('Registration QA'):
            return
        from llamas_pyjamas.CubeViewer.cubeViewRSS import channel_siblings
        from llamas_pyjamas.QA.qa_registration import plot_registration_qa
        siblings = channel_siblings(self._path) or {'': self._path}
        green = siblings.get('green') or next(iter(siblings.values()))
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            out = plot_registration_qa(green)
        except Exception as exc:                  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, 'Registration QA', f'Could not build the QA plot:\n{exc}')
            return
        QApplication.restoreOverrideCursor()
        if out:
            QMessageBox.information(self, 'Registration QA', f'Wrote QA plot:\n{out}')
        else:
            QMessageBox.information(self, 'Registration QA',
                                   'No QA plot produced (no white-light image found).')

    def reset_rough_wcs(self) -> None:
        """Discard the star-tied solution and revert every channel to the rough header WCS."""
        if not self._require_file('Reset WCS'):
            return
        reply = QMessageBox.question(
            self, 'Reset WCS',
            'Discard the star-tied solution for this exposure and revert every channel to the '
            'rough header (TCS) pointing?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        from llamas_pyjamas.Utils.register import reset_rough
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            written = reset_rough(self._path)
        except Exception as exc:                  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, 'Reset WCS', f'Could not reset:\n{exc}')
            return
        QApplication.restoreOverrideCursor()
        self._after_registration(f'Reverted {len(written)} file(s) to the rough header WCS.')

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
