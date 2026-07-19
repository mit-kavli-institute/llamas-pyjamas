"""
Interactive WCS registration for CubeViewer — the manual fallback.

The automated engine (:mod:`llamas_pyjamas.Utils.register`) makes assumptions (Gaia present,
stars bright enough, sky-residual banding manageable) that will not hold for every observer and
every field. This dialog is the escape hatch: the user clicks the stars, and the tool supplies the
*detection* and *matching* the automated path could not — while sharing its tested *centroid,
solve and write* core, so a hand-made solution is identical in kind to an auto one and downstream
combining cannot tell them apart.

Because CubeViewer displays the white-light in an external DS9 over XPA (there is no Qt image
canvas), clicks are read from the DS9 crosshair on demand and overlays are drawn as DS9 regions.

Flow: overlay Gaia through the current (rough) WCS; move the DS9 crosshair onto a star and
**Grab** -> the click is snapped to the flux-weighted fibre centroid (``fibre_centroid`` on the
real fibres, never the rendered hexagons) and paired with a sky position, either auto-paired to
the nearest Gaia source or typed in by hand (for fields with no Gaia); solve (1 pair -> shift,
2 -> +rotation, 3+ -> full, same rotation cap as auto) and redraw Gaia through the new WCS so the
fit is seen to converge; Accept writes via :func:`register._write_frame_solution`.

Two deliberate concessions:
  * **Force raw position** — a last resort that uses the raw clicked position without centroiding,
    for a star too faint/blended for the centroider to trust. Flagged distinctly in the overlay
    and provenance so it never passes for a normal solve.
  * **Hold rotation for the block** — solve rotation once on the best-star frame, then reuse it
    (via :func:`register.register_block` ``fixed_rotation``) so the block's other dithers only
    need their translation, matching the automated block-rotation design.

Classes
-------
InteractiveWCSSession   Headless solve state (pairs, WCS, held rotation) — testable without Qt
RegisterDialog          The Qt dialog around a session, driving DS9
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from llamas_pyjamas.Utils.centroid import fibre_centroid
from llamas_pyjamas.Utils.register import (_axis_pa, _rough_wcs, _write_frame_solution,
                                           query_gaia, solve_wcs)
from llamas_pyjamas.Utils.wcsLlamas import IFU_PA_OFFSET

logger = logging.getLogger(__name__)

#: Our DS9 regions carry this tag so they clear independently of the picker's ``cubeview`` markers.
REG_TAG = 'cubeview-reg'

#: Interactive matching is deliberate, so the pointing error may be large (bad TCS) -- do not let
#: the automated shift cap reject a solve the user explicitly built.
INTERACTIVE_MAX_SHIFT = 120.0


class InteractiveWCSSession:
    """Headless state for one interactive WCS solve, independent of Qt/DS9 so it is unit-testable.

    Works entirely in fibre-map coordinates (1 unit = one fibre spacing), the same frame the
    automated engine uses, so the resulting WCS and per-fibre RA/DEC are written identically.
    """

    def __init__(self, positions, flux, ra, dec, pa, gaia=None, *, centroid_radius=1.5):
        self.positions = np.asarray(positions, dtype=float)          # (N, 2) fibre-map
        self.flux = np.asarray(flux, dtype=float)                    # (N,)
        self.ra, self.dec, self.pa = ra, dec, pa
        self.gaia = gaia if gaia is not None else SkyCoord([], [], unit='deg')
        self.centroid_radius = float(centroid_radius)
        self.pairs: List[dict] = []          # {'xy', 'sky', 'forced', 'source'}
        self.held_rotation: Optional[float] = None
        self.rough = _rough_wcs(ra, dec, pa) if ra is not None else None
        self.wcs = self.rough
        self.rms = float('nan')
        self.n_stars = 0
        self.refined = False

    # ---- clicks -> centroids -> pairs ----
    def grab(self, x_fm: float, y_fm: float, force: bool = False) -> Tuple[Tuple[float, float], bool]:
        """Snap a clicked fibre-map position to the flux-weighted fibre centroid.

        Returns ``((x, y), forced)``. ``forced`` is True when the raw click was used verbatim --
        either because ``force`` was requested (last resort) or the centroid did not converge."""
        if force:
            return (float(x_fm), float(y_fm)), True
        c = fibre_centroid(self.positions[:, 0], self.positions[:, 1], self.flux,
                           guess=(float(x_fm), float(y_fm)), radius=self.centroid_radius)
        if c is None:
            return (float(x_fm), float(y_fm)), True
        return (float(c.x), float(c.y)), False

    def nearest_gaia(self, xy: Tuple[float, float]) -> Optional[Tuple[SkyCoord, float]]:
        """Nearest Gaia source (SkyCoord, separation_arcsec) to a fibre-map point through the
        current WCS, or None when there is no Gaia catalogue / no WCS."""
        if self.wcs is None or len(self.gaia) == 0:
            return None
        sky = self.wcs.pixel_to_world(xy[0], xy[1])
        seps = sky.separation(self.gaia).arcsec
        i = int(np.argmin(seps))
        return self.gaia[i], float(seps[i])

    def add_pair(self, xy, sky: SkyCoord, forced: bool = False, source: str = 'gaia') -> None:
        self.pairs.append({'xy': (float(xy[0]), float(xy[1])), 'sky': sky,
                           'forced': bool(forced), 'source': source})

    def remove_pair(self, index: int) -> None:
        if 0 <= index < len(self.pairs):
            self.pairs.pop(index)

    def clear(self) -> None:
        self.pairs.clear()
        self.wcs = self.rough
        self.rms, self.n_stars, self.refined = float('nan'), 0, False

    # ---- solve ----
    def solve(self, allow_rotation: bool = True, max_rot_deg: float = 3.0) -> Optional[dict]:
        """Solve the WCS from the current pairs. Rotation is fitted only with >=2 pairs, rotation
        allowed, and no held block rotation; otherwise translation only (held rotation applied).
        Returns a summary dict or None if the solve failed."""
        if not self.pairs or self.ra is None:
            return None
        xy = [p['xy'] for p in self.pairs]
        sky = SkyCoord([p['sky'].ra.deg for p in self.pairs] * u.deg,
                       [p['sky'].dec.deg for p in self.pairs] * u.deg)
        base = _rough_wcs(self.ra, self.dec, self.pa, self.held_rotation or 0.0)
        refine = allow_rotation and self.held_rotation is None and len(xy) >= 2
        out = solve_wcs(xy, sky, base, refine_rotation=refine, max_rot_deg=max_rot_deg,
                        max_shift_arcsec=INTERACTIVE_MAX_SHIFT)
        if out is None:
            return None
        self.wcs, self.rms, _rot, self.refined, self.n_stars = out
        return {'n': self.n_stars, 'rms': self.rms, 'refined': self.refined,
                'drot': self.rotation_offset()}

    def rotation_offset(self) -> float:
        """Current PA(+x) minus the calibrated PA, in degrees (the block rotation to hold)."""
        if self.wcs is None or self.rough is None:
            return 0.0
        return ((_axis_pa(self.wcs) - _axis_pa(self.rough) + 180.0) % 360.0) - 180.0

    def hold_rotation(self) -> float:
        """Freeze the current rotation as the block rotation; subsequent solves are shift-only."""
        self.held_rotation = self.rotation_offset()
        return self.held_rotation

    def release_rotation(self) -> None:
        self.held_rotation = None

    def provenance(self) -> dict:
        """FIBERWCS/FIBERMAP provenance for :func:`register._write_frame_solution`."""
        manual = any(p['source'] == 'manual' for p in self.pairs)
        has_gaia = any(p['source'] == 'gaia' for p in self.pairs)
        catalog = ('GaiaDR3+manual' if manual and has_gaia else
                   'manual' if manual else 'GaiaDR3')
        return {'method': 'interactive', 'tier': 'manual' if manual else 'gaia', 'refined': True,
                'pa_offset': float(IFU_PA_OFFSET), 'catalog': catalog,
                'rms': float(self.rms) if np.isfinite(self.rms) else float('nan'),
                'nstars': int(self.n_stars)}


# --------------------------------------------------------------------------- Qt dialog

try:                                          # keep the module importable (and testable) headless
    from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit,
                                 QListWidget, QMessageBox, QPushButton, QVBoxLayout, QWidget)
    _HAVE_QT = True
except Exception:                             # noqa: BLE001 - Qt optional for import/tests
    _HAVE_QT = False


if _HAVE_QT:
    class RegisterDialog(QDialog):
        """Interactive WCS dialog around an :class:`InteractiveWCSSession`, driving DS9.

        Parameters
        ----------
        scene : RSSScene         the loaded scene (fibre positions, flux, pointing, pixel step)
        ds9 : DS9                transport for crosshair reads and region overlays
        path : str               the loaded RSS path (its siblings are written on Accept)
        """

        def __init__(self, scene, ds9, path, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self.scene = scene
            self.ds9 = ds9
            self.path = path
            self.written: List[str] = []
            self.setWindowTitle(f'Refine WCS — {os.path.basename(path)}')
            self.resize(520, 560)

            band = scene.wavelength_range()
            flux = scene.fibre_flux(band[0], band[1])
            gaia = SkyCoord([], [], unit='deg')
            if scene.ra is not None:
                gaia = query_gaia(scene.ra, scene.dec)
            self.session = InteractiveWCSSession(scene.positions, flux, scene.ra, scene.dec,
                                                 scene.pa, gaia)

            self._build_ui()
            self._enter_crosshair_mode()
            self._redraw_overlay()
            self._refresh_status()

        # ---- UI ----
        def _build_ui(self) -> None:
            layout = QVBoxLayout(self)

            info = QLabel('Move the DS9 crosshair onto a star, then Grab. Pair it to Gaia '
                          '(auto) or type RA/DEC, and the fit updates live.')
            info.setWordWrap(True)
            layout.addWidget(info)

            mode_row = QHBoxLayout()
            mode_row.addWidget(QLabel('Pair to:'))
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(['Nearest Gaia', 'Manual RA/DEC'])
            has_gaia = len(self.session.gaia) > 0
            if not has_gaia:
                self.mode_combo.setCurrentIndex(1)
            self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
            mode_row.addWidget(self.mode_combo)
            mode_row.addStretch(1)
            self.force_box = QCheckBox('Force raw position (last resort)')
            self.force_box.setToolTip('Use the clicked position verbatim without centroiding, for '
                                      'a star too faint or blended to centroid. Flagged in the '
                                      'overlay and provenance.')
            mode_row.addWidget(self.force_box)
            layout.addLayout(mode_row)

            coord_row = QHBoxLayout()
            coord_row.addWidget(QLabel('RA'))
            self.ra_edit = QLineEdit()
            self.ra_edit.setPlaceholderText('deg or h:m:s')
            coord_row.addWidget(self.ra_edit)
            coord_row.addWidget(QLabel('Dec'))
            self.dec_edit = QLineEdit()
            self.dec_edit.setPlaceholderText('deg or d:m:s')
            coord_row.addWidget(self.dec_edit)
            self.coord_widgets = (self.ra_edit, self.dec_edit)
            layout.addLayout(coord_row)

            grab_row = QHBoxLayout()
            self.grab_button = QPushButton('Grab star at crosshair')
            self.grab_button.clicked.connect(self._on_grab)
            grab_row.addWidget(self.grab_button)
            self.remove_button = QPushButton('Remove selected')
            self.remove_button.clicked.connect(self._on_remove)
            grab_row.addWidget(self.remove_button)
            self.clear_button = QPushButton('Clear')
            self.clear_button.clicked.connect(self._on_clear)
            grab_row.addWidget(self.clear_button)
            layout.addLayout(grab_row)

            self.pair_list = QListWidget()
            layout.addWidget(self.pair_list, stretch=1)

            self.hold_box = QCheckBox('Hold this rotation for the block')
            self.hold_box.setToolTip('Freeze the solved rotation and reuse it for the other '
                                     'dithers of this pointing (translation only), matching the '
                                     'automated block-rotation design.')
            self.hold_box.toggled.connect(self._on_hold_toggled)
            layout.addWidget(self.hold_box)

            self.status = QLabel('')
            self.status.setWordWrap(True)
            layout.addWidget(self.status)

            btn_row = QHBoxLayout()
            btn_row.addStretch(1)
            self.block_button = QPushButton('Apply to block…')
            self.block_button.setToolTip('Write this rotation to every dither of this pointing '
                                         '(auto translation per frame).')
            self.block_button.clicked.connect(self._on_apply_block)
            btn_row.addWidget(self.block_button)
            self.accept_button = QPushButton('Accept (write this frame)')
            self.accept_button.clicked.connect(self._on_accept)
            btn_row.addWidget(self.accept_button)
            cancel = QPushButton('Cancel')
            cancel.clicked.connect(self.reject)
            btn_row.addWidget(cancel)
            layout.addLayout(btn_row)

            self._on_mode_changed()

        # ---- helpers ----
        def _manual_mode(self) -> bool:
            return self.mode_combo.currentText() == 'Manual RA/DEC'

        def _on_mode_changed(self, *args) -> None:
            for w in self.coord_widgets:
                w.setEnabled(self._manual_mode())

        def _enter_crosshair_mode(self) -> None:
            try:
                self.ds9.set('mode crosshair')
            except Exception as exc:            # noqa: BLE001
                logger.debug('Could not set DS9 crosshair mode: %s', exc)

        def _step(self) -> float:
            return self.scene._step()

        def _fm_to_image(self, x_fm: float, y_fm: float) -> Tuple[float, float]:
            step = self._step()
            return (x_fm / step + 1.0, y_fm / step + 1.0)

        def _image_to_fm(self, x_pix: float, y_pix: float) -> Tuple[float, float]:
            return self.scene._to_world(x_pix, y_pix)

        # ---- interaction ----
        def _on_grab(self) -> None:
            if self.session.ra is None:
                QMessageBox.warning(self, 'Refine WCS', 'This frame has no header pointing, so '
                                    'there is no initial WCS to refine.')
                return
            try:
                x_pix, y_pix = self.ds9.crosshair('image')
            except Exception as exc:            # noqa: BLE001
                QMessageBox.warning(self, 'Refine WCS', f'Could not read the DS9 crosshair: {exc}')
                return
            x_fm, y_fm = self._image_to_fm(x_pix, y_pix)
            xy, forced = self.session.grab(x_fm, y_fm, force=self.force_box.isChecked())

            if self._manual_mode():
                sky = self._parse_manual_coord()
                if sky is None:
                    return
                source = 'manual'
            else:
                near = self.session.nearest_gaia(xy)
                if near is None:
                    QMessageBox.information(self, 'Refine WCS', 'No Gaia sources to pair with — '
                                           'switch to Manual RA/DEC.')
                    return
                sky, sep = near
                source = 'gaia'

            self.session.add_pair(xy, sky, forced=forced, source=source)
            self._resolve_and_refresh()

        def _parse_manual_coord(self) -> Optional[SkyCoord]:
            ra_txt, dec_txt = self.ra_edit.text().strip(), self.dec_edit.text().strip()
            if not ra_txt or not dec_txt:
                QMessageBox.information(self, 'Refine WCS', 'Enter RA and Dec for this star.')
                return None
            try:
                unit = (u.hourangle, u.deg) if (':' in ra_txt or ' ' in ra_txt) else (u.deg, u.deg)
                return SkyCoord(ra_txt, dec_txt, unit=unit)
            except Exception as exc:            # noqa: BLE001
                QMessageBox.warning(self, 'Refine WCS', f'Could not parse RA/Dec: {exc}')
                return None

        def _on_remove(self) -> None:
            row = self.pair_list.currentRow()
            if row >= 0:
                self.session.remove_pair(row)
                self._resolve_and_refresh()

        def _on_clear(self) -> None:
            self.session.clear()
            self._resolve_and_refresh()

        def _on_hold_toggled(self, checked: bool) -> None:
            if checked:
                drot = self.session.hold_rotation()
                self.status.setText(f'Rotation held at {drot:+.2f}° for the block.')
            else:
                self.session.release_rotation()
            self._resolve_and_refresh()

        def _resolve_and_refresh(self) -> None:
            self.session.solve()
            self._redraw_overlay()
            self._refresh_list()
            self._refresh_status()

        # ---- draw / status ----
        def _redraw_overlay(self) -> None:
            step = self._step()
            lines = ['image']
            if len(self.session.gaia) and self.session.wcs is not None:
                gx, gy = self.session.wcs.world_to_pixel(self.session.gaia)
                for x, y in zip(np.atleast_1d(gx), np.atleast_1d(gy)):
                    px, py = self._fm_to_image(float(x), float(y))
                    lines.append(f'circle({px:.2f},{py:.2f},5) # color=green tag={{{REG_TAG}}}')
            for p in self.session.pairs:
                px, py = self._fm_to_image(*p['xy'])
                # Red centroid crosses read clearly on the white star cores and the grey
                # background (yellow washed out on white); forced positions stay magenta so the
                # last-resort ones remain visually distinct.
                colour = 'magenta' if p['forced'] else 'red'
                shape = 'x' if p['forced'] else 'cross'
                lines.append(f'point({px:.2f},{py:.2f}) # point={shape} 16 color={colour} '
                             f'width=3 tag={{{REG_TAG}}}')
            try:
                self.ds9.delete_region_group(REG_TAG)
                self.ds9.set_regions('\n'.join(lines))
            except Exception as exc:            # noqa: BLE001
                logger.debug('Could not draw registration overlay: %s', exc)

        def _refresh_list(self) -> None:
            self.pair_list.clear()
            for i, p in enumerate(self.session.pairs):
                c = p['sky']
                tag = ' [forced]' if p['forced'] else ''
                src = 'Gaia' if p['source'] == 'gaia' else 'manual'
                self.pair_list.addItem(
                    f'{i + 1}. {src}{tag}  {c.ra.deg:.5f}, {c.dec.deg:+.5f}')

        def _refresh_status(self) -> None:
            n = len(self.session.pairs)
            if n == 0:
                self.status.setText('No stars yet. Grab at least one (three+ for a full solve).')
                return
            rms = self.session.rms
            drot = self.session.rotation_offset()
            kind = ('translation+rotation' if self.session.refined else
                    'rotation held' if self.session.held_rotation is not None else 'translation')
            rms_txt = f'{rms:.2f}"' if np.isfinite(rms) else 'n/a'
            self.status.setText(f'{n} star(s) | {kind} | RMS {rms_txt} | rotation {drot:+.2f}°')

        # ---- write ----
        def _on_accept(self) -> None:
            if not self.session.pairs or self.session.wcs is None:
                QMessageBox.information(self, 'Refine WCS', 'Grab at least one star first.')
                return
            from llamas_pyjamas.CubeViewer.cubeViewRSS import channel_siblings
            siblings = channel_siblings(self.path) or {'': self.path}
            det = siblings.get('green') or next(iter(siblings.values()))
            try:
                self.written = _write_frame_solution(det, siblings, self.session.wcs,
                                                     self.session.provenance())
            except Exception as exc:            # noqa: BLE001
                QMessageBox.critical(self, 'Refine WCS', f'Could not write the solution:\n{exc}')
                return
            self.scene.refined_wcs = self.session.wcs   # so the next display shows the new grid
            self._clear_overlay()
            self.accept()

        def _on_apply_block(self) -> None:
            drot = (self.session.held_rotation if self.session.held_rotation is not None
                    else self.session.rotation_offset())
            from llamas_pyjamas.CubeViewer.cubeViewObslog import ObslogDialog
            picker = ObslogDialog(os.path.dirname(self.path), multi=True,
                                  title='Apply WCS rotation to block (pick the dithers)',
                                  parent=self)
            if not picker.exec():
                return
            paths = getattr(picker, 'chosen_files', None) or []
            if not paths:
                return
            reply = QMessageBox.question(
                self, 'Apply to block',
                f'Register {len(paths)} frame(s) holding rotation {drot:+.2f}° '
                '(auto translation per frame)?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
            from llamas_pyjamas.Utils.register import register_block
            results = register_block(paths, fixed_rotation=drot)
            for r in results.values():
                self.written.extend(r.files)
            n_ref = sum(1 for r in results.values() if r.refined)
            self._clear_overlay()
            QMessageBox.information(self, 'Apply to block',
                                   f'Registered {n_ref}/{len(paths)} frame(s) at rotation '
                                   f'{drot:+.2f}°.')
            self.accept()

        def _clear_overlay(self) -> None:
            try:
                self.ds9.delete_region_group(REG_TAG)
            except Exception as exc:            # noqa: BLE001
                logger.debug('Could not clear overlay: %s', exc)

        def closeEvent(self, event) -> None:    # noqa: N802 - Qt naming
            self._clear_overlay()
            super().closeEvent(event)
