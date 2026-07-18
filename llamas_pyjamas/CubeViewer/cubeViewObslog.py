"""Obslog-style file pickers for CubeViewer.

Opening reduced RSS files by their timestamped filename is unfriendly once a run has many
exposures -- ``LLAMAS_2026-05-17_02-49-56.7_RSS_green.fits`` says nothing about *what* it is.
This module lists exposures in a small table keyed on the header (object, exposure time,
observer notes) so the user can pick by what they observed, not by remembering timestamps.

The list is one row per *exposure*, not per file: the three ``_RSS_{blue,green,red}`` planes of
one exposure collapse to a single row. Opening any one pulls in its siblings (see
:func:`RSSScene.open`); applying a sensitivity function calibrates all three.

Two dialogs share the scan and the table:

``ObslogDialog(directory)``
    Single-select. ``File -> Open from obslog``. :attr:`chosen_path` is the representative
    file to load.

``ObslogDialog(directory, multi=True, with_sensfunc=True)``
    Multi-select with a sensitivity-function chooser on top. ``Sensitivity -> Apply``.
    :attr:`chosen_files` is every selected exposure's planes (flattened); :attr:`sensfunc_path`
    is the chosen sensfunc.
"""

import logging
import os
import re
from typing import Dict, List, Optional

from astropy.io import fits
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)

# An RSS plane, at any processing stage: <base>_RSS_<colour>[_FF[_SKYSUB]].fits. The base
# before _RSS_ is the exposure key that groups the three colours into one row.
_RSS_RE = re.compile(r'^(.*)_RSS_(blue|green|red)(?:_FF(?:_SKYSUB)?)?\.fits$', re.IGNORECASE)
_CHANNEL_ORDER = ('blue', 'green', 'red')


def scan_rss_exposures(directory: str) -> List[Dict]:
    """Group the RSS planes in `directory` into one entry per exposure.

    Returns a list of dicts (sorted by base name), each with: ``base`` (the filename stem
    before ``_RSS_``), ``representative`` (the file to open -- green preferred), ``paths`` (all
    plane files, blue/green/red order), ``channels``, and the header summary ``object``,
    ``exptime``, ``notes``. ``error`` is set instead of the header fields if the header could
    not be read. White-light and difference images are skipped.
    """
    if not directory or not os.path.isdir(directory):
        return []

    grouped: Dict[str, Dict[str, str]] = {}
    for name in sorted(os.listdir(directory)):
        low = name.lower()
        if not low.endswith('.fits') or 'white' in low or 'diff' in low:
            continue
        match = _RSS_RE.match(name)
        if not match:
            continue
        base, colour = match.group(1), match.group(2).lower()
        # Keep the least-suffixed plane per (base, colour): a consolidated run has exactly one,
        # but if intermediates linger, prefer the cleanest name for display/opening.
        existing = grouped.setdefault(base, {}).get(colour)
        if existing is None or len(name) < len(os.path.basename(existing)):
            grouped[base][colour] = os.path.join(directory, name)

    entries: List[Dict] = []
    for base in sorted(grouped):
        planes = grouped[base]
        representative = next((planes[c] for c in ('green', 'red', 'blue') if c in planes),
                              next(iter(planes.values())))
        entry = {
            'base': base,
            'representative': representative,
            'paths': [planes[c] for c in _CHANNEL_ORDER if c in planes],
            'channels': [c for c in _CHANNEL_ORDER if c in planes],
            'object': '', 'exptime': None, 'notes': '', 'error': '',
        }
        try:
            hdr = fits.getheader(representative, 0)
            entry['object'] = str(hdr.get('OBJECT', '') or '')
            exptime = hdr.get('SEXPTIME', hdr.get('REXPTIME', hdr.get('EXPTIME')))
            entry['exptime'] = float(exptime) if exptime is not None else None
            notes = hdr.get('OBS-CMNT')
            entry['notes'] = '' if notes is None else str(notes)
        except Exception as err:                           # noqa: BLE001 - a bad file is a row
            entry['error'] = str(err)
            logger.debug('obslog header read failed for %s: %s', representative, err)
        entries.append(entry)
    return entries


class ObslogDialog(QDialog):
    """Table picker over the RSS exposures in a directory.

    Single-select opens one exposure; multi-select (with an optional sensfunc chooser) selects
    a set to flux-calibrate.
    """

    COL_FILE, COL_OBJ, COL_EXP, COL_NOTES = range(4)

    def __init__(self, directory: str, *, multi: bool = False, with_sensfunc: bool = False,
                 title: str = 'Open from obslog', parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 420)
        self.directory = directory or os.getcwd()
        self.multi = multi
        self.with_sensfunc = with_sensfunc
        self.sensfunc_path = ''
        self.chosen_path = ''            # single-select result (representative file)
        self.chosen_files: List[str] = []   # multi-select result (all planes, flattened)
        self._entries: List[Dict] = []

        layout = QVBoxLayout(self)

        # Directory row -- always changeable, so the picker is not tied to the open file's dir.
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel('Directory:'))
        self.dir_label = QLineEdit(self.directory)
        self.dir_label.setReadOnly(True)
        dir_row.addWidget(self.dir_label, 1)
        change_button = QPushButton('Change…')
        change_button.clicked.connect(self._change_directory)
        dir_row.addWidget(change_button)
        layout.addLayout(dir_row)

        # Sensitivity-function chooser (apply flow only).
        if with_sensfunc:
            sens_row = QHBoxLayout()
            sens_row.addWidget(QLabel('Sensitivity function:'))
            self.sens_edit = QLineEdit()
            self.sens_edit.setReadOnly(True)
            self.sens_edit.setPlaceholderText('(none selected)')
            sens_row.addWidget(self.sens_edit, 1)
            sens_button = QPushButton('Browse…')
            sens_button.clicked.connect(self._choose_sensfunc)
            sens_row.addWidget(sens_button)
            layout.addLayout(sens_row)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['File', 'Object', 'Exptime (s)', 'Observer notes'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection if multi
            else QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._update_accept)
        self.table.itemDoubleClicked.connect(self._on_double_click)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self.COL_FILE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_OBJ, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_NOTES, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel)
        self.accept_button = self.buttons.addButton(
            'Apply' if with_sensfunc else 'Open',
            QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._rescan()

    # ------------------------------------------------------------------ scanning

    def _rescan(self) -> None:
        self._entries = scan_rss_exposures(self.directory)
        self.dir_label.setText(self.directory)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self._entries))
        for row, entry in enumerate(self._entries):
            file_item = QTableWidgetItem(os.path.basename(entry['representative']))
            # Carry the entry index so selection survives sorting.
            file_item.setData(Qt.ItemDataRole.UserRole, row)
            if entry['channels']:
                file_item.setToolTip('channels: ' + ', '.join(entry['channels']))
            obj = entry['object'] if not entry['error'] else f"(header error: {entry['error']})"
            exp = '' if entry['exptime'] is None else f"{entry['exptime']:.1f}"
            exp_item = QTableWidgetItem()
            # Numeric sort: store the value in EditRole, the formatted string for display.
            exp_item.setData(Qt.ItemDataRole.EditRole,
                             entry['exptime'] if entry['exptime'] is not None else -1.0)
            exp_item.setData(Qt.ItemDataRole.DisplayRole, exp)
            exp_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, self.COL_FILE, file_item)
            self.table.setItem(row, self.COL_OBJ, QTableWidgetItem(obj))
            self.table.setItem(row, self.COL_EXP, exp_item)
            self.table.setItem(row, self.COL_NOTES, QTableWidgetItem(entry['notes']))
        self.table.setSortingEnabled(True)
        self._update_accept()

    def _change_directory(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, 'Choose directory', self.directory)
        if chosen:
            self.directory = chosen
            self._rescan()

    def _choose_sensfunc(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select sensitivity function', self.directory,
            'Sensitivity FITS (*.fits);;All files (*)')
        if path:
            self.sensfunc_path = path
            self.sens_edit.setText(os.path.basename(path))
            self._update_accept()

    # ------------------------------------------------------------------ selection

    def _selected_entries(self) -> List[Dict]:
        rows = {index.row() for index in self.table.selectionModel().selectedRows()}
        entries = []
        for row in sorted(rows):
            item = self.table.item(row, self.COL_FILE)
            if item is None:
                continue
            idx = item.data(Qt.ItemDataRole.UserRole)
            if idx is not None and 0 <= idx < len(self._entries):
                entries.append(self._entries[idx])
        return entries

    def _update_accept(self) -> None:
        have_selection = bool(self.table.selectionModel().selectedRows())
        ok = have_selection and (not self.with_sensfunc or bool(self.sensfunc_path))
        self.accept_button.setEnabled(ok)

    def _on_double_click(self, _item) -> None:
        if not self.multi and self.table.selectionModel().selectedRows():
            self._accept()

    def _accept(self) -> None:
        entries = self._selected_entries()
        if not entries:
            return
        if self.with_sensfunc and not self.sensfunc_path:
            return
        self.chosen_path = entries[0]['representative']
        self.chosen_files = [p for entry in entries for p in entry['paths']]
        self.accept()
