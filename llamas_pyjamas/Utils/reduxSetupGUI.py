"""
Interactive setup GUI for building a llamas-pyjamas reduction config file.

Scans a raw LLAMAS data directory, tabulates the primary-header metadata of
every MEF exposure (skipping whitelight and difference images), displays the
observer's notes from obslog_usernotes.txt, and lets the user assign files to
pipeline roles (red/green/blue lamp flats, twilight flat, science frames, raw
bias frames). Selected raw biases are median-combined into master bias files
with BiasLlamas, and a ready-to-run config file is generated from the package
template (example_config.txt) with the chosen paths substituted in.

Usage:
    python -m llamas_pyjamas.Utils.reduxSetupGUI [data_dir] [-o config_out.txt]
"""

import argparse
import logging
import os
import re
import shutil
import sys
import traceback
from glob import glob

from astropy.io import fits
from PyQt6 import QtCore, QtGui, QtWidgets

import llamas_pyjamas
from llamas_pyjamas.Bias.llamasBias import BiasLlamas

logger = logging.getLogger(__name__)

TEMPLATE_PATH = os.path.join(os.path.dirname(llamas_pyjamas.__file__), 'example_config.txt')

# Friendly labels for the PRODCATG header values
PRODCATG_LABELS = {
    'CAL.R-BIA': 'Bias',
    'CAL.R-DRK': 'Dark',
    'CAL.R-FLT': 'Flat',
    'CAL.R-SKY': 'SkyFlat',
    'CAL.R-ARC': 'Arc',
    'SCI.R-SL': 'Science',
    'SCI.R-DT': 'Science (dither)',
}

# Role definitions: key -> (display label, multi-file?, row tint, text color)
BLACK = QtGui.QColor('black')
ROLES = {
    'red_flat': ('Red Flat', False, QtGui.QColor(255, 200, 200), BLACK),
    'green_flat': ('Green Flat', False, QtGui.QColor(200, 240, 200), BLACK),
    'blue_flat': ('Blue Flat', False, QtGui.QColor(200, 215, 255), BLACK),
    'red_twilight': ('Red Twilight', False, QtGui.QColor(255, 224, 214), BLACK),
    'green_twilight': ('Green Twilight', False, QtGui.QColor(220, 245, 220), BLACK),
    'blue_twilight': ('Blue Twilight', False, QtGui.QColor(221, 231, 255), BLACK),
    'red_arc': ('Red Arc', False, QtGui.QColor(248, 173, 173), BLACK),
    'green_arc': ('Green Arc', False, QtGui.QColor(168, 226, 168), BLACK),
    'blue_arc': ('Blue Arc', False, QtGui.QColor(173, 196, 250), BLACK),
    'flux_standard': ('Flux Standard', True, QtGui.QColor(255, 224, 130), BLACK),
    'science': ('Science', True, QtGui.QColor(230, 205, 250), BLACK),
    'bias': ('Bias', True, QtGui.QColor(212, 212, 212), BLACK),
    'bad': ('BAD', True, QtGui.QColor(90, 90, 90), QtGui.QColor('white')),
}

NOTES_FILENAME = 'obslog_usernotes.txt'

# Config keys that map 1:1 onto single-file roles
KEY_TO_ROLE = {
    'red_flat_file': 'red_flat',
    'green_flat_file': 'green_flat',
    'blue_flat_file': 'blue_flat',
    'red_twilight_flat': 'red_twilight',
    'green_twilight_flat': 'green_twilight',
    'blue_twilight_flat': 'blue_twilight',
    'red_arc_file': 'red_arc',
    'green_arc_file': 'green_arc',
    'blue_arc_file': 'blue_arc',
}

ARC_HEADER = '# WAVELENGTH CALIBRATION EXPOSURES (selected in the setup GUI)'
BAD_HEADER = '# Exposures flagged as BAD data in the setup GUI (for the record; unused):'


def parse_config(path):
    """Parse a config file into (key -> value string, list of bad-flagged paths).

    Same line format as reduce.py's loader; '#bad:' lines written by this GUI
    are returned separately.
    """
    config, bad = {}, []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#bad:'):
                bad.append(line[len('#bad:'):].strip())
            elif line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config, bad


def scan_directory(data_dir, progress_callback=None):
    """Read primary headers of all relevant MEF files in data_dir.

    Skips filenames containing 'white' or 'diff'. Returns a list of dicts with
    keys: path, filename, object, exptime, type, comment, readmode. Files whose
    headers cannot be read are returned with type 'ERR' rather than raised.
    """
    files = sorted(glob(os.path.join(data_dir, '*.fits')))
    files = [f for f in files
             if 'white' not in os.path.basename(f).lower()
             and 'diff' not in os.path.basename(f).lower()]
    entries = []
    for i, path in enumerate(files):
        if progress_callback and not progress_callback(i, len(files), os.path.basename(path)):
            break
        entry = {'path': path, 'filename': os.path.basename(path), 'object': '',
                 'exptime': None, 'type': '', 'comment': '', 'readmode': '',
                 'standard': None}
        try:
            hdr = fits.getheader(path, 0)
            entry['object'] = str(hdr.get('OBJECT', '') or '')
            exptime = hdr.get('SEXPTIME', hdr.get('REXPTIME'))
            entry['exptime'] = float(exptime) if exptime is not None else None
            prodcatg = str(hdr.get('PRODCATG', '') or '')
            entry['type'] = PRODCATG_LABELS.get(prodcatg, prodcatg)
            comment = hdr.get('OBS-CMNT')
            entry['comment'] = '' if comment is None else str(comment)
            entry['readmode'] = str(hdr.get('READ-MDE', '') or '')
            # A science pointing sitting on a catalogue standard is very probably an
            # exposure of that standard -- headers cannot tell them apart, so this is
            # the only signal. Only SCIENCE frames are checked; calibrations never are.
            if str(hdr.get('OBS TYPE', '')).upper().startswith('SCI'):
                entry['standard'] = _match_standard(hdr)
        except Exception as err:
            entry['type'] = 'ERR'
            entry['comment'] = f'header read failed: {err}'
        entries.append(entry)
    return entries


def _match_standard(header):
    """Crossmatch a header's pointing against the standards catalogue.

    Returns a ``(name, separation_arcsec)`` tuple, or None on no match or if the bundled
    catalogue is unavailable -- a missing catalogue must not break directory scanning.
    """
    try:
        from llamas_pyjamas.Flux.fluxStandards import load_catalog
        match = load_catalog().match_header(header)
    except Exception as err:                            # noqa: BLE001
        logger.debug('standard crossmatch skipped: %s', err)
        return None
    return (match.name, match.separation_arcsec) if match else None


def generate_config(assignments, output_dir, slow_bias=None, fast_bias=None,
                    template_path=TEMPLATE_PATH):
    """Build config-file text from a template.

    assignments maps role key -> path (single roles) or list of paths
    ('science', 'bias', 'bad'). Assigned keys are substituted in place;
    unassigned flat / twilight / arc keys are commented out so the pipeline
    falls back to its defaults. template_path may be a previously generated
    config, in which case hand-edited parameters are preserved; existing
    slow/fast_bias_file lines are kept when no replacement is supplied.
    """
    key_values = {
        # singular twilight_flat is superseded by the per-colour keys;
        # comment it out and let each colour fall back per the template rules
        'twilight_flat': None,
        'science_files': ', '.join(assignments.get('science', [])) or None,
        'flux_standard_files': ', '.join(assignments.get('flux_standard', [])) or None,
        'trace_output_dir': os.path.join(output_dir, 'traces'),
        'extraction_output_dir': os.path.join(output_dir, 'extractions'),
        'cube_output_dir': os.path.join(output_dir, 'cubes'),
        'slow_bias_file': slow_bias,
        'fast_bias_file': fast_bias,
    }
    for key, role in KEY_TO_ROLE.items():
        key_values[key] = assignments.get(role)
    preserve_if_unset = {'slow_bias_file', 'fast_bias_file'}

    with open(template_path, 'r') as f:
        lines = f.readlines()

    out = []
    substituted = set()
    for line in lines:
        # drop any bad-flag block from a previous GUI run; re-appended below
        if line.startswith('#bad:') or line.rstrip('\n') == BAD_HEADER:
            continue
        m = re.match(r'^(#?)\s*([A-Za-z_][A-Za-z0-9_]*)\s*=', line)
        if m and m.group(2) in key_values and m.group(2) not in substituted:
            key = m.group(2)
            substituted.add(key)
            value = key_values[key]
            if value is not None:
                out.append(f'{key} = {value}\n')
            elif not m.group(1) and key not in preserve_if_unset:
                # active template line for an unassigned key: comment it out
                out.append('#' + line)
            else:
                out.append(line)
        else:
            out.append(line)

    # arcs assigned but absent from the template go in an appended section
    new_arcs = {f'{c}_arc_file': assignments.get(f'{c}_arc') for c in ('red', 'green', 'blue')}
    new_arcs = {k: v for k, v in new_arcs.items() if v and k not in substituted}
    if new_arcs:
        if not any(ARC_HEADER in line for line in out):
            out.append('\n#==============================================================================\n'
                       f'{ARC_HEADER}\n'
                       '#==============================================================================\n'
                       '# Per-colour arc keys are not yet consumed by the pipeline; for a new\n'
                       '# wavelength solution set arc_file with generate_new_wavelength_soln = True.\n')
        for key, path in new_arcs.items():
            out.append(f'{key} = {path}\n')

    # flux_standard_files assigned but absent from the template goes in an appended line.
    # An existing config written before this feature has no such key for the substitution
    # loop above to fill, so without this the standards would be silently dropped on rewrite.
    if assignments.get('flux_standard') and 'flux_standard_files' not in substituted:
        out.append('\n# Spectrophotometric standard-star exposures (identified in the setup '
                   'GUI by\n# crossmatching the pointing against the bundled catalogue). '
                   'Extracted like\n# science; used in postprocessing to build a sensitivity '
                   'function.\n')
        out.append('flux_standard_files = ' + ', '.join(assignments['flux_standard']) + '\n')

    if assignments.get('bad'):
        out.append(f'\n{BAD_HEADER}\n')
        for path in assignments['bad']:
            out.append(f'#bad: {path}\n')

    return ''.join(out)


def combine_biases(bias_files_by_mode, output_dir, progress_callback=None):
    """Median-combine raw bias frames into master bias files per readout mode.

    bias_files_by_mode maps 'SLOW'/'FAST' -> list of raw MEF paths. Returns a
    dict mode -> path of the written {slow,fast}_master_bias.fits.
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs = {}
    for mode, files in bias_files_by_mode.items():
        if progress_callback:
            progress_callback(f'Combining {len(files)} {mode} bias frames...')
        bl = BiasLlamas(files)
        bl.bias_path = output_dir
        bl.master_bias()
        combined = os.path.join(output_dir, 'combined_bias.fits')
        final = os.path.join(output_dir, f'{mode.lower()}_master_bias.fits')
        shutil.move(combined, final)
        outputs[mode] = final
    return outputs


class BiasWorker(QtCore.QThread):
    """Runs the bias combination off the GUI thread."""
    status = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, bias_files_by_mode, output_dir, parent=None):
        super().__init__(parent)
        self.bias_files_by_mode = bias_files_by_mode
        self.output_dir = output_dir

    def run(self):
        try:
            outputs = combine_biases(self.bias_files_by_mode, self.output_dir,
                                     progress_callback=self.status.emit)
            self.finished_ok.emit(outputs)
        except Exception:
            self.failed.emit(traceback.format_exc())


class ReduxSetupWindow(QtWidgets.QMainWindow):

    COL_FILE, COL_OBJ, COL_EXP, COL_TYPE, COL_READ, COL_CMNT, COL_ROLE = range(7)

    def __init__(self, data_dir=None, config_out=None):
        super().__init__()
        self.setWindowTitle('LLAMAS Reduction Setup')
        self.resize(1250, 850)
        self.entries = []
        self.roles = {}        # path -> set of role keys
        self.prior_bias = {}   # READ-MDE -> master bias path from a loaded config
        self._loaded_config_path = None
        self._bias_worker = None
        self._ds9 = None
        self._build_ui()
        if config_out:
            self.configEdit.setText(os.path.abspath(config_out))
        if data_dir:
            self.dirEdit.setText(os.path.abspath(data_dir))
            self.load_directory()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # Top bar: data directory
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel('Data directory:'))
        self.dirEdit = QtWidgets.QLineEdit()
        top.addWidget(self.dirEdit, stretch=1)
        browseBtn = QtWidgets.QPushButton('Browse…')
        browseBtn.clicked.connect(self.browse_directory)
        top.addWidget(browseBtn)
        rescanBtn = QtWidgets.QPushButton('Rescan')
        rescanBtn.clicked.connect(self.load_directory)
        top.addWidget(rescanBtn)
        vbox.addLayout(top)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        vbox.addWidget(splitter, stretch=1)

        # Observation table
        tableBox = QtWidgets.QWidget()
        tableLayout = QtWidgets.QVBoxLayout(tableBox)
        tableLayout.setContentsMargins(0, 0, 0, 0)
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ['Filename', 'Object', 'Exp (s)', 'Type', 'Readout', 'Observer Comments', 'Role'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        self.table.setColumnWidth(self.COL_FILE, 330)
        self.table.setColumnWidth(self.COL_OBJ, 180)
        self.table.setColumnWidth(self.COL_EXP, 70)
        self.table.setColumnWidth(self.COL_TYPE, 110)
        self.table.setColumnWidth(self.COL_READ, 70)
        self.table.setColumnWidth(self.COL_CMNT, 300)
        self.table.setColumnWidth(self.COL_ROLE, 110)
        self.table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.table_context_menu)
        tableLayout.addWidget(self.table)

        # Assignment buttons: colors across, calibration type down, plus the
        # multi-file roles stacked alongside to keep the panel compact.
        def role_button(role, text):
            _label, _multi, color, fg = ROLES[role]
            btn = QtWidgets.QPushButton(text)
            btn.setStyleSheet(f'background-color: {color.name()}; color: {fg.name()};')
            btn.clicked.connect(lambda _checked, r=role: self.assign_role(r))
            return btn

        btnRow = QtWidgets.QHBoxLayout()
        calGrid = QtWidgets.QGridLayout()
        calGrid.addWidget(QtWidgets.QLabel('Assign selection as:'), 0, 0)
        for col, color in enumerate(('Red', 'Green', 'Blue'), start=1):
            hdr = QtWidgets.QLabel(color)
            hdr.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            calGrid.addWidget(hdr, 0, col)
        for row, (kind, suffix) in enumerate(
                [('Flat', 'flat'), ('Twilight', 'twilight'), ('Arc', 'arc')], start=1):
            calGrid.addWidget(QtWidgets.QLabel(kind), row, 0)
            for col, color in enumerate(('red', 'green', 'blue'), start=1):
                calGrid.addWidget(role_button(f'{color}_{suffix}', color.capitalize()), row, col)
        btnRow.addLayout(calGrid)
        btnRow.addSpacing(25)

        miscGrid = QtWidgets.QGridLayout()
        miscGrid.addWidget(role_button('science', 'Science'), 0, 0)
        miscGrid.addWidget(role_button('flux_standard', 'Flux Standard'), 1, 0)
        miscGrid.addWidget(role_button('bias', 'Bias'), 2, 0)
        miscGrid.addWidget(role_button('bad', 'Flag Bad'), 0, 1)
        clearBtn = QtWidgets.QPushButton('Clear Role')
        clearBtn.clicked.connect(self.clear_role)
        miscGrid.addWidget(clearBtn, 1, 1)
        btnRow.addLayout(miscGrid)
        btnRow.addStretch(1)
        tableLayout.addLayout(btnRow)
        splitter.addWidget(tableBox)

        # Observer notes pane
        notesBox = QtWidgets.QWidget()
        notesLayout = QtWidgets.QVBoxLayout(notesBox)
        notesLayout.setContentsMargins(0, 0, 0, 0)
        notesLayout.addWidget(QtWidgets.QLabel('Observer notes (obslog_usernotes.txt):'))
        self.notesView = QtWidgets.QPlainTextEdit()
        self.notesView.setReadOnly(True)
        notesLayout.addWidget(self.notesView)
        splitter.addWidget(notesBox)
        splitter.setSizes([600, 200])

        # Bottom bar: outputs and config write
        form = QtWidgets.QGridLayout()
        form.addWidget(QtWidgets.QLabel('Output directory:'), 0, 0)
        self.outdirEdit = QtWidgets.QLineEdit()
        form.addWidget(self.outdirEdit, 0, 1)
        outBrowse = QtWidgets.QPushButton('Browse…')
        outBrowse.clicked.connect(self.browse_outdir)
        form.addWidget(outBrowse, 0, 2)
        form.addWidget(QtWidgets.QLabel('Config file:'), 1, 0)
        self.configEdit = QtWidgets.QLineEdit()
        form.addWidget(self.configEdit, 1, 1)
        loadBtn = QtWidgets.QPushButton('Load Config')
        loadBtn.clicked.connect(self.load_config_clicked)
        form.addWidget(loadBtn, 1, 2)
        self.writeBtn = QtWidgets.QPushButton('Write Config')
        self.writeBtn.clicked.connect(self.write_config)
        form.addWidget(self.writeBtn, 1, 3)
        vbox.addLayout(form)

        self.summaryLabel = QtWidgets.QLabel()
        vbox.addWidget(self.summaryLabel)
        self.update_summary()

    # ------------------------------------------------------- directory scan
    def browse_directory(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select LLAMAS data directory', self.dirEdit.text() or os.path.expanduser('~'))
        if d:
            self.dirEdit.setText(d)
            self.load_directory()

    def browse_outdir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select output directory', self.outdirEdit.text() or self.dirEdit.text())
        if d:
            self.outdirEdit.setText(d)

    def load_directory(self):
        data_dir = self.dirEdit.text().strip()
        if not os.path.isdir(data_dir):
            QtWidgets.QMessageBox.warning(self, 'LLAMAS Setup',
                                          f'Not a directory:\n{data_dir}')
            return
        progress = QtWidgets.QProgressDialog('Reading headers…', 'Cancel', 0, 100, self)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        def cb(i, n, name):
            progress.setMaximum(n)
            progress.setValue(i)
            progress.setLabelText(f'Reading headers… {name}')
            QtWidgets.QApplication.processEvents()
            return not progress.wasCanceled()

        self.entries = scan_directory(data_dir, progress_callback=cb)
        progress.close()
        self.roles = {}
        self.populate_table()
        self.load_notes(data_dir)
        if not self.outdirEdit.text().strip():
            self.outdirEdit.setText(os.path.join(data_dir, 'reduced'))
        if not self.configEdit.text().strip():
            self.configEdit.setText(os.path.join(data_dir, 'llamas_redux_config.txt'))
        self.update_summary()
        # pick up a previous session's config so the user can revise it
        config_path = self.configEdit.text().strip()
        if os.path.isfile(config_path):
            self.load_config(config_path)
        # Recommend standards last, filling only frames the config left unassigned, so a prior
        # session's explicit choices always win over the automatic suggestion.
        self.apply_standard_recommendations()

    def load_notes(self, data_dir):
        notes_path = os.path.join(data_dir, NOTES_FILENAME)
        if os.path.isfile(notes_path):
            try:
                with open(notes_path, 'r', errors='replace') as f:
                    self.notesView.setPlainText(f.read())
                return
            except Exception as err:
                self.notesView.setPlainText(f'(could not read {NOTES_FILENAME}: {err})')
                return
        self.notesView.setPlainText(f'(no {NOTES_FILENAME} found in {data_dir})')

    def populate_table(self):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self.entries))
        for row, e in enumerate(self.entries):
            fileItem = QtWidgets.QTableWidgetItem(e['filename'])
            fileItem.setData(QtCore.Qt.ItemDataRole.UserRole, e['path'])
            self.table.setItem(row, self.COL_FILE, fileItem)
            self.table.setItem(row, self.COL_OBJ, QtWidgets.QTableWidgetItem(e['object']))
            expItem = QtWidgets.QTableWidgetItem()
            if e['exptime'] is not None:
                expItem.setData(QtCore.Qt.ItemDataRole.DisplayRole, round(e['exptime'], 3))
            self.table.setItem(row, self.COL_EXP, expItem)
            self.table.setItem(row, self.COL_TYPE, QtWidgets.QTableWidgetItem(e['type']))
            self.table.setItem(row, self.COL_READ, QtWidgets.QTableWidgetItem(e['readmode']))
            self.table.setItem(row, self.COL_CMNT, QtWidgets.QTableWidgetItem(e['comment']))
            self.table.setItem(row, self.COL_ROLE, QtWidgets.QTableWidgetItem(''))
            if e.get('standard'):
                name, sep = e['standard']
                # Annotate the Type column so the crossmatch is visible where a user looks for
                # the frame's classification. This is independent of the role assignment, so it
                # shows even when a loaded config has (mis)labelled the standard as science. The
                # header's own PRODCATG type is kept in the tooltip.
                typeItem = self.table.item(row, self.COL_TYPE)
                typeItem.setText(f'Flux Std: {name}')
                tip = (f'Matches flux standard {name}, {sep:.1f}" from catalogue position.\n'
                       f'Header type: {e["type"] or "unknown"}.')
                typeItem.setToolTip(tip)
                self.table.item(row, self.COL_OBJ).setToolTip(tip)
        self.table.setSortingEnabled(True)
        # Default to filename order, which for LLAMAS names (ISO date + time) is the order
        # the frames were observed. The user can still re-sort by clicking any header.
        self.table.sortItems(self.COL_FILE, QtCore.Qt.SortOrder.AscendingOrder)

    def apply_standard_recommendations(self):
        """Pre-assign the flux-standard role to crossmatched frames.

        A recommendation, not a decision: it seeds the role so the standards are visible and
        extracted, but the user overrides it with the role buttons or Clear Role like any other
        assignment. Only frames with no role yet are touched, so re-scanning never clobbers a
        choice the user already made.
        """
        recommended = 0
        for e in self.entries:
            if e.get('standard') and not self.roles.get(e['path']):
                self.roles.setdefault(e['path'], set()).add('flux_standard')
                recommended += 1
        if recommended:
            self.refresh_role_column()
            self.update_summary()
            names = ', '.join(sorted({e['standard'][0] for e in self.entries
                                      if e.get('standard')}))
            self.statusBar().showMessage(
                f'Recommended {recommended} exposure(s) as flux standards ({names}) '
                f'— override in the Role column if wrong')

    # ----------------------------------------------------------- ds9 link
    def table_context_menu(self, pos):
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        path = self.table.item(row, self.COL_FILE).data(QtCore.Qt.ItemDataRole.UserRole)
        menu = QtWidgets.QMenu(self)
        actHere = menu.addAction('Display in ds9 (MEF cube)')
        actNew = menu.addAction('Display in ds9 (new frame)')
        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if action is actHere:
            self.send_to_ds9(path, new_frame=False)
        elif action is actNew:
            self.send_to_ds9(path, new_frame=True)

    def send_to_ds9(self, path, new_frame=False):
        """Load a MEF file into ds9 as a multi-extension cube via XPA."""
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')   # pyds9 warns if it can't find the app
                import pyds9
        except ImportError:
            QtWidgets.QMessageBox.warning(
                self, 'LLAMAS Setup',
                'The pyds9 package is required to talk to ds9:\n    pip install pyds9')
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            if self._ds9 is None:
                self._ds9 = pyds9.DS9()   # attaches to a running ds9
            if new_frame:
                self._ds9.set('frame new')
            self._ds9.set(f'file mecube "{path}"')
            self._ds9.set('scale zscale')
            self.statusBar().showMessage(f'Sent to ds9: {os.path.basename(path)}')
        except Exception as err:
            self._ds9 = None   # stale connection; retry fresh next time
            QtWidgets.QMessageBox.warning(
                self, 'LLAMAS Setup',
                f'Could not send to ds9 — is SAOImage ds9 running?\n\n{err}')
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # ---------------------------------------------------------- role logic
    def selected_paths(self):
        paths = []
        for idx in self.table.selectionModel().selectedRows(self.COL_FILE):
            item = self.table.item(idx.row(), self.COL_FILE)
            paths.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
        return paths

    def assign_role(self, role):
        paths = self.selected_paths()
        if not paths:
            QtWidgets.QMessageBox.information(self, 'LLAMAS Setup',
                                              'Select one or more rows first.')
            return
        label, multi = ROLES[role][:2]
        if not multi:
            if len(paths) > 1:
                QtWidgets.QMessageBox.information(
                    self, 'LLAMAS Setup',
                    f'{label} takes a single file; select exactly one row.')
                return
            # single-file role: remove it from any previous holder
            for roleset in self.roles.values():
                roleset.discard(role)
        for p in paths:
            roleset = self.roles.setdefault(p, set())
            if role == 'bad':
                roleset.clear()   # bad data holds no other role
            else:
                roleset.discard('bad')
            roleset.add(role)
        self.refresh_role_column()
        self.update_summary()

    def clear_role(self):
        for p in self.selected_paths():
            self.roles.pop(p, None)
        self.refresh_role_column()
        self.update_summary()

    def refresh_role_column(self):
        default_bg = QtGui.QBrush()
        default_fg = QtGui.QBrush()
        for row in range(self.table.rowCount()):
            path = self.table.item(row, self.COL_FILE).data(QtCore.Qt.ItemDataRole.UserRole)
            # display in ROLES order; tint by the first role a file holds
            held = [r for r in ROLES if r in self.roles.get(path, ())]
            if held:
                label = ', '.join(ROLES[r][0] for r in held)
                _l, _m, color, fgcolor = ROLES[held[0]]
                bg, fg = QtGui.QBrush(color), QtGui.QBrush(fgcolor)
            else:
                label, bg, fg = '', default_bg, default_fg
            self.table.item(row, self.COL_ROLE).setText(label)
            for col in range(self.table.columnCount()):
                self.table.item(row, col).setBackground(bg)
                self.table.item(row, col).setForeground(fg)

    def current_assignments(self):
        """Return dict role -> path (single roles) or list of paths (multi)."""
        assignments = {'flux_standard': [], 'science': [], 'bias': [], 'bad': []}
        for path in sorted(self.roles):
            for role in self.roles[path]:
                if ROLES[role][1]:
                    assignments[role].append(path)
                else:
                    assignments[role] = path
        return assignments

    # ---------------------------------------------------------- config load
    def load_config_clicked(self):
        path = self.configEdit.text().strip()
        if not os.path.isfile(path):
            path, _f = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Select config file', self.dirEdit.text(), 'Config files (*.txt *.cfg);;All files (*)')
            if not path:
                return
            self.configEdit.setText(path)
        self.load_config(path, verbose=True)

    def load_config(self, config_path, verbose=False):
        """Map an existing config file's selections back onto table roles."""
        try:
            config, bad_paths = parse_config(config_path)
        except Exception as err:
            QtWidgets.QMessageBox.warning(self, 'LLAMAS Setup',
                                          f'Could not read config:\n{config_path}\n{err}')
            return

        by_path = {e['path'] for e in self.entries}
        by_name = {e['filename']: e['path'] for e in self.entries}

        def resolve(p):
            p = p.strip()
            if p in by_path:
                return p
            return by_name.get(os.path.basename(p))

        role_paths = {}
        # singular twilight applies to all colours; per-colour keys override
        if config.get('twilight_flat'):
            for c in ('red', 'green', 'blue'):
                role_paths[f'{c}_twilight'] = config['twilight_flat']
        for key, role in KEY_TO_ROLE.items():
            if config.get(key):
                role_paths[role] = config[key]

        self.roles = {}
        unmatched = []
        for role, p in role_paths.items():
            rp = resolve(p)
            if rp:
                self.roles.setdefault(rp, set()).add(role)
            else:
                unmatched.append(f'{ROLES[role][0]}: {p}')
        science = config.get('science_files', '')
        for p in (x for x in science.split(',') if x.strip()):
            rp = resolve(p)
            if rp:
                self.roles.setdefault(rp, set()).add('science')
            else:
                unmatched.append(f'Science: {p.strip()}')
        standards = config.get('flux_standard_files', '')
        for p in (x for x in standards.split(',') if x.strip()):
            rp = resolve(p)
            if rp:
                self.roles.setdefault(rp, set()).add('flux_standard')
            else:
                unmatched.append(f'Flux Standard: {p.strip()}')
        for p in bad_paths:
            rp = resolve(p)
            if rp:
                self.roles[rp] = {'bad'}
            else:
                unmatched.append(f'BAD: {p}')

        # master bias paths can't map to raw frames; carry them through instead
        self.prior_bias = {'SLOW': config.get('slow_bias_file'),
                           'FAST': config.get('fast_bias_file')}
        trace_dir = config.get('trace_output_dir', '')
        if trace_dir:
            self.outdirEdit.setText(os.path.dirname(trace_dir.rstrip(os.sep)))

        self._loaded_config_path = config_path
        self.refresh_role_column()
        self.update_summary()
        msg = f'Loaded selections from {config_path}'
        if unmatched:
            msg += '\n\nEntries not found in this data directory (left unassigned):\n' \
                   + '\n'.join(unmatched)
            QtWidgets.QMessageBox.information(self, 'LLAMAS Setup', msg)
        elif verbose:
            QtWidgets.QMessageBox.information(self, 'LLAMAS Setup', msg)
        self.statusBar().showMessage(f'Loaded config: {config_path}')

    def update_summary(self):
        a = self.current_assignments()

        def rgb(suffix):
            return ' '.join(f"{c[0].upper()}{'✓' if a.get(f'{c}_{suffix}') else '–'}"
                            for c in ('red', 'green', 'blue'))

        readmode = {e['path']: (e['readmode'] or '').upper() for e in self.entries}
        n_slow = sum(1 for p in a['bias'] if readmode.get(p) == 'SLOW')
        n_fast = sum(1 for p in a['bias'] if readmode.get(p) == 'FAST')
        self.summaryLabel.setText(
            f"Assigned — Flats: {rgb('flat')}   Twilights: {rgb('twilight')}   "
            f"Arcs: {rgb('arc')}   Science: {len(a['science'])}   "
            f"Flux std: {len(a['flux_standard'])}   "
            f"Bias: {len(a['bias'])} ({n_slow} slow / {n_fast} fast)   Bad: {len(a['bad'])}")

    # --------------------------------------------------------- config write
    def write_config(self):
        assignments = self.current_assignments()
        if not assignments['science']:
            QtWidgets.QMessageBox.warning(self, 'LLAMAS Setup',
                                          'Select at least one science file.')
            return
        missing = [ROLES[r][0] for r in
                   ('red_flat', 'green_flat', 'blue_flat',
                    'red_twilight', 'green_twilight', 'blue_twilight')
                   if not assignments.get(r)]
        if not assignments['bias'] and not any(self.prior_bias.values()):
            missing.append('Bias')
        if missing:
            reply = QtWidgets.QMessageBox.question(
                self, 'LLAMAS Setup',
                'No selection for: ' + ', '.join(missing) +
                '.\nThe pipeline will fall back to its defaults for these.\n\nContinue?',
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        output_dir = self.outdirEdit.text().strip()
        config_path = self.configEdit.text().strip()
        if not output_dir or not config_path:
            QtWidgets.QMessageBox.warning(self, 'LLAMAS Setup',
                                          'Set the output directory and config file path.')
            return
        # a config loaded this session is being revised on purpose; only
        # prompt before clobbering a file whose contents were never shown
        if os.path.exists(config_path) and config_path != self._loaded_config_path:
            reply = QtWidgets.QMessageBox.question(
                self, 'LLAMAS Setup', f'Overwrite existing file?\n{config_path}',
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        if assignments['bias']:
            by_mode = {}
            readmode = {e['path']: e['readmode'] for e in self.entries}
            for p in assignments['bias']:
                by_mode.setdefault(readmode.get(p) or 'SLOW', []).append(p)
            self.writeBtn.setEnabled(False)
            self._bias_progress = QtWidgets.QProgressDialog(
                'Combining bias frames…', None, 0, 0, self)
            self._bias_progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
            self._bias_progress.setMinimumDuration(0)
            self._bias_worker = BiasWorker(by_mode, output_dir, self)
            self._bias_worker.status.connect(self._bias_progress.setLabelText)
            self._bias_worker.finished_ok.connect(
                lambda outputs: self._finish_config(assignments, output_dir, config_path, outputs))
            self._bias_worker.failed.connect(self._bias_failed)
            self._bias_worker.start()
        else:
            self._finish_config(assignments, output_dir, config_path, {})

    def _bias_failed(self, message):
        self._bias_progress.close()
        self.writeBtn.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, 'LLAMAS Setup',
                                       f'Bias combination failed:\n{message}')

    def _finish_config(self, assignments, output_dir, config_path, bias_outputs):
        if self._bias_worker is not None:
            self._bias_progress.close()
            self.writeBtn.setEnabled(True)
            self._bias_worker = None
        try:
            os.makedirs(output_dir, exist_ok=True)
            # rewrite an existing config in place so hand-edited parameters
            # survive; otherwise start from the package template
            template = config_path if os.path.isfile(config_path) else TEMPLATE_PATH
            text = generate_config(assignments, output_dir,
                                   slow_bias=bias_outputs.get('SLOW') or self.prior_bias.get('SLOW'),
                                   fast_bias=bias_outputs.get('FAST') or self.prior_bias.get('FAST'),
                                   template_path=template)
            with open(config_path, 'w') as f:
                f.write(text)
        except Exception:
            QtWidgets.QMessageBox.critical(self, 'LLAMAS Setup',
                                           f'Failed to write config:\n{traceback.format_exc()}')
            return
        msg = f'Config written to:\n{config_path}'
        if bias_outputs:
            msg += '\n\nMaster bias files:\n' + '\n'.join(sorted(bias_outputs.values()))
        QtWidgets.QMessageBox.information(self, 'LLAMAS Setup', msg)


def main():
    parser = argparse.ArgumentParser(
        description='GUI to build a llamas-pyjamas reduction config from a raw data directory.')
    parser.add_argument('data_dir', nargs='?', default=None,
                        help='Directory of raw LLAMAS MEF files (browse dialog if omitted)')
    parser.add_argument('-o', '--output', default=None,
                        help='Path for the generated config file')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = ReduxSetupWindow(data_dir=args.data_dir, config_out=args.output)
    win.show()
    if not args.data_dir:
        win.browse_directory()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
