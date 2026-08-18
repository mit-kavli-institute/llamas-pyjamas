"""
Observing-log parsing for LLAMAS exposure-metadata repair tools.

Two log formats are supported, auto-detected from the file:

1. The tab-separated ``obs.log`` produced by the observing GUI (canonical):

       file name\tChekSum\tUT\tOBJECT\tEXPTIME\tREAD-MDE\tAIRMASS
       ---------\t------\t...                     (dashes separator row)
       LLAMAS_2026-06-30_19-33-12.8_CAL22_mef.fits\tERR\t...\tLDLS flat\t0.07\tFAST\t1.0 / ...

   OBJECT values contain spaces, so rows are split on tabs only. The AIRMASS
   column is free text whose first float is the start-of-exposure airmass.

2. Hand-written xlsx logs (e.g. LLAMAS_obs-log_20241128.xlsx), columns
   ``file name / local time / read mode / exp. time / type / object / comment``.
   These carry ditto marks ("), blank separator rows, continuation rows with a
   blank filename, and duplicate filenames with conflicting values. Reading
   .xlsx requires openpyxl (installed in the llamas_data_reduction env; it is
   deliberately not a pyproject dependency). CSV exports of either format are
   also accepted; the schema is detected from the header row.

Filenames are matched on the normalized timestamp stem (``.fits`` and ``_mef``
suffixes stripped), which unifies the three spellings seen in real logs.
Conflicting duplicate rows for one file are all kept so the caller can force an
interactive choice - this module never silently picks one.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DITTO_MARKS = {'"', '“', '”', "''"}
VALID_READ_MODES = ('FAST', 'SLOW')

# Normalized header names that identify each schema
_GUI_SIGNATURE = 'read-mde'
_HANDWRITTEN_SIGNATURE = 'read mode'

_FLOAT_RE = re.compile(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
# Two filename eras: LLAMAS_2024-11-29T01_34_17.524[...] and LLAMAS_2026-06-30_19-33-12.8[...]
_TS_PATTERNS = (
    re.compile(r'LLAMAS_(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2}(?:\.\d+)?)'),
    re.compile(r'LLAMAS_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2}(?:\.\d+)?)'),
)


class ObsLogError(Exception):
    """Any problem reading or interpreting an observing log."""


class ObsLogSheetError(ObsLogError):
    """The workbook sheet could not be chosen automatically."""

    def __init__(self, message: str, sheets: list[str]):
        super().__init__(message)
        self.sheets = sheets


@dataclass
class LogRow:
    """One observing-log entry, cleaned. ``*_raw`` fields keep the original text."""
    stem: str | None
    raw_filename: str | None
    read_mode: str | None = None
    read_mode_raw: str | None = None
    exp_time: float | None = None
    exp_time_raw: str | None = None
    obs_type: str | None = None
    object_name: str | None = None
    comment: str | None = None
    airmass: float | None = None
    airmass_raw: str | None = None
    source: str = ''
    row_number: int = 0
    is_continuation: bool = False

    def describe(self) -> str:
        """One-line human summary used in conflict prompts."""
        bits = []
        if self.exp_time is not None:
            bits.append(f"exp {self.exp_time:g}s")
        elif self.exp_time_raw:
            bits.append(f"exp '{self.exp_time_raw}'")
        if self.read_mode:
            bits.append(self.read_mode)
        elif self.read_mode_raw:
            bits.append(f"mode '{self.read_mode_raw}'")
        if self.obs_type:
            bits.append(self.obs_type)
        if self.object_name:
            bits.append(f"'{self.object_name}'")
        if self.comment:
            bits.append(f"comment: {self.comment}")
        return f"{self.source} row {self.row_number}: " + ("  ".join(bits) or "(no values)")

    def harvest_key(self) -> tuple:
        """Values that matter for duplicate collapsing."""
        return (self.read_mode, self.exp_time, self.obs_type, self.object_name)


def normalize_stem(name) -> str | None:
    """Strip '.fits' then '_mef' so all three logged spellings match the file stem."""
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None
    if s.lower().endswith('.fits'):
        s = s[:-5]
    if s.lower().endswith('_mef'):
        s = s[:-4]
    return s or None


def stem_timestamp(stem: str | None) -> datetime | None:
    """Parse the acquisition timestamp out of a normalized stem, either era."""
    if not stem:
        return None
    for pat in _TS_PATTERNS:
        m = pat.search(stem)
        if m:
            date, hh, mm, ss = m.groups()
            try:
                whole = datetime.strptime(f"{date} {hh}:{mm}", "%Y-%m-%d %H:%M")
                return whole.replace(second=int(float(ss)) % 60,
                                     microsecond=int((float(ss) % 1) * 1e6))
            except ValueError:
                return None
    return None


def _clean_read_mode(raw) -> tuple[str | None, str | None]:
    if raw is None:
        return None, None
    text = str(raw).strip()
    if not text:
        return None, None
    mode = text.upper()
    if mode in VALID_READ_MODES:
        return mode, text
    return None, text


def _clean_exp_time(raw) -> tuple[float | None, str | None]:
    if raw is None:
        return None, None
    if isinstance(raw, (int, float)):
        return float(raw), str(raw)
    text = str(raw).strip()
    if not text:
        return None, None
    m = _FLOAT_RE.match(text.rstrip('sS').strip())
    if m and m.group() == text.rstrip('sS').strip():
        return float(m.group()), text
    return None, text


def _first_float(raw) -> float | None:
    if raw is None:
        return None
    m = _FLOAT_RE.search(str(raw))
    return float(m.group()) if m else None


def _norm_header(cell) -> str:
    return str(cell).strip().lower() if cell is not None else ''


def _is_blank(cell) -> bool:
    return cell is None or (isinstance(cell, str) and not cell.strip())


def list_sheets(path: str | Path) -> list[str]:
    """Sheet names of an xlsx log (for interactive selection)."""
    openpyxl = _import_openpyxl()
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    try:
        return list(wb.sheetnames)
    finally:
        wb.close()


def _import_openpyxl():
    try:
        import openpyxl
    except ImportError as exc:
        raise ObsLogError(
            "Reading .xlsx observing logs requires openpyxl (pip install openpyxl). "
            "Alternatively export the sheet to CSV and pass that to --obs-log."
        ) from exc
    return openpyxl


def load_obs_log(path: str | Path, sheet: str | None = None,
                 known_dates: set[str] | None = None) -> list[LogRow]:
    """
    Parse an observing log into LogRows.

    Args:
        path: .xlsx workbook, or a delimited text log (GUI obs.log, CSV export).
        sheet: xlsx sheet name; auto-selected when the workbook has one sheet or
            a sheet name matches one of known_dates.
        known_dates: 'YYYYMMDD' strings from the scanned FITS filenames, used
            for sheet auto-selection.

    Raises:
        ObsLogError / ObsLogSheetError on unreadable input or ambiguous sheets.
    """
    path = Path(path)
    if not path.is_file():
        raise ObsLogError(f"Observing log not found: {path}")
    if path.suffix.lower() == '.xlsx':
        cells, source = _load_xlsx_cells(path, sheet, known_dates)
    else:
        cells, source = _load_text_cells(path)
    if not cells:
        raise ObsLogError(f"Observing log is empty: {path}")

    header = [_norm_header(c) for c in cells[0]]
    if _GUI_SIGNATURE in header:
        return _parse_gui_rows(cells, header, source)
    if _HANDWRITTEN_SIGNATURE in header:
        return _parse_handwritten_rows(cells, header, source)
    raise ObsLogError(
        f"Unrecognized observing-log header in {path.name}: {cells[0]!r}. "
        f"Expected a GUI obs.log ('READ-MDE' column) or a hand-written log "
        f"('read mode' column)."
    )


def _load_text_cells(path: Path) -> tuple[list[list], str]:
    # utf-8-sig: Excel "CSV UTF-8" exports start with a BOM that would
    # otherwise corrupt the first header name and match zero rows.
    with open(path, newline='', encoding='utf-8-sig', errors='replace') as fh:
        first = fh.readline()
        fh.seek(0)
        if '\t' in first:
            rows = [line.rstrip('\r\n').split('\t') for line in fh]
        else:
            rows = list(csv.reader(fh))
    return [r for r in rows if r], path.name


def _load_xlsx_cells(path: Path, sheet: str | None,
                     known_dates: set[str] | None) -> tuple[list[list], str]:
    openpyxl = _import_openpyxl()
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    try:
        names = list(wb.sheetnames)
        if sheet is not None:
            if sheet not in names:
                raise ObsLogSheetError(
                    f"Sheet '{sheet}' not in {path.name}; available: {names}", names)
            chosen = sheet
        elif len(names) == 1:
            chosen = names[0]
        else:
            matches = [n for n in names
                       if known_dates and any(d in n.replace('-', '') for d in known_dates)]
            if len(matches) == 1:
                chosen = matches[0]
                logger.info(f"Auto-selected sheet '{chosen}' matching scanned file dates")
            else:
                raise ObsLogSheetError(
                    f"{path.name} has multiple sheets; pass --log-sheet. "
                    f"Available: {names}", names)
        ws = wb[chosen]
        return [list(row) for row in ws.iter_rows(values_only=True)], f"{path.name}[{chosen}]"
    finally:
        wb.close()


def _parse_gui_rows(cells: list[list], header: list[str], source: str) -> list[LogRow]:
    col = {name: i for i, name in enumerate(header)}

    def get(row, name):
        i = col.get(name)
        return row[i] if i is not None and i < len(row) else None

    rows: list[LogRow] = []
    for n, raw in enumerate(cells[1:], start=2):
        first = str(raw[0]).strip() if raw and raw[0] is not None else ''
        if not first or set(first) == {'-'}:
            continue
        mode, mode_raw = _clean_read_mode(get(raw, 'read-mde'))
        exp, exp_raw = _clean_exp_time(get(raw, 'exptime'))
        obj = get(raw, 'object')
        air_raw = get(raw, 'airmass')
        rows.append(LogRow(
            stem=normalize_stem(first), raw_filename=first,
            read_mode=mode, read_mode_raw=mode_raw,
            exp_time=exp, exp_time_raw=exp_raw,
            object_name=str(obj).strip() if not _is_blank(obj) else None,
            airmass=_first_float(air_raw),
            airmass_raw=str(air_raw).strip() if not _is_blank(air_raw) else None,
            source=source, row_number=n,
        ))
    return rows


def _parse_handwritten_rows(cells: list[list], header: list[str],
                            source: str) -> list[LogRow]:
    col: dict[str, int] = {}
    for i, name in enumerate(header):
        if name == 'file name':
            col['filename'] = i
        elif name == 'read mode':
            col['read_mode'] = i
        elif 'exp' in name:
            col['exp_time'] = i
        elif name == 'type':
            col['obs_type'] = i
        elif name == 'object':
            col['object_name'] = i
        elif name == 'comment':
            col['comment'] = i
        # 'local time' is deliberately ignored: local Chile time with mixed types

    def get(row, name):
        i = col.get(name)
        return row[i] if i is not None and i < len(row) else None

    # Pass 1: drop fully-blank separator rows BEFORE ditto expansion, so a
    # ditto never reaches across a block boundary.
    survivors = [(n, raw) for n, raw in enumerate(cells[1:], start=2)
                 if not all(_is_blank(c) for c in raw)]

    # Pass 2: ditto expansion per column against the previous surviving row.
    ditto_cols = ('read_mode', 'exp_time', 'obs_type', 'object_name', 'comment')
    expanded: list[tuple[int, dict]] = []
    for n, raw in survivors:
        values = {name: get(raw, name) for name in ('filename',) + ditto_cols}
        for name in ditto_cols:
            v = values[name]
            if isinstance(v, str) and v.strip() in DITTO_MARKS:
                values[name] = expanded[-1][1][name] if expanded else None
        expanded.append((n, values))

    # Pass 3: continuation rows (blank filename, data present) attach to the
    # previous named row but keep their OWN values - real logs have
    # continuation rows whose read mode differs from the parent's.
    rows: list[LogRow] = []
    last_named: dict | None = None
    for n, values in expanded:
        filename = values['filename']
        if isinstance(filename, str) and filename.strip() in DITTO_MARKS:
            # A ditto in the FILENAME column means "same file again" - a
            # duplicate row (kept for conflict detection), not a continuation.
            filename = last_named['filename'] if last_named else None
            values = dict(values, filename=filename)
        is_cont = _is_blank(filename)
        if is_cont:
            if last_named is None:
                logger.warning(f"{source} row {n}: data row with no filename and "
                               f"no preceding named row; skipped")
                continue
            filename = last_named['filename']
        else:
            last_named = values
        mode, mode_raw = _clean_read_mode(values['read_mode'])
        exp, exp_raw = _clean_exp_time(values['exp_time'])
        obj = values['object_name']
        otype = values['obs_type']
        comment = values['comment']
        rows.append(LogRow(
            stem=normalize_stem(filename),
            raw_filename=str(filename).strip() if filename is not None else None,
            read_mode=mode, read_mode_raw=mode_raw,
            exp_time=exp, exp_time_raw=exp_raw,
            obs_type=str(otype).strip() if not _is_blank(otype) else None,
            object_name=str(obj).strip() if not _is_blank(obj) else None,
            comment=str(comment).strip() if not _is_blank(comment) else None,
            source=source, row_number=n, is_continuation=is_cont,
        ))
    return rows


def build_log_index(rows: list[LogRow]) -> dict[str, list[LogRow]]:
    """
    Index rows by normalized stem. Duplicate rows whose harvested values are
    identical collapse to one; rows with differing values are ALL kept so the
    caller must disambiguate interactively.
    """
    index: dict[str, list[LogRow]] = {}
    for row in rows:
        if row.stem is None:
            continue
        bucket = index.setdefault(row.stem, [])
        if any(existing.harvest_key() == row.harvest_key() for existing in bucket):
            continue
        bucket.append(row)
    return index


def nearest_row(stem: str, rows: list[LogRow],
                max_offset_s: float = 600.0) -> tuple[LogRow, float] | None:
    """
    Nearest-in-time log row for an unmatched file, as a SUGGESTION only.
    Returns (row, offset_seconds) or None if no row parses within max_offset_s.
    """
    target = stem_timestamp(stem)
    if target is None:
        return None
    best: tuple[LogRow, float] | None = None
    for row in rows:
        ts = stem_timestamp(row.stem)
        if ts is None:
            continue
        off = abs((ts - target).total_seconds())
        if off <= max_offset_s and (best is None or off < best[1]):
            best = (row, off)
    return best
