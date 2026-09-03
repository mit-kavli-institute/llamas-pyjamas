"""
Interactive repair of missing exposure metadata in raw LLAMAS MEF files.

Older LLAMAS data can be missing exposure information where the pipeline
expects it: blank EXPTIME cards, no READ-MDE keyword (added to the DAQ later),
empty OBJECT, and timing/pointing cards populated only inside the HIERARCH
TEL block. This tool scans a directory, prints a concise report, then resolves
the gaps interactively via three paths:

  Case A  the value already exists elsewhere in the file (REXPTIME, HIERARCH
          TEL block, extension EXPTIME) and is propagated to the expected card;
  Case B  the value comes from an observing log (--obs-log; GUI obs.log tab
          format or hand-written xlsx) or is typed in by the user;
  Case C  READ-MDE is inferred statistically from the QA bias-level baselines
          (see readmode_assess.py), with an interpretable confidence.

ORIGINAL FILES ARE NEVER MODIFIED. Repaired copies are written to
--output-dir (default: <input_dir>/modified). Every written value is recorded
in HIERARCH FIXEXP provenance cards and a HISTORY line.

Usage:
    python fix_exposure_info.py /data/ut20241129 --report-only
    python fix_exposure_info.py /data/ut20241129 --obs-log obs.log \\
           --output-dir /scratch/ut20241129_fixed
    python fix_exposure_info.py /data/ut20241129 --obs-log log.xlsx --dry-run

Exit codes:
    0  nothing was missing, or every gap was resolved and written
    1  partial success (some files written, some gaps or skips remain)
    2  gaps found but nothing written (report-only, user quit, non-TTY)
    3  system error (bad paths, unreadable obs log, write failure)

Reading .xlsx logs requires openpyxl (installed in the llamas_data_reduction
env); the GUI tab-separated obs.log format needs only the stdlib.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astropy.io import fits

# Sibling modules (obs_log, readmode_assess) are imported bare; importing the
# llamas_pyjamas package itself would eagerly pull Ray/PypeIt.
sys.path.insert(0, str(Path(__file__).resolve().parent))

EXIT_OK, EXIT_PARTIAL, EXIT_UNRESOLVED, EXIT_ERROR = 0, 1, 2, 3

# Primary cards audited: (key, zero_is_suspect)
PRIMARY_KEYS = [
    ('EXPTIME', True), ('REXPTIME', True), ('SEXPTIME', True), ('AEXPTIME', True),
    ('READ-MDE', False), ('OBJECT', False), ('UTC', False), ('DATE-OBS', False),
    ('DATE', False), ('MJD-OBS', True), ('RA', False), ('DEC', False),
    ('AIRMASS', False),
]
# Donor cards (HIERARCH keywords are addressed without the prefix in astropy)
DONOR_KEYS = ['TEL UTC', 'TEL DATE-OBS', 'TEL RA', 'TEL DEC', 'TEL AIRMASS',
              'OBS EXPTIME REQ']
# Case A propagation for the simple verbatim copies: target -> donor
TEL_PROPAGATION = [
    ('UTC', 'TEL UTC'), ('DATE-OBS', 'TEL DATE-OBS'), ('DATE', 'TEL DATE-OBS'),
    ('RA', 'TEL RA'), ('DEC', 'TEL DEC'), ('AIRMASS', 'TEL AIRMASS'),
]
CARD_COMMENTS = {
    'EXPTIME': '[s] Requested exposure time', 'REXPTIME': '[s] Requested exposure time',
    'AEXPTIME': '[s] Actual exposure time', 'READ-MDE': 'CCD readout mode',
    'OBJECT': 'Object name', 'UTC': 'UT at start of exposure',
    'DATE-OBS': 'UT date at start of exposure', 'DATE': 'UT date at start of exposure',
    'MJD-OBS': 'MJD at start of exposure', 'RA': '[deg] Right ascension (J2000)',
    'DEC': '[deg] Declination (J2000)', 'AIRMASS': 'Airmass at start of exposure',
}
CONFIDENCE_TIER = {'UNDECIDED': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}


def _sexagesimal_to_deg(value, is_ra: bool):
    """TEL-block coordinate -> decimal degrees, or None if it cannot be read.

    RA is hourangle, Dec is degrees. A value that is already numeric is passed through, so
    re-running over a repaired header is a no-op.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        from astropy.coordinates import Angle
        from astropy import units as u
        return float(Angle(str(value).strip(), unit=(u.hourangle if is_ra else u.deg)).deg)
    except Exception:                                   # noqa: BLE001 - bad string => no value
        return None


class UserQuit(Exception):
    """The user chose [q]uit during interactive resolution."""


@dataclass
class CardFinding:
    key: str
    status: str                      # 'ok' | 'blank' | 'absent' | 'suspect_zero'
    current: Any = None

    @property
    def fillable(self) -> bool:
        return self.status in ('blank', 'absent')


@dataclass
class ProposedFix:
    key: str
    location: str                    # 'primary' | 'all_extensions'
    value: Any
    source: str
    case: str                        # 'A' | 'B' | 'C'
    approved: bool = False


@dataclass
class FileAudit:
    path: Path
    stem: str
    findings: dict[str, CardFinding] = field(default_factory=dict)
    fixes: list[ProposedFix] = field(default_factory=list)
    log_rows: list = field(default_factory=list)
    chosen_log_row: Any = None
    ext_exptime_state: str = 'ok'    # 'ok' | 'zero' | 'blank' | 'inconsistent'
    ext_exptime_value: float | None = None
    n_extensions: int = 0
    missing_ext_ids: int = 0
    notes: list[str] = field(default_factory=list)
    scan_error: str | None = None
    skipped: bool = False
    written_to: Path | None = None
    write_error: str | None = None

    def finding(self, key: str) -> CardFinding:
        return self.findings.get(key, CardFinding(key, 'absent'))

    def fix_for(self, key: str, location: str = 'primary') -> ProposedFix | None:
        for f in self.fixes:
            if f.key == key and f.location == location:
                return f
        return None

    def propose(self, key: str, value, source: str, case: str,
                location: str = 'primary') -> ProposedFix | None:
        existing = self.fix_for(key, location)
        if existing is not None:
            if existing.approved:
                return None
            # A declined/unreviewed earlier proposal must not block a better
            # source (e.g. user rejects Case A, then takes the obs-log value).
            existing.value, existing.source, existing.case = value, source, case
            return existing
        fix = ProposedFix(key, location, value, source, case)
        self.fixes.append(fix)
        return fix

    def approved_fix(self, key: str, location: str = 'primary') -> ProposedFix | None:
        fix = self.fix_for(key, location)
        return fix if fix is not None and fix.approved else None

    def resolved_value(self, key: str):
        """Approved-fix value, else the file's own ok value."""
        fix = self.fix_for(key)
        if fix is not None and fix.approved:
            return fix.value
        f = self.finding(key)
        return f.current if f.status == 'ok' else None

    def exptime_resolved(self):
        for key in ('EXPTIME', 'REXPTIME', 'SEXPTIME'):
            v = self.resolved_value(key)
            if v is not None:
                return v
        return None

    def gaps(self) -> list[str]:
        """Cards still missing after resolution (proposed-but-unapproved counts)."""
        out = []
        for key, _ in PRIMARY_KEYS:
            if key in ('SEXPTIME', 'AEXPTIME'):     # never required
                continue
            f = self.finding(key)
            if not f.fillable:
                continue
            fix = self.fix_for(key)
            if fix is None or not fix.approved:
                if key in ('EXPTIME', 'REXPTIME') and self.exptime_resolved() is not None:
                    continue
                out.append(key)
        return out


# ---------------------------------------------------------------- scan phase

def card_status(header, key: str, zero_suspect: bool = False) -> CardFinding:
    if key not in header:
        return CardFinding(key, 'absent')
    val = header[key]
    if val is None or isinstance(val, fits.card.Undefined):
        return CardFinding(key, 'blank')
    if isinstance(val, str) and not val.strip():
        return CardFinding(key, 'blank')
    if zero_suspect and isinstance(val, (int, float)) and float(val) == 0.0:
        return CardFinding(key, 'suspect_zero', val)
    return CardFinding(key, 'ok', val)


def scan_file(path: Path, stem: str) -> FileAudit:
    audit = FileAudit(path=path, stem=stem)
    try:
        with fits.open(path, mode='readonly', memmap=False, lazy_load_hdus=True) as hdul:
            phdr = hdul[0].header
            for key, zero_suspect in PRIMARY_KEYS:
                audit.findings[key] = card_status(phdr, key, zero_suspect)
            for key in DONOR_KEYS:
                audit.findings[key] = card_status(phdr, key)
            ext_values: list[float | None] = []
            for hdu in hdul[1:]:
                if not hasattr(hdu, 'header') or hdu.is_image is False:
                    continue
                audit.n_extensions += 1
                ehdr = hdu.header
                f = card_status(ehdr, 'EXPTIME', zero_suspect=True)
                ext_values.append(float(f.current) if f.status == 'ok' else
                                  (0.0 if f.status == 'suspect_zero' else None))
                if not all(k in ehdr for k in ('BENCH', 'SIDE', 'COLOR')):
                    audit.missing_ext_ids += 1
            filled = {v for v in ext_values if v not in (None, 0.0)}
            if len(filled) == 1 and all(v == next(iter(filled)) for v in ext_values):
                audit.ext_exptime_state = 'ok'
                audit.ext_exptime_value = next(iter(filled))
            elif filled:
                # Any mix of filled with blank/zero/other values: never use as
                # a donor and never let the repair overwrite the filled ones.
                audit.ext_exptime_state = 'inconsistent'
            elif any(v == 0.0 for v in ext_values):
                audit.ext_exptime_state = 'zero'
            else:
                audit.ext_exptime_state = 'blank'
            if audit.missing_ext_ids:
                audit.notes.append(
                    f"{audit.missing_ext_ids} extension(s) lack BENCH/SIDE/COLOR")
    except (OSError, fits.VerifyError) as exc:
        audit.scan_error = str(exc)
    return audit


# ------------------------------------------------------------------- Case A

def first_ok(audit: FileAudit, *keys: str):
    for key in keys:
        f = audit.finding(key)
        if f.status == 'ok':
            return f.current, key
    return None, None


def propose_case_a(audit: FileAudit) -> None:
    if audit.scan_error:
        return
    # Exposure time from in-file donors
    val, src = first_ok(audit, 'REXPTIME', 'SEXPTIME', 'OBS EXPTIME REQ')
    if val is None and audit.ext_exptime_state == 'ok':
        val, src = audit.ext_exptime_value, 'extension EXPTIME'
    if val is not None and audit.finding('EXPTIME').fillable:
        audit.propose('EXPTIME', val, src, 'A')
    if audit.finding('REXPTIME').fillable:
        # A filled EXPTIME (the requested time) outranks SEXPTIME (the
        # measured shutter time) as the REXPTIME donor.
        rex_val, rex_src = first_ok(audit, 'EXPTIME')
        if rex_val is None and src != 'REXPTIME':
            rex_val, rex_src = val, src
        if rex_val is not None:
            audit.propose('REXPTIME', rex_val, rex_src, 'A')
    if audit.ext_exptime_state == 'inconsistent':
        audit.notes.append("extension EXPTIME values disagree; not used as a donor")

    # TEL-block propagation. Verbatim, EXCEPT RA/DEC: the TEL block holds them sexagesimal
    # ('10:00:47.03'), while RA/DEC are decimal degrees by convention -- their own card comments
    # say so, and the whole pipeline reads them with float(). Copying the string across produced
    # headers whose pointing no consumer could parse, which is what silently greyed out the
    # sensitivity-function action on the mar25 standard: the standard is identified by
    # crossmatching RA/DEC, so an unreadable pointing means "not a standard".
    for target, donor in TEL_PROPAGATION:
        if audit.finding(target).fillable:
            d = audit.finding(donor)
            if d.status != 'ok':
                continue
            value = d.current
            if target in ('RA', 'DEC'):
                value = _sexagesimal_to_deg(value, is_ra=(target == 'RA'))
                if value is None:
                    audit.notes.append(
                        f"{target}: could not convert {d.current!r} from the TEL block to "
                        f"decimal degrees; left unfilled rather than written unparseable")
                    continue
            audit.propose(target, value, 'TEL header', 'A')

    # MJD-OBS derived from TEL DATE-OBS (+ TEL UTC when the date has no time part)
    mjd = audit.finding('MJD-OBS')
    if mjd.status in ('blank', 'absent', 'suspect_zero'):
        date_f, utc_f = audit.finding('TEL DATE-OBS'), audit.finding('TEL UTC')
        if date_f.status == 'ok':
            try:
                from astropy.time import Time
                date_str = str(date_f.current).strip()
                if 'T' not in date_str:
                    if utc_f.status != 'ok':
                        raise ValueError("no time component available")
                    date_str = f"{date_str}T{str(utc_f.current).strip()}"
                mjd_val = round(float(Time(date_str, format='isot', scale='utc').mjd), 8)
                audit.propose('MJD-OBS', mjd_val, 'derived from TEL header', 'A')
            except (ValueError, ImportError) as exc:
                audit.notes.append(f"MJD-OBS not derivable: {exc}")


# ------------------------------------------------------------------- Case B

def match_log(audits: list[FileAudit], index: dict) -> None:
    for audit in audits:
        audit.log_rows = index.get(audit.stem, [])
        if len(audit.log_rows) == 1:
            audit.chosen_log_row = audit.log_rows[0]


# FITS header strings must be ASCII; hand-written logs carry curly quotes and
# unicode dashes. Translate the common ones instead of crashing at write time.
_UNICODE_PUNCT = str.maketrans({
    '‐': '-', '‑': '-', '‒': '-', '–': '-', '—': '-',
    '‘': "'", '’': "'", '“': '"', '”': '"', ' ': ' ',
})


def ascii_card_value(value):
    return value.translate(_UNICODE_PUNCT) if isinstance(value, str) else value


def propose_from_log(audit: FileAudit) -> list[ProposedFix]:
    """Build Case B proposals from the chosen log row (must be resolved first)."""
    row = audit.chosen_log_row
    if row is None:
        return []
    # Keep this short: it also becomes a HIERARCH provenance card value, and
    # the log filename is already recorded in the FIXEXP OBSLOG card.
    src = f"obs log row {row.row_number}"
    proposed = []
    exptime = audit.finding('EXPTIME')
    resolved = audit.exptime_resolved()
    if row.exp_time is not None:
        if resolved is None and exptime.status != 'ok':
            if exptime.status == 'suspect_zero' and row.exp_time == 0.0:
                audit.notes.append("log confirms 0.0 s exposure (bias)")
            else:
                for key in ('EXPTIME', 'REXPTIME'):
                    if audit.finding(key).status != 'ok':
                        fix = audit.propose(key, float(row.exp_time), src, 'B')
                        if fix:
                            proposed.append(fix)
        elif resolved is not None and float(resolved) != float(row.exp_time):
            # In-file value (EXPTIME/REXPTIME card, or an approved fix) wins;
            # surface the disagreement so the user can override manually.
            audit.notes.append(
                f"in-file exposure time {resolved} != log {row.exp_time} "
                f"(kept in-file; use manual entry to override)")
    if row.read_mode and audit.finding('READ-MDE').fillable \
            and audit.approved_fix('READ-MDE') is None:
        fix = audit.propose('READ-MDE', row.read_mode, src, 'B')
        if fix:
            proposed.append(fix)
    if row.object_name and audit.finding('OBJECT').fillable \
            and audit.approved_fix('OBJECT') is None:
        fix = audit.propose('OBJECT', ascii_card_value(row.object_name), src, 'B')
        if fix:
            proposed.append(fix)
    if row.airmass is not None and audit.finding('AIRMASS').fillable \
            and audit.approved_fix('AIRMASS') is None:
        fix = audit.propose('AIRMASS', row.airmass, src, 'B')
        if fix:
            proposed.append(fix)
    return proposed


# ------------------------------------------------------------------- report

_LOG_ATTR = {'EXPTIME': 'exp_time', 'REXPTIME': 'exp_time',
             'READ-MDE': 'read_mode', 'OBJECT': 'object_name',
             'AIRMASS': 'airmass'}


def _cell(audit: FileAudit, key: str) -> str:
    f = audit.finding(key)
    fix = audit.fix_for(key)
    if f.status == 'ok':
        return 'ok'
    if fix is not None:
        return {'A': f"A({fix.source.split('(')[0].strip()[:8]})",
                'B': 'LOG', 'C': 'INFER'}[fix.case]
    if f.status == 'suspect_zero':
        return '0.0?'
    if audit.log_rows and len(audit.log_rows) > 1:
        return 'LOG!'
    row, attr = audit.chosen_log_row, _LOG_ATTR.get(key)
    if row is not None and attr and getattr(row, attr) not in (None, ''):
        return 'LOG'
    return f.status

def _exptime_cell(audit: FileAudit) -> str:
    for key in ('EXPTIME', 'REXPTIME', 'SEXPTIME'):
        if audit.finding(key).status == 'ok':
            return f'ok({key[0]})' if key != 'EXPTIME' else 'ok'
    return _cell(audit, 'EXPTIME')


def print_report(audits: list[FileAudit], verbose: bool = False,
                 leftover_log: list | None = None) -> None:
    name_w = max([len(a.stem) for a in audits] + [30])
    print(f"\n{'File (stem)':<{name_w}}  {'EXPTIME':<10} {'READ-MDE':<9} "
          f"{'OBJECT':<8} {'UTC/DATE':<9} RA/DEC/AIRM")
    for a in audits:
        if a.scan_error:
            print(f"{a.stem:<{name_w}}  SCAN ERROR: {a.scan_error}")
            continue
        utc = _cell(a, 'UTC'); date = _cell(a, 'DATE-OBS')
        td = utc if utc == date else f"{utc}/{date}"
        ra = _cell(a, 'RA'); dec = _cell(a, 'DEC'); am = _cell(a, 'AIRMASS')
        radec = ra if (ra == dec == am) else f"{ra}/{dec}/{am}"
        print(f"{a.stem:<{name_w}}  {_exptime_cell(a):<10} "
              f"{_cell(a, 'READ-MDE'):<9} {_cell(a, 'OBJECT'):<8} "
              f"{td:<9} {radec}")
        if verbose:
            for key, _ in PRIMARY_KEYS:
                f = a.finding(key)
                fix = a.fix_for(key)
                extra = f"  -> propose {fix.value!r} ({fix.source})" if fix else ""
                print(f"    {key:<10} {f.status:<12} {f.current!r}{extra}")
            for note in a.notes:
                print(f"    note: {note}")

    n = len(audits)
    def count(pred):
        return sum(1 for a in audits if not a.scan_error and pred(a))
    print(f"\nSummary ({n} files):")
    print(f"  exposure time missing: "
          f"{count(lambda a: a.exptime_resolved() is None and a.fix_for('EXPTIME') is None and a.finding('EXPTIME').status != 'ok')}"
          f"  (in-file recoverable: {count(lambda a: (f := a.fix_for('EXPTIME')) is not None and f.case == 'A')})")
    print(f"  READ-MDE absent: {count(lambda a: a.finding('READ-MDE').fillable)}")
    print(f"  OBJECT unfilled: {count(lambda a: a.finding('OBJECT').fillable)}")
    print(f"  TEL-block donors available: "
          f"{count(lambda a: a.finding('TEL UTC').status == 'ok')}/{n}")
    matched = count(lambda a: len(a.log_rows) == 1)
    conflicts = count(lambda a: len(a.log_rows) > 1)
    if any(a.log_rows for a in audits) or conflicts:
        print(f"  obs log: {matched} matched, {conflicts} with conflicting rows, "
              f"{count(lambda a: not a.log_rows)} unmatched")
    if leftover_log:
        stems = sorted({r.stem for r in leftover_log if r.stem})
        shown = ", ".join(stems[:3]) + ("..." if len(stems) > 3 else "")
        print(f"  obs log rows matching no scanned file: {len(stems)} ({shown})")
    print("Legend: ok | blank (card present, no value) | absent | 0.0? (suspect zero)"
          " | A(src)=in-file donor | LOG=obs log | LOG!=log conflict | INFER=QA inference\n")


# -------------------------------------------------------------- interaction

def make_ask(input_fn=input):
    def ask(prompt: str, choices: str, default: str | None = None) -> str:
        letters = list(choices)
        shown = "/".join(letters)
        while True:
            try:
                raw = input_fn(f"{prompt} [{shown}]{f' ({default})' if default else ''}: ")
            except (EOFError, KeyboardInterrupt):
                print()
                return 'q' if 'q' in letters else letters[-1]
            ans = raw.strip().lower()
            if not ans and default:
                return default
            # Exact single letters only: a word like 'skip' must never be
            # read as its first letter (which could mean 'slow' here).
            if len(ans) == 1 and ans in letters:
                return ans
            print(f"  please answer one of: {shown}")
    return ask


def _approve_file(audit: FileAudit, case: str) -> int:
    n = 0
    for fix in audit.fixes:
        if fix.case == case and not fix.approved:
            fix.approved = True
            n += 1
    return n


def resolve_case_a(audits: list[FileAudit], opts, ask) -> None:
    files = [a for a in audits if any(f.case == 'A' for f in a.fixes)]
    total = sum(1 for a in files for f in a.fixes if f.case == 'A')
    if not files:
        return
    by_key: dict[str, int] = {}
    for a in files:
        for f in a.fixes:
            if f.case == 'A':
                by_key[f.key] = by_key.get(f.key, 0) + 1
    print(f"\n[1/3] Case A - in-file propagation: {total} card fixes across "
          f"{len(files)} files")
    print("      " + ", ".join(f"{k} x{v}" for k, v in sorted(by_key.items())))
    if opts.yes:
        for a in files:
            _approve_file(a, 'A')
        print("      auto-approved (--yes)")
        return
    ans = ask("  Apply: [a]ll files, [r]eview per file, [s]kip category, [q]uit",
              'arsq')
    if ans == 'q':
        raise UserQuit
    if ans == 's':
        return
    if ans == 'a':
        for a in files:
            _approve_file(a, 'A')
        return
    apply_rest = False
    for a in files:
        if apply_rest:
            _approve_file(a, 'A')
            continue
        print(f"  {a.stem}:")
        for f in a.fixes:
            if f.case == 'A':
                print(f"      {f.key} = {f.value!r}  (from {f.source})")
        ans = ask("    Apply this file", 'ynasq')
        if ans == 'q':
            raise UserQuit
        if ans in ('y', 'a'):
            _approve_file(a, 'A')
        if ans == 'a':
            apply_rest = True
        if ans == 's':
            a.skipped = True


def _resolve_conflict(audit: FileAudit, ask, input_fn) -> None:
    print(f"  CONFLICT {audit.stem}: {len(audit.log_rows)} log rows disagree:")
    for i, row in enumerate(audit.log_rows, 1):
        print(f"      {i}) {row.describe()}")
    while True:
        try:
            raw = input_fn(f"    Choose row 1-{len(audit.log_rows)}, "
                           f"[e]dit values, [s]kip file, [q]uit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise UserQuit
        if raw == 'q':
            raise UserQuit
        if raw == 's':
            audit.notes.append("log conflict left unresolved")
            return
        if raw == 'e':
            _manual_entry(audit, ask, input_fn)
            return
        if raw.isdigit() and 1 <= int(raw) <= len(audit.log_rows):
            break
        print(f"    please enter 1-{len(audit.log_rows)}, e, s, or q")
    audit.chosen_log_row = audit.log_rows[int(raw) - 1]
    for fix in propose_from_log(audit):
        fix.approved = True


def _manual_entry(audit: FileAudit, ask, input_fn) -> None:
    if audit.exptime_resolved() is None and audit.finding('EXPTIME').status != 'ok':
        try:
            raw = input_fn(f"  {audit.stem}: EXPTIME seconds [blank=skip]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise UserQuit
        if raw:
            try:
                val = float(raw)
                for key in ('EXPTIME', 'REXPTIME'):
                    if audit.finding(key).status != 'ok':
                        fix = audit.propose(key, val, 'user input', 'B')
                        if fix:
                            fix.approved = True
            except ValueError:
                print(f"    not a number: {raw!r}; skipped")
    if audit.finding('OBJECT').fillable and audit.approved_fix('OBJECT') is None:
        try:
            raw = input_fn(f"  {audit.stem}: OBJECT [blank=skip]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise UserQuit
        if raw:
            fix = audit.propose('OBJECT', ascii_card_value(raw), 'user input', 'B')
            if fix:
                fix.approved = True


def resolve_case_b(audits: list[FileAudit], opts, ask, input_fn,
                   all_log_rows=None) -> None:
    have_log = any(a.log_rows for a in audits)
    matched = [a for a in audits
               if a.chosen_log_row is not None and not a.skipped and not a.scan_error]
    conflicted = [a for a in audits
                  if len(a.log_rows) > 1 and not a.skipped and not a.scan_error]
    unmatched = [a for a in audits
                 if not a.log_rows and not a.skipped and not a.scan_error]
    if have_log:
        proposals = {a.stem: propose_from_log(a) for a in matched}
        n_prop = sum(len(v) for v in proposals.values())
        print(f"\n[2/3] Case B - obs log: {len(matched)} matched, "
              f"{len(conflicted)} conflicting, {len(unmatched)} unmatched")
        if n_prop:
            if opts.yes:
                for a in matched:
                    _approve_file(a, 'B')
                print("      matched values auto-approved (--yes)")
            else:
                ans = ask(f"  Apply {n_prop} matched log values: [a]ll, [r]eview, "
                          f"[s]kip, [q]uit", 'arsq')
                if ans == 'q':
                    raise UserQuit
                if ans == 'a':
                    for a in matched:
                        _approve_file(a, 'B')
                elif ans == 'r':
                    apply_rest = False
                    for a in matched:
                        if not proposals[a.stem]:
                            continue
                        if apply_rest:
                            _approve_file(a, 'B')
                            continue
                        vals = "  ".join(f"{f.key}={f.value!r}"
                                         for f in proposals[a.stem])
                        row = a.chosen_log_row
                        print(f"    {a.stem}: {vals}  ({row.source} row {row.row_number})")
                        ans2 = ask("      Apply", 'ynasq')
                        if ans2 == 'q':
                            raise UserQuit
                        if ans2 in ('y', 'a'):
                            _approve_file(a, 'B')
                        if ans2 == 'a':
                            apply_rest = True
                        if ans2 == 's':
                            a.skipped = True
        for a in conflicted:
            if opts.yes and not sys.stdin.isatty():
                a.notes.append("log conflict skipped (non-interactive)")
                continue
            _resolve_conflict(a, ask, input_fn)
    else:
        print("\n[2/3] Case B - no obs log provided (--obs-log)")

    still_missing = [a for a in audits
                     if not a.skipped and not a.scan_error
                     and ((a.exptime_resolved() is None
                           and a.finding('EXPTIME').status != 'ok')
                          or (a.finding('OBJECT').fillable
                              and a.approved_fix('OBJECT') is None))]
    if still_missing and not opts.yes:
        ans = ask(f"  Manual entry for {len(still_missing)} file(s) still missing "
                  f"EXPTIME/OBJECT: [y]es, [s]kip, [q]uit", 'ysq')
        if ans == 'q':
            raise UserQuit
        if ans == 'y':
            for a in still_missing:
                suggestion = None
                if all_log_rows and not a.log_rows:
                    import obs_log as obs_log_mod
                    suggestion = obs_log_mod.nearest_row(a.stem, all_log_rows)
                if suggestion is not None:
                    row, offset = suggestion
                    print(f"  {a.stem}: no exact log match; nearest row is "
                          f"{offset:.0f} s away:\n      {row.describe()}")
                    if ask("    Use this row", 'yn') == 'y':
                        a.chosen_log_row = row
                        for fix in propose_from_log(a):
                            fix.approved = True
                        continue
                _manual_entry(a, ask, input_fn)


def resolve_case_c(audits: list[FileAudit], opts, ask, input_fn) -> None:
    pending = [a for a in audits
               if not a.skipped and not a.scan_error
               and a.finding('READ-MDE').fillable
               and (a.fix_for('READ-MDE') is None or not a.fix_for('READ-MDE').approved)]
    cross_check = []
    if opts.check_log_readmode:
        cross_check = [a for a in audits
                       if (f := a.fix_for('READ-MDE')) is not None
                       and f.approved and f.case == 'B']
    if not pending and not cross_check:
        return
    print(f"\n[3/3] Case C - READ-MDE: {len(pending)} unresolved"
          + (f", {len(cross_check)} log values to cross-check" if cross_check else ""))

    if opts.assume_readmode:
        for a in pending:
            _set_readmode_fix(a, opts.assume_readmode, 'user assumption')
        print(f"      stamped {opts.assume_readmode} on {len(pending)} file(s) "
              f"(--assume-readmode)")
        return

    try:
        import readmode_assess
    except ImportError as exc:
        print(f"      READ-MDE inference unavailable ({exc}).")
        if not opts.yes:
            for a in pending:
                _manual_readmode(a, ask, reason="inference unavailable")
        return

    min_tier = CONFIDENCE_TIER[opts.readmode_min_confidence.upper()]

    def _assess(a):
        row = a.chosen_log_row
        etype = (row.obs_type or row.object_name) if row is not None else None
        return readmode_assess.assess_readmode(
            a.path, exposure_type=etype, baselines_path=opts.baselines)

    def _prior_default(result):
        return 'SLOW' if result['prior_note'] else None

    # Pre-pass: run inference on every pending file before any prompting so
    # the whole batch can be accepted (or narrowed to the undecided ones)
    # with a single answer.
    assessed, undecided, failed = [], [], []
    n_abstain = 0
    for a in pending:
        try:
            result = _assess(a)
        except Exception as exc:
            print(f"  {a.stem}: inference failed ({exc})")
            failed.append(a)
            continue
        if result['numeric']['n_usable'] == 0:
            # No detector cast a vote at all: there is no estimation to
            # review, so the file is skipped rather than prompted for.
            print(f"  {a.stem}: skipped (all detectors abstained)")
            a.notes.append("READ-MDE not written (all detectors abstained)")
            n_abstain += 1
            continue
        summary = readmode_assess.format_assessment(result, verbose=opts.verbose)
        print(f"  {a.stem}: {summary}")
        if result['mode'] is None:
            if opts.yes:
                a.notes.append("READ-MDE not written (confidence "
                               "UNDECIDED below threshold)")
            else:
                undecided.append((a, result))
        elif opts.yes:
            if CONFIDENCE_TIER[result['confidence']] >= min_tier:
                _set_readmode_fix(a, result['mode'], _infer_source(result))
                print(f"    auto-accepted ({result['confidence']} >= "
                      f"{opts.readmode_min_confidence})")
            else:
                a.notes.append(f"READ-MDE not written (confidence "
                               f"{result['confidence']} below threshold)")
        else:
            assessed.append((a, result))

    if not opts.yes:
        for a in failed:
            _manual_readmode(a, ask, reason="inference failed")

    if assessed or undecided:
        conf_counts: dict[str, int] = {}
        for _, r in assessed:
            conf_counts[r['confidence']] = conf_counts.get(r['confidence'], 0) + 1
        conf_str = ", ".join(
            f"{n} {c}" for c, n in sorted(conf_counts.items(),
                                          key=lambda kv: -CONFIDENCE_TIER[kv[0]]))
        bits = [f"{len(assessed)} inferred" + (f" ({conf_str})" if conf_str else ""),
                f"{len(undecided)} undecided"]
        if n_abstain:
            bits.append(f"{n_abstain} all-abstain (skipped)")
        print("  " + ", ".join(bits))
        ans = ask("  Apply: [a]ccept all inferred, [u]ndecided only, "
                  "[r]eview each, [s]kip, [q]uit", 'aursq')
        if ans == 'q':
            raise UserQuit
        if ans in ('a', 'u'):
            for a, r in assessed:
                _set_readmode_fix(a, r['mode'], _infer_source(r))
            if assessed:
                print(f"      accepted {len(assessed)} inferred value(s)")
            for a, r in undecided:
                if ans == 'u':
                    _manual_readmode(a, ask, reason="inference undecided",
                                     default=_prior_default(r))
                else:
                    a.notes.append("READ-MDE not written (inference undecided)")
        elif ans == 'r':
            apply_rest = False
            for a, r in assessed:
                if apply_rest:
                    _set_readmode_fix(a, r['mode'], _infer_source(r))
                    continue
                print(f"  {a.stem}: "
                      f"{readmode_assess.format_assessment(r, verbose=opts.verbose)}")
                fans = ask("    Accept [y], [a]ll remaining, reject [n], "
                           "[e]nter mode, [s]kip, [q]uit", 'yanesq')
                if fans == 'q':
                    raise UserQuit
                if fans == 'a':
                    apply_rest = True
                if fans in ('y', 'a'):
                    _set_readmode_fix(a, r['mode'], _infer_source(r))
                elif fans == 'e':
                    _manual_readmode(a, ask, reason="user override",
                                     default=_prior_default(r))
            for a, r in undecided:
                _manual_readmode(a, ask, reason="inference undecided",
                                 default=_prior_default(r))
        # ans == 's': skip the category, nothing accepted

    for a in cross_check:
        try:
            result = _assess(a)
        except Exception as exc:
            print(f"  {a.stem}: inference failed ({exc})")
            continue
        summary = readmode_assess.format_assessment(result, verbose=opts.verbose)
        existing = a.fix_for('READ-MDE')
        if result['mode'] and existing and result['mode'] != existing.value:
            print(f"  {a.stem}: log says {existing.value} but {summary}")
            if opts.yes and not sys.stdin.isatty():
                # Batch run: the human log record wins; never abort the
                # whole night over one flagged disagreement.
                a.notes.append(
                    f"READ-MDE disagreement: log {existing.value} kept, "
                    f"inference said {result['mode']} ({result['confidence']})")
                continue
            ans = ask("    Keep [l]og value, use [i]nferred, [s]kip card, [q]uit",
                      'lisq')
            if ans == 'q':
                raise UserQuit
            if ans == 'i':
                existing.value = result['mode']
                existing.source = _infer_source(result)
            elif ans == 's':
                existing.approved = False


def _infer_source(result: dict) -> str:
    num = result['numeric']
    return (f"QA inference ({result['confidence']}, "
            f"{num['votes_slow']}S/{num['votes_fast']}F)")


def _set_readmode_fix(audit: FileAudit, mode: str, source: str) -> None:
    """Create or overwrite the (unapproved) READ-MDE fix and approve it."""
    fix = audit.fix_for('READ-MDE')
    if fix is None:
        fix = audit.propose('READ-MDE', mode, source, 'C')
    fix.value, fix.source, fix.case, fix.approved = mode, source, 'C', True


def _manual_readmode(audit: FileAudit, ask, reason: str,
                     default: str | None = None) -> None:
    ans = ask(f"    Enter READ-MDE for {audit.stem} ({reason}): "
              f"[f]ast, [s]low, s[k]ip, [q]uit", 'fskq',
              default=default and default[0].lower())
    if ans == 'q':
        raise UserQuit
    if ans in ('f', 's'):
        _set_readmode_fix(audit, 'FAST' if ans == 'f' else 'SLOW', 'user input')


# --------------------------------------------------------------- finalizing

def finalize_fixes(audit: FileAudit) -> None:
    """Derived fixes that depend on the resolved exposure time."""
    if not any(f.approved for f in audit.fixes):
        # Nothing was approved for this file, so it will not be written;
        # deriving fixes here would resurrect values the user declined.
        return
    exptime = audit.exptime_resolved()
    if exptime is None:
        return
    primary_fix = next((f for f in audit.fixes
                        if f.key in ('EXPTIME', 'REXPTIME') and f.approved), None)
    src = primary_fix.source if primary_fix else 'primary EXPTIME'
    case = primary_fix.case if primary_fix else 'A'
    if audit.ext_exptime_state in ('zero', 'blank'):
        fix = audit.propose('EXPTIME', float(exptime), src, case,
                            location='all_extensions')
        if fix:
            fix.approved = True
    # AEXPTIME is only un-blanked where the blank card already exists
    if audit.finding('AEXPTIME').status == 'blank' and audit.fix_for('AEXPTIME') is None:
        fix = audit.propose('AEXPTIME', float(exptime), src, case)
        if fix:
            fix.approved = True


# ------------------------------------------------------------------ writing

def set_card(header, key: str, value, comment: str) -> None:
    existing_comment = header.comments[key] if key in header else ''
    header[key] = (value, existing_comment or comment)


def write_repaired(audit: FileAudit, output_dir: Path, overwrite: bool,
                   obs_log_name: str | None) -> Path:
    src, dest = audit.path, output_dir / audit.path.name
    if dest.resolve() == src.resolve():
        raise RuntimeError(f"refusing to write onto the original: {src}")
    if dest.is_symlink() or dest.exists():
        if not overwrite:
            raise FileExistsError(f"{dest} exists (use --overwrite)")
        # Unlink rather than open-for-write: a symlink or hard link here
        # would otherwise make copy2/update write THROUGH it onto its
        # target, which could be an original raw file.
        dest.unlink()
    if dest.is_symlink() or dest.exists():
        raise RuntimeError(f"could not clear existing output path: {dest}")
    shutil.copy2(src, dest)
    approved = [f for f in audit.fixes if f.approved]
    with fits.open(dest, mode='update', memmap=False) as hdul:
        phdr = hdul[0].header
        for fix in approved:
            if fix.location == 'primary':
                set_card(phdr, fix.key, fix.value,
                         CARD_COMMENTS.get(fix.key, 'repaired value'))
            else:
                for hdu in hdul[1:]:
                    if not (hasattr(hdu, 'header') and hdu.is_image):
                        continue
                    current = card_status(hdu.header, fix.key, zero_suspect=True)
                    if current.status == 'ok':
                        continue      # never overwrite a filled extension value
                    set_card(hdu.header, fix.key, fix.value,
                             '[s] Image nominal exposure time')
        # A present-but-blank SEXPTIME/AEXPTIME card poisons downstream
        # hdr.get(key, fallback) chains: astropy returns Undefined for the
        # blank card instead of the fallback (reduxSetupGUI reads SEXPTIME
        # first and then fails on float(Undefined)). Convert blank -> absent
        # in the copy; filled cards are never touched and SEXPTIME is still
        # never fabricated.
        for key in ('SEXPTIME', 'AEXPTIME'):
            if card_status(phdr, key).status == 'blank':
                del phdr[key]
                phdr.add_history(
                    f"fix_exposure_info.py: removed blank {key} card")
        stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        phdr['HIERARCH FIXEXP DATE'] = (stamp, 'fix_exposure_info run (UTC)')
        # No comment: long original filenames would push the card past 80 chars
        phdr['HIERARCH FIXEXP ORIG'] = src.name
        if obs_log_name:
            phdr['HIERARCH FIXEXP OBSLOG'] = (obs_log_name, 'observing log used')
        for fix in approved:
            if fix.location == 'primary':
                # No comment, and the value is truncated to the space the
                # HIERARCH card actually has - astropy raises (not warns) on
                # HIERARCH values that don't fit in 80 characters.
                key = f'FIXEXP SRC {fix.key}'
                room = 80 - len('HIERARCH ') - len(key) - len(" = ''")
                phdr[f'HIERARCH {key}'] = fix.source[:max(room, 10)]
        keys = sorted({f.key for f in approved})
        phdr.add_history(
            f"fix_exposure_info.py {stamp}: filled {', '.join(keys)}; "
            f"original file untouched")
        hdul.flush()
    audit.written_to = dest
    return dest


def verify_written(audit: FileAudit) -> list[str]:
    problems = []
    with fits.open(audit.written_to, mode='readonly', memmap=False) as hdul:
        phdr = hdul[0].header
        for fix in audit.fixes:
            if not fix.approved:
                continue
            headers = [phdr] if fix.location == 'primary' else \
                [h.header for h in hdul[1:] if hasattr(h, 'header') and h.is_image]
            for hdr in headers:
                val = hdr.get(fix.key)
                ok = (abs(float(val) - float(fix.value)) < 1e-6
                      if isinstance(fix.value, (int, float)) and val is not None
                      and not isinstance(val, (str, fits.card.Undefined))
                      else val == fix.value)
                if not ok and fix.location != 'primary' \
                        and val not in (None, 0.0) \
                        and not isinstance(val, fits.card.Undefined):
                    ok = True       # extension kept its own pre-existing value
                if not ok:
                    problems.append(f"{fix.key}: wrote {fix.value!r}, read {val!r}")
                    break
    return problems


# -------------------------------------------------------------------- main

def discover_files(input_dir: Path, pattern: str, output_dir: Path) -> list[Path]:
    files = []
    for p in sorted(input_dir.glob(pattern)):
        if not p.is_file():
            continue
        try:
            if output_dir in p.resolve().parents:
                continue
        except OSError:
            pass
        files.append(p)
    return files


def print_write_plan(audits: list[FileAudit], output_dir: Path) -> int:
    to_write = [a for a in audits
                if not a.skipped and not a.scan_error
                and any(f.approved for f in a.fixes)]
    print(f"\n== Write plan ==")
    print(f"{len(to_write)} file(s) -> {output_dir}   (copies; originals untouched)")
    by_key: dict[str, int] = {}
    for a in to_write:
        for f in a.fixes:
            if f.approved:
                label = f.key + (' (ext)' if f.location != 'primary' else '')
                by_key[label] = by_key.get(label, 0) + 1
    for key, count in sorted(by_key.items()):
        print(f"  {key:<16} x{count}")
    return len(to_write)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Report and repair missing exposure metadata in raw LLAMAS "
                    "MEF files. Originals are never modified; repaired copies "
                    "go to --output-dir.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes: 0 ok/all resolved; 1 partial; 2 unresolved gaps; "
               "3 system error.")
    parser.add_argument('input_dir', help="directory of raw LLAMAS FITS files")
    parser.add_argument('--output-dir',
                        help="where repaired copies go (default: INPUT_DIR/modified)")
    parser.add_argument('--obs-log',
                        help="observing log: GUI obs.log tab format, CSV, or xlsx")
    parser.add_argument('--log-sheet', help="xlsx sheet name")
    parser.add_argument('--glob', default='LLAMAS*_mef.fits',
                        help="file pattern (default: %(default)s)")
    parser.add_argument('--report-only', action='store_true',
                        help="scan and report; no prompts, no writes")
    parser.add_argument('--dry-run', action='store_true',
                        help="full resolution flow, print the write plan, write nothing")
    parser.add_argument('-y', '--yes', action='store_true',
                        help="auto-accept unambiguous proposals; conflicts still prompt")
    parser.add_argument('--assume-readmode', choices=['FAST', 'SLOW'],
                        help="stamp this READ-MDE on files lacking it, skipping inference")
    parser.add_argument('--readmode-min-confidence', choices=['high', 'medium'],
                        default='high',
                        help="auto-accept tier for inferred READ-MDE (default: high)")
    parser.add_argument('--check-log-readmode', action='store_true',
                        help="also run inference to cross-check log-supplied READ-MDE")
    parser.add_argument('--baselines', help="override QA baselines for inference")
    parser.add_argument('--overwrite', action='store_true',
                        help="replace existing files in the output dir")
    parser.add_argument('--verify', action='store_true',
                        help="re-open written files and check every card")
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args(argv)


def main(argv=None, input_fn=input) -> int:
    opts = parse_args(argv)
    input_dir = Path(opts.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"error: not a directory: {input_dir}")
        return EXIT_ERROR
    output_dir = (Path(opts.output_dir).expanduser().resolve()
                  if opts.output_dir else input_dir / 'modified')
    if output_dir == input_dir:
        print("error: --output-dir must differ from the input directory "
              "(originals are never modified)")
        return EXIT_ERROR

    files = discover_files(input_dir, opts.glob, output_dir)
    if not files:
        print(f"error: no files matching {opts.glob!r} in {input_dir}")
        return EXIT_ERROR
    print(f"Scanning {len(files)} file(s) in {input_dir} ...")
    audits = [scan_file(p, _stem_of(p)) for p in files]
    for a in audits:
        propose_case_a(a)

    obs_log_name, log_rows, leftover_log = None, None, None
    if opts.obs_log:
        try:
            import obs_log as obs_log_mod
        except ImportError as exc:
            print(f"error: obs_log module unavailable: {exc}")
            return EXIT_ERROR
        try:
            known_dates = {d for a in audits
                           if (ts := obs_log_mod.stem_timestamp(a.stem))
                           for d in [ts.strftime('%Y%m%d')]}
            log_rows = _load_log_interactive(obs_log_mod, opts, known_dates, input_fn)
            index = obs_log_mod.build_log_index(log_rows)
            match_log(audits, index)
            matched_stems = {a.stem for a in audits if a.log_rows}
            leftover_log = [rows[0] for stem, rows in index.items()
                            if stem not in matched_stems]
            obs_log_name = Path(opts.obs_log).name
        except obs_log_mod.ObsLogError as exc:
            print(f"error: {exc}")
            return EXIT_ERROR

    print_report(audits, verbose=opts.verbose, leftover_log=leftover_log)
    if opts.report_only:
        return EXIT_OK if _no_gaps(audits) else EXIT_UNRESOLVED

    interactive = sys.stdin.isatty()
    if not interactive and not opts.yes:
        print("Interactive resolution required; re-run with --yes, "
              "--report-only, or from a terminal.")
        return EXIT_UNRESOLVED

    ask = make_ask(input_fn)
    try:
        resolve_case_a(audits, opts, ask)
        resolve_case_b(audits, opts, ask, input_fn, all_log_rows=log_rows)
        resolve_case_c(audits, opts, ask, input_fn)
    except UserQuit:
        print("Quit - nothing written.")
        return EXIT_UNRESOLVED
    for a in audits:
        if not a.skipped and not a.scan_error:
            finalize_fixes(a)

    n_planned = print_write_plan(audits, output_dir)
    if opts.dry_run:
        print("(dry run - nothing written)")
        return EXIT_OK if _no_gaps(audits) else EXIT_UNRESOLVED
    if n_planned == 0:
        print("Nothing approved to write.")
        return EXIT_OK if _no_gaps(audits) else EXIT_UNRESOLVED
    if not opts.yes:
        ans = ask("Proceed", 'yn')
        if ans != 'y':
            print("Aborted - nothing written.")
            return EXIT_UNRESOLVED

    output_dir.mkdir(parents=True, exist_ok=True)
    written, failed = 0, 0
    for a in audits:
        if a.skipped or a.scan_error or not any(f.approved for f in a.fixes):
            continue
        try:
            dest = write_repaired(a, output_dir, opts.overwrite, obs_log_name)
            if opts.verify:
                problems = verify_written(a)
                if problems:
                    a.write_error = "; ".join(problems)
                    failed += 1
                    print(f"  VERIFY FAILED {dest.name}: {a.write_error}")
                    # Never leave a file that looks repaired but is not
                    dest.unlink(missing_ok=True)
                    a.written_to = None
                    continue
            written += 1
        except FileExistsError as exc:
            a.write_error = str(exc)
            failed += 1
            print(f"  WRITE FAILED {a.path.name}: {exc}")
        except (OSError, RuntimeError, ValueError, fits.VerifyError) as exc:
            a.write_error = str(exc)
            failed += 1
            print(f"  WRITE FAILED {a.path.name}: {exc}")
            # Remove the half-repaired copy this run created; never leave a
            # file that looks repaired but isn't. Originals are untouched.
            partial = output_dir / a.path.name
            if partial.exists() and partial.resolve() != a.path.resolve():
                partial.unlink()
    _write_summary(audits, output_dir, obs_log_name)
    _final_summary(audits, written, failed, output_dir)

    if failed and not written:
        return EXIT_ERROR
    if failed or not _no_gaps(audits):
        return EXIT_PARTIAL if written else EXIT_UNRESOLVED
    return EXIT_OK


def _stem_of(path: Path) -> str:
    name = path.name
    if name.lower().endswith('.fits'):
        name = name[:-5]
    if name.lower().endswith('_mef'):
        name = name[:-4]
    return name


def _no_gaps(audits: list[FileAudit]) -> bool:
    # Skipped and unreadable files count as unresolved: exit 0 must mean the
    # whole directory is clean, not merely the subset the user reviewed.
    return all(not a.scan_error and not a.gaps() for a in audits)


def _load_log_interactive(obs_log_mod, opts, known_dates, input_fn):
    try:
        return obs_log_mod.load_obs_log(opts.obs_log, sheet=opts.log_sheet,
                                        known_dates=known_dates)
    except obs_log_mod.ObsLogSheetError as exc:
        if not sys.stdin.isatty():
            raise
        print(f"{exc}")
        for i, name in enumerate(exc.sheets, 1):
            print(f"  {i}) {name}")
        raw = input_fn("Sheet number: ").strip()
        if not raw.isdigit() or not 1 <= int(raw) <= len(exc.sheets):
            raise obs_log_mod.ObsLogError(f"invalid sheet choice: {raw!r}")
        return obs_log_mod.load_obs_log(opts.obs_log, sheet=exc.sheets[int(raw) - 1],
                                        known_dates=known_dates)


def _write_summary(audits: list[FileAudit], output_dir: Path,
                   obs_log_name: str | None) -> None:
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    lines = [f"# fix_exposure_info run {stamp}"
             + (f"  (obs log: {obs_log_name})" if obs_log_name else "")]
    for a in audits:
        if a.written_to:
            fixes = "; ".join(f"{f.key}={f.value!r} <- {f.source}"
                              for f in a.fixes if f.approved)
            lines.append(f"{a.written_to.name}: {fixes}")
        elif a.write_error:
            lines.append(f"{a.path.name}: FAILED ({a.write_error})")
        elif a.skipped:
            lines.append(f"{a.path.name}: skipped by user")
        gaps = a.gaps() if not a.scan_error else []
        if gaps:
            lines.append(f"{a.path.name}: remaining gaps: {', '.join(gaps)}")
    try:
        with open(output_dir / 'fix_exposure_info_summary.txt', 'a') as fh:
            fh.write("\n".join(lines) + "\n\n")
    except OSError as exc:
        print(f"  (could not write run summary: {exc})")


def _final_summary(audits: list[FileAudit], written: int, failed: int,
                   output_dir: Path) -> None:
    skipped = sum(1 for a in audits if a.skipped)
    remaining = [a for a in audits
                 if not a.scan_error and not a.skipped and a.gaps()]
    print(f"\nDone: {written} written to {output_dir}, {failed} failed, "
          f"{skipped} skipped.")
    if remaining:
        print(f"{len(remaining)} file(s) still have gaps:")
        for a in remaining[:10]:
            print(f"  {a.stem}: {', '.join(a.gaps())}")
        if len(remaining) > 10:
            print(f"  ... and {len(remaining) - 10} more")
    print("Original files were not modified.")


if __name__ == '__main__':
    sys.exit(main())
