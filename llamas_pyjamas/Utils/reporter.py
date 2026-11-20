"""
llamas_pyjamas.Utils.reporter
=============================
Curated terminal reporting for pipeline runs.

The full pipeline emits ~hundreds of print()/logger.info lines. In curated
mode those all go to the log file; the terminal shows only a compact,
live-updating summary — one line per pipeline phase with an emoji, a tick and
elapsed time on completion, and a braille spinner on the active phase so the
user can see the run is alive. Looping phases (extract / wavelength / sky /
RSS over N science frames) carry a ``[k/N] name`` sub-status. WARNING/ERROR
log records are printed inline above the spinner (and still go to the log);
repeats of the same message within a phase collapse to one ``×N more`` line.
A stage can attach a summary block that prints under its phase's tick line,
and ``finish()`` closes the run with a short report.

Design:
* ``PipelineReporter`` writes to the REAL terminal (``sys.__stdout__``); the
  driver's ``sys.stdout`` is redirected to the log via ``StdoutToLog`` so that
  every existing ``print()`` lands in the log file and none reach the terminal.
* Animation is used only when the stream is a TTY; when output is redirected to
  a file (``> run.log``) the reporter prints plain one-shot lines instead of
  ``\\r`` frames.
* All terminal writes are serialised under one lock so the spinner thread,
  inline warnings and phase transitions never interleave.
* With ``enabled=False`` (verbose mode) the live display is off, but phase
  summaries and the final report still print as plain text. The reporter must
  never break a reduction, so every terminal write swallows its own errors.
"""

import re
import sys
import time
import threading
import logging
from collections import Counter
from contextlib import contextmanager

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Blank the parts of a message that change from detector to detector or file to
# file, so the same warning for 24 cameras shares one key.
_PATTERN_SUBS = (
    (re.compile(r"LLAMAS_\S+?\.(?:fits|pkl)"), "<file>"),
    (re.compile(r"\b(?:red|green|blue)[ _]?[1-4][ _]?[AB]\b", re.IGNORECASE), "<cam>"),
    (re.compile(r"\b[1-4][AB]\b"), "<bs>"),
    (re.compile(r"\b(?:red|green|blue)\b", re.IGNORECASE), "<colour>"),
    (re.compile(r"\d+(?:\.\d+)?"), "#"),
)

# Logger-name component -> label used in the end-of-run warning tally.
_AREA_LABELS = {"rss": "RSS", "reduce": "Pipeline", "stdout": "Pipeline"}


def warning_pattern(msg):
    """Key a warning message by its wording, ignoring camera, file and number."""
    key = str(msg)
    for rx, repl in _PATTERN_SUBS:
        key = rx.sub(repl, key)
    return key.strip()


def logger_area(name):
    """``llamas_pyjamas.Flat.fibreFlat`` -> ``Flat``; ``llamas_pyjamas.rss`` -> ``RSS``."""
    parts = str(name).split(".")
    area = parts[1] if len(parts) > 1 and parts[0] == "llamas_pyjamas" else parts[0]
    return _AREA_LABELS.get(area, area)


def _block_text(title, lines):
    out = [f"      {title}"] if title else []
    out += [f"        {ln}" for ln in lines]
    return "\n".join(out) + "\n"


def _fmt_dt(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class StdoutToLog:
    """File-like object that forwards written lines to a logger (INFO).

    Installed as ``sys.stdout`` during a curated run so existing ``print()``
    calls land in the log file instead of the terminal.
    """

    def __init__(self, logger):
        self._logger = logger
        self._buf = ""

    def write(self, s):
        try:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    self._logger.info(line)
        except Exception:
            pass
        return len(s)

    def flush(self):
        if self._buf.strip():
            try:
                self._logger.info(self._buf.rstrip())
            except Exception:
                pass
            self._buf = ""

    def isatty(self):
        return False


class PipelineReporter:
    """Compact live terminal reporter for a pipeline run.

    Usage::

        r = PipelineReporter(n_frames=5, enabled=True)
        r.start("LLAMAS reduction")
        r.phase("🧾", "Bias subtraction")
        ...
        r.phase("✳️", "Extracting spectra")
        for k, f in enumerate(frames, 1):
            r.frame(k, len(frames), shortname(f))
            ...
        r.summary("Fibre flat", table_lines)   # prints under this phase's tick
        r.report("Output", "5 frames -> 15 RSS")
        r.finish()
    """

    def __init__(self, n_frames=0, enabled=True, stream=None):
        self.enabled = bool(enabled)
        self.stream = stream or sys.__stdout__
        self.n_frames = int(n_frames)
        try:
            self.tty = bool(self.stream and self.stream.isatty())
        except Exception:
            self.tty = False
        self._lock = threading.RLock()
        self._t0 = None
        self._phase_t0 = None
        self._emoji = ""
        self._title = ""
        self._substatus = ""
        self._n_warn = 0
        self._n_err = 0
        self._phase_n = 0
        self._total_phases = 0
        self._spin_i = 0
        self._stop = threading.Event()
        self._thread = None
        self._active = False
        self._paused = False
        self._summaries = []    # (title, lines) blocks waiting for the phase to finish
        self._report_rows = []  # (label, text) rows for finish()
        self._seen = {}         # (level, pattern) -> [first message, repeats] this phase
        self._tally = Counter()  # warnings per logger area
        self.log_path = None

    # -- internal terminal helpers (call under _lock) --------------------
    def _raw(self, text):
        try:
            self.stream.write(text)
            self.stream.flush()
        except Exception:
            pass

    def _clear_line(self):
        if self.tty:
            self._raw("\r\033[2K")

    def _tag(self):
        if self._total_phases:
            return f"[{self._phase_n}/{self._total_phases}] "
        return f"[{self._phase_n}] " if self._phase_n else ""

    def _live_line(self):
        spin = _SPINNER[self._spin_i % len(_SPINNER)]
        sub = f"   {self._substatus}" if self._substatus else ""
        el = _fmt_dt(time.time() - self._phase_t0) if self._phase_t0 else ""
        return f"  {spin} {self._tag()}{self._emoji} {self._title}{sub}   {el}"

    def _redraw(self):
        if not (self.enabled and self.tty and self._active) or self._paused:
            return
        self._clear_line()
        self._raw(self._live_line())

    def _finalize_phase(self, ok=True):
        """Stamp the current phase line with a tick + total elapsed."""
        if not (self.enabled and self._active):
            return
        el = _fmt_dt(time.time() - self._phase_t0) if self._phase_t0 else ""
        mark = "✓" if ok else "✗"
        line = f"  {mark} {self._tag()}{self._emoji} {self._title}   {el}"
        if self.tty:
            self._clear_line()
            self._raw(line + "\n")
        else:
            self._raw(line + "\n")
        self._active = False
        self._flush_phase_extras()

    def _flush_phase_extras(self):
        """Print what belongs under a finished phase: collapsed repeats, then summaries."""
        for (mark, _), (first, n) in self._seen.items():
            if n:
                short = first if len(first) <= 70 else first[:67] + "..."
                self._raw(f'  {mark} ×{n} more like "{short}" — see log\n')
        self._seen = {}
        for title, lines in self._summaries:
            self._raw(_block_text(title, lines))
        self._summaries = []

    # -- spinner thread --------------------------------------------------
    def _spin(self):
        while not self._stop.wait(0.12):
            with self._lock:
                self._spin_i += 1
                self._redraw()

    def _ensure_thread(self):
        if self.tty and self.enabled and self._thread is None:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()

    # -- public API ------------------------------------------------------
    def start(self, title="LLAMAS reduction", subtitle=None, total_phases=0):
        self._t0 = time.time()
        if not self.enabled:
            return
        with self._lock:
            self._total_phases = int(total_phases)
            sub = subtitle
            if sub is None and self.n_frames:
                sub = f"{self.n_frames} science frame" + ("s" if self.n_frames != 1 else "")
            self._raw(f"🔭 {title}" + (f" — {sub}" if sub else "") + "\n")
        self._ensure_thread()

    def phase(self, emoji, title):
        """Finalize the current phase (tick + time) and start a new one."""
        if not self.enabled:
            return
        with self._lock:
            if self._active:
                self._finalize_phase(ok=True)
            self._phase_n += 1
            self._emoji = emoji
            self._title = title
            self._substatus = ""
            self._phase_t0 = time.time()
            self._active = True
            if self.tty:
                self._redraw()
            else:
                self._raw(f"  … {self._tag()}{emoji} {title}\n")
        self._ensure_thread()

    def frame(self, k, total, name=""):
        """Update the per-frame sub-status of the active looping phase."""
        if not self.enabled:
            return
        with self._lock:
            self._substatus = f"[{k}/{total}]" + (f" {name}" if name else "")
            if self.tty:
                self._redraw()
            else:
                self._raw(f"      [{k}/{total}] {name}\n")

    def substatus(self, text):
        if not self.enabled:
            return
        with self._lock:
            self._substatus = text or ""
            self._redraw()

    @contextmanager
    def paused(self):
        """Hold the spinner off the terminal, e.g. while asking the user a question."""
        with self._lock:
            self._paused = True
            if self.enabled:
                self._clear_line()
        try:
            yield
        finally:
            with self._lock:
                self._paused = False
                self._redraw()

    def note(self, text):
        """Print a one-off informational line above the spinner."""
        self._inline("  " + text)

    def summary(self, title, lines):
        """Show a block of lines under the current phase's tick line when it finishes.

        With no active phase, or in verbose mode, the block prints straight away.
        """
        lines = [str(ln) for ln in (lines or [])]
        with self._lock:
            if self.enabled and self._active:
                self._summaries.append((title, lines))
                return
            if self.enabled:
                self._clear_line()
            self._raw(_block_text(title, lines))
            self._redraw()

    def report(self, label, text):
        """Add a row to the end-of-run report printed by ``finish()``.

        A label that is already present keeps its place and takes the new text.
        """
        if not text:
            return
        for i, (lab, _) in enumerate(self._report_rows):
            if lab == str(label):
                self._report_rows[i] = (lab, str(text))
                return
        self._report_rows.append((str(label), str(text)))

    def warn(self, text, area=None, pattern=None):
        """Print a warning above the spinner.

        Given a ``pattern``, only the first warning of that pattern in the phase is
        printed; the rest are counted and summarised when the phase finishes.
        """
        self._n_warn += 1
        if area:
            self._tally[area] += 1
        if not self._repeat(("⚠️ ", pattern), text):
            self._inline(f"  ⚠️  {text}")

    def _repeat(self, key, text):
        """True if ``key`` was already shown this phase (the repeat is counted instead)."""
        if key[1] is None:
            return False
        with self._lock:
            seen = self._seen.get(key)
            if seen is not None:
                seen[1] += 1
                return True
            self._seen[key] = [str(text), 0]
            return False

    def count_warning(self, area=None):
        """Count a warning that a summary block already shows."""
        self._n_warn += 1
        if area:
            self._tally[area] += 1

    def error(self, text, area=None, pattern=None):
        """Print an error above the spinner; repeats collapse like warnings."""
        self._n_err += 1
        if not self._repeat(("❌", pattern), text):
            self._inline(f"  ❌ {text}")

    def _inline(self, text):
        if not self.enabled:
            return
        with self._lock:
            if self.tty:
                self._clear_line()
                self._raw(text + "\n")
                self._redraw()
            else:
                self._raw(text + "\n")

    def finish(self, ok=True, message=None):
        """Close the active phase and print the end-of-run report.

        Prints in verbose mode too, so every run ends with the same short report.
        """
        with self._lock:
            if self.enabled and self._active:
                self._finalize_phase(ok=ok)
            elif self._seen or self._summaries:
                self._flush_phase_extras()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            total = _fmt_dt(time.time() - self._t0) if self._t0 else ""
            if ok:
                head = f"{'⚠️' if self._n_err else '✅'} Done in {total}"
            else:
                head = f"❌ Failed after {total}" + (f": {message}" if message else "")
            rows = list(self._report_rows)
            bits = []
            if self._n_warn:
                areas = ", ".join(f"{a} {n}" for a, n in self._tally.most_common())
                bits.append(str(self._n_warn) + (f" ({areas})" if areas else ""))
            if self._n_err:
                bits.append(f"{self._n_err} error" + ("s" if self._n_err != 1 else ""))
            if bits:
                rows.append(("Warnings" if self._n_warn else "Errors",
                             " · ".join(bits) + " — see log"))
            if self.log_path:
                rows.append(("Log", self.log_path))
            width = max((len(label) for label, _ in rows), default=0)
            self._raw(head + "\n" + "".join(f"   {label:<{width}}  {text}\n"
                                             for label, text in rows))


class ReporterLogHandler(logging.Handler):
    """Logging handler that surfaces WARNING/ERROR records via the reporter.

    Attached to the ``llamas_pyjamas`` parent logger in place of the plain
    console StreamHandler during a curated run, so warnings/errors appear on
    the terminal (above the spinner) while INFO stays in the file only.
    Repeated messages collapse per phase (see ``warning_pattern``). A record
    logged with ``extra={'summarised': True}`` is only counted, because a
    summary block already shows it. In verbose mode it is attached alongside
    the console handler purely to count warnings for the final report.
    """

    def __init__(self, reporter, level=logging.WARNING):
        super().__init__(level=level)
        self.reporter = reporter

    def emit(self, record):
        try:
            msg = record.getMessage()
            area = logger_area(record.name)
            if record.levelno >= logging.ERROR:
                self.reporter.error(msg, area=area, pattern=warning_pattern(msg))
            elif getattr(record, "summarised", False):
                self.reporter.count_warning(area)
            else:
                self.reporter.warn(msg, area=area, pattern=warning_pattern(msg))
        except Exception:
            pass
