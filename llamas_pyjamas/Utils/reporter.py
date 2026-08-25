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
log records are printed inline above the spinner (and still go to the log).

Design:
* ``PipelineReporter`` writes to the REAL terminal (``sys.__stdout__``); the
  driver's ``sys.stdout`` is redirected to the log via ``StdoutToLog`` so that
  every existing ``print()`` lands in the log file and none reach the terminal.
* Animation is used only when the stream is a TTY; when output is redirected to
  a file (``> run.log``) the reporter prints plain one-shot lines instead of
  ``\\r`` frames.
* All terminal writes are serialised under one lock so the spinner thread,
  inline warnings and phase transitions never interleave.
* Everything degrades to a no-op when ``enabled=False`` (verbose mode) or on any
  internal error — the reporter must never break a reduction.
"""

import os
import sys
import time
import threading
import logging

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


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
        if not (self.enabled and self.tty and self._active):
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
        if not self.enabled:
            return
        with self._lock:
            self._t0 = time.time()
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

    def note(self, text):
        """Print a one-off informational line above the spinner."""
        self._inline("  " + text)

    def warn(self, text):
        self._n_warn += 1
        self._inline(f"  ⚠️  {text}")

    def error(self, text):
        self._n_err += 1
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

    def finish(self, ok=True):
        if not self.enabled:
            return
        with self._lock:
            if self._active:
                self._finalize_phase(ok=ok)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            total = _fmt_dt(time.time() - self._t0) if self._t0 else ""
            bits = []
            if self._n_warn:
                bits.append(f"{self._n_warn} ⚠️")
            if self._n_err:
                bits.append(f"{self._n_err} ❌")
            tail = ("  (" + ", ".join(bits) + " — see log)") if bits else ""
            mark = "✅" if ok and not self._n_err else "⚠️"
            self._raw(f"{mark} Done in {total}{tail}\n")


class ReporterLogHandler(logging.Handler):
    """Logging handler that surfaces WARNING/ERROR records via the reporter.

    Attached to the ``llamas_pyjamas`` parent logger in place of the plain
    console StreamHandler during a curated run, so warnings/errors appear on
    the terminal (above the spinner) while INFO stays in the file only.
    """

    def __init__(self, reporter, level=logging.WARNING):
        super().__init__(level=level)
        self.reporter = reporter

    def emit(self, record):
        try:
            msg = record.getMessage()
            if record.levelno >= logging.ERROR:
                self.reporter.error(msg)
            else:
                self.reporter.warn(msg)
        except Exception:
            pass
