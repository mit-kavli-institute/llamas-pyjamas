"""
Crosshair picking — turns DS9 cursor motion into scene selections.

DS9 is put into crosshair mode and its crosshair position is polled; each position is handed
to the :class:`~llamas_pyjamas.CubeViewer.cubeViewScene.SpectralScene`, which decides what
spatial element sits there. Dragging the crosshair therefore scrubs the spectrum panel across
fibres, and a click is just a move that stopped.

**Polling runs on a worker thread.** An XPA round-trip to a local DS9 measures ~9.5 ms median
and up to ~17 ms on this hardware. At 10 Hz that is ~10% duty on the calling thread with
outliers past a 60 fps frame budget, and a selection change adds a second round-trip to draw
the marker — enough to make dragging visibly stutter if it ran on the Qt main thread. The
worker only does XPA I/O; the scene query and all plotting stay on the main thread, reached by
queued signals.

Work is deduplicated on the scene's opaque element identity, not on position: the crosshair
moves continuously but the selection only changes when it crosses into a new fibre, so the
expensive path runs a few times a second at most rather than ten times a second.

Classes
-------
CrosshairPoller  Worker: polls DS9 for crosshair position, off the main thread
ElementPicker    Main-thread facade: position -> scene element -> spectra + marker
"""

import logging
from typing import List, Optional

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot

from llamas_pyjamas.CubeViewer.cubeViewDS9 import DS9, DS9Error
from llamas_pyjamas.CubeViewer.cubeViewScene import Spectrum, SpectralScene

logger = logging.getLogger(__name__)

#: Regions we create carry this tag so they can be removed without disturbing the user's own.
MARKER_TAG = 'cubeview'

DEFAULT_INTERVAL_MS = 100

#: Consecutive XPA failures tolerated before declaring DS9 gone. One transient error while
#: the user is, say, resizing a window should not tear the session down.
FAILURE_LIMIT = 3


class CrosshairPoller(QObject):
    """Polls the DS9 crosshair on a worker thread.

    Emits :attr:`moved` with image coordinates. Never touches the scene or any widget.
    """

    moved = pyqtSignal(float, float)
    failed = pyqtSignal(str)

    def __init__(self, ds9: DS9, interval_ms: int = DEFAULT_INTERVAL_MS) -> None:
        super().__init__()
        self._ds9 = ds9
        self._interval_ms = int(interval_ms)
        self._timer: Optional[QTimer] = None
        self._failures = 0

    @pyqtSlot()
    def begin(self) -> None:
        """Start polling. Must run *in* the worker thread, so the timer belongs to it."""
        self._timer = QTimer()
        self._timer.setInterval(self._interval_ms)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        logger.debug('Crosshair polling every %d ms', self._interval_ms)

    @pyqtSlot()
    def finish(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    @pyqtSlot()
    def _poll(self) -> None:
        try:
            x, y = self._ds9.crosshair('image')
        except DS9Error as exc:
            self._failures += 1
            if self._failures >= FAILURE_LIMIT:
                self.finish()
                self.failed.emit(str(exc))
            return
        self._failures = 0
        self.moved.emit(x, y)

    @pyqtSlot(str)
    def send_regions(self, text: str) -> None:
        """Draw region text, off the main thread. Best-effort."""
        try:
            self._ds9.delete_region_group(MARKER_TAG)
            if text:
                self._ds9.set_regions(text)
        except DS9Error as exc:
            logger.debug('Could not update marker: %s', exc)


class ElementPicker(QObject):
    """Turns DS9 crosshair motion into scene selections.

    Owns the worker thread. Connect :attr:`selectionChanged` to the spectrum panel.

    Parameters
    ----------
    ds9 : DS9
        Transport. Not required to be live until :meth:`start`.
    scene : SpectralScene, optional
        May be set later, or swapped, via :meth:`set_scene`.
    """

    #: Spectra of the newly selected element; empty list when the selection is cleared.
    selectionChanged = pyqtSignal(list)
    #: Human-readable description for the status bar.
    statusChanged = pyqtSignal(str)
    #: DS9 became unreachable; polling has stopped.
    lost = pyqtSignal(str)

    def __init__(self, ds9: DS9, scene: Optional[SpectralScene] = None,
                 interval_ms: int = DEFAULT_INTERVAL_MS, parent=None) -> None:
        super().__init__(parent)
        self._ds9 = ds9
        self._scene = scene
        self._interval_ms = interval_ms
        self._element = None
        self._have_selection = False
        self._thread: Optional[QThread] = None
        self._poller: Optional[CrosshairPoller] = None

    def set_scene(self, scene: Optional[SpectralScene]) -> None:
        """Swap the scene. Clears any current selection, since ids are scene-specific."""
        self._scene = scene
        self._element = None
        self._have_selection = False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self) -> None:
        """Put DS9 into crosshair mode and begin polling.

        Raises
        ------
        DS9Error
            If DS9 cannot be reached — the caller decides how to report that.
        """
        if self.running:
            return
        self._ds9.set('mode crosshair')

        self._thread = QThread()
        self._poller = CrosshairPoller(self._ds9, self._interval_ms)
        self._poller.moveToThread(self._thread)
        self._thread.started.connect(self._poller.begin)
        self._poller.moved.connect(self._on_moved)          # queued onto the main thread
        self._poller.failed.connect(self._on_failed)
        self._thread.start()

    def stop(self) -> None:
        """Stop polling and clear our marker. Safe to call when not running."""
        if self._poller is not None:
            self._poller.send_regions('')
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
        self._poller = None

    @pyqtSlot(float, float)
    def _on_moved(self, x_pix: float, y_pix: float) -> None:
        if self._scene is None:
            return
        element = self._scene.element_at(x_pix, y_pix)

        if element == self._element and self._have_selection:
            return                       # same element: nothing to recompute
        self._element = element
        self._have_selection = True

        if element is None:
            self.selectionChanged.emit([])
            self.statusChanged.emit('no data')
            if self._poller is not None:
                self._poller.send_regions('')
            return

        spectra: List[Spectrum] = self._scene.spectra_at(x_pix, y_pix)
        self.selectionChanged.emit(spectra)
        self.statusChanged.emit(
            f"{spectra[0].label} — {', '.join(s.channel for s in spectra)}"
            if spectra else 'no data'
        )
        if self._poller is not None:
            region = self._scene.marker_region(x_pix, y_pix)
            if region:
                region = self._tagged(region)
            self._poller.send_regions(region)

    @staticmethod
    def _tagged(region: str) -> str:
        """Attach :data:`MARKER_TAG` so the marker can be deleted on its own."""
        if 'tag=' in region:
            return region
        return (f'{region} tag={{{MARKER_TAG}}}' if '#' in region
                else f'{region} # tag={{{MARKER_TAG}}}')

    @pyqtSlot(str)
    def _on_failed(self, message: str) -> None:
        logger.warning('Lost DS9: %s', message)
        self.stop()
        self.lost.emit(message)
