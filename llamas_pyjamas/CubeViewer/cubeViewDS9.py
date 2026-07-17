"""
DS9 transport for CubeViewer, over the XPA messaging system.

CubeViewer drives an already-running SAOImageDS9 by invoking the ``xpaset`` and ``xpaget``
command-line clients through :mod:`subprocess`. This deliberately avoids ``pyds9``, which is
unmaintained, is not declared in this project's dependencies, and does not build on recent
Python versions.

Images are handed to DS9 as FITS bytes piped to ``xpaset <target> fits`` on stdin, so nothing
is written to disk. The CubeViewer builds its images in memory and they stay there.

Note that the XPA *name server* (``xpans``) is bundled inside the SAOImageDS9 application and
is started automatically by DS9, but the XPA *client* tools are a separate SAO package
(https://github.com/ericmandel/xpa) and are frequently absent. :func:`find_xpa_tools` searches
the usual locations and raises an actionable error rather than failing obscurely.

Functions
---------
find_xpa_tools     Locate the xpaset/xpaget/xpaaccess executables
parse_xpans_line   Parse one registration line emitted by ``xpaget xpans``
parse_coordinates  Parse an "x y" coordinate pair as returned by DS9

Classes
-------
DS9Error           Raised for any XPA/DS9 failure
DS9                A connection to one DS9 instance
"""

import logging
import os
import shutil
import subprocess
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple

from astropy.io import fits

logger = logging.getLogger(__name__)

XPA_TOOLS = ('xpaset', 'xpaget', 'xpaaccess')

# Searched in order, after $PATH. The SAOImageDS9 bundle ships xpans but not the client
# tools; the bundle path is included anyway in case a user drops them alongside it.
XPA_SEARCH_DIRS = (
    '/usr/local/bin',
    '/opt/homebrew/bin',
    '/opt/local/bin',
    '/usr/bin',
    '/Applications/SAOImageDS9.app/Contents/MacOS',
    '/Applications/SAOImageDS9.app/Contents/Frameworks/Tksao.framework/Resources',
)

XPA_INSTALL_HINT = (
    "The XPA client tools (xpaset/xpaget) were not found. SAOImageDS9 bundles the XPA name "
    "server (xpans) but not these clients, and there is no Homebrew 'xpa' formula.\n"
    "Install them by building SAO's XPA package:\n"
    "    git clone https://github.com/ericmandel/xpa && cd xpa && ./configure && make && make install\n"
    "or install a packaged build (e.g. conda-forge), then ensure the binaries are on $PATH.\n"
    "Set the CUBEVIEWER_XPA_DIR environment variable to point at them directly if they live "
    "somewhere unusual."
)

DEFAULT_TIMEOUT = 10.0


class DS9Error(RuntimeError):
    """Any failure to locate, reach, or command a DS9 instance."""


def find_xpa_tools(extra_dir: Optional[str] = None) -> Dict[str, str]:
    """Locate the XPA client executables.

    Search order is ``$CUBEVIEWER_XPA_DIR``, then `extra_dir`, then ``$PATH``, then
    :data:`XPA_SEARCH_DIRS`.

    Returns
    -------
    dict
        Mapping of tool name to absolute path, for every tool in :data:`XPA_TOOLS`.

    Raises
    ------
    DS9Error
        If any tool cannot be found, with installation guidance.
    """
    search_dirs: List[str] = []
    env_dir = os.environ.get('CUBEVIEWER_XPA_DIR')
    if env_dir:
        search_dirs.append(env_dir)
    if extra_dir:
        search_dirs.append(extra_dir)

    found: Dict[str, str] = {}
    missing: List[str] = []
    for tool in XPA_TOOLS:
        path = None
        for directory in search_dirs:
            candidate = os.path.join(directory, tool)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                path = candidate
                break
        if path is None:
            path = shutil.which(tool)
        if path is None:
            for directory in XPA_SEARCH_DIRS:
                candidate = os.path.join(directory, tool)
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    path = candidate
                    break
        if path is None:
            missing.append(tool)
        else:
            found[tool] = path

    if missing:
        raise DS9Error(f"Missing XPA client tool(s): {', '.join(missing)}.\n\n{XPA_INSTALL_HINT}")
    return found


def parse_xpans_line(line: str) -> Optional[Tuple[str, str]]:
    """Parse one line of ``xpaget xpans`` output.

    Each registration line looks like::

        DS9 ds9 gs 7f000001:60778 simcoe

    i.e. ``class name access address user``.

    Returns
    -------
    tuple or None
        ``(class, name)``, or None if the line is blank or malformed.
    """
    fields = line.split()
    if len(fields) < 2:
        return None
    return fields[0], fields[1]


def parse_coordinates(text: str) -> Tuple[float, float]:
    """Parse an ``"x y"`` coordinate pair as returned by DS9.

    Raises
    ------
    DS9Error
        If the reply does not contain two parseable floats.
    """
    fields = text.split()
    if len(fields) < 2:
        raise DS9Error(f"Expected two coordinates from DS9, got {text!r}")
    try:
        return float(fields[0]), float(fields[1])
    except ValueError as exc:
        raise DS9Error(f"Un-parseable coordinates from DS9: {text!r}") from exc


class DS9:
    """A connection to one running DS9 instance.

    Parameters
    ----------
    target : str
        XPA target name. ``'ds9'`` matches a single running DS9; use an explicit
        ``name:port`` when several are running (see :meth:`targets`).
    timeout : float
        Per-command timeout in seconds. Keeps a wedged DS9 from freezing the GUI.

    Notes
    -----
    The connection is established lazily; constructing a ``DS9`` does not require DS9 to be
    running. This is deliberate — the CubeViewer spectrum panel is useful without DS9, so
    the application must not hard-require it at start-up (unlike the observing-log GUI,
    which connects in its constructor).
    """

    def __init__(self, target: str = 'ds9', timeout: float = DEFAULT_TIMEOUT) -> None:
        self.target = target
        self.timeout = timeout
        self._tools: Optional[Dict[str, str]] = None

    @property
    def tools(self) -> Dict[str, str]:
        """The located XPA client executables, resolved on first use."""
        if self._tools is None:
            self._tools = find_xpa_tools()
        return self._tools

    def _run(self, argv: Sequence[str], data: Optional[bytes] = None) -> str:
        try:
            proc = subprocess.run(
                list(argv),
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise DS9Error(f"XPA command timed out after {self.timeout}s: {' '.join(argv)}") from exc
        except OSError as exc:
            raise DS9Error(f"Could not execute {argv[0]}: {exc}") from exc

        stderr = proc.stderr.decode('utf-8', 'replace').strip()
        if proc.returncode != 0:
            raise DS9Error(f"XPA command failed ({' '.join(argv)}): {stderr or 'no error text'}")
        # xpaset/xpaget report "no 'xpaset' access points match" on stderr with a zero exit
        # status, so a clean return code alone is not proof of success.
        if stderr and 'match' in stderr:
            raise DS9Error(
                f"No DS9 matching target {self.target!r}: {stderr}. Is DS9 running?"
            )
        if stderr:
            logger.debug("XPA stderr (%s): %s", ' '.join(argv), stderr)
        return proc.stdout.decode('utf-8', 'replace').strip()

    def targets(self) -> List[Tuple[str, str]]:
        """List XPA access points currently registered with the name server.

        Returns
        -------
        list of (class, name)
            Empty if no DS9 is running.
        """
        reply = self._run([self.tools['xpaget'], 'xpans'])
        parsed = (parse_xpans_line(line) for line in reply.splitlines())
        return [entry for entry in parsed if entry is not None]

    def is_alive(self) -> bool:
        """Whether a DS9 matching :attr:`target` is reachable.

        Never raises for the ordinary "DS9 is not running" case — returns False. Missing XPA
        tools still raise :class:`DS9Error`, since that is a setup problem, not a state.
        """
        reply = self._run([self.tools['xpaaccess'], self.target])
        return reply.strip().lower() in ('yes', '1')

    def get(self, command: str) -> str:
        """Send an XPA *get* and return DS9's reply.

        Example: ``ds9.get('crosshair image')`` -> ``'123.5 88.0'``.
        """
        return self._run([self.tools['xpaget'], self.target] + command.split())

    def set(self, command: str, data: Optional[bytes] = None) -> None:
        """Send an XPA *set*.

        Parameters
        ----------
        command : str
            e.g. ``'frame new'``, ``'scale zscale'``, ``'mode crosshair'``.
        data : bytes, optional
            Payload piped to the command's stdin. When None, ``-p`` (no data) is used.
        """
        if data is None:
            argv = [self.tools['xpaset'], '-p', self.target] + command.split()
        else:
            argv = [self.tools['xpaset'], self.target] + command.split()
        self._run(argv, data=data)

    def set_fits(self, hdulist: fits.HDUList, frame: Optional[int] = None,
                 preserve: bool = True) -> None:
        """Display an in-memory HDUList in DS9.

        The HDUList is serialised to bytes and piped to DS9; nothing touches the filesystem.

        Parameters
        ----------
        hdulist : astropy.io.fits.HDUList
            The image to display.
        frame : int, optional
            Frame number to load into. When None, the current frame is reused.
        preserve : bool
            Keep the existing pan, zoom and scale across the load, so that refreshing after a
            wavelength change does not throw away the user's view.
        """
        if preserve:
            # Best-effort: older DS9 builds reject some of these, and losing the pan is not
            # worth failing the display over.
            for setting in ('preserve pan yes', 'preserve regions yes'):
                try:
                    self.set(setting)
                except DS9Error as exc:
                    logger.debug("Could not apply %r: %s", setting, exc)

        if frame is not None:
            self.set(f'frame {frame:d}')

        buffer = BytesIO()
        hdulist.writeto(buffer)
        self.set('fits', data=buffer.getvalue())

    def crosshair(self, system: str = 'image') -> Tuple[float, float]:
        """Return the current crosshair position in the given coordinate system."""
        return parse_coordinates(self.get(f'crosshair {system}'))
