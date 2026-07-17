"""
Spectrophotometric standard-star catalogue and coordinate crossmatch.

Standard stars are not flagged in LLAMAS headers — they carry ``OBS TYPE = SCIENCE`` and the same
``PRODCATG`` as real targets, so they can only be told apart by position. This module loads the
bundled ESO/Oke catalogue (``LUT/standards/``, produced by
:mod:`llamas_pyjamas.Scripts.fetch_standards`) and matches a pointing against it: a science frame
landing within a small radius of a catalogue star is very probably an exposure of that standard.

The bundle carries decimal J2000 coordinates for every star and an Oke reference spectrum for the
subset that has one (both may26 standards, Feige110 and GD108, do). ICRS-vs-J2000 is well under
0.1 arcsec and negligible against the default 30 arcsec radius, which is itself set generously so
that the proper motion of nearby white-dwarf standards over the decades since J2000 cannot push a
real match outside it.

Classes
-------
Standard          One catalogue entry
StandardMatch     A pointing matched to a Standard, with separation
StandardsCatalog  The catalogue, with match() / match_header()

Functions
---------
load_catalog      Load (and cache) the default catalogue
ra_dec_from_header  Pull decimal RA/Dec from a LLAMAS primary header
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from llamas_pyjamas.config import LUT_DIR

logger = logging.getLogger(__name__)

STANDARDS_DIR = os.path.join(LUT_DIR, 'standards')
INDEX_PATH = os.path.join(STANDARDS_DIR, 'index.dat')
FLUX_SUBDIR = 'flux'

DEFAULT_MATCH_RADIUS_ARCSEC = 30.0


@dataclass(frozen=True)
class Standard:
    """One catalogue standard star."""
    name: str
    ra_deg: float
    dec_deg: float
    vmag: Optional[float]
    sptype: str
    flux_file: Optional[str]        # absolute path to the reference spectrum, or None

    @property
    def has_spectrum(self) -> bool:
        return self.flux_file is not None


@dataclass(frozen=True)
class StandardMatch:
    """A pointing matched to a catalogue standard."""
    standard: Standard
    separation_arcsec: float

    @property
    def name(self) -> str:
        return self.standard.name


class StandardsCatalog:
    """The bundled standard-star catalogue, with coordinate crossmatch.

    Parameters
    ----------
    standards : list of Standard
    """

    def __init__(self, standards: List[Standard]) -> None:
        if not standards:
            raise ValueError('empty standards catalogue')
        self.standards = standards
        self._coords = SkyCoord(ra=[s.ra_deg for s in standards] * u.deg,
                                dec=[s.dec_deg for s in standards] * u.deg)

    def __len__(self) -> int:
        return len(self.standards)

    def match(self, ra_deg: float, dec_deg: float,
              radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC) -> Optional[StandardMatch]:
        """Return the nearest standard within `radius_arcsec`, or None.

        Only the single closest catalogue star is considered — standards are far apart on the
        sky, so within a 30 arcsec radius there is never a genuine ambiguity.
        """
        if ra_deg is None or dec_deg is None:
            return None
        if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
            return None

        target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        idx, sep2d, _ = target.match_to_catalog_sky(self._coords)
        separation = float(sep2d.arcsec)
        if separation > radius_arcsec:
            return None
        return StandardMatch(self.standards[int(idx)], separation)

    def match_header(self, header,
                     radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC) -> Optional[StandardMatch]:
        """Crossmatch a FITS primary header's pointing against the catalogue."""
        ra_deg, dec_deg = ra_dec_from_header(header)
        return self.match(ra_deg, dec_deg, radius_arcsec)


def ra_dec_from_header(header):
    """Decimal RA/Dec (degrees) from a LLAMAS primary header, or (None, None).

    LLAMAS writes the pointing as decimal degrees in ``RA``/``DEC`` (ICRS). The sexagesimal
    ``RA-HMS``/``TEL RA`` keys carry the same information; the decimal keys are used directly to
    avoid a parse.
    """
    ra = header.get('RA')
    dec = header.get('DEC')
    try:
        ra = float(ra)
        dec = float(dec)
    except (TypeError, ValueError):
        return None, None
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return None, None
    return ra, dec


def _parse_index(path: str) -> List[Standard]:
    flux_dir = os.path.join(os.path.dirname(path), FLUX_SUBDIR)
    standards = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            name, ra, dec, vmag, sptype, flux = parts[:6]
            flux_path = None
            if flux and flux != '-':
                candidate = os.path.join(flux_dir, flux)
                flux_path = candidate if os.path.exists(candidate) else None
            standards.append(Standard(
                name=name,
                ra_deg=float(ra),
                dec_deg=float(dec),
                vmag=(None if vmag == '-' else float(vmag)),
                sptype=('' if sptype == '-' else sptype),
                flux_file=flux_path,
            ))
    return standards


_CACHE: Optional[StandardsCatalog] = None


def load_catalog(path: str = INDEX_PATH, use_cache: bool = True) -> StandardsCatalog:
    """Load the standards catalogue, cached across calls.

    Raises
    ------
    FileNotFoundError
        If the bundle is absent — run ``python -m llamas_pyjamas.Scripts.fetch_standards``.
    """
    global _CACHE
    if use_cache and _CACHE is not None and path == INDEX_PATH:
        return _CACHE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Standards catalogue not found at {path}. '
            f'Generate it with: python -m llamas_pyjamas.Scripts.fetch_standards')
    catalog = StandardsCatalog(_parse_index(path))
    if path == INDEX_PATH:
        _CACHE = catalog
    logger.info('Loaded %d standard stars from %s', len(catalog), path)
    return catalog
