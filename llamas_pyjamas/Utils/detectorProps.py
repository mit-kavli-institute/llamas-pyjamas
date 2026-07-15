"""Per-detector noise properties from lab characterization.

Each LLAMAS sensor was characterised in the lab (Fe55 gain, read noise, dark
current). Those values live in ``Config/detector_lab_props.csv``, keyed on the
sensor serial number (``CAMSN`` header keyword) — the *physical* identity of the
chip, which is robust to a camera being re-plugged into a different
bench/side/colour slot. This module loads that table once and resolves the gain
and read noise to use for a given extension header.

Resolution order (see :func:`props_for_header`):
    1. explicit header keyword (EGAIN/RDNOISE, ...), if present and > 0
    2. lab table, matched on the header CAMSN serial
    3. caller-supplied defaults (last resort)

Missing/dummy extensions carry ``CAMSN = None`` (a camera under repair); those
fall through to the defaults, which is correct since there is no sensor there.
"""

import csv
import logging
import os

logger = logging.getLogger(__name__)

_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'Config',
                         'detector_lab_props.csv')

# Cached {serial(str): {gain, read_noise, dark, full_well, notes}}; None until loaded.
_TABLE = None


def _load_table():
    """Load and cache the lab-properties CSV keyed on serial number (string)."""
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    table = {}
    try:
        with open(_CSV_PATH, newline='') as fh:
            for row in csv.DictReader(fh):
                serial = (row.get('serial') or '').strip()
                if not serial:
                    continue

                def _num(key):
                    v = (row.get(key) or '').strip()
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None
                table[serial] = {
                    'gain': _num('gain_e_per_adu'),
                    'read_noise': _num('read_noise_e'),
                    'dark': _num('dark_e_per_s'),
                    'full_well': _num('full_well_e'),
                    'notes': (row.get('notes') or '').strip(),
                }
        logger.debug('Loaded %d detector lab records from %s', len(table), _CSV_PATH)
    except FileNotFoundError:
        logger.warning('Detector lab-props CSV not found at %s; using defaults only', _CSV_PATH)
    _TABLE = table
    return _TABLE


def get_props(serial):
    """Return the lab-props dict for ``serial`` (str/int), or ``None`` if absent."""
    if serial is None:
        return None
    return _load_table().get(str(serial).strip())


def props_for_header(hdr, default_gain, default_readnoise):
    """Resolve (gain, read_noise, source) for a FITS extension header.

    ``hdr`` may be an astropy Header or any object with ``.get``. ``source`` is a
    short provenance string for logging: ``'header'``, ``'labtable:<serial>'``,
    or ``'default'``.
    """
    def _hdr_num(keys):
        for k in keys:
            v = hdr.get(k) if hdr is not None else None
            if v is not None:
                try:
                    fv = float(v)
                    if fv > 0:
                        return fv
                except (TypeError, ValueError):
                    pass
        return None

    hdr_gain = _hdr_num(('EGAIN', 'GAIN', 'GAIN1', 'CCDGAIN'))
    hdr_rn = _hdr_num(('RDNOISE', 'RDNOISE1', 'READNOIS', 'RON'))
    if hdr_gain is not None and hdr_rn is not None:
        return hdr_gain, hdr_rn, 'header'

    serial = hdr.get('CAMSN') if hdr is not None else None
    props = get_props(serial)
    if props is not None:
        gain = hdr_gain if hdr_gain is not None else props.get('gain')
        rn = hdr_rn if hdr_rn is not None else props.get('read_noise')
        if gain is not None and gain > 0 and rn is not None and rn > 0:
            return gain, rn, f'labtable:{serial}'

    gain = hdr_gain if hdr_gain is not None else default_gain
    rn = hdr_rn if hdr_rn is not None else default_readnoise
    return gain, rn, 'default'
