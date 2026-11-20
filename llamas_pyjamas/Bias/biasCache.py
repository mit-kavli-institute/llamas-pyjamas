"""In-memory cache of processed master-bias HDUs.

A master bias is about 800 MB, and ``traceLlamasMaster._grab_bias_hdu`` is called once per
camera per frame, so it is trimmed and oriented once per process and kept here.

The cache must not live in ``Trace/traceLlamasMaster.py``. That module is registered with
``cloudpickle.register_pickle_by_value`` (``llamas_pyjamas/__init__.py``), so every pickled
trace carries the module globals its methods reach. A cache held there was written into every
trace saved in the driver (1.6 GB per green shifted-master trace, 2026-11-14), and from there
into every extraction. Functions in this module are pickled by reference, so the cached data
never reaches a pickle.
"""
import os

from llamas_pyjamas.File.llamasIO import process_fits_by_color

# {absolute path: (mtime, HDUList)}. Holds the slow and fast master bias. Per process: Ray
# workers fill their own.
_BIAS_HDU_CACHE = {}


def load_master_bias_hdus(path):
    """``process_fits_by_color(path)`` HDUs, cached per path and reloaded when the mtime changes."""
    try:
        key = os.path.abspath(path)
        mtime = os.path.getmtime(path) if os.path.exists(path) else None
    except Exception:
        key, mtime = str(path), None
    hit = _BIAS_HDU_CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    bias_hdus, _ = process_fits_by_color(path)
    if bias_hdus is not None:
        _BIAS_HDU_CACHE[key] = (mtime, bias_hdus)
    return bias_hdus


def clear_bias_cache():
    """Drop every cached master bias (frees about 1.6 GB once both are loaded)."""
    _BIAS_HDU_CACHE.clear()
