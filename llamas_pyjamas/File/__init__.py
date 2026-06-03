"""File subpackage: FITS I/O and multi-detector data handling.

Provides classes and helpers for reading LLAMAS multi-extension FITS files
(per-camera and all-camera containers) and for generating Row-Stacked Spectra
(RSS) products.
"""

from .llamasIO import *
from .llamasRSS import RSSgeneration