"""Filesystem path constants for the LLAMAS Pyjamas pipeline.

Defines the package root and the standard directories used for input data,
outputs, master calibrations, lookup tables, and bias frames, along with the
paths to the slow- and fast-readout master bias files. Importing this module
also ensures the output directory exists.
"""
import os

# Define the root directory for outputs
#: Absolute path to the package root directory (location of this file).
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
#: Directory containing raw/input data (``Docs/DATA`` under the package root).
DATA_DIR = os.path.join(BASE_DIR, "Docs/DATA")
#: Directory where reduction outputs are written.
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
#: Directory holding master calibration products.
CALIB_DIR = os.path.join(BASE_DIR, "mastercalib")
#: Directory holding lookup tables (e.g. trace and fiber-map LUTs).
LUT_DIR = os.path.join(BASE_DIR, "LUT")
#: Directory holding master bias frames.
BIAS_DIR = os.path.join(BASE_DIR, "Bias")
#: Path to the slow-readout master bias FITS file.
SLOW_BIAS_FILE = os.path.join(BIAS_DIR, 'slow_master_bias.fits')
#: Path to the fast-readout master bias FITS file.
FAST_BIAS_FILE = os.path.join(BIAS_DIR, 'fast_master_bias.fits')
# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)