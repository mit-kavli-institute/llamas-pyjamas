import os

# Define the root directory for outputs
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Docs/DATA")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CALIB_DIR = os.path.join(BASE_DIR, "mastercalib")
LUT_DIR = os.path.join(BASE_DIR, "LUT")
BIAS_DIR = os.path.join(BASE_DIR, "Bias")
SLOW_BIAS_FILE = os.path.join(BIAS_DIR, 'slow_master_bias.fits')
FAST_BIAS_FILE = os.path.join(BIAS_DIR, 'fast_master_bias.fits')
# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)