import os

# Define the root directory for outputs
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Docs/DATA")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)