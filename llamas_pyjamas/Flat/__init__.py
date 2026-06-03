"""Flat subpackage: flat-field calibration for the LLAMAS pipeline.

Provides flat reduction and extraction orchestration, scattered-light
modelling, and fibre-to-fibre flat-field correction utilities.
"""

from .flatProcessing import reduce_flat, produce_flat_extractions

from .scattered2dLlamas import scattered2dLlamas

from .fibre_flat import FibreFlatField, run_fibre_flat