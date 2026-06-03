"""QA subpackage: quality-assurance and diagnostic plotting utilities.

Provides functions for visualising and validating reduction products, including
DS9 display, trace and comb diagnostics, and flat-field QA metrics.
"""
from .llamasQA import * # Assuming plot_ds9 is in plotting.py
from llamas_pyjamas.Trace.traceLlamasMulti import TraceLlamas
__all__ = ['plot_ds9']
