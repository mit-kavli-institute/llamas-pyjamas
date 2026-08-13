"""Unit tests for effective_aperture_pix — the method-aware RN aperture used by the
per-fibre error model in extractLlamas.

Guards the fix for the legacy hard-coded aperture_pix=9 (which over-counted read noise by
sqrt(9/5)=1.34x once the boxcar window shrank to +/-2.5 px). The RN term must scale with the
aperture actually used, and must be method-aware so it stays correct if optimal/Horne returns.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llamas_pyjamas.Extract.extractLlamas import effective_aperture_pix


def test_boxcar_is_twice_halfwidth():
    # current production: halfwidth=2.5 -> 5-px aperture (not the legacy 9)
    assert effective_aperture_pix('boxcar', 2.5) == 5.0
    assert effective_aperture_pix('BOXCAR', 2.5) == 5.0  # case-insensitive


def test_boxcar_legacy_halfwidth_recovers_nine():
    # sanity: the old 9-px window (half=4.5) is exactly where the hard-coded 9 came from
    assert effective_aperture_pix('boxcar', 4.5) == 9.0


def test_optimal_falls_back_to_boxcar_width_when_no_profile():
    # Horne disabled -> no per-column weights available -> conservative boxcar-equivalent width,
    # crucially NOT the legacy 9.
    assert effective_aperture_pix('optimal', 2.5) == 5.0
    assert effective_aperture_pix('horne', 2.5) == 5.0


def test_optimal_prefers_explicit_trace_aperture():
    class _Trace:
        extraction_aperture = 3.7
    assert effective_aperture_pix('optimal', 2.5, _Trace()) == 3.7


def test_default_method_is_boxcar():
    assert effective_aperture_pix() == 2.0 * 2.5
