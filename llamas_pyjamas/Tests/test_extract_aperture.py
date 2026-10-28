"""Unit tests for effective_aperture_pix — the method-aware RN aperture used by the
per-fibre error model in extractLlamas.

The RN term must scale with the aperture actually used (a fixed 9 px over-counts read noise
by sqrt(9/5) for a +/-2.5 px boxcar), and must be method-aware so it stays correct for
optimal/Horne.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llamas_pyjamas.Extract.extractLlamas import effective_aperture_pix


def test_boxcar_is_twice_halfwidth():
    # default halfwidth=2.5 -> 5-px aperture
    assert effective_aperture_pix('boxcar', 2.5) == 5.0
    assert effective_aperture_pix('BOXCAR', 2.5) == 5.0  # case-insensitive


def test_boxcar_legacy_halfwidth_recovers_nine():
    # sanity: a half-width of 4.5 gives the 9-px window
    assert effective_aperture_pix('boxcar', 4.5) == 9.0


def test_optimal_falls_back_to_boxcar_width_when_no_profile():
    # Horne disabled -> no per-column weights available -> boxcar-equivalent width, not a fixed 9.
    assert effective_aperture_pix('optimal', 2.5) == 5.0
    assert effective_aperture_pix('horne', 2.5) == 5.0


def test_optimal_prefers_explicit_trace_aperture():
    class _Trace:
        extraction_aperture = 3.7
    assert effective_aperture_pix('optimal', 2.5, _Trace()) == 3.7


def test_default_method_is_boxcar():
    assert effective_aperture_pix() == 2.0 * 2.5
