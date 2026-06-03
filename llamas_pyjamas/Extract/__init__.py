"""Extract subpackage: spectral extraction for the LLAMAS pipeline.

Provides the :class:`ExtractLlamas` extraction engine along with helpers for
saving and loading extraction products.
"""

from .extractLlamas import ExtractLlamas, save_extractions, load_extractions
__all__ = ['ExtractLlamas']