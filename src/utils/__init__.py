"""
QNNCV Utils Package
==================

This package provides utility functions and compatibility patches for QNNCV.

The compatibility module is automatically imported to ensure that critical
patches (especially for SciPy/Strawberry Fields compatibility) are applied
before any problematic imports can occur.
"""

# Import compatibility module to auto-apply critical patches
from . import compatibility

# Make key functions available at package level
from .compatibility import apply_all_compatibility_patches, apply_scipy_compatibility_immediate

__all__ = [
    'compatibility',
    'apply_all_compatibility_patches',
    'apply_scipy_compatibility_immediate'
]
