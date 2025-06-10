"""
SciPy Compatibility Module

This module provides compatibility for the deprecated scipy.integrate.simps function
which was removed in SciPy 1.14.0 and replaced with scipy.integrate.simpson.

This is needed for Strawberry Fields compatibility with newer SciPy versions.
"""

import scipy.integrate
import sys

def _patch_scipy_simps():
    """
    Patch scipy.integrate to include the missing simps function.
    
    In SciPy >= 1.14.0, the simps function was removed and replaced with simpson.
    This function creates an alias for backward compatibility.
    """
    if not hasattr(scipy.integrate, 'simps'):
        if hasattr(scipy.integrate, 'simpson'):
            # Create alias from simpson to simps
            scipy.integrate.simps = scipy.integrate.simpson  # type: ignore[attr-defined]
            print("Applied SciPy compatibility patch: simps -> simpson")
        else:
            # This should not happen in normal SciPy installations
            print("Warning: Neither simps nor simpson found in scipy.integrate")

# Apply the patch when this module is imported
_patch_scipy_simps() 