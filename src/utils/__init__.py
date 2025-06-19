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
from .quantum_metrics import QuantumMetrics
from .visualization import plot_results, plot_training_history
from .warning_suppression import suppress_all_quantum_warnings
from .tensor_utils import (
    safe_tensor_indexing, 
    batch_gather, 
    ensure_tensor, 
    safe_reduce_mean,
    safe_random_normal
)

__all__ = [
    'compatibility',
    'apply_all_compatibility_patches',
    'apply_scipy_compatibility_immediate',
    'QuantumMetrics',
    'plot_results',
    'plot_training_history',
    'suppress_all_quantum_warnings',
    'safe_tensor_indexing',
    'batch_gather',
    'ensure_tensor',
    'safe_reduce_mean',
    'safe_random_normal'
]
