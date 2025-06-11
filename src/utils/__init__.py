"""
Utility functions for quantum GAN experiments.

This module provides data loading, visualization, and evaluation utilities
for quantum generative adversarial network research.
"""

from .data_utils import load_dataset, load_synthetic_data, create_output_directory
from .visualization import plot_results, plot_training_history
from .metrics import (
    compute_wasserstein_distance,
    compute_mmd,
    compute_coverage_and_precision,
    compute_fid_score,
    save_model,
    load_model
)

# Import compatibility patches first
from .scipy_compat import _patch_scipy_simps
from .tensorflow_compat import configure_tensorflow_for_quantum, suppress_complex_warnings, QuantumExecutionContext

__all__ = [
    'load_dataset',
    'load_synthetic_data',
    'create_output_directory',
    'plot_results',
    'plot_training_history',
    'compute_wasserstein_distance',
    'compute_mmd',
    'compute_coverage_and_precision',
    'compute_fid_score',
    'save_model',
    'load_model',
]
