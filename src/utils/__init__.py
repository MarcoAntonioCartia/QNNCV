"""
Utils package initialization
============================

Utilities for CV Quantum GAN including visualization, monitoring, and compatibility.
"""

# Import visualization functions
from .visualization import (
    plot_wigner_3d,
    plot_wigner_2d,
    plot_wigner_comparison,
    plot_distribution_comparison,
    plot_latent_output_mapping,
    plot_training_curves,
    plot_gradient_norms,
    plot_generation_statistics,
    plot_wasserstein_distance,
    plot_training_dashboard
)

# Import monitoring classes
from .monitoring import (
    TrainingMetrics,
    TrainingMonitor,
    QuantumStateMonitor
)

# Import compatibility utilities
from .scipy_compat import _patch_scipy_simps

# Import compatibility patches
from .compatibility import (
    apply_numpy_compatibility_patches,
    apply_scipy_compatibility_patches,
    apply_tensorflow_compatibility_patches,
    apply_all_compatibility_patches,
    check_environment_compatibility,
    enable_clean_training
)

# Define public API
__all__ = [
    # Visualization functions
    'plot_wigner_3d',
    'plot_wigner_2d', 
    'plot_wigner_comparison',
    'plot_distribution_comparison',
    'plot_latent_output_mapping',
    'plot_training_curves',
    'plot_gradient_norms',
    'plot_generation_statistics',
    'plot_wasserstein_distance',
    'plot_training_dashboard',
    
    # Monitoring classes
    'TrainingMetrics',
    'TrainingMonitor',
    'QuantumStateMonitor',
    
    # Compatibility functions
    '_patch_scipy_simps',
    'apply_numpy_compatibility_patches',
    'apply_scipy_compatibility_patches',
    'apply_tensorflow_compatibility_patches',
    'apply_all_compatibility_patches',
    'check_environment_compatibility',
    'enable_clean_training'
]
