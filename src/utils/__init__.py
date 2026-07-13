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
# (compatibility.enable_clean_training is deliberately NOT imported: the
# warning_suppression import below has always shadowed it, so
# warning_suppression.enable_clean_training is the canonical one.)
from .compatibility import (
    apply_numpy_compatibility_patches,
    apply_scipy_compatibility_patches,
    apply_tensorflow_compatibility_patches,
    apply_all_compatibility_patches,
    check_environment_compatibility
)

# Import warning suppression utilities
from .warning_suppression import (
    suppress_tensorflow_warnings,
    suppress_strawberry_fields_warnings,
    suppress_numpy_warnings,
    suppress_all_quantum_warnings,
    clean_training_output,
    QuantumTrainingLogger,
    setup_clean_environment,
    enable_clean_training,
    test_warning_suppression
)

# Import import checker utilities
try:
    from .import_checker import (
        ImportChecker,
        check_imports,
        init_killoran_environment
    )
    IMPORT_CHECKER_AVAILABLE = True
except ImportError:
    ImportChecker = None
    check_imports = None
    init_killoran_environment = None
    IMPORT_CHECKER_AVAILABLE = False

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

# Add import checker to public API if available
if IMPORT_CHECKER_AVAILABLE:
    __all__.extend(['ImportChecker', 'check_imports', 'init_killoran_environment'])
