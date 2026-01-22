# QNNCV - Quantum Neural Networks for Continuous Variables
# Main package initialization

__version__ = "1.0.0"
__author__ = "QNNCV Team"

# Auto-apply compatibility patches when importing
try:
    from .utils import compatibility
    print("QNNCV: Applying compatibility patches...")
    compatibility.apply_all_compatibility_patches()
except ImportError:
    pass  # Will be handled during setup

# Import main components for easy access
from .models import QuantumSFGenerator, QuantumSFDiscriminator, ClassicalDiscriminator
from .training import QGANTrainer, QGANSFTrainer, TrainerConfig
from .utils import (
    plot_wigner_3d, plot_wigner_2d, plot_distribution_comparison,
    TrainingMonitor, QuantumStateMonitor
)

__all__ = [
    # Models
    'QuantumSFGenerator',
    'QuantumSFDiscriminator', 
    'ClassicalDiscriminator',
    
    # Training
    'QGANTrainer',
    'QGANSFTrainer',
    'TrainerConfig',
    
    # Utils
    'plot_wigner_3d',
    'plot_wigner_2d', 
    'plot_distribution_comparison',
    'TrainingMonitor',
    'QuantumStateMonitor'
]
