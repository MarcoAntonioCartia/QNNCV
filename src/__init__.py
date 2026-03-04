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
from .models import QuantumSFGenerator, QuantumSFDiscriminator, ClassicalDiscriminator, DistributionDiscriminator, QuantumDistributionGenerator

# Try to import Killoran CV-QNN components
try:
    from .models.generators.killoran_cvqnn import KilloranCVQNN
    KILLORAN_AVAILABLE = True
except ImportError:
    KilloranCVQNN = None
    KILLORAN_AVAILABLE = False

from .training import QGANTrainer, QGANSFTrainer, TrainerConfig, DistributionQGANTrainer

# Try to import Killoran trainer components
try:
    from .training.killoran_trainer import KilloranQGANTrainer, KilloranTrainerConfig
    KILLORAN_TRAINER_AVAILABLE = True
except ImportError:
    KilloranQGANTrainer = None
    KilloranTrainerConfig = None
    KILLORAN_TRAINER_AVAILABLE = False

from .utils import (
    plot_wigner_3d, plot_wigner_2d, plot_distribution_comparison,
    TrainingMonitor, QuantumStateMonitor
)

__all__ = [
    # Models
    'QuantumSFGenerator',
    'QuantumSFDiscriminator', 
    'ClassicalDiscriminator',
    'DistributionDiscriminator',
    'QuantumDistributionGenerator',
    
    # Training
    'QGANTrainer',
    'QGANSFTrainer',
    'TrainerConfig',
    'DistributionQGANTrainer',
    
    # Utils
    'plot_wigner_3d',
    'plot_wigner_2d', 
    'plot_distribution_comparison',
    'TrainingMonitor',
    'QuantumStateMonitor'
]

# Add Killoran components to public API if available
if KILLORAN_AVAILABLE:
    __all__.append('KilloranCVQNN')

if KILLORAN_TRAINER_AVAILABLE:
    __all__.extend(['KilloranQGANTrainer', 'KilloranTrainerConfig'])
