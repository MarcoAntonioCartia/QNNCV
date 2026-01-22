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
from .models.generators.quantum_sf_generator import QuantumSFGenerator
from .models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator
from .training.qgan_sf_trainer import QGANSFTrainer

__all__ = [
    'QuantumSFGenerator',
    'QuantumSFDiscriminator', 
    'QGANSFTrainer'
]
