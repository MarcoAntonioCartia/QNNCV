"""
Generators package initialization
==================================

Quantum generators for CV Quantum GAN including the main QuantumSFGenerator
with input encoding capabilities and the Killoran CV-QNN architecture.
"""

# Import the main quantum generator
from .quantum_sf_generator import (
    QuantumSFGenerator,
    create_simple_generator,
    create_1mode_generator
)

# Import the distribution-based quantum generator
from .quantum_distribution_generator import QuantumDistributionGenerator

# Import the Killoran CV-QNN generator
try:
    from .killoran_cvqnn import KilloranCVQNN
    KILLORAN_AVAILABLE = True
except ImportError:
    KilloranCVQNN = None
    KILLORAN_AVAILABLE = False

# Define public API
__all__ = [
    'QuantumSFGenerator',
    'create_simple_generator',
    'create_1mode_generator',
    'QuantumDistributionGenerator'
]

# Add Killoran to public API if available
if KILLORAN_AVAILABLE:
    __all__.append('KilloranCVQNN')
