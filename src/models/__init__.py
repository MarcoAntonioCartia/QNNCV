"""
Neural network models for quantum GANs.

This module contains implementations of classical and quantum generators
and discriminators for adversarial training.
"""

from .generators.classical_generator import ClassicalGenerator
from .discriminators.classical_discriminator import ClassicalDiscriminator

# Quantum imports with fallback handling
try:
    from .generators.quantum_continuous_generator import QuantumContinuousGenerator
    from .generators.quantum_continuous_generator_enhanced import (
        QuantumContinuousGeneratorEnhanced,
        QuantumContinuousGeneratorSimple
    )
    from .discriminators.quantum_continuous_discriminator import QuantumContinuousDiscriminator
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

__all__ = [
    'ClassicalGenerator',
    'ClassicalDiscriminator',
]

if QUANTUM_AVAILABLE:
    __all__.extend([
        'QuantumContinuousGenerator',
        'QuantumContinuousGeneratorEnhanced',
        'QuantumContinuousGeneratorSimple',
        'QuantumContinuousDiscriminator',
    ])
