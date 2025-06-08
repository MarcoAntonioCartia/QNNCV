"""
Generator models for quantum GANs.

This module contains classical and quantum generator implementations
for creating synthetic data samples from latent noise vectors.
"""

from .classical_generator import ClassicalGenerator

# Quantum generator imports with fallback handling
try:
    from .quantum_continuous_generator import QuantumContinuousGenerator
    from .quantum_continuous_generator_enhanced import (
        QuantumContinuousGeneratorEnhanced,
        QuantumContinuousGeneratorSimple
    )
    QUANTUM_GENERATORS_AVAILABLE = True
except ImportError:
    QUANTUM_GENERATORS_AVAILABLE = False

__all__ = ['ClassicalGenerator']

if QUANTUM_GENERATORS_AVAILABLE:
    __all__.extend([
        'QuantumContinuousGenerator',
        'QuantumContinuousGeneratorEnhanced',
        'QuantumContinuousGeneratorSimple',
    ])
