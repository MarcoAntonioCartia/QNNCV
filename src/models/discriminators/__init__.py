"""
Discriminator models for quantum GANs.

This module contains classical and quantum discriminator implementations
for distinguishing between real and generated data samples.
"""

from .classical_discriminator import ClassicalDiscriminator

# Quantum discriminator imports with fallback handling
try:
    from .quantum_continuous_discriminator import QuantumContinuousDiscriminator
    QUANTUM_DISCRIMINATORS_AVAILABLE = True
except ImportError:
    QUANTUM_DISCRIMINATORS_AVAILABLE = False

__all__ = ['ClassicalDiscriminator']

if QUANTUM_DISCRIMINATORS_AVAILABLE:
    __all__.extend([
        'QuantumContinuousDiscriminator',
    ])
