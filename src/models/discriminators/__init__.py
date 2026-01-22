"""
Discriminators package initialization
=====================================

Discriminators for CV Quantum GAN including both quantum and classical options.
"""

# Import quantum discriminator
from .quantum_sf_discriminator import QuantumSFDiscriminator

# Import classical discriminator
from .classical_discriminator import ClassicalDiscriminator

# Define public API
__all__ = [
    'QuantumSFDiscriminator',
    'ClassicalDiscriminator'
]
