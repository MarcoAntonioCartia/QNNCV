"""
Generators package initialization
==================================

Quantum generators for CV Quantum GAN including the main QuantumSFGenerator
with input encoding capabilities.
"""

# Import the main quantum generator
from .quantum_sf_generator import (
    QuantumSFGenerator,
    create_simple_generator,
    create_1mode_generator
)

# Define public API
__all__ = [
    'QuantumSFGenerator',
    'create_simple_generator',
    'create_1mode_generator'
]
