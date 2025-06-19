"""
Complete Quantum GAN Training Framework

This module provides production-ready training infrastructure for Quantum GANs
using Pure Strawberry Fields components with guaranteed gradient flow.
"""

from .quantum_gan_trainer import QuantumGANTrainer
from .data_generators import BimodalDataGenerator, SyntheticDataGenerator

__all__ = [
    'QuantumGANTrainer',
    'BimodalDataGenerator', 
    'SyntheticDataGenerator'
]
