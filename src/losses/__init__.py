"""
Quantum-aware loss functions for GAN training.
"""

from .quantum_gan_loss import QuantumGANLoss, QuantumMeasurementLoss, create_quantum_loss

__all__ = [
    'QuantumGANLoss',
    'QuantumMeasurementLoss',
    'create_quantum_loss'
]
