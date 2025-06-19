"""
Quantum-aware loss functions for GAN training.
"""

from .quantum_gan_loss import (
    QuantumGANLoss, 
    QuantumMeasurementLoss, 
    QuantumWassersteinLoss,
    compute_gradient_penalty,
    create_quantum_loss
)

__all__ = [
    'QuantumGANLoss',
    'QuantumMeasurementLoss',
    'QuantumWassersteinLoss',
    'compute_gradient_penalty',
    'create_quantum_loss'
]
