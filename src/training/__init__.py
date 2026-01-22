"""
Training package initialization
===============================

Training utilities for CV Quantum GAN including the main QGANTrainer
and configuration classes.
"""

# Import the main trainer
from .trainer import (
    QGANTrainer,
    TrainerConfig,
    train_simple_qgan,
    gaussian_data_generator,
    bimodal_data_generator
)

# Import the existing SF trainer (for backward compatibility)
from .qgan_sf_trainer import QGANSFTrainer

# Define public API
__all__ = [
    'QGANTrainer',
    'TrainerConfig',
    'train_simple_qgan',
    'gaussian_data_generator',
    'bimodal_data_generator',
    'QGANSFTrainer'
]
