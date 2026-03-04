"""
Training package initialization
===============================

Training utilities for CV Quantum GAN including the main QGANTrainer
and configuration classes, plus the Killoran CV-QNN trainer.
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

# Import the distribution-based trainer
from .distribution_trainer import DistributionQGANTrainer

# Import the Killoran CV-QNN trainer
try:
    from .killoran_trainer import KilloranQGANTrainer, KilloranTrainerConfig
    KILLORAN_TRAINER_AVAILABLE = True
except ImportError:
    KilloranQGANTrainer = None
    KilloranTrainerConfig = None
    KILLORAN_TRAINER_AVAILABLE = False

# Define public API
__all__ = [
    'QGANTrainer',
    'TrainerConfig',
    'train_simple_qgan',
    'gaussian_data_generator',
    'bimodal_data_generator',
    'QGANSFTrainer',
    'DistributionQGANTrainer'
]

# Add Killoran trainer to public API if available
if KILLORAN_TRAINER_AVAILABLE:
    __all__.extend(['KilloranQGANTrainer', 'KilloranTrainerConfig'])
