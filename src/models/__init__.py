"""
Models package (generators and discriminators).

Deliberately import-free: the live 2D-QGAN path imports
src.models.discriminators.qgan2d_discriminator by full path, and eager
re-exports here would drag the whole legacy 1D generator/discriminator
stack into every training run. Legacy entry points import from the
subpackages (models.generators / models.discriminators) directly.
"""
