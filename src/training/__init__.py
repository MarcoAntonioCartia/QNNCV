"""
Training package (2D QGAN loop + legacy 1D trainers).

Deliberately import-free: the live 2D-QGAN path imports
src.training.qgan_2d by full path, and eager re-exports here dragged the
legacy trainers (trainer, qgan_sf_trainer, distribution_trainer,
killoran_trainer) into every training run. Legacy entry points import
from the modules directly (e.g. training.trainer).
"""
