#!/usr/bin/env python
"""
train_killoran_qgan.py - Killoran CV-QNN Training
==================================================

Train the proper CV-QNN architecture from Killoran et al. (2018)
with Kerr gates for non-Gaussian operations.

EXPRESSIVITY STUDY:
Test how many peaks (modes) a single qumode can learn.

Usage Examples:
--------------
# 2-modal (bimodal) - proven to work
python train_killoran_qgan.py --n-peaks 2 --epochs 300 --n-layers 6 --cutoff-dim 7

# 3-modal - test expressivity
python train_killoran_qgan.py --n-peaks 3 --epochs 400 --n-layers 6 --cutoff-dim 10

# 4-modal - harder
python train_killoran_qgan.py --n-peaks 4 --epochs 500 --n-layers 8 --cutoff-dim 12

# Compare with/without Kerr gate
python train_killoran_qgan.py --n-peaks 3 --use-kerr --exp-name test_kerr
python train_killoran_qgan.py --n-peaks 3 --no-kerr --exp-name test_nokerr

# Custom peak placement
python train_killoran_qgan.py --n-peaks 3 --x-min -3 --x-max 3 --peak-std 0.4

Outputs:
--------
./logs/experiment_name/
├── final_comparison.png      # Distribution comparison
├── training_dashboard.png    # Full training metrics
├── weight_evolution.png      # How weights change
├── wigner_epoch_*.png        # Wigner function snapshots
├── history.json              # All metrics
└── best_weights.npy          # Best model checkpoint
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.killoran_trainer import main

if __name__ == "__main__":
    main()
