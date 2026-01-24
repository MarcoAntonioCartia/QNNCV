#!/usr/bin/env python
"""
train_killoran_qgan.py - Killoran CV-QNN Training
==================================================

Train the proper CV-QNN architecture from Killoran et al. (2018)
with Kerr gates for non-Gaussian operations.

Usage:
    # Test with bimodal target (WITH Kerr gate)
    python train_killoran_qgan.py --epochs 300 --use-kerr --target-type bimodal

    # Test WITHOUT Kerr gate (should fail to learn bimodal)
    python train_killoran_qgan.py --epochs 300 --no-kerr --target-type bimodal

    # Compare both
    python train_killoran_qgan.py --epochs 300 --use-kerr --exp-name bimodal_with_kerr
    python train_killoran_qgan.py --epochs 300 --no-kerr --exp-name bimodal_no_kerr
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.killoran_trainer import main

if __name__ == "__main__":
    main()
