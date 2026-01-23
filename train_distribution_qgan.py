#!/usr/bin/env python
"""
train_distribution_qgan.py - Distribution-Based Quantum GAN Training
=====================================================================

This script trains a quantum generator to produce homodyne probability
distributions that match a target Gaussian distribution.

Usage:
    python train_distribution_qgan.py --epochs 200 --target-mean 2.0 --target-std 0.5

Key difference from scalar QGAN:
- Generator outputs P(x) distribution [num_bins values]
- Discriminator compares distribution shapes
- True distribution-to-distribution adversarial training

Log directory naming follows the same convention as train_qgan.py:
mean{target_mean}_std{target_std}_L{n_layers}_cut{cutoff_dim}_nc{n_critic}_{timestamp}
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.distribution_trainer import DistributionQGANTrainer, DistributionTrainerConfig
from models.generators.quantum_distribution_generator import QuantumDistributionGenerator
from models.discriminators.distribution_discriminator import DistributionDiscriminator


def parse_args():
    parser = argparse.ArgumentParser(description='Distribution-Based QGAN Training')
    
    # Model
    parser.add_argument('--n-layers', type=int, default=2,
                        help='Number of QNN layers (default: 2)')
    parser.add_argument('--cutoff-dim', type=int, default=10,
                        help='Fock space cutoff dimension (default: 10)')
    parser.add_argument('--num-bins', type=int, default=100,
                        help='Number of bins for distribution (default: 100)')
    
    # Target
    parser.add_argument('--target-mean', type=float, default=2.0,
                        help='Target Gaussian mean (default: 2.0)')
    parser.add_argument('--target-std', type=float, default=0.5,
                        help='Target Gaussian std (default: 0.5)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--g-lr', type=float, default=0.005,
                        help='Generator learning rate (default: 0.005)')
    parser.add_argument('--d-lr', type=float, default=0.001,
                        help='Discriminator learning rate (default: 0.001)')
    parser.add_argument('--n-critic', type=int, default=5,
                        help='Discriminator updates per generator update (default: 5)')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Base directory for logs and checkpoints (default: ./logs)')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name. If not provided, auto-generates from params')
    parser.add_argument('--print-every', type=int, default=10,
                        help='Print status every N epochs (default: 10)')
    parser.add_argument('--plot-every', type=int, default=50,
                        help='Plot distributions every N epochs (default: 50)')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=1000,
                        help='Early stopping patience in epochs (default: 1000)')
    
    # Flags
    parser.add_argument('--no-wgan-gp', action='store_true',
                        help='Disable WGAN-GP (use vanilla GAN loss)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    if args.seed is not None:
        import numpy as np
        import tensorflow as tf
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        print(f"Random seed set to {args.seed}")
    
    # Generate experiment directory with consistent naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.exp_name:
        exp_dir = f"{args.log_dir}/{args.exp_name}"
    else:
        # Auto-generate from key parameters (same pattern as train_qgan.py)
        exp_dir = (f"{args.log_dir}/mean{args.target_mean}_std{args.target_std}_"
                   f"L{args.n_layers}_cut{args.cutoff_dim}_nc{args.n_critic}_{timestamp}")
    
    # Use exp_dir instead of args.log_dir from here on
    args.log_dir = exp_dir
    
    # Print configuration
    print("=" * 60)
    print("Distribution-Based QGAN Training")
    print("=" * 60)
    print("\nModel Configuration:")
    print(f"  Quantum modes: 1")
    print(f"  QNN layers: {args.n_layers}")
    print(f"  Cutoff dimension: {args.cutoff_dim}")
    print(f"  Number of bins: {args.num_bins}")
    
    print("\nTarget Distribution:")
    print(f"  Mean: {args.target_mean}")
    print(f"  Std: {args.target_std}")
    
    print("\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Generator LR: {args.g_lr}")
    print(f"  Discriminator LR: {args.d_lr}")
    print(f"  WGAN-GP: {not args.no_wgan_gp}")
    
    print(f"\nLog directory: {args.log_dir}")
    print("=" * 60)
    
    # Create generator
    print("\nInitializing Quantum Distribution Generator...")
    generator = QuantumDistributionGenerator(
        n_modes=1,
        n_layers=args.n_layers,
        cutoff_dim=args.cutoff_dim,
        num_bins=args.num_bins,
        x_min=-5.0,
        x_max=5.0
    )
    print(f"  Generator config: {generator.get_config()}")
    
    # Create discriminator
    print("\nInitializing Distribution Discriminator...")
    discriminator = DistributionDiscriminator(
        input_dim=args.num_bins,
        hidden_dims=[64, 32]
    )
    print(f"  Discriminator config: {discriminator.get_config()}")
    
    # Create trainer config
    config = DistributionTrainerConfig(
        target_mean=args.target_mean,
        target_std=args.target_std,
        batch_size=args.batch_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        n_critic=args.n_critic,
        use_wgan_gp=not args.no_wgan_gp,
        num_bins=args.num_bins
    )
    
    # Create trainer
    trainer = DistributionQGANTrainer(generator, discriminator, config)
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60 + "\n")
    
    summary = trainer.train(
        epochs=args.epochs,
        log_interval=args.print_every,
        checkpoint_interval=args.plot_every,
        log_dir=args.log_dir
    )
    
    # Plot results
    trainer.plot_comparison(save_path=os.path.join(args.log_dir, 'final_comparison.png'))
    
    # Final report
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    print(f"Best Wasserstein: {summary['best_wasserstein']:.4f} at epoch {summary['best_epoch']}")
    print(f"Final Mean: {summary['final_mean']:.3f} (target: {args.target_mean})")
    print(f"Final Std: {summary['final_std']:.3f} (target: {args.target_std})")
    
    return summary


if __name__ == "__main__":
    main()