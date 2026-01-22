#!/usr/bin/env python
"""
Train CV Quantum GAN
====================

Main entry point for training a continuous-variable quantum GAN.

Usage:
    python train_qgan.py                    # Default settings
    python train_qgan.py --epochs 200       # More epochs
    python train_qgan.py --target-mean 3.0  # Different target
    
Architecture:
    - Quantum Generator with displacement input encoding
    - Classical Discriminator (MLP)
    - WGAN-GP loss for stable training
    
The generator uses Solution 3b: Displacement encoding with latent_dim = 2 * n_modes
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# =============================================================================
# SCIPY COMPATIBILITY PATCH - MUST BE APPLIED BEFORE IMPORTING STRAWBERRYFIELDS
# scipy.integrate.simps was renamed to scipy.integrate.simpson in SciPy 1.14+
# =============================================================================
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson
# =============================================================================

import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description='Train CV Quantum GAN')
    
    # Model architecture
    parser.add_argument('--n-modes', type=int, default=1,
                        help='Number of quantum modes (default: 1)')
    parser.add_argument('--n-layers', type=int, default=1,
                        help='Number of QNN layers (default: 1)')
    parser.add_argument('--cutoff-dim', type=int, default=6,
                        help='Fock space cutoff dimension (default: 6)')
    parser.add_argument('--encoding', type=str, default='displacement_full',
                        choices=['displacement_simple', 'displacement_full', 'displacement_squeezing'],
                        help='Input encoding type (default: displacement_full)')
    
    # Target distribution
    parser.add_argument('--target-mean', type=float, default=2.0,
                        help='Target Gaussian mean (default: 2.0)')
    parser.add_argument('--target-std', type=float, default=0.5,
                        help='Target Gaussian std (default: 0.5)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--g-lr', type=float, default=0.001,
                        help='Generator learning rate (default: 0.001)')
    parser.add_argument('--d-lr', type=float, default=0.001,
                        help='Discriminator learning rate (default: 0.001)')
    parser.add_argument('--n-critic', type=int, default=1,
                        help='Discriminator updates per generator update (default: 1)')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory for logs and checkpoints (default: ./logs)')
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
    parser.add_argument('--no-quantum-monitor', action='store_true',
                        help='Disable quantum state monitoring')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        print(f"Random seed set to {args.seed}")
    
    # Print configuration
    print("=" * 60)
    print("CV Quantum GAN Training")
    print("=" * 60)
    print("\nModel Configuration:")
    print(f"  Quantum modes: {args.n_modes}")
    print(f"  QNN layers: {args.n_layers}")
    print(f"  Cutoff dimension: {args.cutoff_dim}")
    print(f"  Encoding type: {args.encoding}")
    
    latent_dim = args.n_modes * (1 if args.encoding == 'displacement_simple' else 
                                 2 if args.encoding == 'displacement_full' else 4)
    print(f"  Latent dimension: {latent_dim}")
    
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
    
    # Import after arg parsing (for faster --help)
    from models.generators import QuantumSFGenerator
    from models.discriminators import ClassicalDiscriminator
    from training import QGANTrainer, TrainerConfig, gaussian_data_generator
    
    # Create generator
    print("\nInitializing Quantum Generator...")
    generator = QuantumSFGenerator(
        n_modes=args.n_modes,
        n_layers=args.n_layers,
        cutoff_dim=args.cutoff_dim,
        output_dim=1,
        encoding_type=args.encoding
    )
    print(f"  Generator config: {generator.get_config()}")
    
    # Create discriminator
    print("\nInitializing Classical Discriminator...")
    discriminator = ClassicalDiscriminator(
        input_dim=1,
        hidden_dims=[32, 32],
        output_dim=1
    )
    print(f"  Discriminator config: {discriminator.get_config()}")
    
    # Create trainer config
    config = TrainerConfig(
        target_mean=args.target_mean,
        target_std=args.target_std,
        batch_size=args.batch_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        n_critic=args.n_critic,
        use_wgan_gp=not args.no_wgan_gp,
        log_dir=args.log_dir,
        print_every=args.print_every,
        plot_every=args.plot_every,
        monitor_quantum_state=not args.no_quantum_monitor,
        patience=args.patience
    )
    
    # Create trainer
    trainer = QGANTrainer(generator, discriminator, config)
    
    # Create data generator
    data_gen = gaussian_data_generator(mean=args.target_mean, std=args.target_std)
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60 + "\n")
    
    summary = trainer.train(data_gen, epochs=args.epochs)
    
    # Final report
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    if summary.get('converged', False):
        print("\n✓ Model CONVERGED successfully!")
    else:
        print("\n⚠ Model did NOT converge to target distribution")
        print(f"  Mean error: {summary.get('mean_error', 'N/A'):.4f}")
        print(f"  Std error: {summary.get('std_error', 'N/A'):.4f}")
    
    return summary


if __name__ == "__main__":
    main()
