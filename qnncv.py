#!/usr/bin/env python
"""
qnncv.py - QNNCV Unified Entry Point
=====================================

Single entry point for all CV-QNN training modes.

Usage:
    # Killoran architecture (N-modal distributions)
    python qnncv.py killoran --n-peaks 3 --epochs 300 --n-layers 7

    # Distribution-based QGAN
    python qnncv.py distribution --target-mean 2.0 --epochs 200

    # Sample-based QGAN (legacy)
    python qnncv.py sample --epochs 100 --batch-size 32
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def setup_killoran_parser(subparsers):
    """Killoran CV-QNN for N-modal distributions."""
    parser = subparsers.add_parser(
        'killoran',
        help='Killoran CV-QNN architecture (multi-modal distributions)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Killoran et al. CV-QNN with Kerr gates for learning N-modal distributions.

Examples:
    python qnncv.py killoran --n-peaks 2 --epochs 300
    python qnncv.py killoran --n-peaks 4 --n-layers 8 --cutoff-dim 10
    python qnncv.py killoran --n-peaks 3 --no-kerr  # Ablation study
        """
    )
    
    # Architecture
    parser.add_argument('--n-layers', type=int, default=6, help='Number of CV-QNN layers')
    parser.add_argument('--cutoff-dim', type=int, default=8, help='Fock space cutoff')
    parser.add_argument('--use-kerr', action='store_true', default=True, help='Use Kerr gate')
    parser.add_argument('--no-kerr', dest='use_kerr', action='store_false', help='Disable Kerr gate')
    
    # Target distribution
    parser.add_argument('--n-peaks', type=int, default=2, help='Number of peaks in target')
    parser.add_argument('--x-min', type=float, default=-2.0, help='Left peak boundary')
    parser.add_argument('--x-max', type=float, default=2.0, help='Right peak boundary')
    parser.add_argument('--peak-std', type=float, default=0.3, help='Peak standard deviation')
    
    # Training
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--g-lr', type=float, default=0.005, help='Generator learning rate')
    parser.add_argument('--d-lr', type=float, default=0.001, help='Discriminator learning rate')
    
    # Output
    parser.add_argument('--exp-name', type=str, default=None, help='Experiment name')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--save-wigner', action='store_true', default=True, help='Save Wigner snapshots')
    parser.add_argument('--no-wigner', dest='save_wigner', action='store_false')
    
    return parser


def setup_distribution_parser(subparsers):
    """Distribution-based QGAN with Hermite transform."""
    parser = subparsers.add_parser(
        'distribution',
        help='Distribution-based QGAN (Hermite transform)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Distribution-based QGAN using Hermite polynomial transform for 
differentiable homodyne measurement simulation.

Examples:
    python qnncv.py distribution --target-mean 2.0 --target-std 0.5
    python qnncv.py distribution --epochs 200 --n-layers 4
        """
    )
    
    # Architecture
    parser.add_argument('--n-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--cutoff-dim', type=int, default=15, help='Fock space cutoff')
    
    # Target
    parser.add_argument('--target-mean', type=float, default=2.0, help='Target mean')
    parser.add_argument('--target-std', type=float, default=0.5, help='Target std')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--g-lr', type=float, default=0.01, help='Generator learning rate')
    parser.add_argument('--d-lr', type=float, default=0.005, help='Discriminator learning rate')
    
    # Output
    parser.add_argument('--exp-name', type=str, default=None, help='Experiment name')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    
    return parser


def setup_sample_parser(subparsers):
    """Sample-based QGAN (legacy approach)."""
    parser = subparsers.add_parser(
        'sample',
        help='Sample-based QGAN (legacy)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Sample-based QGAN for generating point samples from quantum circuits.

Examples:
    python qnncv.py sample --epochs 100 --batch-size 64
        """
    )
    
    # Architecture
    parser.add_argument('--n-modes', type=int, default=1, help='Number of qumodes')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--cutoff-dim', type=int, default=10, help='Fock space cutoff')
    
    # Target
    parser.add_argument('--target-mean', type=float, default=0.0, help='Target mean')
    parser.add_argument('--target-std', type=float, default=1.0, help='Target std')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--g-lr', type=float, default=0.001, help='Generator learning rate')
    parser.add_argument('--d-lr', type=float, default=0.001, help='Discriminator learning rate')
    
    # Output
    parser.add_argument('--exp-name', type=str, default=None, help='Experiment name')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    
    return parser


def run_killoran(args):
    """Run Killoran CV-QNN training."""
    from training.killoran_trainer import (
        KilloranCVQNN, 
        DistributionDiscriminator,
        KilloranQGANTrainer,
        KilloranTrainerConfig
    )
    
    # Build config
    config = KilloranTrainerConfig(
        n_peaks=args.n_peaks,
        x_min=args.x_min,
        x_max=args.x_max,
        peak_std=args.peak_std,
        target_type='nmodal',
        n_layers=args.n_layers,
        cutoff_dim=args.cutoff_dim,
        use_kerr=args.use_kerr,
        batch_size=args.batch_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        save_wigner=args.save_wigner
    )
    
    # Build experiment name
    if args.exp_name:
        exp_name = args.exp_name
    else:
        kerr_str = "kerr" if args.use_kerr else "nokerr"
        exp_name = f"killoran_{kerr_str}_{args.n_peaks}modal_L{args.n_layers}_C{args.cutoff_dim}"
    
    log_dir = os.path.join(args.log_dir, exp_name)
    
    # Create models
    print("\nInitializing Killoran CV-QNN Generator...")
    generator = KilloranCVQNN(
        n_layers=config.n_layers,
        cutoff_dim=config.cutoff_dim,
        use_kerr=config.use_kerr
    )
    
    print("\nInitializing Discriminator...")
    discriminator = DistributionDiscriminator(input_dim=100, hidden_dims=[128, 64, 32])
    
    # Train
    trainer = KilloranQGANTrainer(generator, discriminator, config)
    summary = trainer.train(epochs=args.epochs, log_dir=log_dir)
    
    # Save outputs
    trainer.plot_comparison(save_path=os.path.join(log_dir, 'final_comparison.png'), show=False)
    trainer.plot_full_dashboard(log_dir)
    
    return summary


def run_distribution(args):
    """Run distribution-based QGAN training."""
    from training.distribution_trainer import main as dist_main
    
    # Pass args directly
    sys.argv = ['distribution_trainer.py',
                '--epochs', str(args.epochs),
                '--target-mean', str(args.target_mean),
                '--target-std', str(args.target_std),
                '--n-layers', str(args.n_layers),
                '--cutoff-dim', str(args.cutoff_dim),
                '--batch-size', str(args.batch_size),
                '--g-lr', str(args.g_lr),
                '--d-lr', str(args.d_lr)]
    
    if args.exp_name:
        sys.argv.extend(['--exp-name', args.exp_name])
    if args.log_dir:
        sys.argv.extend(['--log-dir', args.log_dir])
    
    return dist_main()


def run_sample(args):
    """Run sample-based QGAN training."""
    from training.trainer import QGANTrainer, QGANConfig
    from models.generators import QuantumSFGenerator
    from models.discriminators import QuantumSFDiscriminator
    
    # Build config
    config = QGANConfig(
        n_modes=args.n_modes,
        n_layers=args.n_layers,
        cutoff_dim=args.cutoff_dim,
        target_mean=args.target_mean,
        target_std=args.target_std,
        batch_size=args.batch_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr
    )
    
    # Create models
    print("\nInitializing Sample-based Generator...")
    generator = QuantumSFGenerator(
        n_modes=config.n_modes,
        n_layers=config.n_layers,
        cutoff_dim=config.cutoff_dim
    )
    
    print("\nInitializing Discriminator...")
    discriminator = QuantumSFDiscriminator(input_dim=config.n_modes)
    
    # Train
    trainer = QGANTrainer(generator, discriminator, config)
    
    exp_name = args.exp_name or f"sample_L{args.n_layers}_M{args.n_modes}"
    log_dir = os.path.join(args.log_dir, exp_name)
    
    summary = trainer.train(epochs=args.epochs, log_dir=log_dir)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        prog='qnncv',
        description='QNNCV - Continuous Variable Quantum Neural Networks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s killoran --n-peaks 3 --epochs 300
    %(prog)s distribution --target-mean 2.0
    %(prog)s sample --epochs 100

For mode-specific help:
    %(prog)s killoran --help
    %(prog)s distribution --help
    %(prog)s sample --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Training mode')
    
    setup_killoran_parser(subparsers)
    setup_distribution_parser(subparsers)
    setup_sample_parser(subparsers)
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return 1
    
    print("=" * 60)
    print(f"QNNCV - {args.mode.upper()} mode")
    print("=" * 60)
    
    if args.mode == 'killoran':
        return run_killoran(args)
    elif args.mode == 'distribution':
        return run_distribution(args)
    elif args.mode == 'sample':
        return run_sample(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
