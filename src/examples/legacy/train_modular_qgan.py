"""
Example script demonstrating how to train a Quantum GAN using the modular architecture.

This script shows:
1. How to create and configure a quantum GAN
2. How to set up data generation
3. How to train the model
4. How to evaluate and visualize results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Import modular components
from src.models import create_quantum_gan
from src.utils.visualization import plot_results, plot_training_history

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_bimodal_data(batch_size: int = 32) -> tf.Tensor:
    """
    Generate bimodal Gaussian data for training.
    
    Args:
        batch_size: Number of samples to generate
        
    Returns:
        Batch of bimodal data
    """
    # Two Gaussian modes
    mode1 = np.random.normal(-1.5, 0.3, (batch_size // 2, 2))
    mode2 = np.random.normal(1.5, 0.3, (batch_size // 2, 2))
    
    # Combine and shuffle
    data = np.vstack([mode1, mode2])
    np.random.shuffle(data)
    
    return tf.constant(data, dtype=tf.float32)


def create_data_generator(batch_size: int = 32):
    """Create a data generator function."""
    def generator():
        while True:
            yield generate_bimodal_data(batch_size)
    return generator


def main():
    """Main training script."""
    logger.info("="*80)
    logger.info("QUANTUM GAN TRAINING WITH MODULAR ARCHITECTURE")
    logger.info("="*80)
    
    # Configuration
    config = {
        # Data configuration
        'batch_size': 32,
        'epochs': 50,
        'steps_per_epoch': 20,
        
        # Model configuration
        'latent_dim': 6,
        'output_dim': 2,
        'gen_n_modes': 4,
        'gen_layers': 2,
        'disc_n_modes': 2,
        'disc_layers': 2,
        'cutoff_dim': 6,
        
        # Training configuration
        'learning_rate_g': 1e-3,
        'learning_rate_d': 1e-3,
        'n_critic': 5,
        
        # Save configuration
        'save_interval': 10,
        'plot_interval': 5
    }
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"quantum_gan_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    import json
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration saved to {output_dir}/config.json")
    
    # Create Quantum GAN
    logger.info("\nCreating Quantum GAN...")
    qgan = create_quantum_gan(
        gan_type="wasserstein",
        **config
    )
    
    # Create data generator
    data_gen = create_data_generator(config['batch_size'])
    
    # Generate validation data for monitoring
    validation_data = generate_bimodal_data(500)
    
    # Initial evaluation
    logger.info("\nInitial evaluation...")
    initial_metrics = qgan.evaluate(validation_data)
    logger.info("Initial metrics:")
    for key, value in initial_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
    
    # Plot initial state
    initial_generated = qgan.generate(500)
    plot_results(validation_data, initial_generated, epoch=0, 
                save_path=os.path.join(output_dir, 'initial_state.png'))
    
    # Train the model
    logger.info("\nStarting training...")
    try:
        qgan.train(
            data_generator=data_gen,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            steps_per_epoch=config['steps_per_epoch'],
            latent_dim=config['latent_dim'],
            validation_data=validation_data,
            save_interval=config['save_interval'],
            plot_interval=config['plot_interval'],
            checkpoint_dir=output_dir
        )
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    
    # Final evaluation
    logger.info("\nFinal evaluation...")
    final_metrics = qgan.evaluate(validation_data)
    logger.info("Final metrics:")
    for key, value in final_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
    
    # Compare metrics
    logger.info("\nMetric improvements:")
    for key in ['mean_difference', 'std_difference', 'wasserstein_multivariate']:
        if key in initial_metrics and key in final_metrics:
            initial = initial_metrics[key]
            final = final_metrics[key]
            improvement = (initial - final) / initial * 100
            logger.info(f"  {key}: {initial:.4f} → {final:.4f} ({improvement:+.1f}%)")
    
    # Generate final samples
    logger.info("\nGenerating final samples...")
    final_generated = qgan.generate(1000)
    
    # Create final comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Real data
    axes[0].scatter(validation_data[:, 0], validation_data[:, 1], 
                   alpha=0.6, s=20, label='Real')
    axes[0].set_title("Real Data")
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].grid(True, alpha=0.3)
    
    # Initial generated
    axes[1].scatter(initial_generated[:, 0], initial_generated[:, 1], 
                   alpha=0.6, s=20, color='orange', label='Initial')
    axes[1].set_title("Initial Generated")
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].grid(True, alpha=0.3)
    
    # Final generated
    axes[2].scatter(final_generated[:, 0], final_generated[:, 1], 
                   alpha=0.6, s=20, color='green', label='Final')
    axes[2].set_title("Final Generated")
    axes[2].set_xlim(-3, 3)
    axes[2].set_ylim(-3, 3)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150)
    plt.close()
    
    # Save final model
    logger.info(f"\nSaving final model...")
    qgan._save_checkpoint(config['epochs'], output_dir)
    
    # Create summary report
    summary = f"""
QUANTUM GAN TRAINING SUMMARY
===========================

Configuration:
- Epochs: {config['epochs']}
- Batch size: {config['batch_size']}
- Generator modes: {config['gen_n_modes']}
- Discriminator modes: {config['disc_n_modes']}
- Latent dimension: {config['latent_dim']}

Results:
- Mean difference: {initial_metrics.get('mean_difference', 0):.4f} → {final_metrics.get('mean_difference', 0):.4f}
- Std difference: {initial_metrics.get('std_difference', 0):.4f} → {final_metrics.get('std_difference', 0):.4f}
- Wasserstein distance: {initial_metrics.get('wasserstein_multivariate', 0):.4f} → {final_metrics.get('wasserstein_multivariate', 0):.4f}

Output directory: {output_dir}
"""
    
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(summary)
    
    logger.info(summary)
    logger.info("Training completed successfully!")
    
    return qgan, output_dir


def test_loading():
    """Test loading a saved model."""
    logger.info("\nTesting model loading...")
    
    # Create a new GAN instance
    qgan = create_quantum_gan(gan_type="wasserstein")
    
    # Load from checkpoint (you would specify the actual path)
    # qgan.load_checkpoint(epoch=50, checkpoint_dir='quantum_gan_results_...')
    
    # Generate samples
    samples = qgan.generate(100)
    logger.info(f"Generated {samples.shape[0]} samples with shape {samples.shape}")


if __name__ == "__main__":
    # Run main training
    qgan, output_dir = main()
    
    # Optional: test loading
    # test_loading()
    
    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("Run complete!")
