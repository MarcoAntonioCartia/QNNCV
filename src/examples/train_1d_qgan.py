"""
Train 1D QGAN - Simple Training Loop

This script implements a simple training loop for the 1D quantum generator
to test if it can learn to match target distributions.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, Any, Tuple

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.examples.working_1d_qgan import Working1DQuantumGenerator, SimpleClassicalDiscriminator


class Simple1DQGANTrainer:
    """Simple trainer for 1D QGAN."""
    
    def __init__(self, generator: Working1DQuantumGenerator, discriminator: SimpleClassicalDiscriminator):
        self.generator = generator
        self.discriminator = discriminator
        
        # Optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
        # Loss function
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
    def train_step(self, real_data: tf.Tensor, batch_size: int = 32):
        """Single training step."""
        # Generate noise
        noise = tf.random.normal([batch_size, self.generator.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake data
            fake_data = self.generator.generate(noise)
            
            # Discriminator predictions
            real_output = self.discriminator.discriminate(real_data)
            fake_output = self.discriminator.discriminate(fake_data)
            
            # Discriminator loss
            d_loss_real = self.cross_entropy(tf.ones_like(real_output), real_output)
            d_loss_fake = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
            d_loss = d_loss_real + d_loss_fake
            
            # Generator loss
            g_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        
        # Calculate gradients
        g_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        d_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        
        # Apply gradients
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        return {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'real_output': tf.reduce_mean(real_output),
            'fake_output': tf.reduce_mean(fake_output)
        }
    
    def train(self, real_data: np.ndarray, epochs: int = 50, batch_size: int = 32, 
              save_interval: int = 10):
        """Train the QGAN."""
        print(f"🚀 Starting QGAN training for {epochs} epochs...")
        print(f"   Real data shape: {real_data.shape}")
        print(f"   Batch size: {batch_size}")
        
        # Convert to tensor
        real_data_tensor = tf.convert_to_tensor(real_data, dtype=tf.float32)
        
        # Training history
        history = {
            'g_loss': [], 'd_loss': [], 
            'real_output': [], 'fake_output': [],
            'coverage': [], 'std_ratio': []
        }
        
        for epoch in range(epochs):
            # Train step
            metrics = self.train_step(real_data_tensor, batch_size)
            
            # Generate samples for evaluation
            test_noise = tf.random.normal([100, self.generator.latent_dim])
            fake_samples = self.generator.generate(test_noise).numpy()
            
            # Calculate metrics
            real_std = np.std(real_data)
            fake_std = np.std(fake_samples)
            std_ratio = fake_std / real_std if real_std > 0 else 0
            
            real_range = np.max(real_data) - np.min(real_data)
            fake_range = np.max(fake_samples) - np.min(fake_samples)
            coverage = fake_range / real_range if real_range > 0 else 0
            
            # Store history
            history['g_loss'].append(float(metrics['g_loss']))
            history['d_loss'].append(float(metrics['d_loss']))
            history['real_output'].append(float(metrics['real_output']))
            history['fake_output'].append(float(metrics['fake_output']))
            history['coverage'].append(coverage)
            history['std_ratio'].append(std_ratio)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs}: "
                      f"G={metrics['g_loss']:.4f}, D={metrics['d_loss']:.4f}, "
                      f"Cov={coverage:.1%}, Std={std_ratio:.2f}")
            
            # Save intermediate results
            if (epoch + 1) % save_interval == 0:
                self.save_results(real_data, fake_samples, epoch + 1, history)
        
        return history
    
    def save_results(self, real_data: np.ndarray, fake_data: np.ndarray, 
                    epoch: int, history: Dict[str, list]):
        """Save training results."""
        # Create results directory
        os.makedirs("results/training_1d_qgan", exist_ok=True)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram comparison
        ax = axes[0, 0]
        ax.hist(real_data, bins=30, alpha=0.5, density=True, label='Real', color='blue')
        ax.hist(fake_data, bins=30, alpha=0.5, density=True, label='Generated', color='red')
        ax.set_title(f'Distribution Comparison (Epoch {epoch})')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss curves
        ax = axes[0, 1]
        epochs = range(1, len(history['g_loss']) + 1)
        ax.plot(epochs, history['g_loss'], label='Generator Loss', color='red')
        ax.plot(epochs, history['d_loss'], label='Discriminator Loss', color='blue')
        ax.set_title('Training Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Coverage and std ratio
        ax = axes[1, 0]
        ax.plot(epochs, history['coverage'], label='Range Coverage', color='green')
        ax.plot(epochs, history['std_ratio'], label='Std Ratio', color='orange')
        ax.set_title('Quality Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Discriminator outputs
        ax = axes[1, 1]
        ax.plot(epochs, history['real_output'], label='Real Output', color='blue')
        ax.plot(epochs, history['fake_output'], label='Fake Output', color='red')
        ax.set_title('Discriminator Outputs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Output')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = f"results/training_1d_qgan/epoch_{epoch:03d}_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Results saved to: {save_path}")


def create_training_data(distribution_name: str, n_samples: int = 1000) -> np.ndarray:
    """Create training data for different distributions."""
    if distribution_name == "gaussian":
        return np.random.normal(0, 1, (n_samples, 1))
    elif distribution_name == "bimodal":
        n1 = n_samples // 2
        n2 = n_samples - n1
        samples1 = np.random.normal(-2, 0.5, (n1, 1))
        samples2 = np.random.normal(2, 0.5, (n2, 1))
        samples = np.vstack([samples1, samples2])
        np.random.shuffle(samples)
        return samples
    elif distribution_name == "uniform":
        return np.random.uniform(-3, 3, (n_samples, 1))
    else:
        raise ValueError(f"Unknown distribution: {distribution_name}")


def main():
    """Main training function."""
    print("=" * 80)
    print("🚀 TRAINING 1D QUANTUM GAN")
    print("=" * 80)
    
    # Test distributions
    distributions = ["gaussian", "bimodal", "uniform"]
    
    for dist_name in distributions:
        print(f"\n🎯 Training on {dist_name.upper()} distribution...")
        
        # Create training data
        real_data = create_training_data(dist_name, n_samples=1000)
        print(f"   Training data shape: {real_data.shape}")
        print(f"   Data range: [{real_data.min():.3f}, {real_data.max():.3f}]")
        print(f"   Data mean: {real_data.mean():.3f}, std: {real_data.std():.3f}")
        
        # Create models
        generator = Working1DQuantumGenerator(latent_dim=4, cutoff_dim=8)
        discriminator = SimpleClassicalDiscriminator(input_dim=1)
        
        # Create trainer
        trainer = Simple1DQGANTrainer(generator, discriminator)
        
        # Train
        history = trainer.train(
            real_data=real_data,
            epochs=30,  # Start with fewer epochs
            batch_size=32,
            save_interval=10
        )
        
        print(f"✅ Training completed for {dist_name}")
        
        # Final evaluation
        test_noise = tf.random.normal([500, generator.latent_dim])
        final_samples = generator.generate(test_noise).numpy()
        
        print(f"   Final generated range: [{final_samples.min():.3f}, {final_samples.max():.3f}]")
        print(f"   Final generated mean: {final_samples.mean():.3f}, std: {final_samples.std():.3f}")
        
        # Calculate final metrics
        real_range = real_data.max() - real_data.min()
        fake_range = final_samples.max() - final_samples.min()
        coverage = fake_range / real_range if real_range > 0 else 0
        
        real_std = real_data.std()
        fake_std = final_samples.std()
        std_ratio = fake_std / real_std if real_std > 0 else 0
        
        print(f"   Final coverage: {coverage:.1%}")
        print(f"   Final std ratio: {std_ratio:.2f}")
        
        if coverage > 0.5 and std_ratio > 0.5:
            print("   🎉 SUCCESS: Good learning achieved!")
        elif coverage > 0.2 or std_ratio > 0.2:
            print("   ⚠️  PARTIAL: Some learning achieved")
        else:
            print("   ❌ FAILURE: Poor learning")


if __name__ == "__main__":
    main() 