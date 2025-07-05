"""
High-Performance Coordinate Quantum GAN Training - 4 Mode Optimized

Optimized for speed with minimal overhead during training.
All heavy visualizations and monitoring moved to post-training analysis.
"""

import argparse
import numpy as np
import tensorflow as tf
import logging
import sys
import os
from datetime import datetime
import json
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.warning_suppression import suppress_all_quantum_warnings
from src.models.generators.coordinate_quantum_generator import CoordinateQuantumGenerator
from src.models.discriminators.pure_sf_discriminator import PureSFDiscriminator
from src.training.data_generators import BimodalDataGenerator
from src.losses.quantum_gan_loss import QuantumWassersteinLoss

# Suppress warnings
suppress_all_quantum_warnings()

# Configure minimal logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FastCoordinateGANTrainer:
    """High-performance trainer optimized for speed with minimal overhead."""
    
    def __init__(self, 
                 latent_dim: int = 2,
                 data_dim: int = 2,
                 n_modes: int = 4,  # Increased to 4 modes for richer generation
                 mode_centers: list = [(-2.0, -2.0), (2.0, -2.0), (-2.0, 2.0), (2.0, 2.0)],
                 mode_std: float = 0.4):
        """Initialize high-performance trainer."""
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.n_modes = n_modes
        self.mode_centers = mode_centers
        self.mode_std = mode_std
        
        print(f"🚀 Initializing Fast 4-Mode Coordinate Quantum GAN")
        print(f"   Modes: {n_modes}")
        print(f"   Mode centers: {mode_centers}")
        print(f"   Optimization: Maximum speed, minimal overhead")
        
        # Create coordinate generator with 4 modes
        self.generator = CoordinateQuantumGenerator(
            latent_dim=latent_dim,
            output_dim=data_dim,
            n_modes=n_modes,
            layers=1,
            cutoff_dim=6,
            clustering_method='kmeans',
            coordinate_names=['X', 'Y']
        )
        
        # Create discriminator with 4 modes
        self.discriminator = PureSFDiscriminator(
            input_dim=data_dim,
            n_modes=n_modes,
            layers=1,
            cutoff_dim=6
        )
        
        # Use optimized loss function
        self.loss_function = QuantumWassersteinLoss(
            lambda_gp=10.0,
            lambda_entropy=1.0,
            lambda_physics=1.0
        )
        
        # Optimizers with increased learning rate for faster training
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.5)
        
        # Create 4-mode data generator
        self.data_generator = self._create_4_mode_data_generator()
        
        # Minimal training history - only essential metrics
        self.training_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'mode_coverage': [],
            'epoch_times': [],
            'step_times': []
        }
        
        print(f"✅ Fast trainer initialized")
        print(f"   Generator params: {len(self.generator.trainable_variables)}")
        print(f"   Discriminator params: {len(self.discriminator.trainable_variables)}")
    
    def _create_4_mode_data_generator(self):
        """Create optimized 4-mode data generator."""
        class FourModeDataGenerator:
            def __init__(self, mode_centers, mode_std, batch_size=64):
                self.mode_centers = mode_centers
                self.mode_std = mode_std
                self.batch_size = batch_size
                self.n_modes = len(mode_centers)
            
            def generate_batch(self):
                """Generate 4-mode data efficiently."""
                # Distribute samples across 4 modes
                samples_per_mode = self.batch_size // self.n_modes
                remainder = self.batch_size % self.n_modes
                
                batch_data = []
                
                for i, center in enumerate(self.mode_centers):
                    n_samples = samples_per_mode + (1 if i < remainder else 0)
                    if n_samples > 0:
                        # Generate samples for this mode
                        samples = np.random.normal(
                            loc=center, 
                            scale=self.mode_std, 
                            size=(n_samples, 2)
                        )
                        batch_data.append(samples)
                
                # Combine and shuffle
                batch = np.vstack(batch_data)
                np.random.shuffle(batch)
                
                return tf.constant(batch, dtype=tf.float32)
        
        return FourModeDataGenerator(self.mode_centers, self.mode_std)
    
    def fixed_loss_computation(self, real_data: tf.Tensor, fake_data: tf.Tensor):
        """Optimized loss computation with minimal overhead."""
        # Ensure both tensors have the same batch size
        real_batch_size = tf.shape(real_data)[0]
        fake_batch_size = tf.shape(fake_data)[0]
        min_batch_size = tf.minimum(real_batch_size, fake_batch_size)
        
        # Truncate to minimum batch size
        real_truncated = real_data[:min_batch_size]
        fake_truncated = fake_data[:min_batch_size]
        
        # Compute discriminator outputs
        real_output = self.discriminator.discriminate(real_truncated)
        fake_output = self.discriminator.discriminate(fake_truncated)
        
        # Wasserstein distance
        w_distance = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
        
        # Gradient penalty with fixed batch size
        alpha = tf.random.uniform([min_batch_size, 1], 0.0, 1.0)
        interpolated = alpha * real_truncated + (1 - alpha) * fake_truncated
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            interp_output = self.discriminator.discriminate(interpolated)
        
        gradients = gp_tape.gradient(interp_output, interpolated)
        
        if gradients is None:
            gradient_penalty = tf.constant(0.0)
        else:
            gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
            gradient_penalty = 10.0 * tf.reduce_mean(tf.square(gradient_norm - 1.0))
        
        # Quantum regularization
        try:
            quantum_cost = self.generator.compute_quantum_cost()
            entropy_bonus = 1.0 * quantum_cost
        except:
            entropy_bonus = tf.constant(0.0)
        
        # Final losses
        d_loss = -w_distance + gradient_penalty
        g_loss = -tf.reduce_mean(fake_output) - entropy_bonus
        
        return d_loss, g_loss, w_distance, gradient_penalty
    
    def discriminator_train_step(self, real_data: tf.Tensor, z: tf.Tensor):
        """Optimized discriminator training step."""
        with tf.GradientTape() as disc_tape:
            fake_data = self.generator.generate(z)
            d_loss, g_loss, w_distance, gradient_penalty = self.fixed_loss_computation(real_data, fake_data)
            
            # NaN protection
            if tf.math.is_nan(d_loss):
                d_loss = tf.constant(1.0)
        
        # Apply gradients with clipping
        disc_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        
        if disc_gradients is not None:
            # Simple gradient clipping without extensive monitoring
            clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else None 
                               for grad in disc_gradients]
            
            # Filter out None gradients
            valid_grads_and_vars = [(grad, var) for grad, var in zip(clipped_gradients, self.discriminator.trainable_variables) 
                                   if grad is not None and not tf.reduce_any(tf.math.is_nan(grad))]
            
            if valid_grads_and_vars:
                self.discriminator_optimizer.apply_gradients(valid_grads_and_vars)
        
        return d_loss
    
    def generator_train_step(self, z: tf.Tensor):
        """Optimized generator training step."""
        with tf.GradientTape() as gen_tape:
            fake_data = self.generator.generate(z)
            dummy_real = tf.zeros_like(fake_data)
            d_loss, g_loss, w_distance, gradient_penalty = self.fixed_loss_computation(dummy_real, fake_data)
            
            # NaN protection
            if tf.math.is_nan(g_loss):
                g_loss = tf.constant(1.0)
        
        # Apply gradients with clipping
        gen_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        
        if gen_gradients is not None:
            # Simple gradient clipping without extensive monitoring
            clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else None 
                               for grad in gen_gradients]
            
            # Filter out None gradients
            valid_grads_and_vars = [(grad, var) for grad, var in zip(clipped_gradients, self.generator.trainable_variables) 
                                   if grad is not None and not tf.reduce_any(tf.math.is_nan(grad))]
            
            if valid_grads_and_vars:
                self.generator_optimizer.apply_gradients(valid_grads_and_vars)
        
        return g_loss
    
    def calculate_mode_coverage_fast(self, generated_samples: tf.Tensor):
        """Fast mode coverage calculation for 4 modes."""
        gen_np = generated_samples.numpy()
        
        # Calculate distances to all mode centers
        mode_assignments = []
        mode_coverages = []
        
        for i, center in enumerate(self.mode_centers):
            center_np = np.array(center)
            distances = np.linalg.norm(gen_np - center_np, axis=1)
            mode_assignments.append(distances)
        
        # Assign each sample to closest mode
        mode_assignments = np.array(mode_assignments)
        closest_modes = np.argmin(mode_assignments, axis=0)
        
        # Calculate coverage for each mode
        total_samples = len(gen_np)
        for i in range(self.n_modes):
            coverage = np.sum(closest_modes == i) / total_samples
            mode_coverages.append(float(coverage))
        
        # Calculate balance metric (minimum coverage / maximum coverage)
        max_coverage = max(mode_coverages)
        min_coverage = min(mode_coverages)
        balanced_coverage = min_coverage / max_coverage if max_coverage > 0 else 0.0
        
        return {
            'mode_coverages': mode_coverages,
            'balanced_coverage': float(balanced_coverage),
            'total_modes_covered': sum(1 for cov in mode_coverages if cov > 0.05)  # Modes with >5% coverage
        }
    
    def train(self, epochs: int = 20, batch_size: int = 64, save_dir: str = "results/fast_training"):
        """High-performance training loop with minimal overhead."""
        print(f"\n🏃‍♂️ STARTING HIGH-PERFORMANCE 4-MODE TRAINING")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Modes: {self.n_modes}")
        print(f"  Optimization: Speed-focused, minimal I/O")
        print("=" * 60)
        
        # Create minimal results directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Update data generator batch size
        self.data_generator.batch_size = batch_size
        
        # Analyze target data quickly
        print("\n🎯 Quick target data analysis...")
        target_data = self.data_generator.generate_batch().numpy()
        analysis_results = self.generator.analyze_target_data(target_data)
        print(f"✅ Target analysis complete - {len(analysis_results)} clusters detected")
        
        # Training loop with timing
        total_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            epoch_gen_losses = []
            epoch_disc_losses = []
            
            # Reduced steps per epoch for faster training
            steps_per_epoch = 15
            
            for step in range(steps_per_epoch):
                step_start_time = time.time()
                
                # Get real data
                real_data = self.data_generator.generate_batch()
                
                # Train discriminator (reduced from 5 to 3 steps)
                disc_losses = []
                for _ in range(3):
                    z = tf.random.normal([batch_size, self.latent_dim])
                    disc_loss = self.discriminator_train_step(real_data, z)
                    disc_losses.append(float(disc_loss))
                
                # Train generator (1 step)
                z = tf.random.normal([batch_size, self.latent_dim])
                gen_loss = self.generator_train_step(z)
                
                epoch_gen_losses.append(float(gen_loss))
                epoch_disc_losses.append(np.mean(disc_losses))
                
                # Store step timing
                step_time = time.time() - step_start_time
                self.training_history['step_times'].append(step_time)
                
                # Minimal progress output (only every 5 steps)
                if step % 5 == 0:
                    print(f"  Step {step+1:2d}: G={gen_loss:.3f}, D={np.mean(disc_losses):.3f}, Time={step_time:.2f}s")
            
            # Calculate epoch averages
            avg_gen_loss = np.mean(epoch_gen_losses)
            avg_disc_loss = np.mean(epoch_disc_losses)
            
            # Quick mode coverage calculation (only every 5 epochs)
            if (epoch + 1) % 5 == 0:
                test_z = tf.random.normal([500, self.latent_dim])
                test_generated = self.generator.generate(test_z)
                coverage_metrics = self.calculate_mode_coverage_fast(test_generated)
            else:
                coverage_metrics = {'balanced_coverage': 0.0, 'mode_coverages': [0.0] * self.n_modes}
            
            # Store minimal history
            self.training_history['generator_loss'].append(avg_gen_loss)
            self.training_history['discriminator_loss'].append(avg_disc_loss)
            self.training_history['mode_coverage'].append(coverage_metrics)
            
            epoch_time = time.time() - epoch_start_time
            self.training_history['epoch_times'].append(epoch_time)
            
            # Compact progress output
            print(f"Epoch {epoch+1:2d}/{epochs}: G={avg_gen_loss:.3f}, D={avg_disc_loss:.3f}, "
                  f"Balanced={coverage_metrics['balanced_coverage']:.3f}, Time={epoch_time:.1f}s")
        
        total_time = time.time() - total_start_time
        
        # Final results
        print(f"\n🎉 TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Average epoch time: {np.mean(self.training_history['epoch_times']):.1f}s")
        print(f"Average step time: {np.mean(self.training_history['step_times']):.2f}s")
        
        # Final evaluation
        print(f"\n📊 FINAL 4-MODE EVALUATION")
        test_z = tf.random.normal([1000, self.latent_dim])
        test_generated = self.generator.generate(test_z)
        final_coverage = self.calculate_mode_coverage_fast(test_generated)
        
        print(f"Mode coverages: {[f'{cov:.3f}' for cov in final_coverage['mode_coverages']]}")
        print(f"Balanced coverage: {final_coverage['balanced_coverage']:.3f}")
        print(f"Active modes: {final_coverage['total_modes_covered']}/{self.n_modes}")
        
        # Save minimal results
        self.save_fast_results(save_dir, total_time, final_coverage)
        
        return self.training_history
    
    def save_fast_results(self, save_dir: str, total_time: float, final_coverage: dict):
        """Save minimal training results efficiently."""
        results = {
            'training_summary': {
                'total_time_seconds': total_time,
                'total_time_minutes': total_time / 60,
                'average_epoch_time': np.mean(self.training_history['epoch_times']),
                'average_step_time': np.mean(self.training_history['step_times']),
                'total_epochs': len(self.training_history['generator_loss']),
                'total_steps': len(self.training_history['step_times'])
            },
            'final_performance': {
                'final_generator_loss': self.training_history['generator_loss'][-1],
                'final_discriminator_loss': self.training_history['discriminator_loss'][-1],
                'final_coverage': final_coverage
            },
            'configuration': {
                'latent_dim': self.latent_dim,
                'data_dim': self.data_dim,
                'n_modes': self.n_modes,
                'mode_centers': self.mode_centers,
                'mode_std': self.mode_std
            },
            'training_history': self.training_history
        }
        
        results_path = os.path.join(save_dir, "fast_training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Fast results saved to: {results_path}")


def main():
    """Main function for high-performance training."""
    parser = argparse.ArgumentParser(description="Fast 4-Mode Coordinate Quantum GAN Training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--save-dir", type=str, default="results/fast_4mode_training", help="Save directory")
    
    args = parser.parse_args()
    
    print("🚀 FAST 4-MODE COORDINATE QUANTUM GAN")
    print("=" * 60)
    print("High-performance training with minimal overhead")
    print("Optimizations:")
    print("  ✅ No real-time visualizations")
    print("  ✅ Minimal monitoring overhead")
    print("  ✅ 4-mode diverse data generation")
    print("  ✅ Optimized batch processing")
    print("  ✅ Reduced I/O operations")
    print("=" * 60)
    
    # Create fast trainer
    trainer = FastCoordinateGANTrainer()
    
    # Train with high performance
    start_time = time.time()
    results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
    total_time = time.time() - start_time
    
    # Print performance summary
    final_coverage = results['mode_coverage'][-1] if results['mode_coverage'] else {}
    print(f"\n🏆 PERFORMANCE SUMMARY:")
    print("=" * 60)
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Final balanced coverage: {final_coverage.get('balanced_coverage', 0):.3f}")
    print(f"Performance: {len(results['generator_loss']) * 15 / total_time:.1f} steps/second")
    
    if final_coverage.get('balanced_coverage', 0) > 0.6:
        print("✅ SUCCESS: Achieved good 4-mode coverage!")
    else:
        print("⚠️  Room for improvement in mode coverage")


if __name__ == "__main__":
    main()
