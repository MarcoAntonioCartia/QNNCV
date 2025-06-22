"""
Simplified Bimodal Constellation QGAN Training

This script demonstrates the constellation QGAN's ability to handle challenging
bimodal distributions using a simplified training approach that avoids threading issues.

Features:
- Generator-only training to avoid discriminator threading issues
- Progressive visualization during training
- Quantitative mode coverage analysis
- Real-time comparison plots (Real | Generated | Overlay)
"""

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import sys
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.warning_suppression import suppress_all_quantum_warnings
from src.models.generators.optimal_constellation_generator import OptimalConstellationGenerator
from src.training.data_generators import BimodalDataGenerator
from src.utils.visualization import plot_results

# Suppress warnings for clean output
suppress_all_quantum_warnings()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleBimodalConstellationTrainer:
    """
    Simplified trainer for constellation QGAN on bimodal data.
    
    Uses generator-only training to avoid threading issues while still
    demonstrating spatial separation capabilities.
    """
    
    def __init__(self, 
                 latent_dim: int = 6,
                 data_dim: int = 2,
                 n_modes: int = 4,
                 mode1_center: Tuple[float, float] = (-1.5, -1.5),
                 mode2_center: Tuple[float, float] = (1.5, 1.5),
                 mode_std: float = 0.3,
                 enable_state_validation: bool = True):
        """
        Initialize simplified bimodal constellation trainer.
        
        Args:
            latent_dim: Latent space dimension
            data_dim: Data space dimension  
            n_modes: Number of quantum modes (4 for optimal quadrant structure)
            mode1_center: Center of first bimodal cluster
            mode2_center: Center of second bimodal cluster
            mode_std: Standard deviation of each mode
            enable_state_validation: Enable constellation state validation
        """
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.n_modes = n_modes
        self.mode1_center = mode1_center
        self.mode2_center = mode2_center
        self.mode_std = mode_std
        
        # 🏆 Create OPTIMAL constellation generator with FIX 1 + FIX 2 parameters
        self.generator = OptimalConstellationGenerator(
            latent_dim=latent_dim,
            output_dim=data_dim,
            n_modes=n_modes,
            layers=2,
            cutoff_dim=6,
            # 🏆 OPTIMAL PARAMETERS FROM FIX 2
            squeeze_r=1.5,           # 65x better compactness
            squeeze_angle=0.785,     # 45° optimal angle
            modulation_strength=0.3, # Conservative stable modulation
            separation_scale=2.0,    # Proven spatial separation
            enable_state_validation=enable_state_validation
        )
        
        # Optimizer for generator
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
        # Create bimodal data generator
        self.data_generator = BimodalDataGenerator(
            batch_size=32,  # Will be overridden during training
            n_features=data_dim,
            mode1_center=mode1_center,
            mode2_center=mode2_center,
            mode_std=mode_std,
            seed=42  # For reproducible results
        )
        
        # Training history
        self.training_history = {
            'generator_loss': [],
            'mode_coverage_metrics': [],
            'epoch_visualizations': []
        }
        
        logger.info("🏆 Simple Bimodal Constellation Trainer initialized")
        logger.info(f"   Generator parameters: {len(self.generator.trainable_variables)}")
        logger.info(f"   Bimodal centers: {mode1_center} and {mode2_center}")
        logger.info(f"   Mode std: {mode_std}")
    
    def generator_train_step(self, real_data: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """Single generator training step using reconstruction loss."""
        with tf.GradientTape() as gen_tape:
            # Generate fake samples
            fake_data = self.generator.generate(z)
            
            # Simple reconstruction loss - encourage diversity and coverage
            # Loss 1: Encourage samples to be near the real data distribution
            real_mean = tf.reduce_mean(real_data, axis=0)
            fake_mean = tf.reduce_mean(fake_data, axis=0)
            mean_loss = tf.reduce_mean(tf.square(real_mean - fake_mean))
            
            # Loss 2: Encourage variance to match real data
            real_var = tf.math.reduce_variance(real_data, axis=0)
            fake_var = tf.math.reduce_variance(fake_data, axis=0)
            var_loss = tf.reduce_mean(tf.square(real_var - fake_var))
            
            # Loss 3: Encourage bimodal structure by penalizing samples near center
            center_point = tf.constant([[0.0, 0.0]], dtype=tf.float32)
            distances_to_center = tf.norm(fake_data - center_point, axis=1)
            center_penalty = tf.reduce_mean(tf.exp(-distances_to_center))  # Penalize being near center
            
            # Combined loss
            gen_loss = mean_loss + var_loss + 0.1 * center_penalty
        
        # Apply generator gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        
        return gen_loss
    
    def calculate_mode_coverage_metrics(self, generated_samples: tf.Tensor) -> Dict[str, float]:
        """Calculate quantitative mode coverage metrics."""
        gen_np = generated_samples.numpy()
        
        # Calculate distances to each mode center
        mode1_center_np = np.array(self.mode1_center)
        mode2_center_np = np.array(self.mode2_center)
        
        distances_to_mode1 = np.linalg.norm(gen_np - mode1_center_np, axis=1)
        distances_to_mode2 = np.linalg.norm(gen_np - mode2_center_np, axis=1)
        
        # Assign samples to closest mode
        mode1_assignments = distances_to_mode1 < distances_to_mode2
        mode2_assignments = ~mode1_assignments
        
        # Calculate coverage metrics
        mode1_coverage = np.sum(mode1_assignments) / len(gen_np)
        mode2_coverage = np.sum(mode2_assignments) / len(gen_np)
        
        # Calculate average distances within each mode
        if np.sum(mode1_assignments) > 0:
            avg_distance_mode1 = np.mean(distances_to_mode1[mode1_assignments])
        else:
            avg_distance_mode1 = float('inf')
            
        if np.sum(mode2_assignments) > 0:
            avg_distance_mode2 = np.mean(distances_to_mode2[mode2_assignments])
        else:
            avg_distance_mode2 = float('inf')
        
        # Calculate sample spread
        sample_variance = np.var(gen_np, axis=0)
        total_variance = np.sum(sample_variance)
        
        # Calculate distance from center (mode collapse indicator)
        center_distances = np.linalg.norm(gen_np, axis=1)
        avg_center_distance = np.mean(center_distances)
        
        return {
            'mode1_coverage': float(mode1_coverage),
            'mode2_coverage': float(mode2_coverage),
            'avg_distance_mode1': float(avg_distance_mode1),
            'avg_distance_mode2': float(avg_distance_mode2),
            'total_variance': float(total_variance),
            'avg_center_distance': float(avg_center_distance),
            'balanced_coverage': float(min(mode1_coverage, mode2_coverage) / max(mode1_coverage, mode2_coverage)) if max(mode1_coverage, mode2_coverage) > 0 else 0.0
        }
    
    def create_epoch_visualization(self, epoch: int, save_dir: str = "simple_bimodal_results"):
        """Create comprehensive visualization for current epoch."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate samples for visualization
        n_viz_samples = 500
        test_z = tf.random.normal([n_viz_samples, self.latent_dim])
        generated_samples = self.generator.generate(test_z)
        
        # Generate real data for comparison
        self.data_generator.batch_size = n_viz_samples
        real_samples = self.data_generator.generate_batch()
        
        # Create visualization using existing utility
        viz_path = os.path.join(save_dir, f"epoch_{epoch:03d}_comparison.png")
        plot_results(real_samples, generated_samples, epoch=epoch, save_path=viz_path)
        
        # Calculate and log metrics
        metrics = self.calculate_mode_coverage_metrics(generated_samples)
        
        print(f"\n📊 Epoch {epoch} Metrics:")
        print(f"   Mode 1 coverage: {metrics['mode1_coverage']:.3f}")
        print(f"   Mode 2 coverage: {metrics['mode2_coverage']:.3f}")
        print(f"   Balanced coverage: {metrics['balanced_coverage']:.3f}")
        print(f"   Total variance: {metrics['total_variance']:.3f}")
        print(f"   Avg center distance: {metrics['avg_center_distance']:.3f}")
        print(f"   Visualization saved: {viz_path}")
        
        return metrics, viz_path
    
    def train(self, 
              epochs: int = 10,
              batch_size: int = 32,
              steps_per_epoch: int = 50,
              validation_frequency: int = 2,
              save_dir: str = "simple_bimodal_results") -> Dict[str, Any]:
        """
        Simplified training with bimodal data evaluation.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            steps_per_epoch: Steps per epoch
            validation_frequency: Validation frequency (every N epochs)
            save_dir: Directory to save results
            
        Returns:
            Training results and metrics
        """
        print("🚀 SIMPLIFIED BIMODAL CONSTELLATION TRAINING")
        print("=" * 80)
        print(f"Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Validation frequency: every {validation_frequency} epochs")
        print(f"   Results directory: {save_dir}")
        print("=" * 80)
        
        # Update data generator batch size
        self.data_generator.batch_size = batch_size
        
        # Create results directory
        os.makedirs(save_dir, exist_ok=True)
        
        start_time = datetime.now()
        
        # Initial visualization
        print("\n🎨 Creating initial visualization...")
        initial_metrics, initial_viz = self.create_epoch_visualization(0, save_dir)
        self.training_history['mode_coverage_metrics'].append(initial_metrics)
        self.training_history['epoch_visualizations'].append(initial_viz)
        
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch + 1}/{epochs}")
            
            epoch_gen_losses = []
            
            # Training steps for this epoch
            for step in range(steps_per_epoch):
                # Get real data batch
                real_data = self.data_generator.generate_batch()
                
                # Generate latent noise
                z = tf.random.normal([batch_size, self.latent_dim])
                
                # Train generator
                gen_loss = self.generator_train_step(real_data, z)
                epoch_gen_losses.append(float(gen_loss))
                
                # Progress updates
                if step % 10 == 0:
                    print(f"   Step {step + 1}/{steps_per_epoch}: G_loss={gen_loss:.4f}")
            
            # Calculate epoch averages
            avg_gen_loss = np.mean(epoch_gen_losses)
            
            # Store training history
            self.training_history['generator_loss'].append(avg_gen_loss)
            
            print(f"Epoch {epoch + 1} complete: G_loss={avg_gen_loss:.4f}")
            
            # Periodic validation and visualization
            if (epoch + 1) % validation_frequency == 0:
                print(f"\n🔍 Creating validation visualization for epoch {epoch + 1}...")
                metrics, viz_path = self.create_epoch_visualization(epoch + 1, save_dir)
                self.training_history['mode_coverage_metrics'].append(metrics)
                self.training_history['epoch_visualizations'].append(viz_path)
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        print(f"\n🎉 Training completed in {training_duration:.1f} seconds")
        
        # Final comprehensive evaluation
        print("\n🔍 FINAL COMPREHENSIVE EVALUATION")
        final_metrics, final_viz = self.create_epoch_visualization(epochs, save_dir)
        
        # Generate large sample comparison
        print("\n📊 Generating large sample comparison...")
        large_sample_size = 1000
        test_z = tf.random.normal([large_sample_size, self.latent_dim])
        final_generated = self.generator.generate(test_z)
        
        self.data_generator.batch_size = large_sample_size
        final_real = self.data_generator.generate_batch()
        
        final_comparison_path = os.path.join(save_dir, "final_large_sample_comparison.png")
        plot_results(final_real, final_generated, epoch="Final", save_path=final_comparison_path)
        
        # Calculate final comprehensive metrics
        final_comprehensive_metrics = self.calculate_mode_coverage_metrics(final_generated)
        
        print(f"\n🏆 FINAL RESULTS:")
        print(f"   Mode 1 coverage: {final_comprehensive_metrics['mode1_coverage']:.3f}")
        print(f"   Mode 2 coverage: {final_comprehensive_metrics['mode2_coverage']:.3f}")
        print(f"   Balanced coverage: {final_comprehensive_metrics['balanced_coverage']:.3f}")
        print(f"   Total variance: {final_comprehensive_metrics['total_variance']:.3f}")
        print(f"   Avg center distance: {final_comprehensive_metrics['avg_center_distance']:.3f}")
        print(f"   Training duration: {training_duration:.1f}s")
        
        # Prepare results
        training_results = {
            'training_duration_seconds': training_duration,
            'epochs_completed': epochs,
            'final_metrics': final_comprehensive_metrics,
            'training_history': self.training_history,
            'configuration': {
                'batch_size': batch_size,
                'steps_per_epoch': steps_per_epoch,
                'validation_frequency': validation_frequency,
                'mode1_center': self.mode1_center,
                'mode2_center': self.mode2_center,
                'mode_std': self.mode_std
            },
            'visualizations': {
                'final_comparison': final_comparison_path,
                'epoch_progressions': self.training_history['epoch_visualizations']
            }
        }
        
        # Save results
        results_path = os.path.join(save_dir, "training_results.json")
        
        def convert_to_serializable(obj):
            """Convert numpy/tensorflow types to JSON serializable."""
            if isinstance(obj, (np.integer, np.floating, np.bool_, np.complexfloating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tf.Tensor):
                return obj.numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        json_results = convert_to_serializable(training_results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Complete results saved to {results_path}")
        
        return training_results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Simplified Bimodal Constellation QGAN Training"
    )
    
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--steps-per-epoch", type=int, default=50, help="Steps per epoch")
    parser.add_argument("--validation-frequency", type=int, default=2, help="Validation frequency (epochs)")
    parser.add_argument("--mode1-center", nargs=2, type=float, default=[-1.5, -1.5], help="Mode 1 center")
    parser.add_argument("--mode2-center", nargs=2, type=float, default=[1.5, 1.5], help="Mode 2 center")
    parser.add_argument("--mode-std", type=float, default=0.3, help="Mode standard deviation")
    parser.add_argument("--save-dir", type=str, default="simple_bimodal_results", help="Results directory")
    parser.add_argument("--enable-validation", action="store_true", help="Enable state validation")
    
    args = parser.parse_args()
    
    print("🏆 SIMPLIFIED BIMODAL CONSTELLATION QGAN TRAINING")
    print("=" * 80)
    print("Testing constellation QGAN's spatial separation without discriminator")
    print("Using FIX 1 (spatial separation) + FIX 2 (optimal squeezing)")
    print("=" * 80)
    
    # Create trainer
    trainer = SimpleBimodalConstellationTrainer(
        mode1_center=tuple(args.mode1_center),
        mode2_center=tuple(args.mode2_center),
        mode_std=args.mode_std,
        enable_state_validation=args.enable_validation
    )
    
    # Run simplified training
    results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        validation_frequency=args.validation_frequency,
        save_dir=args.save_dir
    )
    
    print("\n" + "=" * 80)
    print("🏆 SIMPLIFIED BIMODAL CONSTELLATION TRAINING COMPLETE")
    print("=" * 80)
    print(f"Check {args.save_dir}/ for comprehensive results and visualizations!")


if __name__ == "__main__":
    main()
