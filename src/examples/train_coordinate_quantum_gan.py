"""
Basic Coordinate Quantum GAN Training

Simple training script to test the coordinate-wise quantum generator
with existing discriminator. No fancy features - just basic functionality.
"""

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import sys
import os
from datetime import datetime
import json
from PIL import Image
import imageio

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
from src.utils.visualization import plot_results
from src.utils.quantum_circuit_visualizer import QuantumCircuitVisualizer
from src.utils.enhanced_quantum_circuit_visualizer import create_enhanced_circuit_visualization

# Suppress warnings
suppress_all_quantum_warnings()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoordinateGANTrainer:
    """Basic trainer for coordinate quantum GAN."""
    
    def __init__(self, 
                 latent_dim: int = 2,
                 data_dim: int = 2,
                 n_modes: int = 2,
                 mode1_center: tuple = (-1.5, -1.5),
                 mode2_center: tuple = (1.5, 1.5),
                 mode_std: float = 0.3):
        """Initialize trainer."""
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.n_modes = n_modes
        self.mode1_center = mode1_center
        self.mode2_center = mode2_center
        self.mode_std = mode_std
        
        # Create coordinate generator
        self.generator = CoordinateQuantumGenerator(
            latent_dim=latent_dim,
            output_dim=data_dim,
            n_modes=n_modes,
            layers=1,
            cutoff_dim=6,
            clustering_method='kmeans',
            coordinate_names=['X', 'Y']
        )
        
        # Create discriminator (existing one)
        self.discriminator = PureSFDiscriminator(
            input_dim=data_dim,
            n_modes=n_modes,
            layers=1,
            cutoff_dim=6
        )
        
        # Use existing Wasserstein loss
        self.loss_function = QuantumWassersteinLoss(
            lambda_gp=10.0,
            lambda_entropy=1.0,
            lambda_physics=1.0
        )
        
        # Optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
        # Data generator
        self.data_generator = BimodalDataGenerator(
            batch_size=32,
            n_features=data_dim,
            mode1_center=mode1_center,
            mode2_center=mode2_center,
            mode_std=mode_std,
            seed=42
        )
        
        # Training history
        self.training_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'mode_coverage': []
        }
        
        logger.info("CoordinateGANTrainer initialized")
        logger.info(f"  Generator params: {len(self.generator.trainable_variables)}")
        logger.info(f"  Discriminator params: {len(self.discriminator.trainable_variables)}")
    
    def visualize_quantum_circuit(self, save_dir: str):
        """Visualize the quantum circuit structure with enhanced visual diagram."""
        print("\n QUANTUM CIRCUIT ANALYSIS")
        print("=" * 60)
        
        # Create basic circuit visualizer for console output
        visualizer = QuantumCircuitVisualizer(self.generator.quantum_circuit)
        visualizer.print_compact_circuit()
        
        # Create enhanced visual circuit diagram
        print("\n🎨 Creating enhanced visual circuit diagram...")
        try:
            create_enhanced_circuit_visualization(
                quantum_circuit=self.generator.quantum_circuit,
                coordinate_generator=self.generator,
                save_dir=os.path.join(save_dir, "circuit_visualization"),
                training_history=None  # No training history yet
            )
            print("✅ Enhanced circuit visualization created!")
        except Exception as e:
            print(f"⚠️ Enhanced visualization failed: {e}")
            print("Continuing with basic visualization...")
        
        print("✅ Quantum circuit visualization complete")
        return visualizer
    
    def analyze_target_data(self, n_samples: int = 50):
        """Analyze target data and set up generator."""
        print("Analyzing target data...")
        
        # Generate target data
        self.data_generator.batch_size = n_samples
        target_data = self.data_generator.generate_batch().numpy()
        
        # Analyze with coordinate generator
        analysis_results = self.generator.analyze_target_data(target_data)
        
        print("Target data analysis complete")
        return analysis_results
    
    def discriminator_train_step(self, real_data: tf.Tensor, z: tf.Tensor):
        """Train discriminator with NaN protection and gradient clipping."""
        with tf.GradientTape() as disc_tape:
            # Generate fake data
            fake_data = self.generator.generate(z)
            
            # Compute loss
            d_loss, g_loss, metrics = self.loss_function(
                real_data, fake_data, self.generator, self.discriminator
            )
            
            # NaN protection
            if tf.math.is_nan(d_loss):
                logger.warning("NaN detected in discriminator loss, using fallback")
                d_loss = tf.constant(1.0)  # Fallback loss
        
        # Apply gradients with clipping
        disc_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        
        # Check for None gradients and clip
        if disc_gradients is not None:
            # Filter out None gradients and clip
            valid_gradients = []
            valid_variables = []
            for grad, var in zip(disc_gradients, self.discriminator.trainable_variables):
                if grad is not None and not tf.reduce_any(tf.math.is_nan(grad)):
                    # Clip gradients to prevent explosion
                    clipped_grad = tf.clip_by_value(grad, -1.0, 1.0)
                    valid_gradients.append(clipped_grad)
                    valid_variables.append(var)
            
            if valid_gradients:
                self.discriminator_optimizer.apply_gradients(
                    zip(valid_gradients, valid_variables)
                )
        
        return d_loss, metrics
    
    def generator_train_step(self, z: tf.Tensor):
        """Train generator with NaN protection and gradient clipping."""
        with tf.GradientTape() as gen_tape:
            # Generate fake data
            fake_data = self.generator.generate(z)
            
            # Dummy real data for loss computation
            dummy_real = tf.zeros_like(fake_data)
            
            # Compute loss
            d_loss, g_loss, metrics = self.loss_function(
                dummy_real, fake_data, self.generator, self.discriminator
            )
            
            # NaN protection
            if tf.math.is_nan(g_loss):
                logger.warning("NaN detected in generator loss, using fallback")
                g_loss = tf.constant(1.0)  # Fallback loss
        
        # Apply gradients with clipping
        gen_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        
        # Check for None gradients and clip
        if gen_gradients is not None:
            # Filter out None gradients and clip
            valid_gradients = []
            valid_variables = []
            for grad, var in zip(gen_gradients, self.generator.trainable_variables):
                if grad is not None and not tf.reduce_any(tf.math.is_nan(grad)):
                    # Clip gradients to prevent explosion
                    clipped_grad = tf.clip_by_value(grad, -1.0, 1.0)
                    valid_gradients.append(clipped_grad)
                    valid_variables.append(var)
            
            if valid_gradients:
                self.generator_optimizer.apply_gradients(
                    zip(valid_gradients, valid_variables)
                )
        
        return g_loss, metrics
    
    def calculate_mode_coverage(self, generated_samples: tf.Tensor):
        """Calculate simple mode coverage metrics."""
        gen_np = generated_samples.numpy()
        
        # Calculate distances to mode centers
        mode1_center_np = np.array(self.mode1_center)
        mode2_center_np = np.array(self.mode2_center)
        
        distances_to_mode1 = np.linalg.norm(gen_np - mode1_center_np, axis=1)
        distances_to_mode2 = np.linalg.norm(gen_np - mode2_center_np, axis=1)
        
        # Assign to closest mode
        mode1_assignments = distances_to_mode1 < distances_to_mode2
        mode2_assignments = ~mode1_assignments
        
        mode1_coverage = np.sum(mode1_assignments) / len(gen_np)
        mode2_coverage = np.sum(mode2_assignments) / len(gen_np)
        
        return {
            'mode1_coverage': float(mode1_coverage),
            'mode2_coverage': float(mode2_coverage),
            'balanced_coverage': float(min(mode1_coverage, mode2_coverage) / max(mode1_coverage, mode2_coverage)) if max(mode1_coverage, mode2_coverage) > 0 else 0.0
        }
    
    def create_initial_comparison_visualization(self, save_dir: str):
        """Create initial comparison visualization like the user's example."""
        print("\n📊 CREATING INITIAL DATA COMPARISON")
        print("=" * 60)
        
        # Generate real and initial fake data
        self.data_generator.batch_size = 500
        real_samples = self.data_generator.generate_batch().numpy()
        
        # Generate initial samples from untrained generator
        test_z = tf.random.normal([500, self.latent_dim])
        generated_samples = self.generator.generate(test_z).numpy()
        
        # Create the three-panel comparison plot like the user's example
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Initial Data Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Real Data
        axes[0].scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.6, c='blue', s=30)
        axes[0].set_title('Real Data', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(-3, 3)
        axes[0].set_ylim(-3, 3)
        
        # Plot 2: Generated Data
        axes[1].scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6, c='red', s=30)
        axes[1].set_title('Generated Data', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Feature 1')
        axes[1].set_ylabel('Feature 2')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(-3, 3)
        axes[1].set_ylim(-3, 3)
        
        # Plot 3: Overlay Comparison
        axes[2].scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.5, c='blue', s=30, label='Real')
        axes[2].scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, c='red', s=30, label='Generated')
        axes[2].set_title('Overlay Comparison', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Feature 1')
        axes[2].set_ylabel('Feature 2')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(-3, 3)
        axes[2].set_ylim(-3, 3)
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = os.path.join(save_dir, "epoch_visualizations", "initial_comparison.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Initial comparison visualization saved to: {viz_path}")
        
        # Calculate and print initial metrics
        coverage_metrics = self.calculate_mode_coverage(tf.constant(generated_samples))
        print(f"📈 Initial metrics:")
        print(f"   Mode 1 coverage: {coverage_metrics['mode1_coverage']:.3f}")
        print(f"   Mode 2 coverage: {coverage_metrics['mode2_coverage']:.3f}")
        print(f"   Balanced coverage: {coverage_metrics['balanced_coverage']:.3f}")
        
        return viz_path
    
    def train(self, epochs: int = 10, batch_size: int = 32, save_dir: str = "results/training"):
        """Basic training loop."""
        print(f"Starting coordinate GAN training...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Create results directory structure
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "circuit_visualization"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "epoch_visualizations"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "training_logs"), exist_ok=True)
        
        # Step 1: Visualize quantum circuit
        self.visualize_quantum_circuit(save_dir)
        
        # Step 2: Analyze target data
        self.analyze_target_data()
        
        # Step 3: Create initial comparison visualization
        self.create_initial_comparison_visualization(save_dir)
        
        # Update data generator batch size
        self.data_generator.batch_size = batch_size
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            epoch_gen_losses = []
            epoch_disc_losses = []
            
            # Training steps per epoch
            steps_per_epoch = 5
            
            for step in range(steps_per_epoch):
                # Get real data
                real_data = self.data_generator.generate_batch()
                
                # Train discriminator (5 steps)
                disc_losses = []
                for _ in range(5):
                    z = tf.random.normal([batch_size, self.latent_dim])
                    disc_loss, _ = self.discriminator_train_step(real_data, z)
                    disc_losses.append(float(disc_loss))
                
                # Train generator (1 step)
                z = tf.random.normal([batch_size, self.latent_dim])
                gen_loss, _ = self.generator_train_step(z)
                
                epoch_gen_losses.append(float(gen_loss))
                epoch_disc_losses.append(np.mean(disc_losses))
                
                if step % 10 == 0:
                    print(f"  Step {step}: G_loss={gen_loss:.4f}, D_loss={np.mean(disc_losses):.4f}")
                
                # Add debugging for the first few steps
                if step < 3:
                    print(f"    DEBUG: Step {step} completed successfully")
            
            # Calculate epoch averages
            avg_gen_loss = np.mean(epoch_gen_losses)
            avg_disc_loss = np.mean(epoch_disc_losses)
            
            # Calculate mode coverage
            test_z = tf.random.normal([500, self.latent_dim])
            test_generated = self.generator.generate(test_z)
            coverage_metrics = self.calculate_mode_coverage(test_generated)
            
            # Store history
            self.training_history['generator_loss'].append(avg_gen_loss)
            self.training_history['discriminator_loss'].append(avg_disc_loss)
            self.training_history['mode_coverage'].append(coverage_metrics)
            
            print(f"Epoch {epoch + 1} complete:")
            print(f"  G_loss: {avg_gen_loss:.4f}, D_loss: {avg_disc_loss:.4f}")
            print(f"  Mode1 coverage: {coverage_metrics['mode1_coverage']:.3f}")
            print(f"  Mode2 coverage: {coverage_metrics['mode2_coverage']:.3f}")
            print(f"  Balanced coverage: {coverage_metrics['balanced_coverage']:.3f}")
            
            # Save visualization every few epochs
            if (epoch + 1) % 2 == 0:
                self.create_visualization(epoch + 1, save_dir)
        
        # Final results
        print(f"\nTraining complete!")
        
        # Create enhanced circuit visualization
        print("\n🎨 CREATING ENHANCED CIRCUIT VISUALIZATION")
        print("=" * 60)
        try:
            create_enhanced_circuit_visualization(
                quantum_circuit=self.generator.quantum_circuit,
                coordinate_generator=self.generator,
                save_dir=os.path.join(save_dir, "circuit_visualization"),
                training_history=self.training_history
            )
        except Exception as e:
            print(f"⚠️ Enhanced visualization failed: {e}")
            print("Continuing with basic results...")
        
        self.save_results(save_dir)
        
        return self.training_history
    
    def create_visualization(self, epoch: int, save_dir: str):
        """Create simple visualization."""
        # Generate samples
        test_z = tf.random.normal([500, self.latent_dim])
        generated_samples = self.generator.generate(test_z)
        
        # Generate real samples
        self.data_generator.batch_size = 500
        real_samples = self.data_generator.generate_batch()
        
        # Create plot
        viz_path = os.path.join(save_dir, f"epoch_{epoch:03d}_comparison.png")
        plot_results(real_samples, generated_samples, epoch=epoch, save_path=viz_path)
        
        print(f"  Visualization saved: {viz_path}")
    
    def save_results(self, save_dir: str):
        """Save training results."""
        results = {
            'training_history': self.training_history,
            'final_coverage': self.training_history['mode_coverage'][-1] if self.training_history['mode_coverage'] else {},
            'configuration': {
                'latent_dim': self.latent_dim,
                'data_dim': self.data_dim,
                'n_modes': self.n_modes,
                'mode1_center': self.mode1_center,
                'mode2_center': self.mode2_center,
                'mode_std': self.mode_std
            }
        }
        
        results_path = os.path.join(save_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Coordinate Quantum GAN")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--save-dir", type=str, default="coordinate_gan_results", help="Save directory")
    
    args = parser.parse_args()
    
    print("COORDINATE QUANTUM GAN TRAINING")
    print("=" * 50)
    print("Basic training script - testing coordinate generator")
    print("=" * 50)
    
    # Create trainer
    trainer = CoordinateGANTrainer()
    
    # Train
    results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
    
    # Print final results
    final_coverage = results['mode_coverage'][-1] if results['mode_coverage'] else {}
    print(f"\nFINAL RESULTS:")
    print(f"Mode 1 coverage: {final_coverage.get('mode1_coverage', 0):.3f}")
    print(f"Mode 2 coverage: {final_coverage.get('mode2_coverage', 0):.3f}")
    print(f"Balanced coverage: {final_coverage.get('balanced_coverage', 0):.3f}")
    
    if final_coverage.get('balanced_coverage', 0) > 0.5:
        print("SUCCESS: Achieved balanced mode coverage!")
    else:
        print("Mode collapse still present - needs further work")


if __name__ == "__main__":
    main()
