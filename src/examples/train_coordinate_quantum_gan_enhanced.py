"""
Enhanced Coordinate Quantum GAN Training with Comprehensive Diagnostics

This enhanced training script provides:
- Quantum circuit visualization
- Detailed loss component analysis
- Epoch-by-epoch visualizations
- Coordinate generation diagnostics
- Comprehensive results saving to results/training/
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
from typing import Dict, Any, List, Tuple

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

# Suppress warnings
suppress_all_quantum_warnings()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedCoordinateGANTrainer:
    """Enhanced trainer with comprehensive diagnostics."""
    
    def __init__(self, 
                 latent_dim: int = 6,
                 data_dim: int = 2,
                 n_modes: int = 4,
                 mode1_center: tuple = (-1.5, -1.5),
                 mode2_center: tuple = (1.5, 1.5),
                 mode_std: float = 0.3,
                 results_dir: str = "results/training"):
        """Initialize enhanced trainer."""
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.n_modes = n_modes
        self.mode1_center = mode1_center
        self.mode2_center = mode2_center
        self.mode_std = mode_std
        self.results_dir = results_dir
        
        # Create results subdirectories
        self.circuit_viz_dir = os.path.join(results_dir, "circuit_visualization")
        self.loss_components_dir = os.path.join(results_dir, "loss_components")
        self.epoch_viz_dir = os.path.join(results_dir, "epoch_visualizations")
        self.coord_diag_dir = os.path.join(results_dir, "coordinate_diagnostics")
        self.training_logs_dir = os.path.join(results_dir, "training_logs")
        
        for dir_path in [self.circuit_viz_dir, self.loss_components_dir, 
                        self.epoch_viz_dir, self.coord_diag_dir, self.training_logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Create coordinate generator
        self.generator = CoordinateQuantumGenerator(
            latent_dim=latent_dim,
            output_dim=data_dim,
            n_modes=n_modes,
            layers=2,
            cutoff_dim=6,
            clustering_method='kmeans',
            coordinate_names=['X', 'Y']
        )
        
        # Create discriminator
        self.discriminator = PureSFDiscriminator(
            input_dim=data_dim,
            n_modes=n_modes,
            layers=3,
            cutoff_dim=6
        )
        
        # Use existing Wasserstein loss
        self.loss_function = QuantumWassersteinLoss(
            lambda_gp=10.0,
            lambda_entropy=0.1,  # Reduced from 1.0 to help with large losses
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
        
        # Enhanced training history
        self.training_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'mode_coverage': [],
            'loss_components': {
                'w_distance': [],
                'gradient_penalty': [],
                'entropy_bonus': [],
                'trace_penalty': [],
                'norm_penalty': []
            },
            'coordinate_diagnostics': {
                'quantum_param_ranges': [],
                'measurement_ranges': [],
                'decoder_output_ranges': [],
                'final_output_ranges': []
            }
        }
        
        logger.info("Enhanced CoordinateGANTrainer initialized")
        logger.info(f"  Generator params: {len(self.generator.trainable_variables)}")
        logger.info(f"  Discriminator params: {len(self.discriminator.trainable_variables)}")
        logger.info(f"  Results directory: {results_dir}")
    
    def visualize_quantum_circuit(self):
        """Visualize the quantum circuit and save to results."""
        print("\n🔬 QUANTUM CIRCUIT ANALYSIS")
        print("=" * 60)
        
        # Create circuit visualizer
        visualizer = QuantumCircuitVisualizer(self.generator.quantum_circuit)
        
        # Save circuit diagram to file
        circuit_file = os.path.join(self.circuit_viz_dir, "circuit_diagram.txt")
        
        # Redirect print output to file
        import io
        from contextlib import redirect_stdout
        
        with open(circuit_file, 'w') as f:
            with redirect_stdout(f):
                visualizer.print_circuit_diagram(show_values=True)
                print("\n" + "="*80)
                visualizer.print_compact_circuit()
                print("\n" + "="*80)
                visualizer.print_parameter_list(show_values=True)
        
        # Also print to console
        visualizer.print_compact_circuit()
        
        # Export parameter info as JSON
        param_info = visualizer.export_parameter_info()
        param_file = os.path.join(self.circuit_viz_dir, "circuit_parameters.json")
        with open(param_file, 'w') as f:
            json.dump(param_info, f, indent=2)
        
        print(f"✅ Circuit visualization saved to: {circuit_file}")
        print(f"✅ Circuit parameters saved to: {param_file}")
        
        return param_info
    
    def analyze_target_data(self, n_samples: int = 1000):
        """Analyze target data and set up generator."""
        print("\n🎯 TARGET DATA ANALYSIS")
        print("=" * 60)
        
        # Generate target data
        self.data_generator.batch_size = n_samples
        target_data = self.data_generator.generate_batch().numpy()
        
        # Analyze with coordinate generator
        analysis_results = self.generator.analyze_target_data(target_data)
        
        # Save analysis results
        analysis_file = os.path.join(self.coord_diag_dir, "target_data_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump({
                'target_data_stats': {
                    'shape': target_data.shape,
                    'mean': target_data.mean(axis=0).tolist(),
                    'std': target_data.std(axis=0).tolist(),
                    'min': target_data.min(axis=0).tolist(),
                    'max': target_data.max(axis=0).tolist()
                },
                'cluster_analysis': analysis_results
            }, f, indent=2)
        
        print(f"✅ Target data analysis saved to: {analysis_file}")
        return analysis_results
    
    def discriminator_train_step(self, real_data: tf.Tensor, z: tf.Tensor):
        """Enhanced discriminator training step with diagnostics."""
        with tf.GradientTape() as disc_tape:
            # Generate fake data
            fake_data = self.generator.generate(z)
            
            # Compute loss with detailed components
            d_loss, g_loss, metrics = self.loss_function(
                real_data, fake_data, self.generator, self.discriminator
            )
        
        # Apply gradients
        disc_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )
        
        return d_loss, metrics, fake_data
    
    def generator_train_step(self, z: tf.Tensor):
        """Enhanced generator training step with diagnostics."""
        with tf.GradientTape() as gen_tape:
            # Generate fake data
            fake_data = self.generator.generate(z)
            
            # Dummy real data for loss computation
            dummy_real = tf.zeros_like(fake_data)
            
            # Compute loss with detailed components
            d_loss, g_loss, metrics = self.loss_function(
                dummy_real, fake_data, self.generator, self.discriminator
            )
        
        # Apply gradients
        gen_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        
        return g_loss, metrics, fake_data
    
    def analyze_coordinate_generation(self, z: tf.Tensor) -> Dict[str, Any]:
        """Analyze the coordinate generation pipeline."""
        # Step 1: Encode latent to quantum parameters
        quantum_params = self.generator.input_encoder(z)
        
        # Step 2: Extract measurements (simplified version)
        displacements = tf.reshape(quantum_params, [tf.shape(z)[0], self.n_modes, 2])
        x_quadratures = displacements[:, :, 0]
        quantum_noise = tf.random.normal(tf.shape(x_quadratures), stddev=0.1)
        measurements = x_quadratures + quantum_noise
        
        # Step 3: Apply coordinate decoders
        coordinate_outputs = {}
        for coord_name, decoder_info in self.generator.coordinate_decoders.items():
            if decoder_info['n_modes'] > 0:
                mode_indices = [m['mode_idx'] for m in decoder_info['modes']]
                coord_measurements = tf.gather(measurements, mode_indices, axis=1)
                coord_output = decoder_info['decoder'](coord_measurements)
                coord_output = tf.squeeze(coord_output)
                if len(coord_output.shape) > 1:
                    coord_output = coord_output[:, 0]
                coordinate_outputs[coord_name] = coord_output
        
        # Step 4: Final output
        final_output = self.generator.generate(z)
        
        # Analyze ranges
        diagnostics = {
            'quantum_param_ranges': {
                'min': float(quantum_params.numpy().min()),
                'max': float(quantum_params.numpy().max()),
                'mean': float(quantum_params.numpy().mean()),
                'std': float(quantum_params.numpy().std())
            },
            'measurement_ranges': {
                'min': float(measurements.numpy().min()),
                'max': float(measurements.numpy().max()),
                'mean': float(measurements.numpy().mean()),
                'std': float(measurements.numpy().std())
            },
            'decoder_output_ranges': {},
            'final_output_ranges': {
                'min': final_output.numpy().min(axis=0).tolist(),
                'max': final_output.numpy().max(axis=0).tolist(),
                'mean': final_output.numpy().mean(axis=0).tolist(),
                'std': final_output.numpy().std(axis=0).tolist()
            }
        }
        
        # Analyze decoder outputs
        for coord_name, coord_output in coordinate_outputs.items():
            diagnostics['decoder_output_ranges'][coord_name] = {
                'min': float(coord_output.numpy().min()),
                'max': float(coord_output.numpy().max()),
                'mean': float(coord_output.numpy().mean()),
                'std': float(coord_output.numpy().std())
            }
        
        return diagnostics
    
    def calculate_mode_coverage(self, generated_samples: tf.Tensor):
        """Calculate mode coverage metrics."""
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
    
    def create_epoch_visualization(self, epoch: int, real_samples: tf.Tensor, generated_samples: tf.Tensor):
        """Create comprehensive epoch visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Epoch {epoch}', fontsize=16, fontweight='bold')
        
        # Plot 1: Real data
        axes[0].scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.6, c='blue', s=30)
        axes[0].set_title('Real Data')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Generated data
        axes[1].scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6, c='red', s=30)
        axes[1].set_title('Generated Data')
        axes[1].set_xlabel('Feature 1')
        axes[1].set_ylabel('Feature 2')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Overlay comparison
        axes[2].scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.5, c='blue', s=30, label='Real')
        axes[2].scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, c='red', s=30, label='Generated')
        axes[2].set_title('Overlay Comparison')
        axes[2].set_xlabel('Feature 1')
        axes[2].set_ylabel('Feature 2')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(self.epoch_viz_dir, f"epoch_{epoch:03d}_comparison.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return viz_path
    
    def plot_loss_components(self):
        """Plot detailed loss component evolution."""
        if not self.training_history['loss_components']['w_distance']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Loss Component Analysis', fontsize=16, fontweight='bold')
        
        epochs = range(len(self.training_history['generator_loss']))
        
        # Plot 1: Generator and Discriminator losses
        axes[0, 0].plot(epochs, self.training_history['generator_loss'], label='Generator', color='red')
        axes[0, 0].plot(epochs, self.training_history['discriminator_loss'], label='Discriminator', color='blue')
        axes[0, 0].set_title('Overall Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Wasserstein distance
        axes[0, 1].plot(epochs, self.training_history['loss_components']['w_distance'], color='green')
        axes[0, 1].set_title('Wasserstein Distance')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Distance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Gradient penalty
        axes[0, 2].plot(epochs, self.training_history['loss_components']['gradient_penalty'], color='orange')
        axes[0, 2].set_title('Gradient Penalty')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Penalty')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Entropy bonus
        axes[1, 0].plot(epochs, self.training_history['loss_components']['entropy_bonus'], color='purple')
        axes[1, 0].set_title('Entropy Bonus (Quantum Cost)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Bonus')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Mode coverage
        mode1_coverage = [mc['mode1_coverage'] for mc in self.training_history['mode_coverage']]
        mode2_coverage = [mc['mode2_coverage'] for mc in self.training_history['mode_coverage']]
        balanced_coverage = [mc['balanced_coverage'] for mc in self.training_history['mode_coverage']]
        
        axes[1, 1].plot(epochs, mode1_coverage, label='Mode 1', color='red')
        axes[1, 1].plot(epochs, mode2_coverage, label='Mode 2', color='blue')
        axes[1, 1].plot(epochs, balanced_coverage, label='Balanced', color='green', linewidth=2)
        axes[1, 1].set_title('Mode Coverage')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Coverage')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Coordinate diagnostics
        if self.training_history['coordinate_diagnostics']['final_output_ranges']:
            final_ranges = self.training_history['coordinate_diagnostics']['final_output_ranges']
            x_means = [fr['mean'][0] for fr in final_ranges]
            y_means = [fr['mean'][1] for fr in final_ranges]
            x_stds = [fr['std'][0] for fr in final_ranges]
            y_stds = [fr['std'][1] for fr in final_ranges]
            
            axes[1, 2].plot(epochs, x_means, label='X mean', color='red')
            axes[1, 2].plot(epochs, y_means, label='Y mean', color='blue')
            axes[1, 2].fill_between(epochs, np.array(x_means) - np.array(x_stds), 
                                   np.array(x_means) + np.array(x_stds), alpha=0.3, color='red')
            axes[1, 2].fill_between(epochs, np.array(y_means) - np.array(y_stds), 
                                   np.array(y_means) + np.array(y_stds), alpha=0.3, color='blue')
            axes[1, 2].set_title('Output Coordinate Statistics')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Value')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save loss analysis
        loss_path = os.path.join(self.loss_components_dir, "loss_component_analysis.png")
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return loss_path
    
    def train(self, epochs: int = 10, batch_size: int = 32):
        """Enhanced training loop with comprehensive diagnostics."""
        print(f"\n🚀 ENHANCED COORDINATE GAN TRAINING")
        print("=" * 60)
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Step 1: Visualize quantum circuit
        circuit_info = self.visualize_quantum_circuit()
        
        # Step 2: Analyze target data
        target_analysis = self.analyze_target_data()
        
        # Update data generator batch size
        self.data_generator.batch_size = batch_size
        
        print(f"\n📊 TRAINING PROGRESS")
        print("=" * 60)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            epoch_gen_losses = []
            epoch_disc_losses = []
            epoch_loss_components = {
                'w_distance': [],
                'gradient_penalty': [],
                'entropy_bonus': [],
                'trace_penalty': [],
                'norm_penalty': []
            }
            
            # Training steps per epoch
            steps_per_epoch = 50
            
            for step in range(steps_per_epoch):
                # Get real data
                real_data = self.data_generator.generate_batch()
                
                # Train discriminator (5 steps)
                disc_losses = []
                for _ in range(5):
                    z = tf.random.normal([batch_size, self.latent_dim])
                    disc_loss, disc_metrics, _ = self.discriminator_train_step(real_data, z)
                    disc_losses.append(float(disc_loss))
                    
                    # Store loss components
                    for key in epoch_loss_components.keys():
                        if key in disc_metrics:
                            epoch_loss_components[key].append(float(disc_metrics[key]))
                
                # Train generator (1 step)
                z = tf.random.normal([batch_size, self.latent_dim])
                gen_loss, gen_metrics, fake_data = self.generator_train_step(z)
                
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
            
            # Analyze coordinate generation
            coord_diagnostics = self.analyze_coordinate_generation(test_z[:100])  # Smaller sample for diagnostics
            
            # Store history
            self.training_history['generator_loss'].append(avg_gen_loss)
            self.training_history['discriminator_loss'].append(avg_disc_loss)
            self.training_history['mode_coverage'].append(coverage_metrics)
            self.training_history['coordinate_diagnostics']['final_output_ranges'].append(coord_diagnostics['final_output_ranges'])
            
            # Store loss components
            for key in epoch_loss_components.keys():
                if epoch_loss_components[key]:
                    self.training_history['loss_components'][key].append(np.mean(epoch_loss_components[key]))
                else:
                    self.training_history['loss_components'][key].append(0.0)
            
            print(f"Epoch {epoch + 1} complete:")
            print(f"  G_loss: {avg_gen_loss:.4f}, D_loss: {avg_disc_loss:.4f}")
            print(f"  Mode1 coverage: {coverage_metrics['mode1_coverage']:.3f}")
            print(f"  Mode2 coverage: {coverage_metrics['mode2_coverage']:.3f}")
            print(f"  Balanced coverage: {coverage_metrics['balanced_coverage']:.3f}")
            print(f"  Output range: X=[{coord_diagnostics['final_output_ranges']['min'][0]:.3f}, {coord_diagnostics['final_output_ranges']['max'][0]:.3f}], Y=[{coord_diagnostics['final_output_ranges']['min'][1]:.3f}, {coord_diagnostics['final_output_ranges']['max'][1]:.3f}]")
            
            # Create epoch visualization
            real_samples = self.data_generator.generate_batch()
            viz_path = self.create_epoch_visualization(epoch + 1, real_samples, test_generated)
            print(f"  Visualization saved: {viz_path}")
        
        # Final analysis
        print(f"\n📈 FINAL ANALYSIS")
        print("=" * 60)
        
        # Plot loss components
        loss_plot_path = self.plot_loss_components()
        print(f"✅ Loss analysis saved: {loss_plot_path}")
        
        # Save comprehensive results
        self.save_comprehensive_results()
        
        return self.training_history
    
    def save_comprehensive_results(self):
        """Save comprehensive training results."""
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
            },
            'final_diagnostics': self.training_history['coordinate_diagnostics']
        }
        
        results_path = os.path.join(self.training_logs_dir, "comprehensive_training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Comprehensive results saved to: {results_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced Coordinate Quantum GAN Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--results-dir", type=str, default="results/training", help="Results directory")
    
    args = parser.parse_args()
    
    print("🎯 ENHANCED COORDINATE QUANTUM GAN TRAINING")
    print("=" * 60)
    print("Comprehensive diagnostics and visualization")
    print("=" * 60)
    
    # Create enhanced trainer
    trainer = EnhancedCoordinateGANTrainer(results_dir=args.results_dir)
    
    # Train with enhanced diagnostics
    results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Print final results
    final_coverage = results['mode_coverage'][-1] if results['mode_coverage'] else {}
    print(f"\n🎉 FINAL RESULTS:")
    print(f"Mode 1 coverage: {final_coverage.get('mode1_coverage', 0):.3f}")
    print(f"Mode 2 coverage: {final_coverage.get('mode2_coverage', 0):.3f}")
    print(f"Balanced coverage: {final_coverage.get('balanced_coverage', 0):.3f}")
    
    if final_coverage.get('balanced_coverage', 0) > 0.5:
        print("✅ SUCCESS: Achieved balanced mode coverage!")
    else:
        print("❌ Mode collapse still present - needs further work")
    
    print(f"\n📁 All results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
