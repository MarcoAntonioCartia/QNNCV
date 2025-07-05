"""
Comprehensive Bimodal Constellation QGAN Training with Full Diagnostics

This script provides the complete diagnostic framework to understand exactly
where mode collapse occurs in the constellation QGAN architecture.

Features:
- Proper Wasserstein loss with entropy regularization
- Real-time quantum mode parameter tracking
- 8D measurement space visualization with PCA/t-SNE
- Per-mode contribution analysis to final output
- Constellation encoding verification (4-quadrant structure)
- Progressive diagnostic reports every epoch
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
from src.models.discriminators.pure_sf_discriminator import PureSFDiscriminator
from src.training.data_generators import BimodalDataGenerator
from src.losses.quantum_gan_loss import QuantumWassersteinLoss
from src.utils.visualization import plot_results
from src.utils.quantum_mode_diagnostics import run_full_diagnostic_analysis

# Suppress warnings for clean output
suppress_all_quantum_warnings()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DiagnosticBimodalConstellationTrainer:
    """
    Comprehensive trainer with full diagnostic framework.
    
    Integrates all 4 diagnostic tools to understand exactly where
    mode collapse occurs in the constellation QGAN architecture.
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
        Initialize diagnostic bimodal constellation trainer.
        
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
        
        # Create discriminator
        self.discriminator = PureSFDiscriminator(
            input_dim=data_dim,
            n_modes=n_modes,
            layers=3,
            cutoff_dim=6
        )
        
        # 🎯 PROPER WASSERSTEIN LOSS WITH ENTROPY REGULARIZATION
        self.loss_function = QuantumWassersteinLoss(
            lambda_gp=10.0,      # Gradient penalty weight
            lambda_entropy=1.0,  # MODE COLLAPSE FIX - entropy regularization
            lambda_physics=1.0   # Physics constraint weight
        )
        
        # Optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
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
            'discriminator_loss': [],
            'wasserstein_distance': [],
            'entropy_bonus': [],
            'mode_coverage_metrics': [],
            'epoch_visualizations': [],
            'diagnostic_results': []
        }
        
        logger.info("🔬 Diagnostic Bimodal Constellation Trainer initialized")
        logger.info(f"   Generator parameters: {len(self.generator.trainable_variables)}")
        logger.info(f"   Discriminator parameters: {len(self.discriminator.trainable_variables)}")
        logger.info(f"   Using Wasserstein loss with entropy regularization")
        logger.info(f"   Bimodal centers: {mode1_center} and {mode2_center}")
        logger.info(f"   Mode std: {mode_std}")
    
    def discriminator_train_step(self, real_data: tf.Tensor, z: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Single discriminator training step with Wasserstein loss."""
        with tf.GradientTape() as disc_tape:
            # Generate fake samples
            fake_data = self.generator.generate(z)
            
            # Use proper Wasserstein loss
            d_loss, g_loss, metrics = self.loss_function(
                real_data, fake_data, self.generator, self.discriminator
            )
        
        # Apply discriminator gradients
        disc_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )
        
        return d_loss, metrics
    
    def generator_train_step(self, z: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Single generator training step with Wasserstein loss."""
        with tf.GradientTape() as gen_tape:
            # Generate fake samples
            fake_data = self.generator.generate(z)
            
            # Create dummy real data for loss computation
            dummy_real = tf.zeros_like(fake_data)
            
            # Use proper Wasserstein loss
            d_loss, g_loss, metrics = self.loss_function(
                dummy_real, fake_data, self.generator, self.discriminator
            )
        
        # Apply generator gradients
        gen_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        
        return g_loss, metrics
    
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
    
    def create_epoch_visualization(self, epoch: int, save_dir: str = "diagnostic_bimodal_results"):
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
              discriminator_steps: int = 5,
              validation_frequency: int = 2,
              diagnostic_frequency: int = 2,
              save_dir: str = "diagnostic_bimodal_results") -> Dict[str, Any]:
        """
        Comprehensive training with full diagnostic analysis.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            discriminator_steps: Number of discriminator steps per generator step
            validation_frequency: Validation frequency (every N epochs)
            diagnostic_frequency: Full diagnostic analysis frequency (every N epochs)
            save_dir: Directory to save results
            
        Returns:
            Training results and metrics
        """
        print("🔬 COMPREHENSIVE DIAGNOSTIC BIMODAL CONSTELLATION TRAINING")
        print("=" * 80)
        print(f"Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Discriminator steps per generator step: {discriminator_steps}")
        print(f"   Validation frequency: every {validation_frequency} epochs")
        print(f"   Diagnostic frequency: every {diagnostic_frequency} epochs")
        print(f"   Results directory: {save_dir}")
        print(f"   Using Wasserstein loss with entropy regularization")
        print("=" * 80)
        
        # Update data generator batch size
        self.data_generator.batch_size = batch_size
        
        # Create results directory
        os.makedirs(save_dir, exist_ok=True)
        
        start_time = datetime.now()
        
        # Initial visualization and diagnostics
        print("\n🎨 Creating initial visualization...")
        initial_metrics, initial_viz = self.create_epoch_visualization(0, save_dir)
        self.training_history['mode_coverage_metrics'].append(initial_metrics)
        self.training_history['epoch_visualizations'].append(initial_viz)
        
        # Initial diagnostic analysis
        print("\n🔬 Running initial diagnostic analysis...")
        test_z = tf.random.normal([100, self.latent_dim])
        self.data_generator.batch_size = 100
        target_data = self.data_generator.generate_batch()
        
        initial_diagnostics, _ = run_full_diagnostic_analysis(
            self.generator, test_z, target_data, epoch=0, 
            save_dir=os.path.join(save_dir, "diagnostics")
        )
        self.training_history['diagnostic_results'].append(initial_diagnostics)
        
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch + 1}/{epochs}")
            
            epoch_gen_losses = []
            epoch_disc_losses = []
            epoch_w_distances = []
            epoch_entropy_bonuses = []
            
            # Training steps for this epoch
            steps_per_epoch = 50  # Fixed number of steps per epoch
            
            for step in range(steps_per_epoch):
                # Get real data batch
                self.data_generator.batch_size = batch_size
                real_data = self.data_generator.generate_batch()
                
                # Train discriminator multiple times
                disc_losses = []
                w_distances = []
                entropy_bonuses = []
                
                for _ in range(discriminator_steps):
                    z = tf.random.normal([batch_size, self.latent_dim])
                    disc_loss, disc_metrics = self.discriminator_train_step(real_data, z)
                    disc_losses.append(float(disc_loss))
                    w_distances.append(float(disc_metrics.get('w_distance', 0)))
                    entropy_bonuses.append(float(disc_metrics.get('entropy_bonus', 0)))
                
                # Train generator once
                z = tf.random.normal([batch_size, self.latent_dim])
                gen_loss, gen_metrics = self.generator_train_step(z)
                
                epoch_gen_losses.append(float(gen_loss))
                epoch_disc_losses.append(np.mean(disc_losses))
                epoch_w_distances.append(np.mean(w_distances))
                epoch_entropy_bonuses.append(np.mean(entropy_bonuses))
                
                # Progress updates
                if step % 10 == 0:
                    print(f"   Step {step + 1}/{steps_per_epoch}: "
                          f"G_loss={gen_loss:.4f}, D_loss={np.mean(disc_losses):.4f}, "
                          f"W_dist={np.mean(w_distances):.4f}")
            
            # Calculate epoch averages
            avg_gen_loss = np.mean(epoch_gen_losses)
            avg_disc_loss = np.mean(epoch_disc_losses)
            avg_w_distance = np.mean(epoch_w_distances)
            avg_entropy_bonus = np.mean(epoch_entropy_bonuses)
            
            # Store training history
            self.training_history['generator_loss'].append(avg_gen_loss)
            self.training_history['discriminator_loss'].append(avg_disc_loss)
            self.training_history['wasserstein_distance'].append(avg_w_distance)
            self.training_history['entropy_bonus'].append(avg_entropy_bonus)
            
            print(f"Epoch {epoch + 1} complete: G_loss={avg_gen_loss:.4f}, D_loss={avg_disc_loss:.4f}, "
                  f"W_dist={avg_w_distance:.4f}, Entropy={avg_entropy_bonus:.4f}")
            
            # Periodic validation and visualization
            if (epoch + 1) % validation_frequency == 0:
                print(f"\n🔍 Creating validation visualization for epoch {epoch + 1}...")
                metrics, viz_path = self.create_epoch_visualization(epoch + 1, save_dir)
                self.training_history['mode_coverage_metrics'].append(metrics)
                self.training_history['epoch_visualizations'].append(viz_path)
            
            # Periodic full diagnostic analysis
            if (epoch + 1) % diagnostic_frequency == 0:
                print(f"\n🔬 Running full diagnostic analysis for epoch {epoch + 1}...")
                test_z = tf.random.normal([100, self.latent_dim])
                self.data_generator.batch_size = 100
                target_data = self.data_generator.generate_batch()
                
                diagnostics, _ = run_full_diagnostic_analysis(
                    self.generator, test_z, target_data, epoch=epoch + 1,
                    save_dir=os.path.join(save_dir, "diagnostics")
                )
                self.training_history['diagnostic_results'].append(diagnostics)
                
                # Print key diagnostic insights
                if 'parameter_evolution' in diagnostics:
                    param_analysis = diagnostics['parameter_evolution']
                    if 'mode_diversity' in param_analysis:
                        disp_diversity = param_analysis['mode_diversity'].get('displacement_r', {})
                        print(f"   🔬 Mode diversity (displacement): CV = {disp_diversity.get('coefficient_of_variation', 0):.3f}")
                
                if 'measurement_separation' in diagnostics:
                    meas_analysis = diagnostics['measurement_separation']
                    if 'mode_separation' in meas_analysis:
                        min_dist = meas_analysis['mode_separation'].get('min_distance', 0)
                        print(f"   🔬 Measurement space separation: min = {min_dist:.3f}")
                
                if 'constellation_validation' in diagnostics:
                    const_validation = diagnostics['constellation_validation']
                    validation_passed = const_validation.get('validation_passed', False)
                    print(f"   🔬 Constellation structure: {'✅ VALID' if validation_passed else '❌ INVALID'}")
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        print(f"\n🎉 Training completed in {training_duration:.1f} seconds")
        
        # Final comprehensive evaluation
        print("\n🔍 FINAL COMPREHENSIVE EVALUATION")
        final_metrics, final_viz = self.create_epoch_visualization(epochs, save_dir)
        
        # Final diagnostic analysis
        print("\n🔬 FINAL DIAGNOSTIC ANALYSIS")
        test_z = tf.random.normal([200, self.latent_dim])
        self.data_generator.batch_size = 200
        target_data = self.data_generator.generate_batch()
        
        final_diagnostics, _ = run_full_diagnostic_analysis(
            self.generator, test_z, target_data, epoch=epochs,
            save_dir=os.path.join(save_dir, "diagnostics")
        )
        
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
        
        # Print diagnostic summary
        print(f"\n🔬 DIAGNOSTIC SUMMARY:")
        if final_diagnostics:
            if 'parameter_evolution' in final_diagnostics:
                param_analysis = final_diagnostics['parameter_evolution']
                if 'spatial_separation' in param_analysis:
                    spatial_sep = param_analysis['spatial_separation']
                    print(f"   Spatial separation: min={spatial_sep.get('min_distance', 0):.3f}, "
                          f"mean={spatial_sep.get('mean_distance', 0):.3f}")
            
            if 'mode_contributions' in final_diagnostics:
                contrib_analysis = final_diagnostics['mode_contributions']
                if 'analysis_summary' in contrib_analysis:
                    summary = contrib_analysis['analysis_summary']
                    print(f"   Most important mode: {summary.get('most_important_mode', 'N/A')}")
                    print(f"   Contribution balance: {summary.get('contribution_balance', 0):.3f}")
        
        # Prepare results
        training_results = {
            'training_duration_seconds': training_duration,
            'epochs_completed': epochs,
            'final_metrics': final_comprehensive_metrics,
            'final_diagnostics': final_diagnostics,
            'training_history': self.training_history,
            'configuration': {
                'batch_size': batch_size,
                'discriminator_steps': discriminator_steps,
                'validation_frequency': validation_frequency,
                'diagnostic_frequency': diagnostic_frequency,
                'mode1_center': self.mode1_center,
                'mode2_center': self.mode2_center,
                'mode_std': self.mode_std,
                'loss_function': 'QuantumWassersteinLoss',
                'entropy_regularization': True
            },
            'visualizations': {
                'final_comparison': final_comparison_path,
                'epoch_progressions': self.training_history['epoch_visualizations']
            }
        }
        
        # Save results
        results_path = os.path.join(save_dir, "complete_diagnostic_training_results.json")
        
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
        
        print(f"✅ Complete diagnostic results saved to {results_path}")
        
        return training_results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Diagnostic Bimodal Constellation QGAN Training"
    )
    
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--discriminator-steps", type=int, default=5, help="Discriminator steps per generator step")
    parser.add_argument("--validation-frequency", type=int, default=2, help="Validation frequency (epochs)")
    parser.add_argument("--diagnostic-frequency", type=int, default=2, help="Full diagnostic frequency (epochs)")
    parser.add_argument("--mode1-center", nargs=2, type=float, default=[-1.5, -1.5], help="Mode 1 center")
    parser.add_argument("--mode2-center", nargs=2, type=float, default=[1.5, 1.5], help="Mode 2 center")
    parser.add_argument("--mode-std", type=float, default=0.3, help="Mode standard deviation")
    parser.add_argument("--save-dir", type=str, default="diagnostic_bimodal_results", help="Results directory")
    parser.add_argument("--enable-validation", action="store_true", help="Enable state validation")
    
    args = parser.parse_args()
    
    print("🔬 COMPREHENSIVE DIAGNOSTIC BIMODAL CONSTELLATION TRAINING")
    print("=" * 80)
    print("Complete diagnostic framework to understand mode collapse")
    print("Using Wasserstein loss + entropy regularization + 4 diagnostic tools")
    print("=" * 80)
    
    # Create trainer
    trainer = DiagnosticBimodalConstellationTrainer(
        mode1_center=tuple(args.mode1_center),
        mode2_center=tuple(args.mode2_center),
        mode_std=args.mode_std,
        enable_state_validation=args.enable_validation
    )
    
    # Run comprehensive diagnostic training
    results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        discriminator_steps=args.discriminator_steps,
        validation_frequency=args.validation_frequency,
        diagnostic_frequency=args.diagnostic_frequency,
        save_dir=args.save_dir
    )
    
    print("\n" + "=" * 80)
    print("🔬 COMPREHENSIVE DIAGNOSTIC TRAINING COMPLETE")
    print("=" * 80)
    print(f"Check {args.save_dir}/ for complete diagnostic analysis!")
    print("Key diagnostic outputs:")
    print(f"  - Parameter evolution plots")
    print(f"  - 8D measurement space visualization (PCA/t-SNE)")
    print(f"  - Per-mode contribution analysis")
    print(f"  - Constellation structure validation")
    print(f"  - Complete JSON results with all metrics")


if __name__ == "__main__":
    main()
