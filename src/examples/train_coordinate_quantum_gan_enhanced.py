"""
Enhanced Coordinate Quantum GAN Training with Comprehensive Debugging

This script provides extensive debugging and monitoring capabilities:
- Gradient flow tracking
- Generated data evolution visualization
- Parameter evolution monitoring
- GIF creation for training progression
- Comprehensive validation metrics
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
from typing import Dict, List, Any, Optional
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
from src.utils.visualization import plot_results
from src.utils.quantum_circuit_visualizer import QuantumCircuitVisualizer
from src.utils.enhanced_quantum_circuit_visualizer import create_enhanced_circuit_visualization
from src.utils.quantum_training_health_checker import QuantumTrainingHealthChecker

# Suppress warnings
suppress_all_quantum_warnings()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GradientFlowMonitor:
    """Monitor gradient flow through the network."""
    
    def __init__(self):
        self.gradient_history = []
    
    def monitor_gradients(self, gradients, variables, step_info=""):
        """Monitor gradient statistics."""
        gradient_stats = {
            'step_info': step_info,
            'total_variables': len(variables),
            'valid_gradients': 0,
            'gradient_norms': [],
            'gradient_means': [],
            'nan_gradients': 0,
            'zero_gradients': 0
        }
        
        for i, (grad, var) in enumerate(zip(gradients, variables)):
            if grad is not None:
                gradient_stats['valid_gradients'] += 1
                
                # Check for NaN
                if tf.reduce_any(tf.math.is_nan(grad)):
                    gradient_stats['nan_gradients'] += 1
                else:
                    grad_norm = tf.norm(grad)
                    grad_mean = tf.reduce_mean(tf.abs(grad))
                    
                    gradient_stats['gradient_norms'].append(float(grad_norm))
                    gradient_stats['gradient_means'].append(float(grad_mean))
                    
                    # Check for effectively zero gradients
                    if grad_norm < 1e-8:
                        gradient_stats['zero_gradients'] += 1
        
        # Calculate aggregate statistics
        if gradient_stats['gradient_norms']:
            gradient_stats['avg_gradient_norm'] = np.mean(gradient_stats['gradient_norms'])
            gradient_stats['max_gradient_norm'] = np.max(gradient_stats['gradient_norms'])
            gradient_stats['avg_gradient_mean'] = np.mean(gradient_stats['gradient_means'])
        else:
            gradient_stats['avg_gradient_norm'] = 0.0
            gradient_stats['max_gradient_norm'] = 0.0
            gradient_stats['avg_gradient_mean'] = 0.0
        
        gradient_stats['gradient_flow_ratio'] = gradient_stats['valid_gradients'] / gradient_stats['total_variables']
        
        self.gradient_history.append(gradient_stats)
        return gradient_stats


class ParameterEvolutionTracker:
    """Track parameter evolution over training."""
    
    def __init__(self):
        self.parameter_history = []
        self.initial_parameters = None
    
    def track_parameters(self, variables, epoch, step):
        """Track current parameter values."""
        current_params = {}
        total_change = 0.0
        
        for i, var in enumerate(variables):
            param_name = var.name if hasattr(var, 'name') else f'param_{i}'
            param_value = var.numpy()
            current_params[param_name] = {
                'value': param_value.copy(),
                'norm': float(np.linalg.norm(param_value)),
                'mean': float(np.mean(param_value)),
                'std': float(np.std(param_value))
            }
            
            # Calculate change from initial
            if self.initial_parameters is not None and param_name in self.initial_parameters:
                change = np.linalg.norm(param_value - self.initial_parameters[param_name]['value'])
                current_params[param_name]['change_from_initial'] = float(change)
                total_change += change
        
        # Store initial parameters on first call
        if self.initial_parameters is None:
            self.initial_parameters = current_params.copy()
            total_change = 0.0
        
        param_snapshot = {
            'epoch': epoch,
            'step': step,
            'timestamp': time.time(),
            'parameters': current_params,
            'total_parameter_change': total_change
        }
        
        self.parameter_history.append(param_snapshot)
        return param_snapshot


class SampleEvolutionTracker:
    """Track generated sample evolution."""
    
    def __init__(self):
        self.sample_history = []
    
    def track_samples(self, generated_samples, epoch, step):
        """Track generated samples for evolution analysis."""
        samples_np = generated_samples.numpy()
        
        sample_stats = {
            'epoch': epoch,
            'step': step,
            'timestamp': time.time(),
            'samples': samples_np.copy(),
            'mean': np.mean(samples_np, axis=0),
            'std': np.std(samples_np, axis=0),
            'min': np.min(samples_np, axis=0),
            'max': np.max(samples_np, axis=0),
            'sample_count': len(samples_np)
        }
        
        self.sample_history.append(sample_stats)
        return sample_stats


class EnhancedCoordinateGANTrainer:
    """Enhanced trainer with comprehensive debugging capabilities."""
    
    def __init__(self, 
                 latent_dim: int = 2,
                 data_dim: int = 2,
                 n_modes: int = 4,
                 mode1_center: tuple = (-1.5, -1.5),
                 mode2_center: tuple = (1.5, 1.5),
                 mode_std: float = 0.3):
        """Initialize enhanced trainer."""
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
        
        # Create discriminator
        self.discriminator = PureSFDiscriminator(
            input_dim=data_dim,
            n_modes=n_modes,
            layers=1,
            cutoff_dim=6
        )
        
        # Use fixed Wasserstein loss
        self.loss_function = QuantumWassersteinLoss(
            lambda_gp=10.0,
            lambda_entropy=1.0,
            lambda_physics=1.0
        )
        
        # Optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.5)
        
        # Data generator
        self.data_generator = BimodalDataGenerator(
            batch_size=32,
            n_features=data_dim,
            mode1_center=mode1_center,
            mode2_center=mode2_center,
            mode_std=mode_std,
            seed=42
        )
        
        # Enhanced debugging and monitoring
        self.gradient_monitor = GradientFlowMonitor()
        self.parameter_tracker = ParameterEvolutionTracker()
        self.sample_tracker = SampleEvolutionTracker()
        
        # Training history with enhanced metrics
        self.training_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'mode_coverage': [],
            'gradient_flow': [],
            'parameter_evolution': [],
            'loss_components': [],
            'sample_evolution': []
        }
        
        logger.info("Enhanced CoordinateGANTrainer initialized")
        logger.info(f"  Generator params: {len(self.generator.trainable_variables)}")
        logger.info(f"  Discriminator params: {len(self.discriminator.trainable_variables)}")
    
    def fixed_loss_computation(self, real_data: tf.Tensor, fake_data: tf.Tensor):
        """Fixed loss computation with proper batch size handling."""
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
        
        metrics = {
            'w_distance': w_distance,
            'gradient_penalty': gradient_penalty,
            'entropy_bonus': entropy_bonus,
            'd_loss': d_loss,
            'g_loss': g_loss
        }
        
        return d_loss, g_loss, metrics
    
    def discriminator_train_step(self, real_data: tf.Tensor, z: tf.Tensor):
        """Enhanced discriminator training step with monitoring."""
        with tf.GradientTape() as disc_tape:
            # Generate fake data
            fake_data = self.generator.generate(z)
            
            # Use fixed loss computation
            d_loss, g_loss, metrics = self.fixed_loss_computation(real_data, fake_data)
            
            # NaN protection
            if tf.math.is_nan(d_loss):
                logger.warning("NaN detected in discriminator loss, using fallback")
                d_loss = tf.constant(1.0)
        
        # Compute gradients
        disc_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        
        # Monitor gradient flow
        grad_stats = self.gradient_monitor.monitor_gradients(
            disc_gradients, self.discriminator.trainable_variables, "discriminator"
        )
        
        # Apply gradients with clipping
        if disc_gradients is not None:
            valid_gradients = []
            valid_variables = []
            for grad, var in zip(disc_gradients, self.discriminator.trainable_variables):
                if grad is not None and not tf.reduce_any(tf.math.is_nan(grad)):
                    clipped_grad = tf.clip_by_value(grad, -1.0, 1.0)
                    valid_gradients.append(clipped_grad)
                    valid_variables.append(var)
            
            if valid_gradients:
                self.discriminator_optimizer.apply_gradients(
                    zip(valid_gradients, valid_variables)
                )
        
        return d_loss, metrics, grad_stats
    
    def generator_train_step(self, z: tf.Tensor, epoch: int, step: int):
        """Enhanced generator training step with monitoring."""
        with tf.GradientTape() as gen_tape:
            # Generate fake data
            fake_data = self.generator.generate(z)
            
            # Create dummy real data for loss computation
            dummy_real = tf.zeros_like(fake_data)
            
            # Use fixed loss computation
            d_loss, g_loss, metrics = self.fixed_loss_computation(dummy_real, fake_data)
            
            # NaN protection
            if tf.math.is_nan(g_loss):
                logger.warning("NaN detected in generator loss, using fallback")
                g_loss = tf.constant(1.0)
        
        # Compute gradients
        gen_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        
        # Monitor gradient flow
        grad_stats = self.gradient_monitor.monitor_gradients(
            gen_gradients, self.generator.trainable_variables, "generator"
        )
        
        # Track parameter evolution
        param_stats = self.parameter_tracker.track_parameters(
            self.generator.trainable_variables, epoch, step
        )
        
        # Track sample evolution
        sample_stats = self.sample_tracker.track_samples(fake_data, epoch, step)
        
        # Apply gradients with clipping
        if gen_gradients is not None:
            valid_gradients = []
            valid_variables = []
            for grad, var in zip(gen_gradients, self.generator.trainable_variables):
                if grad is not None and not tf.reduce_any(tf.math.is_nan(grad)):
                    clipped_grad = tf.clip_by_value(grad, -1.0, 1.0)
                    valid_gradients.append(clipped_grad)
                    valid_variables.append(var)
            
            if valid_gradients:
                self.generator_optimizer.apply_gradients(
                    zip(valid_gradients, valid_variables)
                )
        
        return g_loss, metrics, grad_stats, param_stats, sample_stats
    
    def calculate_mode_coverage(self, generated_samples: tf.Tensor):
        """Calculate mode coverage metrics."""
        gen_np = generated_samples.numpy()
        
        mode1_center_np = np.array(self.mode1_center)
        mode2_center_np = np.array(self.mode2_center)
        
        distances_to_mode1 = np.linalg.norm(gen_np - mode1_center_np, axis=1)
        distances_to_mode2 = np.linalg.norm(gen_np - mode2_center_np, axis=1)
        
        mode1_assignments = distances_to_mode1 < distances_to_mode2
        mode2_assignments = ~mode1_assignments
        
        mode1_coverage = np.sum(mode1_assignments) / len(gen_np)
        mode2_coverage = np.sum(mode2_assignments) / len(gen_np)
        
        return {
            'mode1_coverage': float(mode1_coverage),
            'mode2_coverage': float(mode2_coverage),
            'balanced_coverage': float(min(mode1_coverage, mode2_coverage) / max(mode1_coverage, mode2_coverage)) if max(mode1_coverage, mode2_coverage) > 0 else 0.0
        }
    
    def create_epoch_visualization(self, epoch: int, save_dir: str):
        """Create comprehensive epoch visualization."""
        # Generate samples for visualization
        test_z = tf.random.normal([500, self.latent_dim])
        generated_samples = self.generator.generate(test_z)
        
        # Generate real samples
        self.data_generator.batch_size = 50
        real_samples = self.data_generator.generate_batch()
        
        # Create enhanced visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Epoch {epoch} - Training Progress Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Real vs Generated Data
        axes[0, 0].scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.5, c='blue', s=30, label='Real')
        axes[0, 0].scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, c='red', s=30, label='Generated')
        axes[0, 0].set_title('Real vs Generated Data')
        axes[0, 0].set_xlabel('Feature 1')
        axes[0, 0].set_ylabel('Feature 2')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(-3, 3)
        axes[0, 0].set_ylim(-3, 3)
        
        # Plot 2: Loss Evolution
        if len(self.training_history['generator_loss']) > 0:
            epochs_so_far = range(1, len(self.training_history['generator_loss']) + 1)
            axes[0, 1].plot(epochs_so_far, self.training_history['generator_loss'], 'r-', label='Generator')
            axes[0, 1].plot(epochs_so_far, self.training_history['discriminator_loss'], 'b-', label='Discriminator')
            axes[0, 1].set_title('Loss Evolution')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Mode Coverage Evolution
        if len(self.training_history['mode_coverage']) > 0:
            epochs_so_far = range(1, len(self.training_history['mode_coverage']) + 1)
            mode1_coverage = [mc['mode1_coverage'] for mc in self.training_history['mode_coverage']]
            mode2_coverage = [mc['mode2_coverage'] for mc in self.training_history['mode_coverage']]
            balanced_coverage = [mc['balanced_coverage'] for mc in self.training_history['mode_coverage']]
            
            axes[0, 2].plot(epochs_so_far, mode1_coverage, 'r-', label='Mode 1')
            axes[0, 2].plot(epochs_so_far, mode2_coverage, 'b-', label='Mode 2')
            axes[0, 2].plot(epochs_so_far, balanced_coverage, 'g-', label='Balanced')
            axes[0, 2].set_title('Mode Coverage Evolution')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Coverage')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_ylim(0, 1)
        
        # Plot 4: Gradient Flow
        if len(self.training_history['gradient_flow']) > 0:
            recent_grads = self.training_history['gradient_flow'][-10:]  # Last 10 gradient measurements
            avg_norms = [g.get('avg_gradient_norm', 0) for g in recent_grads]
            flow_ratios = [g.get('gradient_flow_ratio', 0) for g in recent_grads]
            
            ax_twin = axes[1, 0].twinx()
            axes[1, 0].bar(range(len(avg_norms)), avg_norms, alpha=0.7, color='blue', label='Avg Gradient Norm')
            ax_twin.plot(range(len(flow_ratios)), flow_ratios, 'ro-', label='Flow Ratio')
            
            axes[1, 0].set_title('Recent Gradient Flow')
            axes[1, 0].set_xlabel('Recent Steps')
            axes[1, 0].set_ylabel('Gradient Norm', color='blue')
            ax_twin.set_ylabel('Flow Ratio', color='red')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Parameter Evolution
        if len(self.training_history['parameter_evolution']) > 0:
            param_changes = [pe.get('total_parameter_change', 0) for pe in self.training_history['parameter_evolution']]
            if param_changes:
                axes[1, 1].plot(param_changes, 'g-')
                axes[1, 1].set_title('Total Parameter Change')
                axes[1, 1].set_xlabel('Training Steps')
                axes[1, 1].set_ylabel('Parameter Change')
                axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Sample Distribution Evolution
        if len(self.sample_tracker.sample_history) > 0:
            recent_samples = self.sample_tracker.sample_history[-5:]  # Last 5 sample batches
            for i, sample_data in enumerate(recent_samples):
                samples = sample_data['samples']
                alpha = 0.2 + 0.6 * (i / len(recent_samples))  # Fade effect
                axes[1, 2].scatter(samples[:50, 0], samples[:50, 1], alpha=alpha, s=20, 
                                 label=f'Step {sample_data["step"]}')
            
            axes[1, 2].set_title('Sample Evolution (Recent)')
            axes[1, 2].set_xlabel('Feature 1')
            axes[1, 2].set_ylabel('Feature 2')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_xlim(-3, 3)
            axes[1, 2].set_ylim(-3, 3)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(save_dir, "epoch_visualizations", f"epoch_{epoch:03d}_analysis.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  📊 Enhanced visualization saved: {viz_path}")
        return viz_path
    
    def create_training_gif(self, save_dir: str, fps: int = 2):
        """Create GIF from epoch visualizations."""
        print("\n🎬 CREATING TRAINING PROGRESSION GIF")
        print("=" * 60)
        
        viz_dir = os.path.join(save_dir, "epoch_visualizations")
        gif_path = os.path.join(save_dir, "training_progression.gif")
        
        # Collect all epoch visualization files
        viz_files = []
        for file in os.listdir(viz_dir):
            if file.startswith("epoch_") and file.endswith("_analysis.png"):
                viz_files.append(os.path.join(viz_dir, file))
        
        viz_files.sort()  # Ensure chronological order
        
        if len(viz_files) > 0:
            # Create GIF
            images = []
            for viz_file in viz_files:
                img = Image.open(viz_file)
                images.append(img)
            
            # Save GIF
            imageio.mimsave(gif_path, images, fps=fps)
            print(f"✅ Training progression GIF created: {gif_path}")
            print(f"   Frames: {len(images)}, FPS: {fps}")
        else:
            print("❌ No visualization files found for GIF creation")
        
        return gif_path
    
    def print_debug_summary(self, epoch: int, grad_stats: Dict, param_stats: Dict, sample_stats: Dict):
        """Print comprehensive debug summary."""
        print(f"\n🔍 DEBUG SUMMARY - Epoch {epoch}")
        print("-" * 50)
        
        # Gradient Flow Analysis
        if grad_stats:
            print(f"📊 Gradient Flow:")
            print(f"   Valid gradients: {grad_stats.get('valid_gradients', 0)}/{grad_stats.get('total_variables', 0)}")
            print(f"   Flow ratio: {grad_stats.get('gradient_flow_ratio', 0):.3f}")
            print(f"   Avg gradient norm: {grad_stats.get('avg_gradient_norm', 0):.6f}")
            print(f"   Max gradient norm: {grad_stats.get('max_gradient_norm', 0):.6f}")
            print(f"   NaN gradients: {grad_stats.get('nan_gradients', 0)}")
            print(f"   Zero gradients: {grad_stats.get('zero_gradients', 0)}")
        
        # Parameter Evolution
        if param_stats:
            print(f"🔧 Parameter Evolution:")
            print(f"   Total parameter change: {param_stats.get('total_parameter_change', 0):.6f}")
            print(f"   Parameters tracked: {len(param_stats.get('parameters', {}))}")
        
        # Sample Evolution
        if sample_stats:
            samples = sample_stats.get('samples', np.array([]))
            if len(samples) > 0:
                print(f"📈 Generated Samples:")
                print(f"   Sample range: X=[{samples[:, 0].min():.3f}, {samples[:, 0].max():.3f}], "
                      f"Y=[{samples[:, 1].min():.3f}, {samples[:, 1].max():.3f}]")
                print(f"   Sample mean: ({samples[:, 0].mean():.3f}, {samples[:, 1].mean():.3f})")
                print(f"   Sample std: ({samples[:, 0].std():.3f}, {samples[:, 1].std():.3f})")
    
    def train(self, epochs: int = 5, batch_size: int = 16, save_dir: str = "results/training", health_check: bool = False):
        """Enhanced training loop with comprehensive monitoring."""
        print(f"🚀 STARTING ENHANCED COORDINATE GAN TRAINING")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps per epoch: 5")
        print(f"  Save directory: {save_dir}")
        print(f"  Health check: {health_check}")
        print("=" * 60)
        
        # Health Check
        if health_check:
            print("\n🏥 RUNNING PRE-TRAINING HEALTH CHECK")
            print("=" * 60)
            
            health_checker = QuantumTrainingHealthChecker()
            
            # Create configuration for health check
            config = {
                'n_modes': self.n_modes,
                'cutoff_dim': 6,  # From generator initialization
                'batch_size': batch_size,
                'layers': 1,
                'epochs': epochs,
                'steps_per_epoch': 5,
                'latent_dim': self.latent_dim,
                'output_dim': self.data_dim
            }
            
            # Run health check
            health_result = health_checker.pre_training_health_check(config)
            
            print(f"\n📋 HEALTH CHECK RESULTS:")
            print(f"  Safe to proceed: {'✅' if health_result.safe_to_proceed else '❌'} {health_result.safe_to_proceed}")
            print(f"  Risk level: {health_result.risk_level}")
            print(f"  Estimated memory: {health_result.estimated_memory_gb:.2f}GB")
            print(f"  Estimated time: {health_result.estimated_time_hours:.2f} hours")
            print(f"  Confidence score: {health_result.confidence_score:.2f}")
            
            if health_result.warnings:
                print(f"\n⚠️ WARNINGS:")
                for warning in health_result.warnings:
                    print(f"    • {warning}")
            
            if health_result.recommendations:
                print(f"\n💡 RECOMMENDATIONS:")
                for rec in health_result.recommendations:
                    print(f"    • {rec}")
            
            # Apply optimizations if needed
            if not health_result.safe_to_proceed:
                print(f"\n🛑 TRAINING ABORTED: System not safe for training")
                print(f"Please follow the recommendations above and try again.")
                return None
            
            # Apply optimized configuration
            if health_result.optimized_config != config:
                print(f"\n🔧 APPLYING OPTIMIZED CONFIGURATION:")
                old_batch_size = batch_size
                batch_size = health_result.optimized_config.get('batch_size', batch_size)
                if batch_size != old_batch_size:
                    print(f"    • Batch size: {old_batch_size} → {batch_size}")
                    self.data_generator.batch_size = batch_size
            
            # Start health monitoring
            health_checker.start_training_monitoring(epochs)
            
            print("=" * 60)
        
        # Create results directory structure
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "circuit_visualization"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "epoch_visualizations"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "training_logs"), exist_ok=True)
        
        # Step 1: Circuit Analysis
        print("\n🔬 QUANTUM CIRCUIT ANALYSIS")
        print("=" * 60)
        visualizer = QuantumCircuitVisualizer(self.generator.quantum_circuit)
        visualizer.print_compact_circuit()
        
        try:
            create_enhanced_circuit_visualization(
                quantum_circuit=self.generator.quantum_circuit,
                coordinate_generator=self.generator,
                save_dir=os.path.join(save_dir, "circuit_visualization"),
                training_history=None
            )
            print("✅ Enhanced circuit visualization created!")
        except Exception as e:
            print(f"⚠️ Enhanced visualization failed: {e}")
        
        # Step 2: Target Data Analysis
        print("\n🎯 TARGET DATA ANALYSIS")
        print("=" * 60)
        self.data_generator.batch_size = 50
        target_data = self.data_generator.generate_batch().numpy()
        analysis_results = self.generator.analyze_target_data(target_data)
        
        # Step 3: Initial Comparison
        print("\n📊 INITIAL DATA COMPARISON")
        print("=" * 60)
        self.data_generator.batch_size = 500
        real_samples = self.data_generator.generate_batch().numpy()
        
        test_z = tf.random.normal([500, self.latent_dim])
        generated_samples = self.generator.generate(test_z).numpy()
        
        coverage_metrics = self.calculate_mode_coverage(tf.constant(generated_samples))
        print(f"📈 Initial metrics:")
        print(f"   Mode 1 coverage: {coverage_metrics['mode1_coverage']:.3f}")
        print(f"   Mode 2 coverage: {coverage_metrics['mode2_coverage']:.3f}")
        print(f"   Balanced coverage: {coverage_metrics['balanced_coverage']:.3f}")
        
        # Update data generator batch size for training
        self.data_generator.batch_size = batch_size
        
        # Training Loop
        print(f"\n🏃 TRAINING LOOP")
        print("=" * 60)
        
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch + 1}/{epochs}")
            print("-" * 30)
            
            epoch_gen_losses = []
            epoch_disc_losses = []
            epoch_grad_stats = []
            epoch_param_stats = []
            epoch_sample_stats = []
            
            # Training steps per epoch
            steps_per_epoch = 5
            
            for step in range(steps_per_epoch):
                # Get real data
                real_data = self.data_generator.generate_batch()
                
                # Train discriminator (3 steps)
                disc_losses = []
                for _ in range(3):
                    z = tf.random.normal([batch_size, self.latent_dim])
                    disc_loss, disc_metrics, disc_grad_stats = self.discriminator_train_step(real_data, z)
                    disc_losses.append(float(disc_loss))
                    epoch_grad_stats.append(disc_grad_stats)
                
                # Train generator (1 step)
                z = tf.random.normal([batch_size, self.latent_dim])
                gen_loss, gen_metrics, gen_grad_stats, param_stats, sample_stats = self.generator_train_step(
                    z, epoch + 1, step
                )
                
                epoch_gen_losses.append(float(gen_loss))
                epoch_disc_losses.append(np.mean(disc_losses))
                epoch_grad_stats.append(gen_grad_stats)
                epoch_param_stats.append(param_stats)
                epoch_sample_stats.append(sample_stats)
                
                if step == 0:  # Print first step details
                    print(f"  Step {step + 1}: G_loss={gen_loss:.4f}, D_loss={np.mean(disc_losses):.4f}")
                    self.print_debug_summary(epoch + 1, gen_grad_stats, param_stats, sample_stats)
            
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
            self.training_history['gradient_flow'].extend(epoch_grad_stats)
            self.training_history['parameter_evolution'].extend(epoch_param_stats)
            self.training_history['sample_evolution'].extend(epoch_sample_stats)
            
            print(f"✅ Epoch {epoch + 1} complete:")
            print(f"   G_loss: {avg_gen_loss:.4f}, D_loss: {avg_disc_loss:.4f}")
            print(f"   Mode1 coverage: {coverage_metrics['mode1_coverage']:.3f}")
            print(f"   Mode2 coverage: {coverage_metrics['mode2_coverage']:.3f}")
            print(f"   Balanced coverage: {coverage_metrics['balanced_coverage']:.3f}")
            
            # Create epoch visualization
            self.create_epoch_visualization(epoch + 1, save_dir)
        
        # Final results
        print(f"\n🎉 TRAINING COMPLETE!")
        print("=" * 60)
        
        # Create training GIF
        self.create_training_gif(save_dir)
        
        # Enhanced final analysis
        print(f"\n📊 FINAL ANALYSIS")
        print("=" * 60)
        try:
            create_enhanced_circuit_visualization(
                quantum_circuit=self.generator.quantum_circuit,
                coordinate_generator=self.generator,
                save_dir=os.path.join(save_dir, "circuit_visualization"),
                training_history=self.training_history
            )
            print("✅ Final enhanced circuit visualization created!")
        except Exception as e:
            print(f"⚠️ Final visualization failed: {e}")
        
        # Save comprehensive results
        self.save_enhanced_results(save_dir)
        
        return self.training_history
    
    def save_enhanced_results(self, save_dir: str):
        """Save comprehensive training results with debugging data."""
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
            'debugging_summary': {
                'total_gradient_measurements': len(self.training_history['gradient_flow']),
                'total_parameter_snapshots': len(self.training_history['parameter_evolution']),
                'total_sample_snapshots': len(self.training_history['sample_evolution']),
                'final_gradient_flow_ratio': self.training_history['gradient_flow'][-1].get('gradient_flow_ratio', 0) if self.training_history['gradient_flow'] else 0,
                'final_parameter_change': self.training_history['parameter_evolution'][-1].get('total_parameter_change', 0) if self.training_history['parameter_evolution'] else 0
            }
        }
        
        results_path = os.path.join(save_dir, "enhanced_training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)  # default=str to handle numpy types
        
        print(f"✅ Enhanced results saved to: {results_path}")
        
        # Save gradient flow analysis
        gradient_analysis_path = os.path.join(save_dir, "gradient_flow_analysis.json")
        with open(gradient_analysis_path, 'w') as f:
            json.dump(self.training_history['gradient_flow'], f, indent=2, default=str)
        
        print(f"✅ Gradient flow analysis saved to: {gradient_analysis_path}")


def main():
    """Main function for enhanced training."""
    parser = argparse.ArgumentParser(description="Enhanced Coordinate Quantum GAN Training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--save-dir", type=str, default="results/enhanced_training", help="Save directory")
    parser.add_argument("--health-check", action="store_true", help="Run pre-training health check")
    
    args = parser.parse_args()
    
    print("🚀 ENHANCED COORDINATE QUANTUM GAN TRAINING")
    print("=" * 60)
    print("Comprehensive debugging and monitoring system")
    print("Features:")
    print("  ✅ Gradient flow tracking")
    print("  ✅ Parameter evolution monitoring")
    print("  ✅ Generated data evolution")
    print("  ✅ Training progression GIF")
    print("  ✅ Comprehensive validation metrics")
    if args.health_check:
        print("  ✅ Pre-training health check")
        print("  ✅ Memory safety analysis")
        print("  ✅ Training time estimation")
        print("  ✅ Hardware performance benchmarking")
    print("=" * 60)
    
    # Create enhanced trainer
    trainer = EnhancedCoordinateGANTrainer()
    
    # Train with enhanced monitoring
    results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        health_check=args.health_check
    )
    
    # Print final comprehensive results
    final_coverage = results['mode_coverage'][-1] if results['mode_coverage'] else {}
    print(f"\n🏆 FINAL RESULTS:")
    print("=" * 60)
    print(f"Mode 1 coverage: {final_coverage.get('mode1_coverage', 0):.3f}")
    print(f"Mode 2 coverage: {final_coverage.get('mode2_coverage', 0):.3f}")
    print(f"Balanced coverage: {final_coverage.get('balanced_coverage', 0):.3f}")
    
    # Learning validation
    if len(results['parameter_evolution']) > 0:
        total_param_change = results['parameter_evolution'][-1].get('total_parameter_change', 0)
        print(f"Total parameter change: {total_param_change:.6f}")
        
        if total_param_change > 1e-6:
            print("✅ LEARNING DETECTED: Parameters evolved significantly!")
        else:
            print("❌ LIMITED LEARNING: Parameters changed minimally")
    
    # Gradient flow validation
    if len(results['gradient_flow']) > 0:
        final_flow_ratio = results['gradient_flow'][-1].get('gradient_flow_ratio', 0)
        print(f"Final gradient flow ratio: {final_flow_ratio:.3f}")
        
        if final_flow_ratio > 0.5:
            print("✅ GRADIENT FLOW: Good gradient propagation!")
        else:
            print("❌ GRADIENT ISSUES: Poor gradient flow detected")
    
    # Mode coverage validation
    balanced_coverage = final_coverage.get('balanced_coverage', 0)
    if balanced_coverage > 0.5:
        print("✅ SUCCESS: Achieved balanced mode coverage!")
    else:
        print("❌ MODE COLLAPSE: Unbalanced mode coverage")
    
    print("=" * 60)
    print("Check the generated GIF and visualizations for detailed analysis!")


if __name__ == "__main__":
    main()
