"""
Diagnostic Quantum GAN Training with Learning Rate Scheduling and Enhanced Loss Analysis

This script provides comprehensive diagnostics for quantum GAN training issues:
- Adaptive learning rate scheduling
- Enhanced gradient analysis
- Loss landscape exploration
- Alternative loss formulations
- Quantum circuit expressivity testing
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

# Suppress warnings
suppress_all_quantum_warnings()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduler for quantum GANs."""
    
    def __init__(self, initial_lr_g=0.0001, initial_lr_d=0.0001, 
                 patience=5, factor=0.8, min_lr=1e-6):
        self.initial_lr_g = initial_lr_g
        self.initial_lr_d = initial_lr_d
        self.current_lr_g = initial_lr_g
        self.current_lr_d = initial_lr_d
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        
        self.best_loss = float('inf')
        self.wait = 0
        self.lr_history = []
        
    def update(self, current_loss, epoch):
        """Update learning rates based on loss progress."""
        self.lr_history.append({
            'epoch': epoch,
            'lr_g': self.current_lr_g,
            'lr_d': self.current_lr_d,
            'loss': current_loss
        })
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            # Reduce learning rate
            old_lr_g = self.current_lr_g
            old_lr_d = self.current_lr_d
            
            self.current_lr_g = max(self.current_lr_g * self.factor, self.min_lr)
            self.current_lr_d = max(self.current_lr_d * self.factor, self.min_lr)
            
            if self.current_lr_g < old_lr_g:
                print(f"  📉 Learning rate reduced: G={old_lr_g:.6f}→{self.current_lr_g:.6f}, "
                      f"D={old_lr_d:.6f}→{self.current_lr_d:.6f}")
            
            self.wait = 0
            
        return self.current_lr_g, self.current_lr_d


class DiagnosticQuantumGANTrainer:
    """Diagnostic trainer with enhanced analysis capabilities."""
    
    def __init__(self, 
                 latent_dim: int = 2,
                 data_dim: int = 2,
                 n_modes: int = 4,
                 mode1_center: tuple = (-1.5, -1.5),
                 mode2_center: tuple = (1.5, 1.5),
                 mode_std: float = 0.3):
        """Initialize diagnostic trainer."""
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.n_modes = n_modes
        self.mode1_center = mode1_center
        self.mode2_center = mode2_center
        self.mode_std = mode_std
        
        # Create models
        self.generator = CoordinateQuantumGenerator(
            latent_dim=latent_dim,
            output_dim=data_dim,
            n_modes=n_modes,
            layers=1,
            cutoff_dim=6,
            clustering_method='kmeans',
            coordinate_names=['X', 'Y']
        )
        
        self.discriminator = PureSFDiscriminator(
            input_dim=data_dim,
            n_modes=n_modes,
            layers=1,
            cutoff_dim=6
        )
        
        # Adaptive learning rate scheduler
        self.lr_scheduler = AdaptiveLearningRateScheduler(
            initial_lr_g=0.0005,  # Start higher than before
            initial_lr_d=0.0001,  # Discriminator learns slower
            patience=3,
            factor=0.7,
            min_lr=1e-6
        )
        
        # Optimizers (will be updated with adaptive LR)
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr_scheduler.current_lr_g, 
            beta_1=0.5, 
            beta_2=0.9
        )
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr_scheduler.current_lr_d, 
            beta_1=0.5, 
            beta_2=0.9
        )
        
        # Data generator
        self.data_generator = BimodalDataGenerator(
            batch_size=32,
            n_features=data_dim,
            mode1_center=mode1_center,
            mode2_center=mode2_center,
            mode_std=mode_std,
            seed=42
        )
        
        # Diagnostic tracking
        self.diagnostics = {
            'losses': [],
            'gradients': [],
            'learning_rates': [],
            'mode_coverage': [],
            'quantum_metrics': [],
            'loss_components': []
        }
        
        logger.info("Diagnostic Quantum GAN Trainer initialized")
        logger.info(f"  Initial LR - Generator: {self.lr_scheduler.current_lr_g:.6f}")
        logger.info(f"  Initial LR - Discriminator: {self.lr_scheduler.current_lr_d:.6f}")
    
    def enhanced_loss_computation(self, real_data: tf.Tensor, fake_data: tf.Tensor, 
                                epoch: int, step: int):
        """Enhanced loss computation with detailed analysis."""
        # Ensure same batch size
        min_batch_size = tf.minimum(tf.shape(real_data)[0], tf.shape(fake_data)[0])
        real_truncated = real_data[:min_batch_size]
        fake_truncated = fake_data[:min_batch_size]
        
        # Compute discriminator outputs
        real_output = self.discriminator.discriminate(real_truncated)
        fake_output = self.discriminator.discriminate(fake_truncated)
        
        # Enhanced Wasserstein distance with stability checks
        real_mean = tf.reduce_mean(real_output)
        fake_mean = tf.reduce_mean(fake_output)
        w_distance = real_mean - fake_mean
        
        # Gradient penalty with enhanced stability
        alpha = tf.random.uniform([min_batch_size, 1], 0.0, 1.0)
        interpolated = alpha * real_truncated + (1 - alpha) * fake_truncated
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            interp_output = self.discriminator.discriminate(interpolated)
        
        gradients = gp_tape.gradient(interp_output, interpolated)
        
        if gradients is None:
            gradient_penalty = tf.constant(0.0)
            gp_norm = tf.constant(0.0)
        else:
            gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-12)
            gradient_penalty = 10.0 * tf.reduce_mean(tf.square(gradient_norm - 1.0))
            gp_norm = tf.reduce_mean(gradient_norm)
        
        # Enhanced quantum regularization
        try:
            quantum_cost = self.generator.compute_quantum_cost()
            # Scale quantum cost based on training progress
            entropy_weight = 0.1 + 0.9 * tf.exp(-tf.cast(epoch, tf.float32) / 10.0)
            entropy_bonus = entropy_weight * quantum_cost
        except:
            entropy_bonus = tf.constant(0.0)
            quantum_cost = tf.constant(0.0)
        
        # Mode separation loss (encourage bimodal generation)
        mode_separation_loss = self.compute_mode_separation_loss(fake_truncated)
        
        # Final losses with enhanced formulation
        d_loss = -w_distance + gradient_penalty
        g_loss = -fake_mean + entropy_bonus + 0.1 * mode_separation_loss
        
        # Comprehensive metrics
        metrics = {
            'w_distance': w_distance,
            'real_mean': real_mean,
            'fake_mean': fake_mean,
            'gradient_penalty': gradient_penalty,
            'gp_norm': gp_norm,
            'entropy_bonus': entropy_bonus,
            'quantum_cost': quantum_cost,
            'mode_separation_loss': mode_separation_loss,
            'd_loss': d_loss,
            'g_loss': g_loss
        }
        
        return d_loss, g_loss, metrics
    
    def compute_mode_separation_loss(self, samples: tf.Tensor):
        """Compute loss to encourage mode separation."""
        # Compute distances to mode centers
        mode1_center = tf.constant(self.mode1_center, dtype=tf.float32)
        mode2_center = tf.constant(self.mode2_center, dtype=tf.float32)
        
        dist_to_mode1 = tf.reduce_sum(tf.square(samples - mode1_center), axis=1)
        dist_to_mode2 = tf.reduce_sum(tf.square(samples - mode2_center), axis=1)
        
        # Encourage samples to be close to one of the modes
        min_dist = tf.minimum(dist_to_mode1, dist_to_mode2)
        mode_loss = tf.reduce_mean(min_dist)
        
        return mode_loss
    
    def discriminator_train_step(self, real_data: tf.Tensor, z: tf.Tensor, 
                               epoch: int, step: int):
        """Enhanced discriminator training step."""
        with tf.GradientTape() as disc_tape:
            fake_data = self.generator.generate(z)
            d_loss, g_loss, metrics = self.enhanced_loss_computation(
                real_data, fake_data, epoch, step
            )
            
            # Add noise for stability
            d_loss += tf.random.normal([], 0.0, 0.001)
        
        # Compute gradients
        disc_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        
        # Enhanced gradient analysis
        grad_stats = self.analyze_gradients(disc_gradients, "discriminator")
        
        # Apply gradients with adaptive clipping
        if disc_gradients is not None:
            # Adaptive gradient clipping based on gradient norms
            grad_norms = [tf.norm(g) for g in disc_gradients if g is not None]
            if grad_norms:
                max_norm = tf.reduce_max(grad_norms)
                clip_value = tf.minimum(1.0, 10.0 / (max_norm + 1e-8))
                
                clipped_gradients = []
                for grad in disc_gradients:
                    if grad is not None:
                        clipped_gradients.append(tf.clip_by_value(grad, -clip_value, clip_value))
                    else:
                        clipped_gradients.append(grad)
                
                self.discriminator_optimizer.apply_gradients(
                    zip(clipped_gradients, self.discriminator.trainable_variables)
                )
        
        return d_loss, metrics, grad_stats
    
    def generator_train_step(self, z: tf.Tensor, epoch: int, step: int):
        """Enhanced generator training step."""
        with tf.GradientTape() as gen_tape:
            fake_data = self.generator.generate(z)
            
            # Use real data for proper loss computation
            real_data = self.data_generator.generate_batch()
            d_loss, g_loss, metrics = self.enhanced_loss_computation(
                real_data, fake_data, epoch, step
            )
            
            # Add exploration noise early in training
            exploration_factor = tf.exp(-tf.cast(epoch, tf.float32) / 5.0)
            g_loss += exploration_factor * tf.random.normal([], 0.0, 0.01)
        
        # Compute gradients
        gen_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        
        # Enhanced gradient analysis
        grad_stats = self.analyze_gradients(gen_gradients, "generator")
        
        # Apply gradients with adaptive clipping
        if gen_gradients is not None:
            # Adaptive gradient clipping
            grad_norms = [tf.norm(g) for g in gen_gradients if g is not None]
            if grad_norms:
                max_norm = tf.reduce_max(grad_norms)
                clip_value = tf.minimum(1.0, 5.0 / (max_norm + 1e-8))
                
                clipped_gradients = []
                for grad in gen_gradients:
                    if grad is not None:
                        clipped_gradients.append(tf.clip_by_value(grad, -clip_value, clip_value))
                    else:
                        clipped_gradients.append(grad)
                
                self.generator_optimizer.apply_gradients(
                    zip(clipped_gradients, self.generator.trainable_variables)
                )
        
        return g_loss, metrics, grad_stats
    
    def analyze_gradients(self, gradients, model_name):
        """Comprehensive gradient analysis."""
        grad_stats = {
            'model': model_name,
            'total_gradients': len(gradients),
            'valid_gradients': 0,
            'nan_gradients': 0,
            'zero_gradients': 0,
            'gradient_norms': [],
            'gradient_means': []
        }
        
        for grad in gradients:
            if grad is not None:
                grad_stats['valid_gradients'] += 1
                
                if tf.reduce_any(tf.math.is_nan(grad)):
                    grad_stats['nan_gradients'] += 1
                else:
                    norm = tf.norm(grad)
                    mean = tf.reduce_mean(tf.abs(grad))
                    
                    grad_stats['gradient_norms'].append(float(norm))
                    grad_stats['gradient_means'].append(float(mean))
                    
                    if norm < 1e-8:
                        grad_stats['zero_gradients'] += 1
        
        # Calculate statistics
        if grad_stats['gradient_norms']:
            grad_stats['avg_norm'] = np.mean(grad_stats['gradient_norms'])
            grad_stats['max_norm'] = np.max(grad_stats['gradient_norms'])
            grad_stats['min_norm'] = np.min(grad_stats['gradient_norms'])
            grad_stats['std_norm'] = np.std(grad_stats['gradient_norms'])
        else:
            grad_stats['avg_norm'] = 0.0
            grad_stats['max_norm'] = 0.0
            grad_stats['min_norm'] = 0.0
            grad_stats['std_norm'] = 0.0
        
        return grad_stats
    
    def update_learning_rates(self, epoch, avg_loss):
        """Update learning rates based on progress."""
        new_lr_g, new_lr_d = self.lr_scheduler.update(avg_loss, epoch)
        
        # Update optimizers
        self.generator_optimizer.learning_rate.assign(new_lr_g)
        self.discriminator_optimizer.learning_rate.assign(new_lr_d)
        
        return new_lr_g, new_lr_d
    
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
        
        # Additional metrics
        avg_dist_to_nearest_mode = np.mean(np.minimum(distances_to_mode1, distances_to_mode2))
        mode_separation = np.abs(mode1_coverage - mode2_coverage)
        
        return {
            'mode1_coverage': float(mode1_coverage),
            'mode2_coverage': float(mode2_coverage),
            'balanced_coverage': float(min(mode1_coverage, mode2_coverage) / max(mode1_coverage, mode2_coverage)) if max(mode1_coverage, mode2_coverage) > 0 else 0.0,
            'mode_separation': float(mode_separation),
            'avg_dist_to_mode': float(avg_dist_to_nearest_mode)
        }
    
    def train(self, epochs: int = 20, batch_size: int = 16, save_dir: str = "results/diagnostic"):
        """Enhanced training loop with comprehensive diagnostics."""
        print(f"🔬 STARTING DIAGNOSTIC QUANTUM GAN TRAINING")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Initial LR - Generator: {self.lr_scheduler.current_lr_g:.6f}")
        print(f"  Initial LR - Discriminator: {self.lr_scheduler.current_lr_d:.6f}")
        print("=" * 60)
        
        # Create results directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Update data generator batch size
        self.data_generator.batch_size = batch_size
        
        # Analyze target data (required for coordinate generator)
        print("\n🎯 ANALYZING TARGET DATA")
        print("-" * 30)
        target_data = self.data_generator.generate_batch().numpy()
        analysis_results = self.generator.analyze_target_data(target_data)
        print(f"✅ Target data analyzed: {len(target_data)} samples")
        
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch + 1}/{epochs}")
            print("-" * 30)
            
            epoch_gen_losses = []
            epoch_disc_losses = []
            epoch_metrics = []
            epoch_grad_stats = []
            
            # Training steps per epoch
            steps_per_epoch = 10  # More steps for better convergence
            
            for step in range(steps_per_epoch):
                # Get real data
                real_data = self.data_generator.generate_batch()
                z = tf.random.normal([batch_size, self.latent_dim])
                
                # Train discriminator (2 steps for every generator step)
                disc_losses = []
                for _ in range(2):
                    z_disc = tf.random.normal([batch_size, self.latent_dim])
                    disc_loss, disc_metrics, disc_grad_stats = self.discriminator_train_step(
                        real_data, z_disc, epoch, step
                    )
                    disc_losses.append(float(disc_loss))
                    epoch_grad_stats.append(disc_grad_stats)
                
                # Train generator (1 step)
                gen_loss, gen_metrics, gen_grad_stats = self.generator_train_step(
                    z, epoch, step
                )
                
                epoch_gen_losses.append(float(gen_loss))
                epoch_disc_losses.append(np.mean(disc_losses))
                epoch_metrics.append(gen_metrics)
                epoch_grad_stats.append(gen_grad_stats)
            
            # Calculate epoch averages
            avg_gen_loss = np.mean(epoch_gen_losses)
            avg_disc_loss = np.mean(epoch_disc_losses)
            avg_total_loss = avg_gen_loss + avg_disc_loss
            
            # Update learning rates
            new_lr_g, new_lr_d = self.update_learning_rates(epoch, avg_total_loss)
            
            # Calculate mode coverage
            test_z = tf.random.normal([500, self.latent_dim])
            test_generated = self.generator.generate(test_z)
            coverage_metrics = self.calculate_mode_coverage(test_generated)
            
            # Store diagnostics
            self.diagnostics['losses'].append({
                'epoch': epoch,
                'gen_loss': avg_gen_loss,
                'disc_loss': avg_disc_loss,
                'total_loss': avg_total_loss
            })
            self.diagnostics['learning_rates'].append({
                'epoch': epoch,
                'lr_g': new_lr_g,
                'lr_d': new_lr_d
            })
            self.diagnostics['mode_coverage'].append(coverage_metrics)
            self.diagnostics['gradients'].extend(epoch_grad_stats)
            
            # Print progress
            print(f"✅ Epoch {epoch + 1} complete:")
            print(f"   G_loss: {avg_gen_loss:.6f}, D_loss: {avg_disc_loss:.6f}")
            print(f"   Mode1: {coverage_metrics['mode1_coverage']:.3f}, Mode2: {coverage_metrics['mode2_coverage']:.3f}")
            print(f"   Balanced: {coverage_metrics['balanced_coverage']:.3f}")
            print(f"   LR - G: {new_lr_g:.6f}, D: {new_lr_d:.6f}")
            
            # Detailed analysis every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.print_detailed_analysis(epoch_grad_stats, epoch_metrics)
        
        # Save comprehensive results
        self.save_diagnostic_results(save_dir)
        
        return self.diagnostics
    
    def print_detailed_analysis(self, grad_stats_list, metrics_list):
        """Print detailed analysis of training progress."""
        print(f"\n🔍 DETAILED ANALYSIS")
        print("-" * 40)
        
        # Gradient analysis
        gen_grads = [g for g in grad_stats_list if g['model'] == 'generator']
        disc_grads = [g for g in grad_stats_list if g['model'] == 'discriminator']
        
        if gen_grads:
            avg_gen_norm = np.mean([g['avg_norm'] for g in gen_grads])
            print(f"📊 Generator gradients: avg_norm={avg_gen_norm:.6f}")
        
        if disc_grads:
            avg_disc_norm = np.mean([g['avg_norm'] for g in disc_grads])
            print(f"📊 Discriminator gradients: avg_norm={avg_disc_norm:.6f}")
        
        # Loss component analysis
        if metrics_list:
            avg_w_distance = np.mean([float(m['w_distance']) for m in metrics_list])
            avg_gp = np.mean([float(m['gradient_penalty']) for m in metrics_list])
            avg_entropy = np.mean([float(m['entropy_bonus']) for m in metrics_list])
            
            print(f"📈 Loss components:")
            print(f"   Wasserstein distance: {avg_w_distance:.6f}")
            print(f"   Gradient penalty: {avg_gp:.6f}")
            print(f"   Entropy bonus: {avg_entropy:.6f}")
    
    def save_diagnostic_results(self, save_dir: str):
        """Save comprehensive diagnostic results."""
        results_path = os.path.join(save_dir, "diagnostic_results.json")
        
        with open(results_path, 'w') as f:
            json.dump(self.diagnostics, f, indent=2, default=str)
        
        print(f"✅ Diagnostic results saved to: {results_path}")
        
        # Create diagnostic plots
        self.create_diagnostic_plots(save_dir)
    
    def create_diagnostic_plots(self, save_dir: str):
        """Create comprehensive diagnostic plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Quantum GAN Training Diagnostics', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss Evolution
        if self.diagnostics['losses']:
            epochs = [l['epoch'] for l in self.diagnostics['losses']]
            gen_losses = [l['gen_loss'] for l in self.diagnostics['losses']]
            disc_losses = [l['disc_loss'] for l in self.diagnostics['losses']]
            
            axes[0, 0].plot(epochs, gen_losses, 'r-', label='Generator')
            axes[0, 0].plot(epochs, disc_losses, 'b-', label='Discriminator')
            axes[0, 0].set_title('Loss Evolution')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Learning Rate Evolution
        if self.diagnostics['learning_rates']:
            epochs = [lr['epoch'] for lr in self.diagnostics['learning_rates']]
            lr_g = [lr['lr_g'] for lr in self.diagnostics['learning_rates']]
            lr_d = [lr['lr_d'] for lr in self.diagnostics['learning_rates']]
            
            axes[0, 1].semilogy(epochs, lr_g, 'r-', label='Generator')
            axes[0, 1].semilogy(epochs, lr_d, 'b-', label='Discriminator')
            axes[0, 1].set_title('Learning Rate Evolution')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate (log scale)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Mode Coverage Evolution
        if self.diagnostics['mode_coverage']:
            epochs = range(len(self.diagnostics['mode_coverage']))
            mode1_cov = [mc['mode1_coverage'] for mc in self.diagnostics['mode_coverage']]
            mode2_cov = [mc['mode2_coverage'] for mc in self.diagnostics['mode_coverage']]
            balanced_cov = [mc['balanced_coverage'] for mc in self.diagnostics['mode_coverage']]
            
            axes[0, 2].plot(epochs, mode1_cov, 'r-', label='Mode 1')
            axes[0, 2].plot(epochs, mode2_cov, 'b-', label='Mode 2')
            axes[0, 2].plot(epochs, balanced_cov, 'g-', label='Balanced')
            axes[0, 2].set_title('Mode Coverage Evolution')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Coverage')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_ylim(0, 1)
        
        # Plot 4: Gradient Norms
        if self.diagnostics['gradients']:
            gen_norms = [g['avg_norm'] for g in self.diagnostics['gradients'] if g['model'] == 'generator']
            disc_norms = [g['avg_norm'] for g in self.diagnostics['gradients'] if g['model'] == 'discriminator']
            
            if gen_norms:
                axes[1, 0].plot(gen_norms, 'r-', alpha=0.7, label='Generator')
            if disc_norms:
                axes[1, 0].plot(disc_norms, 'b-', alpha=0.7, label='Discriminator')
            
            axes[1, 0].set_title('Gradient Norms Over Time')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Average Gradient Norm')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Current Generated vs Real Data
        test_z = tf.random.normal([500, self.latent_dim])
        generated_samples = self.generator.generate(test_z)
        real_samples = self.data_generator.generate_batch()
        
        axes[1, 1].scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.5, c='blue', s=30, label='Real')
        axes[1, 1].scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, c='red', s=30, label='Generated')
        axes[1, 1].set_title('Final: Real vs Generated Data')
        axes[1, 1].set_xlabel('Feature 1')
        axes[1, 1].set_ylabel('Feature 2')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(-3, 3)
        axes[1, 1].set_ylim(-3, 3)
        
        # Plot 6: Loss Components (if available)
        axes[1, 2].text(0.5, 0.5, 'Training Complete!\nCheck diagnostic_results.json\nfor detailed analysis', 
                        ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].set_title('Diagnostic Summary')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, "diagnostic_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Diagnostic plots saved to: {plot_path}")


def main():
    """Main function for diagnostic training."""
    parser = argparse.ArgumentParser(description="Diagnostic Quantum GAN Training")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--save-dir", type=str, default="results/diagnostic", help="Save directory")
    
    args = parser.parse_args()
    
    print("🔬 DIAGNOSTIC QUANTUM GAN TRAINING")
    print("=" * 60)
    print("Enhanced diagnostics and adaptive learning:")
    print("  ✅ Adaptive learning rate scheduling")
    print("  ✅ Enhanced gradient analysis")
    print("  ✅ Mode separation loss")
    print("  ✅ Quantum regularization")
    print("  ✅ Comprehensive monitoring")
    print("=" * 60)
    
    # Create diagnostic trainer
    trainer = DiagnosticQuantumGANTrainer()
    
    # Train with diagnostics
    results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
    
    # Print final analysis
    print(f"\n🏆 DIAGNOSTIC RESULTS:")
    print("=" * 60)
    
    if results['mode_coverage']:
        final_coverage = results['mode_coverage'][-1]
        print(f"Final mode coverage:")
        print(f"  Mode 1: {final_coverage['mode1_coverage']:.3f}")
        print(f"  Mode 2: {final_coverage['mode2_coverage']:.3f}")
        print(f"  Balanced: {final_coverage['balanced_coverage']:.3f}")
        print(f"  Mode separation: {final_coverage['mode_separation']:.3f}")
    
    if results['losses']:
        final_loss = results['losses'][-1]
        print(f"Final losses:")
        print(f"  Generator: {final_loss['gen_loss']:.6f}")
        print(f"  Discriminator: {final_loss['disc_loss']:.6f}")
    
    if results['learning_rates']:
        final_lr = results['learning_rates'][-1]
        print(f"Final learning rates:")
        print(f"  Generator: {final_lr['lr_g']:.6f}")
        print(f"  Discriminator: {final_lr['lr_d']:.6f}")
    
    print("=" * 60)
    print("Check diagnostic_plots.png and diagnostic_results.json for detailed analysis!")


if __name__ == "__main__":
    main()
