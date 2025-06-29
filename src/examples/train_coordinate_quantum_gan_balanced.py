"""
Balanced Quantum GAN Training - Fixed Discriminator Gradient Collapse

This script addresses the discriminator gradient collapse issue identified in diagnostics:
- Balanced discriminator/generator learning rates
- Improved training ratios
- Discriminator regularization
- Enhanced mode separation targeting
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

# Suppress warnings
suppress_all_quantum_warnings()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BalancedLearningRateScheduler:
    """Balanced learning rate scheduler that prevents discriminator domination."""
    
    def __init__(self, initial_lr_g=0.0003, initial_lr_d=0.00003, 
                 patience=5, factor=0.8, min_lr=1e-7):
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
        
        # Track discriminator gradient health
        self.disc_grad_history = []
        
    def update(self, current_loss, epoch, disc_grad_norm=None, gen_grad_norm=None):
        """Update learning rates with discriminator gradient monitoring."""
        self.lr_history.append({
            'epoch': epoch,
            'lr_g': self.current_lr_g,
            'lr_d': self.current_lr_d,
            'loss': current_loss,
            'disc_grad_norm': disc_grad_norm,
            'gen_grad_norm': gen_grad_norm
        })
        
        # Check for discriminator gradient collapse
        if disc_grad_norm is not None and disc_grad_norm < 1e-6:
            # Increase discriminator learning rate to revive gradients
            old_lr_d = self.current_lr_d
            self.current_lr_d = min(self.current_lr_d * 1.5, self.initial_lr_d * 2)
            if self.current_lr_d > old_lr_d:
                print(f"  📈 Discriminator LR increased to revive gradients: {old_lr_d:.6f}→{self.current_lr_d:.6f}")
        
        # Standard loss-based adaptation
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            # Reduce learning rates
            old_lr_g = self.current_lr_g
            old_lr_d = self.current_lr_d
            
            self.current_lr_g = max(self.current_lr_g * self.factor, self.min_lr)
            self.current_lr_d = max(self.current_lr_d * self.factor, self.min_lr)
            
            if self.current_lr_g < old_lr_g:
                print(f"  📉 Learning rates reduced: G={old_lr_g:.6f}→{self.current_lr_g:.6f}, "
                      f"D={old_lr_d:.6f}→{self.current_lr_d:.6f}")
            
            self.wait = 0
            
        return self.current_lr_g, self.current_lr_d


class BalancedQuantumGANTrainer:
    """Balanced trainer that prevents discriminator gradient collapse."""
    
    def __init__(self, 
                 latent_dim: int = 2,
                 data_dim: int = 2,
                 n_modes: int = 4,
                 mode1_center: tuple = (-1.5, -1.5),
                 mode2_center: tuple = (1.5, 1.5),
                 mode_std: float = 0.3):
        """Initialize balanced trainer."""
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
        
        # Balanced learning rate scheduler
        self.lr_scheduler = BalancedLearningRateScheduler(
            initial_lr_g=0.0003,  # Slightly reduced from diagnostic
            initial_lr_d=0.00003,  # Much lower to prevent domination (10x lower)
            patience=4,
            factor=0.8,
            min_lr=1e-7
        )
        
        # Optimizers with balanced settings
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
        
        # Training tracking
        self.training_history = {
            'losses': [],
            'gradients': [],
            'learning_rates': [],
            'mode_coverage': [],
            'balance_metrics': []
        }
        
        logger.info("Balanced Quantum GAN Trainer initialized")
        logger.info(f"  Initial LR - Generator: {self.lr_scheduler.current_lr_g:.6f}")
        logger.info(f"  Initial LR - Discriminator: {self.lr_scheduler.current_lr_d:.6f}")
        logger.info(f"  LR Ratio (G/D): {self.lr_scheduler.current_lr_g/self.lr_scheduler.current_lr_d:.1f}:1")
    
    def balanced_loss_computation(self, real_data: tf.Tensor, fake_data: tf.Tensor, 
                                epoch: int, step: int):
        """Balanced loss computation that prevents discriminator domination."""
        # Ensure same batch size
        min_batch_size = tf.minimum(tf.shape(real_data)[0], tf.shape(fake_data)[0])
        real_truncated = real_data[:min_batch_size]
        fake_truncated = fake_data[:min_batch_size]
        
        # Compute discriminator outputs
        real_output = self.discriminator.discriminate(real_truncated)
        fake_output = self.discriminator.discriminate(fake_truncated)
        
        # Balanced Wasserstein distance
        real_mean = tf.reduce_mean(real_output)
        fake_mean = tf.reduce_mean(fake_output)
        w_distance = real_mean - fake_mean
        
        # Softer gradient penalty to prevent discriminator over-optimization
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
            # Reduced gradient penalty weight to prevent discriminator domination
            gradient_penalty = 5.0 * tf.reduce_mean(tf.square(gradient_norm - 1.0))
            gp_norm = tf.reduce_mean(gradient_norm)
        
        # Enhanced quantum regularization
        try:
            quantum_cost = self.generator.compute_quantum_cost()
            # Adaptive entropy weight
            entropy_weight = 0.05 + 0.15 * tf.exp(-tf.cast(epoch, tf.float32) / 8.0)
            entropy_bonus = entropy_weight * quantum_cost
        except:
            entropy_bonus = tf.constant(0.0)
            quantum_cost = tf.constant(0.0)
        
        # Improved mode separation loss with actual target centers
        mode_separation_loss = self.compute_targeted_mode_loss(fake_truncated)
        
        # Discriminator regularization to prevent over-optimization
        disc_reg = 0.001 * (tf.reduce_mean(tf.square(real_output)) + tf.reduce_mean(tf.square(fake_output)))
        
        # Balanced losses
        d_loss = -w_distance + gradient_penalty + disc_reg
        g_loss = -fake_mean + entropy_bonus + 0.2 * mode_separation_loss
        
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
            'disc_reg': disc_reg,
            'd_loss': d_loss,
            'g_loss': g_loss
        }
        
        return d_loss, g_loss, metrics
    
    def compute_targeted_mode_loss(self, samples: tf.Tensor):
        """Compute loss that specifically targets the actual mode centers."""
        # Use actual mode centers from data generator
        mode1_center = tf.constant(self.mode1_center, dtype=tf.float32)
        mode2_center = tf.constant(self.mode2_center, dtype=tf.float32)
        
        # Compute distances to each mode
        dist_to_mode1 = tf.reduce_sum(tf.square(samples - mode1_center), axis=1)
        dist_to_mode2 = tf.reduce_sum(tf.square(samples - mode2_center), axis=1)
        
        # Encourage samples to be close to one of the modes
        min_dist = tf.minimum(dist_to_mode1, dist_to_mode2)
        
        # Also encourage balanced assignment to both modes
        prob_mode1 = tf.nn.sigmoid(-dist_to_mode1 + dist_to_mode2)
        prob_mode2 = 1.0 - prob_mode1
        
        # Balance loss - encourage 50/50 split
        avg_prob_mode1 = tf.reduce_mean(prob_mode1)
        balance_loss = tf.square(avg_prob_mode1 - 0.5)
        
        # Combined mode loss
        proximity_loss = tf.reduce_mean(min_dist)
        total_mode_loss = proximity_loss + 2.0 * balance_loss
        
        return total_mode_loss
    
    def discriminator_train_step(self, real_data: tf.Tensor, z: tf.Tensor, 
                               epoch: int, step: int):
        """Balanced discriminator training step."""
        with tf.GradientTape() as disc_tape:
            fake_data = self.generator.generate(z)
            d_loss, g_loss, metrics = self.balanced_loss_computation(
                real_data, fake_data, epoch, step
            )
            
            # Add small noise for stability (reduced)
            d_loss += tf.random.normal([], 0.0, 0.0005)
        
        # Compute gradients
        disc_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        
        # Analyze gradients
        grad_stats = self.analyze_gradients(disc_gradients, "discriminator")
        
        # Gentle gradient clipping to preserve learning signal
        if disc_gradients is not None:
            grad_norms = [tf.norm(g) for g in disc_gradients if g is not None]
            if grad_norms:
                max_norm = tf.reduce_max(grad_norms)
                # More conservative clipping
                clip_value = tf.minimum(0.5, 5.0 / (max_norm + 1e-8))
                
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
            d_loss, g_loss, metrics = self.balanced_loss_computation(
                real_data, fake_data, epoch, step
            )
            
            # Reduced exploration noise
            exploration_factor = tf.exp(-tf.cast(epoch, tf.float32) / 8.0)
            g_loss += exploration_factor * tf.random.normal([], 0.0, 0.005)
        
        # Compute gradients
        gen_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        
        # Analyze gradients
        grad_stats = self.analyze_gradients(gen_gradients, "generator")
        
        # Conservative gradient clipping for generator
        if gen_gradients is not None:
            grad_norms = [tf.norm(g) for g in gen_gradients if g is not None]
            if grad_norms:
                max_norm = tf.reduce_max(grad_norms)
                clip_value = tf.minimum(1.0, 8.0 / (max_norm + 1e-8))
                
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
        """Analyze gradient health."""
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
        else:
            grad_stats['avg_norm'] = 0.0
            grad_stats['max_norm'] = 0.0
            grad_stats['min_norm'] = 0.0
        
        return grad_stats
    
    def update_learning_rates(self, epoch, avg_loss, disc_grad_norm, gen_grad_norm):
        """Update learning rates with gradient health monitoring."""
        new_lr_g, new_lr_d = self.lr_scheduler.update(
            avg_loss, epoch, disc_grad_norm, gen_grad_norm
        )
        
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
    
    def train(self, epochs: int = 15, batch_size: int = 16, save_dir: str = "results/balanced"):
        """Balanced training loop."""
        print(f"⚖️ STARTING BALANCED QUANTUM GAN TRAINING")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Generator LR: {self.lr_scheduler.current_lr_g:.6f}")
        print(f"  Discriminator LR: {self.lr_scheduler.current_lr_d:.6f}")
        print(f"  LR Ratio (G/D): {self.lr_scheduler.current_lr_g/self.lr_scheduler.current_lr_d:.1f}:1")
        print("=" * 60)
        
        # Create results directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Update data generator batch size
        self.data_generator.batch_size = batch_size
        
        # Analyze target data
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
            
            # Balanced training: 1:1 ratio to prevent discriminator domination
            steps_per_epoch = 8
            
            for step in range(steps_per_epoch):
                # Get real data
                real_data = self.data_generator.generate_batch()
                z = tf.random.normal([batch_size, self.latent_dim])
                
                # Train discriminator (1 step)
                z_disc = tf.random.normal([batch_size, self.latent_dim])
                disc_loss, disc_metrics, disc_grad_stats = self.discriminator_train_step(
                    real_data, z_disc, epoch, step
                )
                epoch_grad_stats.append(disc_grad_stats)
                
                # Train generator (1 step) - balanced ratio
                gen_loss, gen_metrics, gen_grad_stats = self.generator_train_step(
                    z, epoch, step
                )
                
                epoch_gen_losses.append(float(gen_loss))
                epoch_disc_losses.append(float(disc_loss))
                epoch_metrics.append(gen_metrics)
                epoch_grad_stats.append(gen_grad_stats)
            
            # Calculate epoch averages
            avg_gen_loss = np.mean(epoch_gen_losses)
            avg_disc_loss = np.mean(epoch_disc_losses)
            avg_total_loss = avg_gen_loss + avg_disc_loss
            
            # Calculate gradient health
            disc_grads = [g for g in epoch_grad_stats if g['model'] == 'discriminator']
            gen_grads = [g for g in epoch_grad_stats if g['model'] == 'generator']
            
            avg_disc_grad_norm = np.mean([g['avg_norm'] for g in disc_grads]) if disc_grads else 0.0
            avg_gen_grad_norm = np.mean([g['avg_norm'] for g in gen_grads]) if gen_grads else 0.0
            
            # Update learning rates with gradient monitoring
            new_lr_g, new_lr_d = self.update_learning_rates(
                epoch, avg_total_loss, avg_disc_grad_norm, avg_gen_grad_norm
            )
            
            # Calculate mode coverage
            test_z = tf.random.normal([500, self.latent_dim])
            test_generated = self.generator.generate(test_z)
            coverage_metrics = self.calculate_mode_coverage(test_generated)
            
            # Store training history
            self.training_history['losses'].append({
                'epoch': epoch,
                'gen_loss': avg_gen_loss,
                'disc_loss': avg_disc_loss,
                'total_loss': avg_total_loss
            })
            self.training_history['learning_rates'].append({
                'epoch': epoch,
                'lr_g': new_lr_g,
                'lr_d': new_lr_d
            })
            self.training_history['mode_coverage'].append(coverage_metrics)
            self.training_history['gradients'].extend(epoch_grad_stats)
            self.training_history['balance_metrics'].append({
                'epoch': epoch,
                'disc_grad_norm': avg_disc_grad_norm,
                'gen_grad_norm': avg_gen_grad_norm,
                'lr_ratio': new_lr_g / new_lr_d if new_lr_d > 0 else 0
            })
            
            # Print progress with gradient health
            print(f"✅ Epoch {epoch + 1} complete:")
            print(f"   G_loss: {avg_gen_loss:.6f}, D_loss: {avg_disc_loss:.6f}")
            print(f"   Mode1: {coverage_metrics['mode1_coverage']:.3f}, Mode2: {coverage_metrics['mode2_coverage']:.3f}")
            print(f"   Balanced: {coverage_metrics['balanced_coverage']:.3f}")
            print(f"   LR - G: {new_lr_g:.6f}, D: {new_lr_d:.6f}")
            print(f"   Grad norms - G: {avg_gen_grad_norm:.6f}, D: {avg_disc_grad_norm:.6f}")
            
            # Detailed analysis every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.print_balance_analysis(epoch_grad_stats, epoch_metrics)
        
        # Save results
        self.save_results(save_dir)
        
        return self.training_history
    
    def print_balance_analysis(self, grad_stats_list, metrics_list):
        """Print balance analysis."""
        print(f"\n⚖️ BALANCE ANALYSIS")
        print("-" * 40)
        
        # Gradient health
        gen_grads = [g for g in grad_stats_list if g['model'] == 'generator']
        disc_grads = [g for g in grad_stats_list if g['model'] == 'discriminator']
        
        if gen_grads and disc_grads:
            avg_gen_norm = np.mean([g['avg_norm'] for g in gen_grads])
            avg_disc_norm = np.mean([g['avg_norm'] for g in disc_grads])
            
            print(f"📊 Gradient Health:")
            print(f"   Generator norm: {avg_gen_norm:.6f}")
            print(f"   Discriminator norm: {avg_disc_norm:.6f}")
            print(f"   Ratio (G/D): {avg_gen_norm/avg_disc_norm:.2f}" if avg_disc_norm > 0 else "   Ratio: inf")
            
            if avg_disc_norm < 1e-6:
                print("   ⚠️ Discriminator gradient collapse detected!")
            elif avg_disc_norm > avg_gen_norm * 10:
                print("   ⚠️ Discriminator domination detected!")
            else:
                print("   ✅ Balanced gradient flow")
        
        # Loss components
        if metrics_list:
            avg_w_distance = np.mean([float(m['w_distance']) for m in metrics_list])
            avg_gp = np.mean([float(m['gradient_penalty']) for m in metrics_list])
            avg_mode_loss = np.mean([float(m['mode_separation_loss']) for m in metrics_list])
            
            print(f"📈 Loss Components:")
            print(f"   Wasserstein distance: {avg_w_distance:.6f}")
            print(f"   Gradient penalty: {avg_gp:.6f}")
            print(f"   Mode separation: {avg_mode_loss:.6f}")
    
    def save_results(self, save_dir: str):
        """Save training results."""
        results_path = os.path.join(save_dir, "balanced_training_results.json")
        
        with open(results_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        print(f"✅ Results saved to: {results_path}")
        
        # Create visualization
        self.create_balance_plots(save_dir)
    
    def create_balance_plots(self, save_dir: str):
        """Create balance analysis plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Balanced Quantum GAN Training Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss Evolution
        if self.training_history['losses']:
            epochs = [l['epoch'] for l in self.training_history['losses']]
            gen_losses = [l['gen_loss'] for l in self.training_history['losses']]
            disc_losses = [l['disc_loss'] for l in self.training_history['losses']]
            
            axes[0, 0].plot(epochs, gen_losses, 'r-', label='Generator')
            axes[0, 0].plot(epochs, disc_losses, 'b-', label='Discriminator')
            axes[0, 0].set_title('Loss Evolution')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Gradient Health
        if self.training_history['balance_metrics']:
            epochs = [bm['epoch'] for bm in self.training_history['balance_metrics']]
            gen_grad_norms = [bm['gen_grad_norm'] for bm in self.training_history['balance_metrics']]
            disc_grad_norms = [bm['disc_grad_norm'] for bm in self.training_history['balance_metrics']]
            
            axes[0, 1].semilogy(epochs, gen_grad_norms, 'r-', label='Generator')
            axes[0, 1].semilogy(epochs, disc_grad_norms, 'b-', label='Discriminator')
            axes[0, 1].set_title('Gradient Norms (Log Scale)')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Gradient Norm')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Mode Coverage Evolution
        if self.training_history['mode_coverage']:
            epochs = range(len(self.training_history['mode_coverage']))
            mode1_cov = [mc['mode1_coverage'] for mc in self.training_history['mode_coverage']]
            mode2_cov = [mc['mode2_coverage'] for mc in self.training_history['mode_coverage']]
            balanced_cov = [mc['balanced_coverage'] for mc in self.training_history['mode_coverage']]
            
            axes[0, 2].plot(epochs, mode1_cov, 'r-', label='Mode 1')
            axes[0, 2].plot(epochs, mode2_cov, 'b-', label='Mode 2')
            axes[0, 2].plot(epochs, balanced_cov, 'g-', label='Balanced')
            axes[0, 2].set_title('Mode Coverage Evolution')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Coverage')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_ylim(0, 1)
        
        # Plot 4: Learning Rate Ratio
        if self.training_history['balance_metrics']:
            epochs = [bm['epoch'] for bm in self.training_history['balance_metrics']]
            lr_ratios = [bm['lr_ratio'] for bm in self.training_history['balance_metrics']]
            
            axes[1, 0].plot(epochs, lr_ratios, 'g-', label='LR Ratio (G/D)')
            axes[1, 0].set_title('Learning Rate Ratio Evolution')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Ratio')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Final Generated vs Real Data
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
        
        # Plot 6: Balance Summary
        axes[1, 2].text(0.5, 0.5, 'Balanced Training Complete!\n\nKey Improvements:\n• 10:1 LR ratio (G:D)\n• 1:1 training ratio\n• Gradient health monitoring\n• Discriminator regularization', 
                        ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=11)
        axes[1, 2].set_title('Balance Summary')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, "balanced_training_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Balance plots saved to: {plot_path}")


def main():
    """Main function for balanced training."""
    parser = argparse.ArgumentParser(description="Balanced Quantum GAN Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--save-dir", type=str, default="results/balanced", help="Save directory")
    
    args = parser.parse_args()
    
    print("⚖️ BALANCED QUANTUM GAN TRAINING")
    print("=" * 60)
    print("Fixes for discriminator gradient collapse:")
    print("  ✅ 10:1 Generator/Discriminator learning rate ratio")
    print("  ✅ 1:1 training step ratio (vs 2:1 before)")
    print("  ✅ Discriminator gradient health monitoring")
    print("  ✅ Reduced gradient penalty weight")
    print("  ✅ Discriminator regularization")
    print("  ✅ Targeted mode separation loss")
    print("=" * 60)
    
    # Create balanced trainer
    trainer = BalancedQuantumGANTrainer()
    
    # Train with balance monitoring
    results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )
    
    # Print final analysis
    print(f"\n🏆 BALANCED TRAINING RESULTS:")
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
    
    if results['balance_metrics']:
        final_balance = results['balance_metrics'][-1]
        print(f"Final gradient health:")
        print(f"  Generator norm: {final_balance['gen_grad_norm']:.6f}")
        print(f"  Discriminator norm: {final_balance['disc_grad_norm']:.6f}")
        print(f"  LR ratio (G/D): {final_balance['lr_ratio']:.1f}:1")
        
        # Health assessment
        if final_balance['disc_grad_norm'] < 1e-6:
            print("  ❌ Discriminator gradient collapse still present")
        elif final_balance['disc_grad_norm'] > final_balance['gen_grad_norm'] * 10:
            print("  ⚠️ Discriminator still dominating")
        else:
            print("  ✅ Balanced gradient flow achieved!")
    
    print("=" * 60)
    print("Check balanced_training_plots.png for detailed analysis!")


if __name__ == "__main__":
    main()
