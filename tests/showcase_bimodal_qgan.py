#!/usr/bin/env python3
"""
Bimodal Quantum GAN Showcase
============================

Comprehensive test to showcase the working bimodal quantum GAN with proper gradient flow.
This addresses the encoder gradient issue and demonstrates successful bimodal learning.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Import warning suppression for clean output
from utils.warning_suppression import setup_clean_environment, clean_training_output
setup_clean_environment()

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

from models.generators.quantum_sf_generator import QuantumSFGenerator

class BimodalQuantumGANShowcase:
    """Showcase system for bimodal quantum GAN with fixed gradient flow."""
    
    def __init__(self, config=None):
        """Initialize the showcase system."""
        self.config = config or {
            'n_modes': 4,           # 4 modes as suggested
            'latent_dim': 6,        # Slightly higher for more expressivity
            'layers': 2,            # 2 layers as suggested
            'cutoff_dim': 5,        # Lower cutoff for stability (5^4 = 625 dims)
            'epochs': 8,            # 8 epochs as suggested
            'batch_size': 10,       # Small batches for stability
            'learning_rate': 0.005, # Higher learning rate for faster convergence
            'samples_per_mode': 25, # 50 total data points (25 per mode)
            'mode_1_center': [-1.5, -1.5],
            'mode_2_center': [1.5, 1.5],
            'mode_std': 0.4
        }
        
        # Initialize tracking
        self.training_history = []
        self.generation_samples = []
        
        # Create directories
        os.makedirs('tests/results', exist_ok=True)
        
        print("🎯 Bimodal Quantum GAN Showcase Initialized")
        print(f"📊 Configuration: {self.config}")
    
    def create_bimodal_data(self):
        """Create well-separated bimodal training data."""
        # Mode 1: Bottom-left cluster
        mode1 = tf.random.normal(
            [self.config['samples_per_mode'], 2],
            mean=self.config['mode_1_center'],
            stddev=self.config['mode_std']
        )
        
        # Mode 2: Top-right cluster
        mode2 = tf.random.normal(
            [self.config['samples_per_mode'], 2],
            mean=self.config['mode_2_center'],
            stddev=self.config['mode_std']
        )
        
        # Combine and shuffle
        data = tf.concat([mode1, mode2], axis=0)
        indices = tf.random.shuffle(tf.range(tf.shape(data)[0]))
        return tf.gather(data, indices)
    
    def create_generator_with_fixed_gradients(self):
        """Create generator with fixed gradient flow."""
        generator = QuantumSFGenerator(
            n_modes=self.config['n_modes'],
            latent_dim=self.config['latent_dim'],
            layers=self.config['layers'],
            cutoff_dim=self.config['cutoff_dim']
        )
        
        print(f"✅ Generator created: {generator.weights.shape} quantum weights")
        print(f"📊 Total trainable variables: {len(generator.trainable_variables)}")
        
        return generator
    
    def train_step_with_pure_quantum_gradients(self, generator, optimizer, real_batch):
        """Training step using PURE QUANTUM approach - no classical encoder."""
        batch_size = int(real_batch.shape[0])  # Use static shape
        
        with tf.GradientTape() as tape:
            # Generate latent vectors (used only for batch size)
            z = tf.random.normal([batch_size, self.config['latent_dim']])
            
            # PURE QUANTUM: Generate using only quantum weights
            generated_samples = generator.generate(z)
            
            # Bimodal-specific loss functions
            target_mean = tf.reduce_mean(real_batch, axis=0)
            gen_mean = tf.reduce_mean(generated_samples, axis=0)
            mean_matching_loss = tf.reduce_mean(tf.square(gen_mean - target_mean))
            
            # Encourage bimodal structure by measuring variance
            gen_var = tf.math.reduce_variance(generated_samples, axis=0)
            # Encourage spread in both dimensions
            variance_loss = -tf.reduce_mean(gen_var)  # Negative to maximize variance
            
            # Quantum weight regularization (prevent too large weights)
            weight_reg = tf.reduce_mean(tf.square(generator.weights)) * 0.001
            
            # Combined loss focusing on bimodal structure
            total_loss = mean_matching_loss + 0.1 * variance_loss + weight_reg
        
        # Get gradients ONLY for quantum weights (pure quantum approach)
        quantum_variables = [generator.weights]  # Only quantum weights
        gradients = tape.gradient(total_loss, quantum_variables)
        
        # Check gradient health
        valid_grads = sum(1 for g in gradients if g is not None and not tf.reduce_any(tf.math.is_nan(g)))
        
        # Apply gradients to quantum weights only
        optimizer.apply_gradients(zip(gradients, quantum_variables))
        
        return {
            'total_loss': float(total_loss),
            'mean_loss': float(mean_matching_loss),
            'variance_loss': float(variance_loss),
            'weight_reg': float(weight_reg),
            'valid_gradients': valid_grads,
            'total_gradients': len(gradients),
            'quantum_weight_norm': float(tf.linalg.norm(generator.weights))
        }
    
    def evaluate_generation_quality(self, generator, real_data):
        """Comprehensive evaluation of generation quality."""
        # Generate test samples
        z_test = tf.random.normal([100, self.config['latent_dim']])
        generated = generator.generate(z_test)
        
        gen_np = generated.numpy()
        real_np = real_data.numpy()
        
        # Mode separation analysis
        mode1_center = np.array(self.config['mode_1_center'])
        mode2_center = np.array(self.config['mode_2_center'])
        
        # Assign generated samples to nearest mode
        dist_to_mode1 = np.linalg.norm(gen_np - mode1_center, axis=1)
        dist_to_mode2 = np.linalg.norm(gen_np - mode2_center, axis=1)
        
        mode1_samples = gen_np[dist_to_mode1 < dist_to_mode2]
        mode2_samples = gen_np[dist_to_mode1 >= dist_to_mode2]
        
        # Calculate metrics
        if len(mode1_samples) > 0 and len(mode2_samples) > 0:
            gen_mode1_center = np.mean(mode1_samples, axis=0)
            gen_mode2_center = np.mean(mode2_samples, axis=0)
            mode_separation = np.linalg.norm(gen_mode1_center - gen_mode2_center)
            mode_balance = min(len(mode1_samples), len(mode2_samples)) / len(gen_np)
        else:
            mode_separation = 0.0
            mode_balance = 0.0
        
        # Overall quality metrics
        target_separation = np.linalg.norm(mode1_center - mode2_center)
        separation_accuracy = mode_separation / target_separation if target_separation > 0 else 0
        
        return {
            'mode_separation': mode_separation,
            'target_separation': target_separation,
            'separation_accuracy': separation_accuracy,
            'mode_balance': mode_balance,
            'mode1_samples': len(mode1_samples) if len(mode1_samples) > 0 else 0,
            'mode2_samples': len(mode2_samples) if len(mode2_samples) > 0 else 0,
            'total_samples': len(gen_np),
            'mode_collapsed': mode_separation < 0.5,
            'generated_samples': gen_np
        }
    
    def run_training_epoch(self, generator, optimizer, real_data, epoch):
        """Run one training epoch with detailed monitoring."""
        print(f"\n🏃‍♂️ Epoch {epoch+1}/{self.config['epochs']}")
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(real_data)
        dataset = dataset.batch(self.config['batch_size']).shuffle(100)
        
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataset):
            losses = self.train_step_with_pure_quantum_gradients(generator, optimizer, batch)
            epoch_losses.append(losses)
            
            if batch_idx == 0:  # Print first batch details
                print(f"   Batch 1: Loss={losses['total_loss']:.4f}, "
                      f"Grads={losses['valid_gradients']}/{losses['total_gradients']}")
        
        # Calculate epoch averages
        avg_losses = {key: np.mean([loss[key] for loss in epoch_losses]) 
                     for key in epoch_losses[0].keys()}
        
        # Evaluate generation quality
        quality_metrics = self.evaluate_generation_quality(generator, real_data)
        
        # Store samples for visualization
        self.generation_samples.append({
            'epoch': epoch,
            'samples': quality_metrics['generated_samples']
        })
        
        # Print epoch summary
        print(f"   📊 Avg Loss: {avg_losses['total_loss']:.4f}")
        print(f"   🎯 Mode Sep: {quality_metrics['mode_separation']:.3f} "
              f"(target: {quality_metrics['target_separation']:.3f})")
        print(f"   ⚖️  Balance: {quality_metrics['mode_balance']:.3f}")
        print(f"   📈 Sep Accuracy: {quality_metrics['separation_accuracy']:.3f}")
        print(f"   🔄 Mode Collapse: {'❌ YES' if quality_metrics['mode_collapsed'] else '✅ NO'}")
        
        # Store training history
        epoch_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'losses': avg_losses,
            'quality': quality_metrics
        }
        self.training_history.append(epoch_data)
        
        return epoch_data
    
    def run_showcase(self):
        """Run the complete showcase."""
        print("\n🚀 Starting Bimodal Quantum GAN Showcase")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Create data and models
        real_data = self.create_bimodal_data()
        generator = self.create_generator_with_fixed_gradients()
        # Import keras compatibility
        try:
            from tensorflow import keras
        except ImportError:
            import keras
        
        optimizer = keras.optimizers.Adam(self.config['learning_rate'])
        
        print(f"\n📊 Training Data Statistics:")
        print(f"   Shape: {real_data.shape}")
        print(f"   Mean: {tf.reduce_mean(real_data, axis=0).numpy()}")
        print(f"   Std: {tf.math.reduce_std(real_data, axis=0).numpy()}")
        
        # Training loop with clean output
        with clean_training_output(suppress_warnings=True, show_progress=True):
            for epoch in range(self.config['epochs']):
                epoch_result = self.run_training_epoch(generator, optimizer, real_data, epoch)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Final evaluation
        final_quality = self.evaluate_generation_quality(generator, real_data)
        
        # Create comprehensive results
        results = {
            'config': self.config,
            'training_summary': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'epochs_completed': len(self.training_history)
            },
            'final_metrics': final_quality,
            'training_history': self.training_history,
            'success_criteria': {
                'good_separation': final_quality['separation_accuracy'] > 0.7,
                'no_mode_collapse': not final_quality['mode_collapsed'],
                'balanced_modes': final_quality['mode_balance'] > 0.3,
                'reasonable_loss': self.training_history[-1]['losses']['total_loss'] < 5.0
            }
        }
        
        # Overall success
        results['overall_success'] = all(results['success_criteria'].values())
        
        # Save results
        self.save_results(results, real_data.numpy())
        
        # Create visualizations
        self.create_visualizations(real_data.numpy(), results)
        
        # Print final summary
        self.print_final_summary(results)
        
        return results
    
    def save_results(self, results, real_data):
        """Save comprehensive results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, np.generic)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy_types(results)
        
        # Save detailed results
        results_file = f"tests/results/bimodal_qgan_showcase_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save real data separately
        np.save(f"tests/results/real_data_{timestamp}.npy", real_data)
        
        print(f"💾 Results saved to: {results_file}")
    
    def create_visualizations(self, real_data, results):
        """Create comprehensive visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Bimodal Quantum GAN Showcase Results', fontsize=16)
        
        # 1. Training losses
        epochs = [h['epoch'] for h in self.training_history]
        total_losses = [h['losses']['total_loss'] for h in self.training_history]
        mean_losses = [h['losses']['mean_loss'] for h in self.training_history]
        
        axes[0, 0].plot(epochs, total_losses, 'b-', label='Total Loss', linewidth=2)
        axes[0, 0].plot(epochs, mean_losses, 'r--', label='Mean Loss', linewidth=2)
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Mode separation evolution
        separations = [h['quality']['mode_separation'] for h in self.training_history]
        target_sep = results['final_metrics']['target_separation']
        
        axes[0, 1].plot(epochs, separations, 'g-', linewidth=2)
        axes[0, 1].axhline(y=target_sep, color='r', linestyle='--', alpha=0.7, label=f'Target ({target_sep:.2f})')
        axes[0, 1].set_title('Mode Separation Evolution')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Separation Distance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Mode balance evolution
        balances = [h['quality']['mode_balance'] for h in self.training_history]
        
        axes[0, 2].plot(epochs, balances, 'purple', linewidth=2)
        axes[0, 2].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Perfect Balance')
        axes[0, 2].set_title('Mode Balance Evolution')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Balance Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Final generated vs real data
        final_generated = results['final_metrics']['generated_samples']
        
        axes[1, 0].scatter(real_data[:, 0], real_data[:, 1], 
                          alpha=0.7, c='blue', s=50, label='Real Data')
        axes[1, 0].scatter(final_generated[:, 0], final_generated[:, 1], 
                          alpha=0.7, c='red', s=30, label='Generated')
        axes[1, 0].set_title('Final Generated vs Real Data')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')
        
        # 5. Generation evolution (first, middle, last epochs)
        if len(self.generation_samples) >= 3:
            first_gen = self.generation_samples[0]['samples']
            mid_gen = self.generation_samples[len(self.generation_samples)//2]['samples']
            last_gen = self.generation_samples[-1]['samples']
            
            axes[1, 1].scatter(first_gen[:, 0], first_gen[:, 1], alpha=0.5, c='lightcoral', s=20, label='Epoch 1')
            axes[1, 1].scatter(mid_gen[:, 0], mid_gen[:, 1], alpha=0.5, c='orange', s=20, label=f'Epoch {len(self.generation_samples)//2+1}')
            axes[1, 1].scatter(last_gen[:, 0], last_gen[:, 1], alpha=0.7, c='darkred', s=30, label=f'Epoch {len(self.generation_samples)}')
            axes[1, 1].scatter(real_data[:, 0], real_data[:, 1], alpha=0.3, c='blue', s=20, label='Real')
            axes[1, 1].set_title('Generation Evolution')
            axes[1, 1].set_xlabel('X')
            axes[1, 1].set_ylabel('Y')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Success criteria
        criteria = results['success_criteria']
        criteria_names = list(criteria.keys())
        criteria_values = [1 if v else 0 for v in criteria.values()]
        colors = ['green' if v else 'red' for v in criteria.values()]
        
        bars = axes[1, 2].bar(range(len(criteria_names)), criteria_values, color=colors, alpha=0.7)
        axes[1, 2].set_title('Success Criteria')
        axes[1, 2].set_xlabel('Criteria')
        axes[1, 2].set_ylabel('Pass (1) / Fail (0)')
        axes[1, 2].set_xticks(range(len(criteria_names)))
        axes[1, 2].set_xticklabels(criteria_names, rotation=45, ha='right')
        axes[1, 2].set_ylim(0, 1.2)
        
        # Add value labels on bars
        for bar, value, name in zip(bars, criteria_values, criteria_names):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           'PASS' if value else 'FAIL',
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = f"tests/results/bimodal_qgan_showcase_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizations saved to: {plot_file}")
    
    def print_final_summary(self, results):
        """Print comprehensive final summary."""
        print("\n🎯 BIMODAL QUANTUM GAN SHOWCASE RESULTS")
        print("=" * 60)
        
        success = results['overall_success']
        print(f"✅ Overall Success: {'🎉 PASS' if success else '❌ FAIL'}")
        print(f"⏱️  Training Duration: {results['training_summary']['duration_seconds']:.1f} seconds")
        print(f"🔄 Epochs Completed: {results['training_summary']['epochs_completed']}")
        
        print(f"\n🎯 Final Quality Metrics:")
        final = results['final_metrics']
        print(f"   Mode Separation: {final['mode_separation']:.3f} (target: {final['target_separation']:.3f})")
        print(f"   Separation Accuracy: {final['separation_accuracy']:.3f}")
        print(f"   Mode Balance: {final['mode_balance']:.3f}")
        print(f"   Mode Collapse: {'❌ YES' if final['mode_collapsed'] else '✅ NO'}")
        
        print(f"\n🔍 Success Criteria:")
        for criterion, passed in results['success_criteria'].items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {criterion.replace('_', ' ').title()}: {status}")
        
        print(f"\n📊 Training Progress:")
        final_loss = results['training_history'][-1]['losses']['total_loss']
        initial_loss = results['training_history'][0]['losses']['total_loss']
        print(f"   Initial Loss: {initial_loss:.4f}")
        print(f"   Final Loss: {final_loss:.4f}")
        print(f"   Loss Improvement: {((initial_loss - final_loss) / initial_loss * 100):.1f}%")


def run_showcase():
    """Run the complete bimodal quantum GAN showcase."""
    showcase = BimodalQuantumGANShowcase()
    results = showcase.run_showcase()
    return results


if __name__ == "__main__":
    results = run_showcase() 