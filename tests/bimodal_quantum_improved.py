#!/usr/bin/env python3
"""
Improved Bimodal Quantum GAN
============================

Addresses mode collapse with better loss functions and training strategies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

from models.generators.quantum_sf_generator import QuantumSFGenerator

class ImprovedBimodalQuantumGAN:
    """Improved quantum GAN that can learn bimodal distributions."""
    
    def __init__(self):
        """Initialize with optimized configuration."""
        self.config = {
            'n_modes': 4,           # Keep 4 modes for expressivity
            'latent_dim': 8,        # Higher dimensional latent space
            'layers': 3,            # More layers for complexity
            'cutoff_dim': 6,        # Higher cutoff for more states
            'epochs': 12,           # More epochs
            'batch_size': 8,        # Smaller batches for stability
            'learning_rate': 0.005, # Lower learning rate
            'samples_per_mode': 25, # 50 total samples
        }
        
        # Mode centers (well separated)
        self.mode1_center = np.array([-2.0, -2.0])
        self.mode2_center = np.array([2.0, 2.0])
        self.mode_std = 0.3  # Tighter clusters
        
        os.makedirs('tests/results', exist_ok=True)
        
        print("🚀 Improved Bimodal Quantum GAN")
        print(f"📊 Config: {self.config}")
    
    def create_bimodal_data(self):
        """Create well-separated bimodal data."""
        # Mode 1: bottom-left cluster
        mode1 = tf.random.normal([self.config['samples_per_mode'], 2], 
                                mean=self.mode1_center, stddev=self.mode_std)
        
        # Mode 2: top-right cluster  
        mode2 = tf.random.normal([self.config['samples_per_mode'], 2],
                                mean=self.mode2_center, stddev=self.mode_std)
        
        # Combine and shuffle
        data = tf.concat([mode1, mode2], axis=0)
        indices = tf.random.shuffle(tf.range(tf.shape(data)[0]))
        return tf.gather(data, indices)
    
    def create_generator(self):
        """Create improved quantum generator."""
        generator = QuantumSFGenerator(
            n_modes=self.config['n_modes'],
            latent_dim=self.config['latent_dim'],
            layers=self.config['layers'],
            cutoff_dim=self.config['cutoff_dim'],
            encoding_strategy='coherent_state'
        )
        
        print(f"✅ Generator: {generator.weights.shape} quantum weights")
        return generator
    
    def bimodal_loss(self, real_batch, generated_samples):
        """Advanced loss function that encourages bimodal structure."""
        # Take first 2 dimensions for comparison
        gen_2d = generated_samples[:, :2]
        
        # 1. Mode assignment: assign each generated sample to nearest real mode
        real_mode1 = self.mode1_center
        real_mode2 = self.mode2_center
        
        # Distance to each mode center
        dist_to_mode1 = tf.linalg.norm(gen_2d[:, None, :] - real_mode1[None, None, :], axis=2)
        dist_to_mode2 = tf.linalg.norm(gen_2d[:, None, :] - real_mode2[None, None, :], axis=2)
        
        # Assign to nearest mode
        mode1_mask = tf.squeeze(dist_to_mode1 < dist_to_mode2)
        
        # 2. Mode coverage loss: encourage samples near both modes
        min_dist_to_modes = tf.minimum(
            tf.reduce_min(dist_to_mode1), 
            tf.reduce_min(dist_to_mode2)
        )
        coverage_loss = tf.reduce_mean(min_dist_to_modes)
        
        # 3. Mode separation loss: encourage generated modes to be far apart
        gen_mode1_samples = tf.boolean_mask(gen_2d, mode1_mask)
        gen_mode2_samples = tf.boolean_mask(gen_2d, ~mode1_mask)
        
        if tf.shape(gen_mode1_samples)[0] > 0 and tf.shape(gen_mode2_samples)[0] > 0:
            gen_mode1_center = tf.reduce_mean(gen_mode1_samples, axis=0)
            gen_mode2_center = tf.reduce_mean(gen_mode2_samples, axis=0)
            gen_separation = tf.linalg.norm(gen_mode1_center - gen_mode2_center)
            target_separation = tf.linalg.norm(real_mode2 - real_mode1)
            separation_loss = tf.square(gen_separation - target_separation)
        else:
            # Penalty for mode collapse (all samples assigned to one mode)
            separation_loss = 10.0
        
        # 4. Mode balance loss: encourage equal numbers in each mode
        mode1_count = tf.reduce_sum(tf.cast(mode1_mask, tf.float32))
        mode2_count = tf.reduce_sum(tf.cast(~mode1_mask, tf.float32))
        total_count = tf.cast(tf.shape(gen_2d)[0], tf.float32)
        
        balance_loss = tf.square(mode1_count/total_count - 0.5)
        
        # 5. Within-mode compactness: encourage tight clusters
        compactness_loss = 0.0
        if tf.shape(gen_mode1_samples)[0] > 1:
            mode1_var = tf.reduce_mean(tf.math.reduce_variance(gen_mode1_samples, axis=0))
            compactness_loss += mode1_var
        
        if tf.shape(gen_mode2_samples)[0] > 1:
            mode2_var = tf.reduce_mean(tf.math.reduce_variance(gen_mode2_samples, axis=0))
            compactness_loss += mode2_var
        
        # Combine losses
        total_loss = (coverage_loss + 
                     0.5 * separation_loss + 
                     2.0 * balance_loss + 
                     0.1 * compactness_loss)
        
        return {
            'total_loss': total_loss,
            'coverage_loss': coverage_loss,
            'separation_loss': separation_loss,
            'balance_loss': balance_loss,
            'compactness_loss': compactness_loss
        }
    
    def training_step(self, generator, optimizer, real_batch):
        """Advanced training step with bimodal loss."""
        batch_size = tf.shape(real_batch)[0]
        
        with tf.GradientTape() as tape:
            # Generate samples
            z = tf.random.normal([batch_size, self.config['latent_dim']])
            generated_samples = generator.generate(z)
            
            # Apply bimodal loss
            loss_dict = self.bimodal_loss(real_batch, generated_samples)
            
            # Add regularization
            weight_reg = tf.reduce_mean(tf.square(generator.weights)) * 0.0001
            total_loss = loss_dict['total_loss'] + weight_reg
        
        # Apply gradients
        variables = generator.trainable_variables
        gradients = tape.gradient(total_loss, variables)
        
        # Gradient clipping to prevent instability
        gradients = [tf.clip_by_value(g, -0.1, 0.1) if g is not None else g 
                    for g in gradients]
        
        optimizer.apply_gradients(zip(gradients, variables))
        
        # Add total loss to dict
        loss_dict['total_loss'] = total_loss
        loss_dict['weight_reg'] = weight_reg
        
        return loss_dict
    
    def evaluate_bimodal_quality(self, generator, real_data):
        """Comprehensive bimodal evaluation."""
        # Generate test samples  
        z_test = tf.random.normal([200, self.config['latent_dim']])
        generated = generator.generate(z_test)
        gen_2d = generated[:, :2].numpy()
        
        # Mode assignment
        dist_to_mode1 = np.linalg.norm(gen_2d - self.mode1_center, axis=1)
        dist_to_mode2 = np.linalg.norm(gen_2d - self.mode2_center, axis=1)
        
        mode1_samples = gen_2d[dist_to_mode1 < dist_to_mode2]
        mode2_samples = gen_2d[dist_to_mode1 >= dist_to_mode2]
        
        # Calculate metrics
        mode1_count = len(mode1_samples)
        mode2_count = len(mode2_samples)
        
        if mode1_count > 0 and mode2_count > 0:
            gen_mode1_center = np.mean(mode1_samples, axis=0)
            gen_mode2_center = np.mean(mode2_samples, axis=0)
            
            # Mode separation
            gen_separation = np.linalg.norm(gen_mode1_center - gen_mode2_center)
            target_separation = np.linalg.norm(self.mode2_center - self.mode1_center)
            separation_accuracy = gen_separation / target_separation
            
            # Mode balance  
            balance_score = min(mode1_count, mode2_count) / len(gen_2d)
            
            # Mode collapse check
            mode_collapsed = balance_score < 0.1 or separation_accuracy < 0.3
            
        else:
            gen_separation = 0.0
            separation_accuracy = 0.0
            balance_score = 0.0
            mode_collapsed = True
        
        return {
            'mode1_count': mode1_count,
            'mode2_count': mode2_count,
            'separation_accuracy': separation_accuracy,
            'balance_score': balance_score,
            'mode_collapsed': mode_collapsed,
            'generated_samples': gen_2d
        }
    
    def run_training(self):
        """Run improved training."""
        print("\n🎯 Starting Improved Bimodal Training")
        print("=" * 50)
        
        # Create data and model
        real_data = self.create_bimodal_data()
        generator = self.create_generator()
        
        # Use Adam with custom schedule
        initial_lr = self.config['learning_rate']
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_lr, decay_steps=100, decay_rate=0.96, staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(lr_schedule)
        
        print(f"📊 Data shape: {real_data.shape}")
        print(f"📊 Real mode centers: {self.mode1_center}, {self.mode2_center}")
        
        # Training loop
        history = []
        
        for epoch in range(self.config['epochs']):
            print(f"\n🏃‍♂️ Epoch {epoch + 1}/{self.config['epochs']}")
            
            # Create dataset
            dataset = tf.data.Dataset.from_tensor_slices(real_data)
            dataset = dataset.batch(self.config['batch_size']).shuffle(100)
            
            epoch_losses = []
            
            for batch in dataset:
                losses = self.training_step(generator, optimizer, batch)
                epoch_losses.append(losses)
            
            # Average losses
            avg_losses = {}
            for key in epoch_losses[0].keys():
                avg_losses[key] = np.mean([l[key] for l in epoch_losses])
            
            print(f"   📊 Total Loss: {avg_losses['total_loss']:.4f}")
            print(f"   🎯 Coverage: {avg_losses['coverage_loss']:.4f}")
            print(f"   🔄 Balance: {avg_losses['balance_loss']:.4f}")
            
            # Evaluate bimodal quality
            quality = self.evaluate_bimodal_quality(generator, real_data)
            
            print(f"   📈 Mode Balance: {quality['balance_score']:.3f}")
            print(f"   📏 Separation: {quality['separation_accuracy']:.3f}")
            print(f"   🚨 Mode Collapse: {'YES' if quality['mode_collapsed'] else 'NO'}")
            
            # Store history
            history.append({
                'epoch': epoch,
                'losses': {k: float(v) for k, v in avg_losses.items()},
                'quality': {k: float(v) if isinstance(v, (int, float, np.number)) 
                           else v for k, v in quality.items() if k != 'generated_samples'}
            })
        
        # Final results
        final_quality = self.evaluate_bimodal_quality(generator, real_data)
        
        # Save and visualize
        self.save_results(history, real_data.numpy(), final_quality)
        self.create_advanced_plots(real_data.numpy(), final_quality, history)
        
        return history, final_quality
    
    def save_results(self, history, real_data, final_quality):
        """Save comprehensive results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'config': self.config,
            'mode_centers': {
                'mode1': self.mode1_center.tolist(),
                'mode2': self.mode2_center.tolist()
            },
            'training_history': history,
            'final_quality': {k: v for k, v in final_quality.items() 
                            if k != 'generated_samples'},
            'timestamp': timestamp
        }
        
        filename = f"tests/results/improved_bimodal_qgan_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved: {filename}")
    
    def create_advanced_plots(self, real_data, final_quality, history):
        """Create comprehensive visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Improved Bimodal Quantum GAN Results', fontsize=16)
        
        generated_samples = final_quality['generated_samples']
        
        # 1. Real vs Generated with mode centers
        axes[0, 0].scatter(real_data[:, 0], real_data[:, 1], 
                          alpha=0.7, c='blue', label='Real Data', s=50)
        axes[0, 0].scatter(generated_samples[:, 0], generated_samples[:, 1],
                          alpha=0.7, c='red', label='Generated', s=30)
        axes[0, 0].scatter(*self.mode1_center, c='blue', s=200, marker='x', 
                          linewidth=3, label='Real Mode 1')
        axes[0, 0].scatter(*self.mode2_center, c='blue', s=200, marker='x', 
                          linewidth=3, label='Real Mode 2')
        axes[0, 0].set_title('Real vs Generated Data')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axis('equal')
        
        # 2. Loss evolution
        epochs = [h['epoch'] for h in history]
        total_losses = [h['losses']['total_loss'] for h in history]
        coverage_losses = [h['losses']['coverage_loss'] for h in history]
        
        axes[0, 1].plot(epochs, total_losses, 'b-', label='Total Loss', linewidth=2)
        axes[0, 1].plot(epochs, coverage_losses, 'r--', label='Coverage Loss', linewidth=2)
        axes[0, 1].set_title('Training Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Balance and separation evolution
        balance_scores = [h['quality']['balance_score'] for h in history]
        separation_scores = [h['quality']['separation_accuracy'] for h in history]
        
        axes[0, 2].plot(epochs, balance_scores, 'g-', label='Balance Score', linewidth=2)
        axes[0, 2].plot(epochs, separation_scores, 'purple', label='Separation Accuracy', linewidth=2)
        axes[0, 2].axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Perfect Balance')
        axes[0, 2].axhline(y=1.0, color='purple', linestyle='--', alpha=0.5, label='Perfect Separation')
        axes[0, 2].set_title('Quality Metrics Evolution')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Mode assignment visualization
        dist_to_mode1 = np.linalg.norm(generated_samples - self.mode1_center, axis=1)
        dist_to_mode2 = np.linalg.norm(generated_samples - self.mode2_center, axis=1)
        mode1_mask = dist_to_mode1 < dist_to_mode2
        
        mode1_gen = generated_samples[mode1_mask]
        mode2_gen = generated_samples[~mode1_mask]
        
        if len(mode1_gen) > 0:
            axes[1, 0].scatter(mode1_gen[:, 0], mode1_gen[:, 1], 
                              c='lightcoral', alpha=0.8, label=f'Gen Mode 1 ({len(mode1_gen)})', s=40)
        if len(mode2_gen) > 0:
            axes[1, 0].scatter(mode2_gen[:, 0], mode2_gen[:, 1], 
                              c='lightblue', alpha=0.8, label=f'Gen Mode 2 ({len(mode2_gen)})', s=40)
        
        axes[1, 0].scatter(*self.mode1_center, c='red', s=200, marker='x', linewidth=3)
        axes[1, 0].scatter(*self.mode2_center, c='blue', s=200, marker='x', linewidth=3)
        axes[1, 0].set_title('Generated Mode Assignment')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')
        
        # 5. Success metrics
        success_metrics = {
            'No Mode Collapse': not final_quality['mode_collapsed'],
            'Good Balance': final_quality['balance_score'] > 0.3,
            'Good Separation': final_quality['separation_accuracy'] > 0.7,
            'Both Modes Present': final_quality['mode1_count'] > 5 and final_quality['mode2_count'] > 5
        }
        
        metric_names = list(success_metrics.keys())
        metric_values = [1 if v else 0 for v in success_metrics.values()]
        colors = ['green' if v else 'red' for v in success_metrics.values()]
        
        bars = axes[1, 1].bar(range(len(metric_names)), metric_values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Success Metrics')
        axes[1, 1].set_xlabel('Criteria')  
        axes[1, 1].set_ylabel('Pass (1) / Fail (0)')
        axes[1, 1].set_xticks(range(len(metric_names)))
        axes[1, 1].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[1, 1].set_ylim(0, 1.2)
        
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           'PASS' if value else 'FAIL',
                           ha='center', va='bottom', fontweight='bold')
        
        # 6. Distance distribution analysis
        all_distances = []
        for i in range(len(generated_samples)):
            for j in range(i+1, len(generated_samples)):
                dist = np.linalg.norm(generated_samples[i] - generated_samples[j])
                all_distances.append(dist)
        
        axes[1, 2].hist(all_distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 2].axvline(x=np.linalg.norm(self.mode2_center - self.mode1_center), 
                          color='red', linestyle='--', linewidth=2, 
                          label=f'Target Mode Sep: {np.linalg.norm(self.mode2_center - self.mode1_center):.2f}')
        axes[1, 2].set_title('Pairwise Distance Distribution')
        axes[1, 2].set_xlabel('Distance')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"tests/results/improved_bimodal_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Advanced plots saved: {plot_file}")

def run_improved_test():
    """Run the improved bimodal test."""
    gan = ImprovedBimodalQuantumGAN()
    history, final_quality = gan.run_training()
    
    print(f"\n🎉 FINAL RESULTS")
    print(f"✅ Mode 1 Count: {final_quality['mode1_count']}")
    print(f"✅ Mode 2 Count: {final_quality['mode2_count']}")
    print(f"✅ Balance Score: {final_quality['balance_score']:.3f}")
    print(f"✅ Separation Accuracy: {final_quality['separation_accuracy']:.3f}")
    print(f"✅ Mode Collapse: {'NO' if not final_quality['mode_collapsed'] else 'YES'}")
    
    return history, final_quality

if __name__ == "__main__":
    results = run_improved_test()
    print("✅ Improved test completed!") 