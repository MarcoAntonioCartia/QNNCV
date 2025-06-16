#!/usr/bin/env python3
"""
Improved Bimodal Quantum GAN - Addresses Mode Collapse
====================================================

This version implements advanced loss functions to prevent mode collapse.
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

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

from models.generators.quantum_sf_generator import QuantumSFGenerator

def create_improved_bimodal_data():
    """Create well-separated bimodal data."""
    mode1_center = np.array([-2.0, -2.0])
    mode2_center = np.array([2.0, 2.0])
    mode_std = 0.4
    samples_per_mode = 25
    
    # Mode 1
    mode1 = tf.random.normal([samples_per_mode, 2], 
                            mean=mode1_center, stddev=mode_std)
    # Mode 2
    mode2 = tf.random.normal([samples_per_mode, 2],
                            mean=mode2_center, stddev=mode_std)
    
    # Combine and shuffle
    data = tf.concat([mode1, mode2], axis=0)
    indices = tf.random.shuffle(tf.range(tf.shape(data)[0]))
    return tf.gather(data, indices), mode1_center, mode2_center

def create_quantum_generator():
    """Create quantum generator with improved config."""
    config = {
        'n_modes': 4,
        'latent_dim': 8,
        'layers': 3,
        'cutoff_dim': 6
    }
    
    generator = QuantumSFGenerator(
        n_modes=config['n_modes'],
        latent_dim=config['latent_dim'],
        layers=config['layers'],
        cutoff_dim=config['cutoff_dim'],
        encoding_strategy='coherent_state'
    )
    
    print(f"✅ Quantum Generator created: {generator.weights.shape}")
    return generator, config

def anti_collapse_loss(generated_samples, mode1_center, mode2_center):
    """Loss function designed to prevent mode collapse."""
    # Take first 2 dimensions
    gen_2d = generated_samples[:, :2]
    
    # 1. Encourage samples to spread out (high variance)
    gen_mean = tf.reduce_mean(gen_2d, axis=0)
    gen_var = tf.reduce_mean(tf.square(gen_2d - gen_mean))
    variance_reward = -tf.exp(-gen_var)  # Reward high variance
    
    # 2. Push samples away from center point
    center_point = (mode1_center + mode2_center) / 2.0
    distances_from_center = tf.linalg.norm(gen_2d - center_point, axis=1)
    center_penalty = tf.reduce_mean(tf.exp(-distances_from_center * 0.5))
    
    # 3. Encourage samples near target modes
    dist_to_mode1 = tf.linalg.norm(gen_2d - mode1_center, axis=1)
    dist_to_mode2 = tf.linalg.norm(gen_2d - mode2_center, axis=1)
    min_mode_dist = tf.minimum(dist_to_mode1, dist_to_mode2)
    mode_attraction = tf.reduce_mean(min_mode_dist)
    
    # 4. Bimodality encouragement - samples should be either near mode1 OR mode2
    mode_assignment = tf.cast(dist_to_mode1 < dist_to_mode2, tf.float32)
    mode_balance = tf.abs(tf.reduce_mean(mode_assignment) - 0.5)  # Encourage 50-50 split
    
    # Combine losses
    total_loss = (2.0 * center_penalty + 
                 0.5 * mode_attraction + 
                 1.0 * mode_balance + 
                 variance_reward)
    
    return {
        'total_loss': total_loss,
        'center_penalty': center_penalty,
        'mode_attraction': mode_attraction,
        'mode_balance': mode_balance,
        'variance_reward': variance_reward
    }

def training_step(generator, optimizer, real_batch, mode1_center, mode2_center):
    """Improved training step."""
    batch_size = tf.shape(real_batch)[0]
    
    with tf.GradientTape() as tape:
        # Generate samples
        z = tf.random.normal([batch_size, 8])  # latent_dim = 8
        generated_samples = generator.generate(z)
        
        # Apply anti-collapse loss
        loss_dict = anti_collapse_loss(generated_samples, mode1_center, mode2_center)
        
        # Small regularization
        weight_reg = tf.reduce_mean(tf.square(generator.weights)) * 0.0001
        total_loss = loss_dict['total_loss'] + weight_reg
    
    # Apply gradients with clipping
    variables = generator.trainable_variables
    gradients = tape.gradient(total_loss, variables)
    gradients = [tf.clip_by_value(g, -0.05, 0.05) if g is not None else g 
                for g in gradients]
    
    optimizer.apply_gradients(zip(gradients, variables))
    
    loss_dict['total_loss'] = total_loss
    loss_dict['weight_reg'] = weight_reg
    return loss_dict

def evaluate_generation(generator, mode1_center, mode2_center):
    """Evaluate generation quality."""
    z_test = tf.random.normal([100, 8])
    generated = generator.generate(z_test)
    gen_2d = generated[:, :2].numpy()
    
    # Mode assignment
    dist_to_mode1 = np.linalg.norm(gen_2d - mode1_center, axis=1)
    dist_to_mode2 = np.linalg.norm(gen_2d - mode2_center, axis=1)
    
    mode1_mask = dist_to_mode1 < dist_to_mode2
    mode1_count = np.sum(mode1_mask)
    mode2_count = len(gen_2d) - mode1_count
    
    # Calculate quality metrics
    if mode1_count > 0 and mode2_count > 0:
        mode1_samples = gen_2d[mode1_mask]
        mode2_samples = gen_2d[~mode1_mask]
        
        gen_mode1_center = np.mean(mode1_samples, axis=0)
        gen_mode2_center = np.mean(mode2_samples, axis=0)
        
        gen_separation = np.linalg.norm(gen_mode1_center - gen_mode2_center)
        target_separation = np.linalg.norm(mode2_center - mode1_center)
        separation_accuracy = gen_separation / target_separation
        
        balance_score = min(mode1_count, mode2_count) / len(gen_2d)
        mode_collapsed = balance_score < 0.2 or separation_accuracy < 0.4
    else:
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

def run_improved_training():
    """Run improved training with anti-collapse techniques."""
    print("🚀 Improved Bimodal Quantum GAN Training")
    print("=" * 50)
    
    # Setup
    os.makedirs('tests/results', exist_ok=True)
    
    # Create data and model
    real_data, mode1_center, mode2_center = create_improved_bimodal_data()
    generator, config = create_quantum_generator()
    
    # Use slower learning rate and custom schedule
    initial_lr = 0.003
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr, decay_steps=50, decay_rate=0.95, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    
    print(f"📊 Data shape: {real_data.shape}")
    print(f"📊 Mode centers: {mode1_center} and {mode2_center}")
    print(f"📊 Target separation: {np.linalg.norm(mode2_center - mode1_center):.2f}")
    
    # Training loop
    epochs = 15
    batch_size = 8
    history = []
    
    for epoch in range(epochs):
        print(f"\n🏃‍♂️ Epoch {epoch + 1}/{epochs}")
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(real_data)
        dataset = dataset.batch(batch_size).shuffle(100)
        
        epoch_losses = []
        
        for batch in dataset:
            losses = training_step(generator, optimizer, batch, mode1_center, mode2_center)
            epoch_losses.append(losses)
        
        # Average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([float(l[key]) for l in epoch_losses])
        
        print(f"   📊 Total Loss: {avg_losses['total_loss']:.4f}")
        print(f"   🎯 Center Penalty: {avg_losses['center_penalty']:.4f}")
        print(f"   ⚖️  Mode Balance: {avg_losses['mode_balance']:.4f}")
        
        # Evaluate quality
        quality = evaluate_generation(generator, mode1_center, mode2_center)
        
        print(f"   📈 Balance Score: {quality['balance_score']:.3f}")
        print(f"   📏 Separation: {quality['separation_accuracy']:.3f}")
        print(f"   🚨 Mode Collapse: {'YES' if quality['mode_collapsed'] else 'NO'}")
        print(f"   📊 Counts: Mode1={quality['mode1_count']}, Mode2={quality['mode2_count']}")
        
        # Store history
        history.append({
            'epoch': epoch,
            'losses': avg_losses,
            'quality': {k: v for k, v in quality.items() if k != 'generated_samples'}
        })
    
    # Final evaluation
    final_quality = evaluate_generation(generator, mode1_center, mode2_center)
    
    # Create visualization
    create_results_plot(real_data.numpy(), final_quality, mode1_center, mode2_center, history)
    
    # Save results
    save_improved_results(history, final_quality, config, mode1_center, mode2_center)
    
    return history, final_quality

def create_results_plot(real_data, final_quality, mode1_center, mode2_center, history):
    """Create comprehensive results plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Improved Bimodal Quantum GAN Results', fontsize=14)
    
    generated_samples = final_quality['generated_samples']
    
    # Plot 1: Real vs Generated
    axes[0, 0].scatter(real_data[:, 0], real_data[:, 1], 
                      alpha=0.7, c='blue', label='Real Data', s=50)
    axes[0, 0].scatter(generated_samples[:, 0], generated_samples[:, 1],
                      alpha=0.7, c='red', label='Generated', s=30)
    axes[0, 0].scatter(*mode1_center, c='blue', s=200, marker='x', 
                      linewidth=3, label='Target Mode 1')
    axes[0, 0].scatter(*mode2_center, c='blue', s=200, marker='x', 
                      linewidth=3, label='Target Mode 2')
    axes[0, 0].set_title('Real vs Generated Data')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # Plot 2: Mode assignment
    dist_to_mode1 = np.linalg.norm(generated_samples - mode1_center, axis=1)
    dist_to_mode2 = np.linalg.norm(generated_samples - mode2_center, axis=1)
    mode1_mask = dist_to_mode1 < dist_to_mode2
    
    mode1_gen = generated_samples[mode1_mask]
    mode2_gen = generated_samples[~mode1_mask]
    
    if len(mode1_gen) > 0:
        axes[0, 1].scatter(mode1_gen[:, 0], mode1_gen[:, 1], 
                          c='lightcoral', alpha=0.8, label=f'Gen Mode 1 ({len(mode1_gen)})', s=40)
    if len(mode2_gen) > 0:
        axes[0, 1].scatter(mode2_gen[:, 0], mode2_gen[:, 1], 
                          c='lightblue', alpha=0.8, label=f'Gen Mode 2 ({len(mode2_gen)})', s=40)
    
    axes[0, 1].scatter(*mode1_center, c='red', s=200, marker='x', linewidth=3)
    axes[0, 1].scatter(*mode2_center, c='blue', s=200, marker='x', linewidth=3)
    axes[0, 1].set_title('Generated Mode Assignment')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # Plot 3: Training metrics
    epochs = [h['epoch'] for h in history]
    total_losses = [h['losses']['total_loss'] for h in history]
    balance_scores = [h['quality']['balance_score'] for h in history]
    
    ax1 = axes[1, 0]
    ax1.plot(epochs, total_losses, 'b-', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, balance_scores, 'r-', linewidth=2)
    ax2.set_ylabel('Balance Score', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.axhline(y=0.4, color='r', linestyle='--', alpha=0.5)
    
    # Plot 4: Success metrics
    success_metrics = {
        'No Collapse': not final_quality['mode_collapsed'],
        'Good Balance': final_quality['balance_score'] > 0.3,
        'Good Separation': final_quality['separation_accuracy'] > 0.6,
        'Both Modes': final_quality['mode1_count'] > 5 and final_quality['mode2_count'] > 5
    }
    
    metric_names = list(success_metrics.keys())
    metric_values = [1 if v else 0 for v in success_metrics.values()]
    colors = ['green' if v else 'red' for v in success_metrics.values()]
    
    bars = axes[1, 1].bar(range(len(metric_names)), metric_values, color=colors, alpha=0.7)
    axes[1, 1].set_title('Success Criteria')
    axes[1, 1].set_ylabel('Pass (1) / Fail (0)')
    axes[1, 1].set_xticks(range(len(metric_names)))
    axes[1, 1].set_xticklabels(metric_names, rotation=45, ha='right')
    axes[1, 1].set_ylim(0, 1.2)
    
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       'PASS' if value else 'FAIL',
                       ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f"tests/results/improved_bimodal_results_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Results plot saved: {plot_file}")

def save_improved_results(history, final_quality, config, mode1_center, mode2_center):
    """Save comprehensive results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'config': config,
        'mode_centers': {
            'mode1': mode1_center.tolist(),
            'mode2': mode2_center.tolist()
        },
        'training_history': history,
        'final_quality': {k: v for k, v in final_quality.items() 
                        if k != 'generated_samples'},
        'timestamp': timestamp,
        'success_summary': {
            'mode_collapsed': final_quality['mode_collapsed'],
            'balance_score': final_quality['balance_score'],
            'separation_accuracy': final_quality['separation_accuracy'],
            'mode1_count': final_quality['mode1_count'],
            'mode2_count': final_quality['mode2_count']
        }
    }
    
    filename = f"tests/results/improved_bimodal_results_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved: {filename}")

def main():
    """Main function to run improved bimodal test."""
    print("Starting Improved Bimodal Quantum GAN Test...")
    
    history, final_quality = run_improved_training()
    
    print(f"\n🎉 FINAL RESULTS SUMMARY")
    print("=" * 40)
    print(f"✅ Mode 1 Samples: {final_quality['mode1_count']}")
    print(f"✅ Mode 2 Samples: {final_quality['mode2_count']}")
    print(f"✅ Balance Score: {final_quality['balance_score']:.3f}")
    print(f"✅ Separation Accuracy: {final_quality['separation_accuracy']:.3f}")
    print(f"✅ Mode Collapse: {'NO' if not final_quality['mode_collapsed'] else 'YES'}")
    
    # Overall success assessment
    success_criteria = [
        not final_quality['mode_collapsed'],
        final_quality['balance_score'] > 0.3,
        final_quality['separation_accuracy'] > 0.6,
        final_quality['mode1_count'] > 5,
        final_quality['mode2_count'] > 5
    ]
    
    overall_success = sum(success_criteria) >= 3  # At least 3/5 criteria
    print(f"🎯 Overall Success: {'PASS' if overall_success else 'NEEDS IMPROVEMENT'}")
    
    return history, final_quality

if __name__ == "__main__":
    results = main()
    print("✅ Improved test completed!") 