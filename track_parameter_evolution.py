"""
Track Parameter Evolution in Quantum GAN Training

This script monitors how quantum parameters evolve during training to identify
if parameters are actually being updated and if they're differentiating between modes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import json

from models.generators.quantum_pure_generator import QuantumPureGenerator
from models.discriminators.quantum_pure_discriminator import QuantumPureDiscriminator
from utils.quantum_bimodal_loss import QuantumBimodalLoss, BimodalDiscriminatorLoss
from utils.warning_suppression import setup_clean_environment

setup_clean_environment()


class ParameterTracker:
    """Track parameter evolution during training."""
    
    def __init__(self):
        self.history = {
            'generator': {},
            'discriminator': {},
            'metrics': []
        }
        self.epoch = 0
    
    def record_parameters(self, generator, discriminator, metrics=None):
        """Record current parameter values."""
        # Generator parameters
        for i, var in enumerate(generator.trainable_variables):
            var_name = f"g_var_{i}_{var.name}"
            if var_name not in self.history['generator']:
                self.history['generator'][var_name] = []
            self.history['generator'][var_name].append({
                'epoch': self.epoch,
                'values': var.numpy().copy(),
                'mean': float(np.mean(var.numpy())),
                'std': float(np.std(var.numpy())),
                'min': float(np.min(var.numpy())),
                'max': float(np.max(var.numpy())),
                'norm': float(np.linalg.norm(var.numpy()))
            })
        
        # Discriminator parameters
        for i, var in enumerate(discriminator.trainable_variables):
            var_name = f"d_var_{i}_{var.name}"
            if var_name not in self.history['discriminator']:
                self.history['discriminator'][var_name] = []
            self.history['discriminator'][var_name].append({
                'epoch': self.epoch,
                'values': var.numpy().copy(),
                'mean': float(np.mean(var.numpy())),
                'std': float(np.std(var.numpy())),
                'min': float(np.min(var.numpy())),
                'max': float(np.max(var.numpy())),
                'norm': float(np.linalg.norm(var.numpy()))
            })
        
        # Metrics
        if metrics:
            self.history['metrics'].append({
                'epoch': self.epoch,
                **metrics
            })
        
        self.epoch += 1
    
    def analyze_evolution(self):
        """Analyze how parameters evolved."""
        print("\n" + "="*60)
        print("PARAMETER EVOLUTION ANALYSIS")
        print("="*60)
        
        # Generator analysis
        print("\n📊 GENERATOR PARAMETERS:")
        for var_name, history in self.history['generator'].items():
            print(f"\n{var_name}:")
            
            # Check if parameters changed
            initial_values = history[0]['values']
            final_values = history[-1]['values']
            
            # Compute change metrics
            absolute_change = np.abs(final_values - initial_values)
            relative_change = absolute_change / (np.abs(initial_values) + 1e-8)
            
            print(f"  Initial norm: {history[0]['norm']:.6f}")
            print(f"  Final norm: {history[-1]['norm']:.6f}")
            print(f"  Max absolute change: {np.max(absolute_change):.6f}")
            print(f"  Mean absolute change: {np.mean(absolute_change):.6f}")
            print(f"  Max relative change: {np.max(relative_change):.6f}")
            
            # Check if frozen
            if np.max(absolute_change) < 1e-6:
                print("  ⚠️  WARNING: Parameters appear FROZEN!")
            
            # Special analysis for mode parameters
            if "mode1" in var_name or "mode2" in var_name:
                print(f"  Initial values: {initial_values}")
                print(f"  Final values: {final_values}")
        
        # Discriminator analysis
        print("\n📊 DISCRIMINATOR PARAMETERS:")
        for var_name, history in self.history['discriminator'].items():
            print(f"\n{var_name}:")
            
            initial_values = history[0]['values']
            final_values = history[-1]['values']
            
            absolute_change = np.abs(final_values - initial_values)
            
            print(f"  Initial norm: {history[0]['norm']:.6f}")
            print(f"  Final norm: {history[-1]['norm']:.6f}")
            print(f"  Max absolute change: {np.max(absolute_change):.6f}")
            
            if np.max(absolute_change) < 1e-6:
                print("  ⚠️  WARNING: Parameters appear FROZEN!")
    
    def plot_evolution(self):
        """Plot parameter evolution over time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Generator parameter norms
        ax = axes[0, 0]
        for var_name, history in self.history['generator'].items():
            epochs = [h['epoch'] for h in history]
            norms = [h['norm'] for h in history]
            ax.plot(epochs, norms, label=var_name.split('_')[-1], marker='o')
        ax.set_title('Generator Parameter Norms')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('L2 Norm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Discriminator parameter norms
        ax = axes[0, 1]
        for var_name, history in self.history['discriminator'].items():
            epochs = [h['epoch'] for h in history]
            norms = [h['norm'] for h in history]
            ax.plot(epochs, norms, label=var_name.split('_')[-1], marker='s')
        ax.set_title('Discriminator Parameter Norms')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('L2 Norm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Mode parameters evolution
        ax = axes[1, 0]
        mode1_history = None
        mode2_history = None
        
        for var_name, history in self.history['generator'].items():
            if "mode1" in var_name:
                mode1_history = history
            elif "mode2" in var_name:
                mode2_history = history
        
        if mode1_history and mode2_history:
            epochs = [h['epoch'] for h in mode1_history]
            
            # Plot first 3 parameters of each mode
            for i in range(min(3, len(mode1_history[0]['values']))):
                mode1_vals = [h['values'][i] for h in mode1_history]
                mode2_vals = [h['values'][i] for h in mode2_history]
                
                ax.plot(epochs, mode1_vals, f'C{i}-', label=f'Mode1[{i}]', marker='o')
                ax.plot(epochs, mode2_vals, f'C{i}--', label=f'Mode2[{i}]', marker='s')
        
        ax.set_title('Mode Parameters Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Parameter Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Loss evolution
        ax = axes[1, 1]
        if self.history['metrics']:
            epochs = [m['epoch'] for m in self.history['metrics']]
            d_losses = [m.get('d_loss', 0) for m in self.history['metrics']]
            g_losses = [m.get('g_loss', 0) for m in self.history['metrics']]
            
            ax.plot(epochs, d_losses, 'b-', label='D Loss', marker='o')
            ax.plot(epochs, g_losses, 'r-', label='G Loss', marker='s')
            ax.set_title('Training Losses')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('parameter_evolution.png', dpi=150)
        plt.show()
    
    def save_history(self, filename=None):
        """Save parameter history to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"parameter_evolution_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        history_json = convert_numpy(self.history)
        
        with open(filename, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"\n💾 Parameter history saved to: {filename}")


def train_with_tracking(epochs=10, batch_size=8):
    """Train quantum GAN while tracking parameter evolution."""
    print("\n" + "="*60)
    print("🔍 QUANTUM GAN TRAINING WITH PARAMETER TRACKING")
    print("="*60)
    
    # Configuration
    config = {
        'n_modes': 4,
        'latent_dim': 6,
        'layers': 2,
        'cutoff_dim': 6,
        'mode1_center': [-2.0, -2.0],
        'mode2_center': [2.0, 2.0],
        'mode_std': 0.3
    }
    
    # Create models
    generator = QuantumPureGenerator(
        n_modes=config['n_modes'],
        latent_dim=config['latent_dim'],
        layers=config['layers'],
        cutoff_dim=config['cutoff_dim'],
        mode_centers=[config['mode1_center'], config['mode2_center']]
    )
    
    discriminator = QuantumPureDiscriminator(
        n_modes=2,
        input_dim=2,
        layers=config['layers'],
        cutoff_dim=config['cutoff_dim']
    )
    
    # Create optimizers
    try:
        from tensorflow import keras
    except ImportError:
        import keras
    
    g_optimizer = keras.optimizers.Adam(0.002)
    d_optimizer = keras.optimizers.Adam(0.002)
    
    # Create loss functions
    g_loss_fn = QuantumBimodalLoss(
        mode1_center=config['mode1_center'],
        mode2_center=config['mode2_center']
    )
    d_loss_fn = BimodalDiscriminatorLoss()
    
    # Create tracker
    tracker = ParameterTracker()
    
    # Record initial parameters
    print("\n📸 Recording initial parameters...")
    tracker.record_parameters(generator, discriminator)
    
    # Create training data
    mode1 = tf.random.normal([50, 2], mean=config['mode1_center'], stddev=config['mode_std'])
    mode2 = tf.random.normal([50, 2], mean=config['mode2_center'], stddev=config['mode_std'])
    real_data = tf.concat([mode1, mode2], axis=0)
    
    # Training loop
    print("\n🏃 Starting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Shuffle data
        indices = tf.random.shuffle(tf.range(tf.shape(real_data)[0]))
        real_data_shuffled = tf.gather(real_data, indices)
        
        epoch_metrics = {
            'd_loss': 0.0,
            'g_loss': 0.0,
            'd_grads_norm': 0.0,
            'g_grads_norm': 0.0
        }
        
        n_batches = 0
        
        # Train on batches
        for i in range(0, len(real_data_shuffled), batch_size):
            real_batch = real_data_shuffled[i:i+batch_size]
            if len(real_batch) < batch_size:
                continue
            
            # Train discriminator
            with tf.GradientTape() as d_tape:
                z = tf.random.normal([batch_size, config['latent_dim']])
                fake_samples = generator.generate(z)
                d_loss, _ = d_loss_fn(real_batch, fake_samples, discriminator)
            
            d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
            
            # Train generator
            with tf.GradientTape() as g_tape:
                z = tf.random.normal([batch_size, config['latent_dim']])
                fake_samples = generator.generate(z)
                
                # Fool discriminator
                fake_output = discriminator.discriminate(fake_samples)
                g_fool_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(
                        tf.ones_like(fake_output), fake_output
                    )
                )
                
                # Bimodal loss
                g_bimodal_loss, _ = g_loss_fn(real_batch, fake_samples, generator)
                g_loss = g_fool_loss + g_bimodal_loss
            
            g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
            
            # Update metrics
            epoch_metrics['d_loss'] += float(d_loss)
            epoch_metrics['g_loss'] += float(g_loss)
            
            # Compute gradient norms
            d_grad_norm = sum(tf.norm(g).numpy() for g in d_grads if g is not None)
            g_grad_norm = sum(tf.norm(g).numpy() for g in g_grads if g is not None)
            
            epoch_metrics['d_grads_norm'] += d_grad_norm
            epoch_metrics['g_grads_norm'] += g_grad_norm
            
            n_batches += 1
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
        
        print(f"  D Loss: {epoch_metrics['d_loss']:.4f}, G Loss: {epoch_metrics['g_loss']:.4f}")
        print(f"  D Grad Norm: {epoch_metrics['d_grads_norm']:.4f}, G Grad Norm: {epoch_metrics['g_grads_norm']:.4f}")
        
        # Record parameters
        tracker.record_parameters(generator, discriminator, epoch_metrics)
    
    print("\n✅ Training complete!")
    
    # Analyze evolution
    tracker.analyze_evolution()
    
    # Plot evolution
    tracker.plot_evolution()
    
    # Save history
    tracker.save_history()
    
    return tracker


if __name__ == "__main__":
    tracker = train_with_tracking(epochs=20)
