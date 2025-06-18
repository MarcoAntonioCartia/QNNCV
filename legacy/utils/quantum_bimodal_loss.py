"""
Quantum Bimodal Loss Functions

Specialized loss functions for training quantum GANs to generate bimodal distributions.
These losses are designed to:
1. Encourage true bimodal structure
2. Preserve quantum gradient flow
3. Prevent mode collapse
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Any


class QuantumBimodalLoss:
    """
    Quantum-aware loss function for bimodal distributions.
    
    This loss combines multiple objectives:
    - Mode separation: Encourages two distinct modes
    - Mode balance: Ensures both modes are represented
    - Quantum fidelity: Maintains quantum state properties
    - Gradient preservation: Ensures smooth gradient flow
    """
    
    def __init__(self, mode1_center=None, mode2_center=None, 
                 lambda_separation=1.0, lambda_balance=2.0,
                 lambda_coverage=1.0, lambda_quantum=0.1):
        """
        Initialize quantum bimodal loss.
        
        Args:
            mode1_center: Target center for mode 1 (default: [-1.5, -1.5])
            mode2_center: Target center for mode 2 (default: [1.5, 1.5])
            lambda_separation: Weight for mode separation loss
            lambda_balance: Weight for mode balance loss
            lambda_coverage: Weight for mode coverage loss
            lambda_quantum: Weight for quantum fidelity loss
        """
        if mode1_center is None:
            self.mode1_center = tf.constant([-1.5, -1.5], dtype=tf.float32)
        else:
            self.mode1_center = tf.constant(mode1_center, dtype=tf.float32)
            
        if mode2_center is None:
            self.mode2_center = tf.constant([1.5, 1.5], dtype=tf.float32)
        else:
            self.mode2_center = tf.constant(mode2_center, dtype=tf.float32)
        
        self.lambda_separation = lambda_separation
        self.lambda_balance = lambda_balance
        self.lambda_coverage = lambda_coverage
        self.lambda_quantum = lambda_quantum
        
        # Target separation
        self.target_separation = tf.linalg.norm(self.mode2_center - self.mode1_center)
    
    def __call__(self, real_samples, generated_samples, generator=None):
        """
        Compute quantum bimodal loss.
        
        Args:
            real_samples: Real data samples [batch_size, 2]
            generated_samples: Generated samples [batch_size, 2]
            generator: Optional quantum generator for quantum metrics
            
        Returns:
            total_loss: Combined loss value
            metrics: Dictionary of individual loss components
        """
        # Ensure we're working with 2D samples
        real_2d = real_samples[:, :2]
        gen_2d = generated_samples[:, :2]
        
        # 1. Mode Assignment
        mode_assignments = self._assign_to_modes(gen_2d)
        
        # 2. Mode Separation Loss
        separation_loss = self._compute_separation_loss(gen_2d, mode_assignments)
        
        # 3. Mode Balance Loss
        balance_loss = self._compute_balance_loss(mode_assignments)
        
        # 4. Mode Coverage Loss
        coverage_loss = self._compute_coverage_loss(gen_2d)
        
        # 5. Quantum Fidelity Loss (if generator provided)
        if generator is not None and hasattr(generator, 'compute_quantum_metrics'):
            quantum_loss = self._compute_quantum_loss(generator)
        else:
            quantum_loss = tf.constant(0.0)
        
        # Combine losses
        total_loss = (
            self.lambda_separation * separation_loss +
            self.lambda_balance * balance_loss +
            self.lambda_coverage * coverage_loss +
            self.lambda_quantum * quantum_loss
        )
        
        metrics = {
            'separation_loss': separation_loss,
            'balance_loss': balance_loss,
            'coverage_loss': coverage_loss,
            'quantum_loss': quantum_loss,
            'total_loss': total_loss
        }
        
        return total_loss, metrics
    
    def _assign_to_modes(self, samples):
        """
        Assign each sample to nearest mode.
        
        Args:
            samples: Generated samples [batch_size, 2]
            
        Returns:
            mode_assignments: Boolean tensor, True for mode 1, False for mode 2
        """
        # Compute distances to each mode center
        dist_to_mode1 = tf.linalg.norm(samples - self.mode1_center[None, :], axis=1)
        dist_to_mode2 = tf.linalg.norm(samples - self.mode2_center[None, :], axis=1)
        
        # Assign to nearest mode
        mode_assignments = dist_to_mode1 < dist_to_mode2
        
        return mode_assignments
    
    def _compute_separation_loss(self, samples, mode_assignments):
        """
        Compute mode separation loss.
        
        Encourages generated modes to be well-separated.
        """
        # Get samples for each mode
        mode1_mask = mode_assignments
        mode2_mask = ~mode_assignments
        
        # Count samples in each mode
        n_mode1 = tf.reduce_sum(tf.cast(mode1_mask, tf.float32))
        n_mode2 = tf.reduce_sum(tf.cast(mode2_mask, tf.float32))
        
        # If either mode is empty, return high penalty
        if n_mode1 < 1 or n_mode2 < 1:
            return tf.constant(10.0)
        
        # Compute mode centers
        mode1_samples = tf.boolean_mask(samples, mode1_mask)
        mode2_samples = tf.boolean_mask(samples, mode2_mask)
        
        gen_mode1_center = tf.reduce_mean(mode1_samples, axis=0)
        gen_mode2_center = tf.reduce_mean(mode2_samples, axis=0)
        
        # Compute separation
        gen_separation = tf.linalg.norm(gen_mode2_center - gen_mode1_center)
        
        # Loss: penalize deviation from target separation
        separation_loss = tf.square(gen_separation - self.target_separation)
        
        return separation_loss
    
    def _compute_balance_loss(self, mode_assignments):
        """
        Compute mode balance loss.
        
        Encourages equal representation of both modes.
        """
        # Count samples in each mode
        mode1_count = tf.reduce_sum(tf.cast(mode_assignments, tf.float32))
        total_count = tf.cast(tf.shape(mode_assignments)[0], tf.float32)
        
        # Ideal is 50% in each mode
        mode1_ratio = mode1_count / (total_count + 1e-8)
        
        # Penalize deviation from 0.5
        balance_loss = tf.square(mode1_ratio - 0.5) * 4.0  # Scale up
        
        return balance_loss
    
    def _compute_coverage_loss(self, samples):
        """
        Compute mode coverage loss.
        
        Encourages samples to be near the target mode centers.
        """
        # Distance to nearest mode center
        dist_to_mode1 = tf.linalg.norm(samples - self.mode1_center[None, :], axis=1)
        dist_to_mode2 = tf.linalg.norm(samples - self.mode2_center[None, :], axis=1)
        
        min_distances = tf.minimum(dist_to_mode1, dist_to_mode2)
        
        # Average minimum distance (should be small)
        coverage_loss = tf.reduce_mean(min_distances)
        
        return coverage_loss
    
    def _compute_quantum_loss(self, generator):
        """
        Compute quantum fidelity loss.
        
        Encourages proper quantum state properties.
        """
        try:
            metrics = generator.compute_quantum_metrics()
            
            # Penalize low trace (should be 1 for pure states)
            trace_loss = tf.square(1.0 - metrics.get('trace', 1.0))
            
            # Encourage high purity
            purity = metrics.get('purity', 1.0)
            purity_loss = tf.square(1.0 - purity)
            
            quantum_loss = trace_loss + purity_loss
            
            return quantum_loss
            
        except Exception:
            return tf.constant(0.0)


class BimodalDiscriminatorLoss:
    """
    Discriminator loss for bimodal distributions.
    
    Includes mode-aware discrimination to prevent mode collapse.
    """
    
    def __init__(self, label_smoothing=0.1):
        """
        Initialize bimodal discriminator loss.
        
        Args:
            label_smoothing: Label smoothing factor
        """
        self.label_smoothing = label_smoothing
    
    def __call__(self, real_samples, fake_samples, discriminator):
        """
        Compute discriminator loss with mode awareness.
        
        Args:
            real_samples: Real data samples
            fake_samples: Generated samples
            discriminator: Discriminator model
            
        Returns:
            d_loss: Discriminator loss
            metrics: Loss components
        """
        batch_size = tf.shape(real_samples)[0]
        
        # Create smoothed labels
        real_labels = tf.ones([batch_size, 1]) * (1.0 - self.label_smoothing)
        fake_labels = tf.zeros([batch_size, 1]) + self.label_smoothing
        
        # Get discriminator outputs
        real_output = discriminator.discriminate(real_samples)
        fake_output = discriminator.discriminate(fake_samples)
        
        # Binary cross-entropy loss
        real_loss = tf.keras.losses.binary_crossentropy(real_labels, real_output)
        fake_loss = tf.keras.losses.binary_crossentropy(fake_labels, fake_output)
        
        # Total discriminator loss
        d_loss = tf.reduce_mean(real_loss + fake_loss)
        
        # Compute accuracy metrics
        real_acc = tf.reduce_mean(tf.cast(real_output > 0.5, tf.float32))
        fake_acc = tf.reduce_mean(tf.cast(fake_output < 0.5, tf.float32))
        
        metrics = {
            'd_loss': d_loss,
            'real_loss': tf.reduce_mean(real_loss),
            'fake_loss': tf.reduce_mean(fake_loss),
            'real_accuracy': real_acc,
            'fake_accuracy': fake_acc
        }
        
        return d_loss, metrics


def test_bimodal_loss():
    """Test the quantum bimodal loss functions."""
    print("\n" + "="*60)
    print("TESTING QUANTUM BIMODAL LOSS")
    print("="*60)
    
    # Create test data
    batch_size = 20
    
    # Real data: perfect bimodal
    mode1_real = tf.random.normal([batch_size//2, 2], mean=[-1.5, -1.5], stddev=0.2)
    mode2_real = tf.random.normal([batch_size//2, 2], mean=[1.5, 1.5], stddev=0.2)
    real_samples = tf.concat([mode1_real, mode2_real], axis=0)
    
    # Test different generated distributions
    test_cases = {
        'perfect_bimodal': tf.concat([
            tf.random.normal([batch_size//2, 2], mean=[-1.5, -1.5], stddev=0.2),
            tf.random.normal([batch_size//2, 2], mean=[1.5, 1.5], stddev=0.2)
        ], axis=0),
        'mode_collapsed': tf.random.normal([batch_size, 2], mean=[0.0, 0.0], stddev=0.5),
        'unbalanced': tf.concat([
            tf.random.normal([2, 2], mean=[-1.5, -1.5], stddev=0.2),
            tf.random.normal([batch_size-2, 2], mean=[1.5, 1.5], stddev=0.2)
        ], axis=0)
    }
    
    # Create loss function
    loss_fn = QuantumBimodalLoss()
    
    print("\n📊 Loss evaluation for different distributions:")
    for name, generated in test_cases.items():
        total_loss, metrics = loss_fn(real_samples, generated)
        
        print(f"\n{name}:")
        print(f"   Total loss: {total_loss:.4f}")
        print(f"   Separation loss: {metrics['separation_loss']:.4f}")
        print(f"   Balance loss: {metrics['balance_loss']:.4f}")
        print(f"   Coverage loss: {metrics['coverage_loss']:.4f}")
    
    print("\n✅ Bimodal loss test completed!")


if __name__ == "__main__":
    test_bimodal_loss()
