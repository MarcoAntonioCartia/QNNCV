"""
Quantum Wasserstein Loss for Raw Measurements

Enhanced loss function that operates directly on raw quantum measurements
rather than classical outputs, preserving quantum information in the optimization.

Features:
- Wasserstein distance on raw measurement space
- Quantum regularization terms
- Gradient penalty for Lipschitz constraint
- Transformation matrix regularization
"""

import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)


class QuantumRawMeasurementWassersteinLoss:
    """
    Wasserstein loss operating on raw quantum measurements.
    
    This loss function computes Wasserstein distance directly on the
    raw measurement space, preserving quantum information that would
    be lost in classical post-processing.
    """
    
    def __init__(self, lambda_gp=10.0, lambda_transform=1.0, lambda_quantum=0.5):
        """
        Initialize quantum raw measurement Wasserstein loss.
        
        Args:
            lambda_gp (float): Gradient penalty weight
            lambda_transform (float): Transformation regularization weight
            lambda_quantum (float): Quantum regularization weight
        """
        self.lambda_gp = lambda_gp
        self.lambda_transform = lambda_transform
        self.lambda_quantum = lambda_quantum
        
        logger.info(f"Quantum Raw Measurement Wasserstein Loss initialized:")
        logger.info(f"  - Gradient penalty: {lambda_gp}")
        logger.info(f"  - Transformation regularization: {lambda_transform}")
        logger.info(f"  - Quantum regularization: {lambda_quantum}")
    
    def __call__(self, real_data, generator, discriminator, batch_size=None):
        """
        Compute quantum-enhanced Wasserstein loss on raw measurements.
        
        Args:
            real_data: Real training data [batch_size, input_dim]
            generator: Pure quantum generator
            discriminator: Pure quantum discriminator
            batch_size: Optional batch size override
            
        Returns:
            tuple: (discriminator_loss, generator_loss, metrics_dict)
        """
        if batch_size is None:
            batch_size = tf.shape(real_data)[0]
        
        # 1. Generate fake data
        z = tf.random.normal([batch_size, generator.latent_dim])
        fake_data = generator.generate(z)
        
        # 2. Get raw measurements from both real and fake data
        real_measurements = discriminator.get_raw_measurements(real_data)
        fake_measurements = discriminator.get_raw_measurements(fake_data)
        
        # 3. Compute Wasserstein distance on raw measurement space
        w_distance = self._compute_raw_measurement_wasserstein(
            real_measurements, fake_measurements, discriminator
        )
        
        # 4. Gradient penalty on raw measurement space
        gradient_penalty = self._compute_raw_measurement_gradient_penalty(
            real_measurements, fake_measurements, discriminator
        )
        
        # 5. Transformation regularization
        transform_reg = self._compute_transformation_regularization(
            generator, discriminator
        )
        
        # 6. Quantum regularization
        quantum_reg = self._compute_quantum_regularization(
            generator, discriminator
        )
        
        # 7. Compute final losses
        d_loss = -w_distance + gradient_penalty + transform_reg + quantum_reg
        g_loss = -self._compute_generator_loss_on_measurements(
            generator, discriminator, batch_size
        ) + transform_reg + quantum_reg
        
        # 8. Compile metrics
        metrics = {
            'w_distance': w_distance,
            'gradient_penalty': gradient_penalty,
            'transform_regularization': transform_reg,
            'quantum_regularization': quantum_reg,
            'total_d_loss': d_loss,
            'total_g_loss': g_loss,
            'raw_measurement_dim': real_measurements.shape[1]
        }
        
        return d_loss, g_loss, metrics
    
    def _compute_raw_measurement_wasserstein(self, real_measurements, fake_measurements, discriminator):
        """
        Compute Wasserstein distance on raw measurement space.
        
        Args:
            real_measurements: Raw measurements from real data [batch_size, measurement_dim]
            fake_measurements: Raw measurements from fake data [batch_size, measurement_dim]
            discriminator: Discriminator network
            
        Returns:
            tensor: Wasserstein distance estimate
        """
        # Discriminate on raw measurements directly
        real_scores = discriminator.classify_measurements(real_measurements)
        fake_scores = discriminator.classify_measurements(fake_measurements)
        
        # Wasserstein distance approximation
        w_distance = tf.reduce_mean(real_scores) - tf.reduce_mean(fake_scores)
        
        return w_distance
    
    def _compute_raw_measurement_gradient_penalty(self, real_measurements, fake_measurements, discriminator):
        """
        Compute gradient penalty on raw measurement space.
        
        Args:
            real_measurements: Real raw measurements
            fake_measurements: Fake raw measurements
            discriminator: Discriminator network
            
        Returns:
            tensor: Gradient penalty
        """
        batch_size = tf.shape(real_measurements)[0]
        measurement_dim = tf.shape(real_measurements)[1]
        
        # Random interpolation between real and fake measurements
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interpolated_measurements = alpha * real_measurements + (1 - alpha) * fake_measurements
        
        # Compute gradients with respect to interpolated measurements
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_measurements)
            interp_scores = discriminator.classify_measurements(interpolated_measurements)
        
        gradients = gp_tape.gradient(interp_scores, interpolated_measurements)
        
        # Gradient penalty
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-12)
        gradient_penalty = self.lambda_gp * tf.reduce_mean(tf.square(gradient_norm - 1.0))
        
        return gradient_penalty
    
    def _compute_transformation_regularization(self, generator, discriminator):
        """
        Compute regularization for transformation matrices.
        
        Args:
            generator: Pure quantum generator
            discriminator: Pure quantum discriminator
            
        Returns:
            tensor: Transformation regularization loss
        """
        # Generator transformation regularization
        gen_reg = generator.compute_transformation_regularization()
        
        # Discriminator transformation regularization
        disc_reg = discriminator.compute_transformation_regularization()
        
        total_transform_reg = self.lambda_transform * (gen_reg + disc_reg)
        
        return total_transform_reg
    
    def _compute_quantum_regularization(self, generator, discriminator):
        """
        Compute quantum-specific regularization terms.
        
        Args:
            generator: Pure quantum generator
            discriminator: Pure quantum discriminator
            
        Returns:
            tensor: Quantum regularization loss
        """
        # 1. Encourage diverse quantum states (entropy)
        entropy_reg = self._compute_quantum_entropy_regularization(generator)
        
        # 2. Encourage physically valid quantum states
        physics_reg = self._compute_physics_regularization(generator, discriminator)
        
        # 3. Measurement consistency regularization
        measurement_reg = self._compute_measurement_consistency_regularization(generator)
        
        total_quantum_reg = self.lambda_quantum * (entropy_reg + physics_reg + measurement_reg)
        
        return total_quantum_reg
    
    def _compute_quantum_entropy_regularization(self, generator):
        """Encourage high entropy in quantum states for diversity."""
        try:
            # Generate a small batch to compute entropy
            z_entropy = tf.random.normal([4, generator.latent_dim])
            quantum_encoding = generator.transform_latent(z_entropy)
            quantum_states = generator.execute_quantum_circuit(quantum_encoding)
            
            # Compute average entropy
            entropies = []
            for state in quantum_states:
                ket = state.ket()
                prob_amplitudes = tf.abs(ket) ** 2 + 1e-12
                entropy = -tf.reduce_sum(prob_amplitudes * tf.math.log(prob_amplitudes))
                entropies.append(entropy)
            
            avg_entropy = tf.reduce_mean(tf.stack(entropies))
            
            # Encourage high entropy (diverse states)
            entropy_reg = -avg_entropy  # Negative because we want to maximize entropy
            
            return entropy_reg
            
        except Exception as e:
            logger.warning(f"Entropy regularization failed: {e}")
            return tf.constant(0.0)
    
    def _compute_physics_regularization(self, generator, discriminator):
        """Encourage physically valid quantum states."""
        try:
            # Generate states and check physical validity
            z_physics = tf.random.normal([2, generator.latent_dim])
            quantum_encoding = generator.transform_latent(z_physics)
            quantum_states = generator.execute_quantum_circuit(quantum_encoding)
            
            # Check normalization
            norm_penalties = []
            for state in quantum_states:
                ket = state.ket()
                norm = tf.reduce_sum(tf.abs(ket) ** 2)
                norm_penalty = tf.square(norm - 1.0)
                norm_penalties.append(norm_penalty)
            
            avg_norm_penalty = tf.reduce_mean(tf.stack(norm_penalties))
            
            return avg_norm_penalty
            
        except Exception as e:
            logger.warning(f"Physics regularization failed: {e}")
            return tf.constant(0.0)
    
    def _compute_measurement_consistency_regularization(self, generator):
        """Encourage consistency in measurement outcomes."""
        try:
            # Generate the same latent vector multiple times and check measurement consistency
            z_base = tf.random.normal([1, generator.latent_dim])
            z_repeated = tf.tile(z_base, [3, 1])  # Repeat 3 times
            
            raw_measurements = generator.get_raw_measurements(z_repeated)
            
            # Compute variance across repeated measurements (should be low due to quantum noise only)
            measurement_var = tf.reduce_mean(tf.math.reduce_variance(raw_measurements, axis=0))
            
            # Encourage low variance (consistent measurements for same input)
            consistency_reg = measurement_var
            
            return consistency_reg
            
        except Exception as e:
            logger.warning(f"Measurement consistency regularization failed: {e}")
            return tf.constant(0.0)
    
    def _compute_generator_loss_on_measurements(self, generator, discriminator, batch_size):
        """
        Compute generator loss based on raw measurements.
        
        Args:
            generator: Pure quantum generator
            discriminator: Pure quantum discriminator
            batch_size: Batch size
            
        Returns:
            tensor: Generator loss
        """
        # Generate fake data
        z = tf.random.normal([batch_size, generator.latent_dim])
        fake_data = generator.generate(z)
        
        # Get raw measurements from fake data
        fake_measurements = discriminator.get_raw_measurements(fake_data)
        
        # Discriminate on raw measurements
        fake_scores = discriminator.classify_measurements(fake_measurements)
        
        # Generator wants to fool discriminator
        g_loss = tf.reduce_mean(fake_scores)
        
        return g_loss
    
    def compute_measurement_statistics(self, real_data, generator, discriminator):
        """
        Compute useful statistics about raw measurements for monitoring.
        
        Args:
            real_data: Real training data
            generator: Pure quantum generator
            discriminator: Pure quantum discriminator
            
        Returns:
            dict: Measurement statistics
        """
        batch_size = tf.shape(real_data)[0]
        
        # Generate fake data
        z = tf.random.normal([batch_size, generator.latent_dim])
        fake_data = generator.generate(z)
        
        # Get raw measurements
        real_measurements = discriminator.get_raw_measurements(real_data)
        fake_measurements = discriminator.get_raw_measurements(fake_data)
        
        # Compute statistics
        stats = {
            'real_measurements_mean': tf.reduce_mean(real_measurements, axis=0),
            'real_measurements_std': tf.math.reduce_std(real_measurements, axis=0),
            'fake_measurements_mean': tf.reduce_mean(fake_measurements, axis=0),
            'fake_measurements_std': tf.math.reduce_std(fake_measurements, axis=0),
            'measurement_distance': tf.reduce_mean(tf.abs(
                tf.reduce_mean(real_measurements, axis=0) - 
                tf.reduce_mean(fake_measurements, axis=0)
            )),
            'measurement_correlation': self._compute_measurement_correlation(
                real_measurements, fake_measurements
            )
        }
        
        return stats
    
    def _compute_measurement_correlation(self, real_measurements, fake_measurements):
        """Compute correlation between real and fake measurement distributions."""
        try:
            # Flatten measurements
            real_flat = tf.reshape(real_measurements, [-1])
            fake_flat = tf.reshape(fake_measurements, [-1])
            
            # Compute correlation coefficient
            real_mean = tf.reduce_mean(real_flat)
            fake_mean = tf.reduce_mean(fake_flat)
            
            numerator = tf.reduce_mean((real_flat - real_mean) * (fake_flat - fake_mean))
            
            real_var = tf.reduce_mean(tf.square(real_flat - real_mean))
            fake_var = tf.reduce_mean(tf.square(fake_flat - fake_mean))
            
            denominator = tf.sqrt(real_var * fake_var) + 1e-12
            
            correlation = numerator / denominator
            
            return correlation
            
        except Exception as e:
            logger.warning(f"Correlation computation failed: {e}")
            return tf.constant(0.0)


class QuantumMMDLossRawMeasurements:
    """
    Alternative: Maximum Mean Discrepancy loss on raw measurements.
    
    Useful for comparison with Wasserstein approach.
    """
    
    def __init__(self, sigma=1.0, lambda_transform=1.0, lambda_quantum=0.5):
        """
        Initialize MMD loss for raw measurements.
        
        Args:
            sigma (float): Gaussian kernel bandwidth
            lambda_transform (float): Transformation regularization weight
            lambda_quantum (float): Quantum regularization weight
        """
        self.sigma = sigma
        self.lambda_transform = lambda_transform
        self.lambda_quantum = lambda_quantum
    
    def __call__(self, real_data, generator, discriminator, batch_size=None):
        """Compute MMD loss on raw measurement space."""
        if batch_size is None:
            batch_size = tf.shape(real_data)[0]
        
        # Generate fake data
        z = tf.random.normal([batch_size, generator.latent_dim])
        fake_data = generator.generate(z)
        
        # Get raw measurements
        real_measurements = discriminator.get_raw_measurements(real_data)
        fake_measurements = discriminator.get_raw_measurements(fake_data)
        
        # Compute MMD on raw measurements
        mmd_loss = self._compute_mmd(real_measurements, fake_measurements)
        
        # Add regularization
        transform_reg = self.lambda_transform * (
            generator.compute_transformation_regularization() +
            discriminator.compute_transformation_regularization()
        )
        
        # For MMD, both generator and discriminator use the same loss
        total_loss = mmd_loss + transform_reg
        
        metrics = {
            'mmd_loss': mmd_loss,
            'transform_regularization': transform_reg,
            'total_loss': total_loss
        }
        
        return total_loss, total_loss, metrics
    
    def _compute_mmd(self, x, y):
        """Compute MMD between two measurement distributions."""
        # Gaussian kernel
        def gaussian_kernel(a, b):
            return tf.exp(-tf.reduce_sum(tf.square(a - b), axis=1) / (2 * self.sigma**2))
        
        # MMD computation
        x_kernel = tf.reduce_mean(gaussian_kernel(x[:, None], x[None, :]))
        y_kernel = tf.reduce_mean(gaussian_kernel(y[:, None], y[None, :]))
        xy_kernel = tf.reduce_mean(gaussian_kernel(x[:, None], y[None, :]))
        
        mmd = x_kernel + y_kernel - 2 * xy_kernel
        
        return mmd


def test_quantum_raw_measurement_loss():
    """Test the quantum raw measurement loss functions."""
    print("\n" + "="*60)
    print("TESTING QUANTUM RAW MEASUREMENT LOSSES")
    print("="*60)
    
    # Import the components (would need actual imports in practice)
    # from pure_quantum_generator import PureQuantumGenerator
    # from pure_quantum_discriminator import PureQuantumDiscriminator
    
    print("⚠️  Note: This test requires PureQuantumGenerator and PureQuantumDiscriminator")
    print("         Run this after creating those components.")
    
    # Create loss function
    loss_fn = QuantumRawMeasurementWassersteinLoss(
        lambda_gp=10.0,
        lambda_transform=1.0,
        lambda_quantum=0.5
    )
    
    print(f"✅ Quantum Wasserstein loss created")
    
    # Create MMD loss for comparison
    mmd_loss_fn = QuantumMMDLossRawMeasurements(
        sigma=1.0,
        lambda_transform=1.0,
        lambda_quantum=0.5
    )
    
    print(f"✅ Quantum MMD loss created")
    
    print(f"\n🎯 Loss functions ready for testing with actual quantum models")
    
    return loss_fn, mmd_loss_fn


if __name__ == "__main__":
    test_quantum_raw_measurement_loss()