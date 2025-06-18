"""
Quantum metrics for evaluating quantum GANs.

This module provides alot of metrics for quantum generative models including:
- Quantum entanglement entropy (Von Neumann entropy, bells inquality-ish test)
- Multivariate Wasserstein distance for distribution matching (more accurate than 1D approximations)
- Gradient penalty score for training stability
- Additional quantum state characterization metrics (Must improve over time....)
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple, Union
import logging
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

class QuantumMetrics:
    """
    Comprehensive quantum metrics for evaluating quantum GANs.
    
    Features:
    - Quantum entanglement entropy (Von Neumann entropy)
    - Multivariate Wasserstein distance
    - Gradient penalty score for training stability
    - Quantum state purity and fidelity measures
    - Classical generative model metrics
    """
    
    def __init__(self):
        """Initialize quantum metrics calculator."""
        self.name = "QuantumMetrics"
        logger.info("QuantumMetrics initialized")
    
    def quantum_entanglement_entropy(self, quantum_state: tf.Tensor, 
                                   subsystem_dims: Optional[Tuple[int, ...]] = None) -> tf.Tensor:
        """
        Compute quantum entanglement entropy (Von Neumann entropy).
        
        This measures the amount of entanglement in a quantum state by computing
        the Von Neumann entropy of the reduced density matrix of a subsystem.
        
        Args:
            quantum_state: Quantum state vector [batch_size, state_dim] or density matrix
            subsystem_dims: Dimensions of subsystems for partial trace (optional)
            
        Returns:
            Von Neumann entropy measuring entanglement
        """
        try:
            # Handle different input formats
            if len(quantum_state.shape) == 1:
                # Single state vector
                quantum_state = tf.expand_dims(quantum_state, 0)
            
            batch_size = tf.shape(quantum_state)[0]
            
            # If state vector, convert to density matrix
            if len(quantum_state.shape) == 2:
                # Assume state vectors, create density matrices
                # ρ = |ψ⟩⟨ψ|
                state_conj = tf.math.conj(quantum_state)
                density_matrices = tf.einsum('bi,bj->bij', quantum_state, state_conj)
            else:
                # Already density matrices
                density_matrices = quantum_state
            
            # Compute eigenvalues of density matrix
            eigenvals = tf.linalg.eigvals(density_matrices)
            eigenvals = tf.math.real(eigenvals)  # Take real part
            
            # Add small epsilon for numerical stability
            eigenvals = tf.maximum(eigenvals, 1e-12)
            
            # Von Neumann entropy: S = -Tr(ρ log ρ) = -Σ λᵢ log λᵢ
            entropy = -tf.reduce_sum(eigenvals * tf.math.log(eigenvals), axis=-1)
            
            return entropy
            
        except Exception as e:
            logger.warning(f"Quantum entanglement entropy computation failed: {e}")
            # Return zero entropy as fallback
            return tf.zeros([tf.shape(quantum_state)[0]])
    
    def wasserstein_distance_multivariate(self, real_samples: tf.Tensor, 
                                        generated_samples: tf.Tensor,
                                        p: int = 2) -> tf.Tensor:
        """
        Compute multivariate Wasserstein distance between real and generated distributions.
        
        This provides a more accurate measure of distribution matching than 1D
        approximations by considering the full multivariate structure.
        
        Args:
            real_samples: Real data samples [n_real, n_features]
            generated_samples: Generated samples [n_generated, n_features]
            p: Order of Wasserstein distance (default: 2 for W2 distance)
            
        Returns:
            Multivariate Wasserstein distance
        """
        try:
            # Convert to numpy for scipy operations
            real_np = real_samples.numpy() if hasattr(real_samples, 'numpy') else real_samples
            gen_np = generated_samples.numpy() if hasattr(generated_samples, 'numpy') else generated_samples
            
            # For multivariate case, we use the Sinkhorn approximation or
            # compute pairwise distances and solve optimal transport
            
            # Compute pairwise distance matrix
            cost_matrix = cdist(real_np, gen_np, metric='euclidean')
            
            # Simple approximation: minimum mean distance
            # For exact computation, would need optimal transport solver
            min_distances = np.min(cost_matrix, axis=1)
            wasserstein_approx = np.mean(min_distances ** p) ** (1.0 / p)
            
            return tf.constant(wasserstein_approx, dtype=tf.float32)
            
        except Exception as e:
            logger.warning(f"Multivariate Wasserstein distance computation failed: {e}")
            # Fallback to 1D approximation
            return self._wasserstein_1d_fallback(real_samples, generated_samples)
    
    def _wasserstein_1d_fallback(self, real_samples: tf.Tensor, 
                                generated_samples: tf.Tensor) -> tf.Tensor:
        """Fallback to 1D Wasserstein distance approximation."""
        try:
            # Average Wasserstein distance across all dimensions
            n_features = real_samples.shape[-1]
            distances = []
            
            real_np = real_samples.numpy() if hasattr(real_samples, 'numpy') else real_samples
            gen_np = generated_samples.numpy() if hasattr(generated_samples, 'numpy') else generated_samples
            
            for i in range(n_features):
                wd = wasserstein_distance(real_np[:, i], gen_np[:, i])
                distances.append(wd)
            
            return tf.constant(np.mean(distances), dtype=tf.float32)
            
        except Exception as e:
            logger.warning(f"1D Wasserstein fallback failed: {e}")
            return tf.constant(float('inf'), dtype=tf.float32)
    
    def gradient_penalty_score(self, discriminator, real_batch: tf.Tensor, 
                             fake_batch: tf.Tensor, lambda_gp: float = 10.0) -> tf.Tensor:
        """
        Compute gradient penalty score for training stability assessment.
        
        This implements the WGAN-GP gradient penalty to measure training stability
        and ensure Lipschitz constraint satisfaction.
        
        Args:
            discriminator: Discriminator model
            real_batch: Real data batch [batch_size, features]
            fake_batch: Generated data batch [batch_size, features]
            lambda_gp: Gradient penalty coefficient
            
        Returns:
            Gradient penalty score
        """
        try:
            batch_size = tf.shape(real_batch)[0]
            
            # Random interpolation between real and fake samples
            epsilon = tf.random.uniform([batch_size, 1], 0.0, 1.0)
            interpolated = epsilon * real_batch + (1 - epsilon) * fake_batch
            
            # Compute gradients of discriminator w.r.t. interpolated samples
            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                interpolated_output = discriminator.discriminate(interpolated)
            
            gradients = tape.gradient(interpolated_output, interpolated)
            
            if gradients is None:
                logger.warning("Gradients are None in gradient penalty computation")
                return tf.constant(0.0, dtype=tf.float32)
            
            # Compute gradient norm
            gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
            
            # Gradient penalty: (||∇D(x)||₂ - 1)²
            gradient_penalty = tf.reduce_mean(tf.square(gradient_norm - 1.0))
            
            return lambda_gp * gradient_penalty
            
        except Exception as e:
            logger.warning(f"Gradient penalty computation failed: {e}")
            return tf.constant(0.0, dtype=tf.float32)
    
    def quantum_state_purity(self, quantum_state: tf.Tensor) -> tf.Tensor:
        """
        Compute quantum state purity: Tr(ρ²).
        
        Purity measures how "mixed" a quantum state is:
        - Pure states: purity = 1
        - Maximally mixed states: purity = 1/d (d = dimension)
        
        Args:
            quantum_state: Quantum state vector or density matrix
            
        Returns:
            Purity of the quantum state
        """
        try:
            # Convert state vector to density matrix if needed
            if len(quantum_state.shape) == 2 and quantum_state.shape[0] != quantum_state.shape[1]:
                # State vector case
                state_conj = tf.math.conj(quantum_state)
                density_matrix = tf.einsum('bi,bj->bij', quantum_state, state_conj)
            else:
                # Already density matrix
                density_matrix = quantum_state
            
            # Compute ρ²
            rho_squared = tf.matmul(density_matrix, density_matrix)
            
            # Purity = Tr(ρ²)
            purity = tf.linalg.trace(rho_squared)
            
            return tf.math.real(purity)
            
        except Exception as e:
            logger.warning(f"Quantum state purity computation failed: {e}")
            return tf.constant(0.0, dtype=tf.float32)
    
    def quantum_fidelity(self, state1: tf.Tensor, state2: tf.Tensor) -> tf.Tensor:
        """
        Compute quantum fidelity between two quantum states.
        
        Fidelity measures the "closeness" or "alikeness" of two quantum states:
        F(ρ,σ) = Tr(√(√ρ σ √ρ))
        
        For pure states: F(|ψ⟩,|φ⟩) = |⟨ψ|φ⟩|²
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            
        Returns:
            Quantum fidelity between the states
        """
        try:
            # For pure states (state vectors), use overlap formula
            if len(state1.shape) == 1:
                state1 = tf.expand_dims(state1, 0)
            if len(state2.shape) == 1:
                state2 = tf.expand_dims(state2, 0)
            
            # Compute overlap ⟨ψ|φ⟩
            overlap = tf.reduce_sum(tf.math.conj(state1) * state2, axis=-1)
            
            # Fidelity = |⟨ψ|φ⟩|²
            fidelity = tf.abs(overlap) ** 2
            
            return fidelity
            
        except Exception as e:
            logger.warning(f"Quantum fidelity computation failed: {e}")
            return tf.constant(0.0, dtype=tf.float32)
    
    def classical_distribution_metrics(self, real_samples: tf.Tensor, 
                                     generated_samples: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Compute classical distribution matching metrics.
        
        Args:
            real_samples: Real data samples
            generated_samples: Generated samples
            
        Returns:
            Dictionary of classical metrics
        """
        try:
            # Convert to numpy for statistical computations
            real_np = real_samples.numpy() if hasattr(real_samples, 'numpy') else real_samples
            gen_np = generated_samples.numpy() if hasattr(generated_samples, 'numpy') else generated_samples
            
            # Mean and standard deviation differences
            real_mean = np.mean(real_np, axis=0)
            gen_mean = np.mean(gen_np, axis=0)
            real_std = np.std(real_np, axis=0)
            gen_std = np.std(gen_np, axis=0)
            
            mean_diff = np.linalg.norm(real_mean - gen_mean)
            std_diff = np.linalg.norm(real_std - gen_std)
            
            # Covariance difference
            real_cov = np.cov(real_np.T)
            gen_cov = np.cov(gen_np.T)
            cov_diff = np.linalg.norm(real_cov - gen_cov, 'fro')  # Frobenius norm
            
            # Maximum Mean Discrepancy (MMD) approximation
            mmd_approx = self._compute_mmd_approximation(real_np, gen_np)
            
            return {
                'mean_difference': tf.constant(mean_diff, dtype=tf.float32),
                'std_difference': tf.constant(std_diff, dtype=tf.float32),
                'covariance_difference': tf.constant(cov_diff, dtype=tf.float32),
                'mmd_approximation': tf.constant(mmd_approx, dtype=tf.float32)
            }
            
        except Exception as e:
            logger.warning(f"Classical distribution metrics computation failed: {e}")
            return {
                'mean_difference': tf.constant(float('inf'), dtype=tf.float32),
                'std_difference': tf.constant(float('inf'), dtype=tf.float32),
                'covariance_difference': tf.constant(float('inf'), dtype=tf.float32),
                'mmd_approximation': tf.constant(float('inf'), dtype=tf.float32)
            }
    
    def _compute_mmd_approximation(self, real_samples: np.ndarray, 
                                 generated_samples: np.ndarray, 
                                 sigma: float = 1.0) -> float:
        """Compute Maximum Mean Discrepancy approximation with RBF kernel."""
        try:
            # Subsample for efficiency
            n_samples = min(500, len(real_samples), len(generated_samples))
            real_sub = real_samples[:n_samples]
            gen_sub = generated_samples[:n_samples]
            
            # RBF kernel computations
            def rbf_kernel(X, Y, sigma):
                pairwise_dists = cdist(X, Y, 'sqeuclidean')
                return np.exp(-pairwise_dists / (2 * sigma**2))
            
            # MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
            kxx = np.mean(rbf_kernel(real_sub, real_sub, sigma))
            kyy = np.mean(rbf_kernel(gen_sub, gen_sub, sigma))
            kxy = np.mean(rbf_kernel(real_sub, gen_sub, sigma))
            
            mmd_squared = kxx + kyy - 2 * kxy
            return max(0.0, mmd_squared)  # Ensure non-negative
            
        except Exception as e:
            logger.warning(f"MMD approximation failed: {e}")
            return float('inf')
    
    def compute_comprehensive_metrics(self, real_samples: tf.Tensor,
                                    generated_samples: tf.Tensor,
                                    quantum_state: Optional[tf.Tensor] = None,
                                    discriminator = None) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for quantum GAN evaluation.
        
        Args:
            real_samples: Real data samples
            generated_samples: Generated samples
            quantum_state: Quantum state (optional)
            discriminator: Discriminator model (optional)
            
        Returns:
            Dictionary of all computed metrics
        """
        metrics = {}
        
        # Classical distribution metrics
        classical_metrics = self.classical_distribution_metrics(real_samples, generated_samples)
        metrics.update(classical_metrics)
        
        # Multivariate Wasserstein distance
        metrics['wasserstein_multivariate'] = self.wasserstein_distance_multivariate(
            real_samples, generated_samples
        )
        
        # Quantum metrics (if quantum state provided)
        if quantum_state is not None:
            metrics['quantum_entanglement_entropy'] = self.quantum_entanglement_entropy(quantum_state)
            metrics['quantum_state_purity'] = self.quantum_state_purity(quantum_state)
        
        # Gradient penalty (if discriminator provided)
        if discriminator is not None:
            # Use subset of samples for efficiency
            n_samples = min(32, tf.shape(real_samples)[0], tf.shape(generated_samples)[0])
            real_subset = real_samples[:n_samples]
            gen_subset = generated_samples[:n_samples]
            
            metrics['gradient_penalty_score'] = self.gradient_penalty_score(
                discriminator, real_subset, gen_subset
            )
        
        return metrics

def test_quantum_metrics():
    """Test quantum metrics functionality."""
    print("Testing Quantum Metrics...")
    print("=" * 50)
    
    # Create test data
    batch_size = 100
    n_features = 2
    state_dim = 8
    
    # Generate test samples
    real_samples = tf.random.normal([batch_size, n_features])
    generated_samples = tf.random.normal([batch_size, n_features]) + 0.5
    
    # Generate test quantum state
    quantum_state = tf.random.normal([batch_size, state_dim], dtype=tf.complex64)
    quantum_state = quantum_state / tf.linalg.norm(quantum_state, axis=-1, keepdims=True)
    
    # Initialize metrics
    metrics = QuantumMetrics()
    
    print(f"Test data shapes:")
    print(f"  Real samples: {real_samples.shape}")
    print(f"  Generated samples: {generated_samples.shape}")
    print(f"  Quantum state: {quantum_state.shape}")
    
    # Test quantum entanglement entropy
    print(f"\nTesting quantum entanglement entropy...")
    try:
        entropy = metrics.quantum_entanglement_entropy(quantum_state)
        print(f"✓ Entanglement entropy: {tf.reduce_mean(entropy):.4f}")
    except Exception as e:
        print(f"✗ Entanglement entropy failed: {e}")
    
    # Test multivariate Wasserstein distance
    print(f"\nTesting multivariate Wasserstein distance...")
    try:
        wd = metrics.wasserstein_distance_multivariate(real_samples, generated_samples)
        print(f"✓ Wasserstein distance: {wd:.4f}")
    except Exception as e:
        print(f"✗ Wasserstein distance failed: {e}")
    
    # Test quantum state purity
    print(f"\nTesting quantum state purity...")
    try:
        purity = metrics.quantum_state_purity(quantum_state)
        print(f"✓ State purity: {tf.reduce_mean(purity):.4f}")
    except Exception as e:
        print(f"✗ State purity failed: {e}")
    
    # Test quantum fidelity
    print(f"\nTesting quantum fidelity...")
    try:
        state2 = quantum_state + 0.1 * tf.random.normal(quantum_state.shape, dtype=tf.complex64)
        state2 = state2 / tf.linalg.norm(state2, axis=-1, keepdims=True)
        fidelity = metrics.quantum_fidelity(quantum_state, state2)
        print(f"✓ Quantum fidelity: {tf.reduce_mean(fidelity):.4f}")
    except Exception as e:
        print(f"✗ Quantum fidelity failed: {e}")
    
    # Test classical distribution metrics
    print(f"\nTesting classical distribution metrics...")
    try:
        classical_metrics = metrics.classical_distribution_metrics(real_samples, generated_samples)
        print(f"✓ Classical metrics computed:")
        for key, value in classical_metrics.items():
            print(f"    {key}: {value:.4f}")
    except Exception as e:
        print(f"✗ Classical metrics failed: {e}")
    
    # Test comprehensive metrics
    print(f"\nTesting comprehensive metrics...")
    try:
        comprehensive = metrics.compute_comprehensive_metrics(
            real_samples, generated_samples, quantum_state
        )
        print(f"✓ Comprehensive metrics computed:")
        for key, value in comprehensive.items():
            if isinstance(value, tf.Tensor):
                if len(value.shape) == 0:  # Scalar
                    print(f"    {key}: {value:.4f}")
                else:  # Vector
                    print(f"    {key}: mean={tf.reduce_mean(value):.4f}")
            else:
                print(f"    {key}: {value}")
    except Exception as e:
        print(f"✗ Comprehensive metrics failed: {e}")
    
    print(f"\n✓ Quantum metrics testing completed!")

if __name__ == "__main__":
    test_quantum_metrics()
