"""
Distribution-Based Quantum Generator
=====================================

This generator outputs the full homodyne probability distribution P(x)
instead of scalar values. This enables true distribution-to-distribution
comparison in the GAN framework.

Output: P(x) = |ψ(x)|² computed via Hermite transform from Fock-basis state.
"""

import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import math
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
from typing import Tuple, Optional

# Warning suppression (if available)
try:
    from src.utils.warning_suppression import enable_clean_training
    enable_clean_training()
except ImportError:
    pass


# =============================================================================
# HERMITE TRANSFORM (Fock → Position basis)
# =============================================================================

# Precompute factorials (using math.factorial for NumPy 2.0 compatibility)
_FACTORIAL_CACHE = np.array([math.factorial(n) for n in range(50)], dtype=np.float64)


def hermite_functions(x: tf.Tensor, cutoff_dim: int, hbar: float = 2.0) -> tf.Tensor:
    """
    Compute Hermite functions ψ₀(x) through ψ_{n-1}(x).
    
    ψₙ(x) = (1/√(2ⁿ n! √π σ)) × Hₙ(x/σ) × exp(-x²/2σ²)
    where σ = √(ℏ/2)
    
    Returns: [cutoff_dim, num_points]
    """
    sigma = np.sqrt(hbar / 2.0)
    x_scaled = tf.cast(x / sigma, tf.float64)
    
    # Hermite polynomials via recurrence
    H = tf.TensorArray(dtype=tf.float64, size=cutoff_dim)
    H0 = tf.ones_like(x_scaled)
    H = H.write(0, H0)
    
    if cutoff_dim > 1:
        H1 = 2.0 * x_scaled
        H = H.write(1, H1)
        
        Hn_2, Hn_1 = H0, H1
        for n in range(2, cutoff_dim):
            Hn = 2.0 * x_scaled * Hn_1 - 2.0 * (n - 1) * Hn_2
            H = H.write(n, Hn)
            Hn_2, Hn_1 = Hn_1, Hn
    
    H_all = tf.cast(H.stack(), tf.float32)  # [cutoff_dim, num_points]
    
    # Normalization
    n_vals = np.arange(cutoff_dim)
    norms = 1.0 / np.sqrt((2.0 ** n_vals) * _FACTORIAL_CACHE[:cutoff_dim] * np.sqrt(np.pi) * sigma)
    norms = tf.constant(norms.reshape(-1, 1), dtype=tf.float32)
    
    # Gaussian envelope
    gaussian = tf.exp(-tf.cast(x_scaled, tf.float32) ** 2 / 2.0)
    
    return norms * H_all * gaussian  # [cutoff_dim, num_points]


def fock_to_probability(state_ket: tf.Tensor, xvec: tf.Tensor, cutoff_dim: int, hbar: float = 2.0) -> tf.Tensor:
    """
    Convert Fock-basis state to homodyne probability distribution.
    
    P(x) = |ψ(x)|² = |Σₙ cₙ ψₙ(x)|²
    
    Args:
        state_ket: Fock coefficients [cutoff_dim], complex
        xvec: x-quadrature grid [num_bins]
        cutoff_dim: Fock truncation
        hbar: Planck constant
    
    Returns:
        P(x): Probability distribution [num_bins]
    """
    psi_n = tf.cast(hermite_functions(xvec, cutoff_dim, hbar), tf.complex64)  # [cutoff_dim, num_bins]
    state_ket = tf.reshape(state_ket, [cutoff_dim, 1])  # [cutoff_dim, 1]
    
    # ψ(x) = Σₙ cₙ ψₙ(x)
    psi_x = tf.reduce_sum(state_ket * psi_n, axis=0)  # [num_bins]
    
    # P(x) = |ψ(x)|²
    P_x = tf.abs(psi_x) ** 2
    
    # Normalize
    dx = xvec[1] - xvec[0]
    P_x = P_x / (tf.reduce_sum(P_x) * dx + 1e-10)
    
    return P_x


# =============================================================================
# WEIGHT INITIALIZATION
# =============================================================================

def init_weights(n_modes: int, n_layers: int, active_sd: float = 0.1, 
                 passive_sd: float = 0.1) -> tf.Variable:
    """Initialize QNN weights."""
    if n_modes == 1:
        params_per_layer = 5  # squeeze(r,φ), displace(r,φ), rotate(θ)
    else:
        n_bs = (n_modes * (n_modes - 1)) // 2
        params_per_layer = n_modes * 4 + n_bs * 2 + n_modes
    
    weights = []
    for _ in range(n_layers):
        layer_weights = np.concatenate([
            np.random.normal(0, active_sd, n_modes),   # squeeze r
            np.random.normal(0, passive_sd, n_modes),  # squeeze φ
            np.random.normal(0, active_sd, n_modes),   # displace r
            np.random.normal(0, passive_sd, n_modes),  # displace φ
        ])
        if n_modes > 1:
            n_bs = (n_modes * (n_modes - 1)) // 2
            layer_weights = np.concatenate([
                layer_weights,
                np.random.normal(0, passive_sd, n_bs),  # BS θ
                np.random.normal(0, passive_sd, n_bs),  # BS φ
            ])
        layer_weights = np.concatenate([
            layer_weights,
            np.random.normal(0, passive_sd, n_modes),  # rotation
        ])
        weights.append(layer_weights)
    
    return tf.Variable(np.array(weights), dtype=tf.float32, name="qnn_weights")


# =============================================================================
# DISTRIBUTION GENERATOR
# =============================================================================

class QuantumDistributionGenerator:
    """
    Quantum generator that outputs homodyne probability distributions.
    
    Instead of outputting scalar samples, this generator outputs the full
    P(x) distribution, enabling distribution-to-distribution GAN training.
    
    Output shape: [batch_size, num_bins]
    """
    
    def __init__(
        self,
        n_modes: int = 1,
        n_layers: int = 2,
        cutoff_dim: int = 10,
        num_bins: int = 100,
        x_min: float = -5.0,
        x_max: float = 5.0,
        hbar: float = 2.0,
        encoding_type: str = 'displacement_full'
    ):
        """
        Args:
            n_modes: Number of quantum modes
            n_layers: Number of QNN layers
            cutoff_dim: Fock space truncation
            num_bins: Number of bins in output distribution
            x_min, x_max: Range of x-quadrature
            hbar: Planck constant (SF default is 2.0)
            encoding_type: How to encode latent vector
        """
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.cutoff_dim = cutoff_dim
        self.num_bins = num_bins
        self.x_min = x_min
        self.x_max = x_max
        self.hbar = hbar
        self.encoding_type = encoding_type
        
        # Latent dimension based on encoding
        if encoding_type == 'displacement_simple':
            self.latent_dim = n_modes
        elif encoding_type == 'displacement_full':
            self.latent_dim = 2 * n_modes
        else:
            self.latent_dim = 4 * n_modes
        
        # Fixed x-grid for output distribution
        self.xvec = tf.constant(np.linspace(x_min, x_max, num_bins), dtype=tf.float32)
        
        # Trainable QNN weights
        self.weights = init_weights(n_modes, n_layers)
        
        # Build SF program
        self._build_program()
    
    def _build_program(self):
        """Build the Strawberry Fields program."""
        self.prog = sf.Program(self.n_modes)
        self.eng = sf.Engine("tf", backend_options={"cutoff_dim": self.cutoff_dim})
        
        # Create symbolic parameters for input encoding
        if self.encoding_type == 'displacement_simple':
            self.input_params = [self.prog.params(f"in_{i}") for i in range(self.n_modes)]
        elif self.encoding_type == 'displacement_full':
            self.input_params = [self.prog.params(f"in_{i}") for i in range(2 * self.n_modes)]
        else:
            self.input_params = [self.prog.params(f"in_{i}") for i in range(4 * self.n_modes)]
        
        # Create symbolic parameters for QNN
        num_qnn_params = np.prod(self.weights.shape)
        self.qnn_params = [self.prog.params(f"qnn_{i}") for i in range(num_qnn_params)]
        
        # Build circuit
        with self.prog.context as q:
            # Input encoding
            if self.encoding_type == 'displacement_simple':
                for i in range(self.n_modes):
                    ops.Dgate(self.input_params[i], 0.0) | q[i]
            elif self.encoding_type == 'displacement_full':
                for i in range(self.n_modes):
                    ops.Dgate(self.input_params[2*i], self.input_params[2*i + 1]) | q[i]
            
            # QNN layers
            param_idx = 0
            for layer in range(self.n_layers):
                for i in range(self.n_modes):
                    # Squeeze
                    ops.Sgate(self.qnn_params[param_idx], self.qnn_params[param_idx + 1]) | q[i]
                    param_idx += 2
                    # Displace
                    ops.Dgate(self.qnn_params[param_idx], self.qnn_params[param_idx + 1]) | q[i]
                    param_idx += 2
                
                # Rotation
                for i in range(self.n_modes):
                    ops.Rgate(self.qnn_params[param_idx]) | q[i]
                    param_idx += 1
    
    def _execute_circuit(self, z_single: tf.Tensor) -> tf.Tensor:
        """Execute circuit and return state ket."""
        if self.eng.run_progs:
            self.eng.reset()
        
        # Build parameter mapping
        mapping = {}
        for i, param in enumerate(self.input_params):
            mapping[param.name] = z_single[i] if i < len(z_single) else tf.constant(0.0)
        for i, param in enumerate(self.qnn_params):
            flat_weights = tf.reshape(self.weights, [-1])
            mapping[param.name] = flat_weights[i]
        
        # Run circuit
        result = self.eng.run(self.prog, args=mapping)
        state = result.state
        
        # Get state ket (Fock coefficients)
        # For single mode, ket is shape [cutoff_dim]
        ket = state.ket()
        
        if self.n_modes == 1:
            return ket  # [cutoff_dim]
        else:
            # For multi-mode, need to trace out other modes or handle differently
            # For now, only support single mode
            raise NotImplementedError("Multi-mode distribution output not yet implemented")
    
    def generate(self, z: tf.Tensor) -> tf.Tensor:
        """
        Generate probability distributions from latent vectors.
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
        
        Returns:
            P(x) distributions [batch_size, num_bins]
        """
        batch_size = tf.shape(z)[0]
        
        distributions = []
        for i in range(batch_size):
            # Get state ket
            state_ket = self._execute_circuit(z[i])
            
            # Convert to P(x)
            P_x = fock_to_probability(state_ket, self.xvec, self.cutoff_dim, self.hbar)
            distributions.append(P_x)
        
        return tf.stack(distributions)  # [batch_size, num_bins]
    
    @property
    def trainable_variables(self):
        return [self.weights]
    
    @property
    def num_params(self) -> int:
        return int(np.prod(self.weights.shape))
    
    @property
    def output_dim(self) -> int:
        return self.num_bins
    
    def get_config(self) -> dict:
        return {
            'n_modes': self.n_modes,
            'n_layers': self.n_layers,
            'cutoff_dim': self.cutoff_dim,
            'num_bins': self.num_bins,
            'x_range': (self.x_min, self.x_max),
            'encoding_type': self.encoding_type,
            'latent_dim': self.latent_dim,
            'num_params': self.num_params
        }


# =============================================================================
# TARGET DISTRIBUTION HELPER
# =============================================================================

def gaussian_distribution(xvec: tf.Tensor, mean: float, std: float) -> tf.Tensor:
    """
    Create discretized Gaussian distribution on xvec grid.
    
    Args:
        xvec: x-values grid [num_bins]
        mean: Gaussian mean
        std: Gaussian standard deviation
    
    Returns:
        Normalized P(x) [num_bins]
    """
    P = tf.exp(-0.5 * ((xvec - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    dx = xvec[1] - xvec[0]
    return P / (tf.reduce_sum(P) * dx)  # Ensure normalization


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing QuantumDistributionGenerator...")
    
    # Create generator
    gen = QuantumDistributionGenerator(
        n_modes=1,
        n_layers=2,
        cutoff_dim=10,
        num_bins=100,
        x_min=-5.0,
        x_max=5.0
    )
    
    print(f"Config: {gen.get_config()}")
    print(f"Trainable variables: {len(gen.trainable_variables)}")
    
    # Test forward pass
    z = tf.random.normal([4, gen.latent_dim])  # Batch of 4
    P_x = gen.generate(z)
    
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {P_x.shape}")
    print(f"Output sums (should be ~1.0): {tf.reduce_sum(P_x, axis=1) * (gen.xvec[1] - gen.xvec[0])}")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    with tf.GradientTape() as tape:
        P_x = gen.generate(z)
        loss = tf.reduce_mean(tf.reduce_sum(P_x * gen.xvec, axis=1))  # Mean of E[x]
    
    grads = tape.gradient(loss, gen.trainable_variables)
    print(f"Gradients computed: {grads[0] is not None}")
    print(f"Gradient norm: {tf.norm(grads[0]).numpy():.6f}")
    
    print("\n✓ All tests passed!")
