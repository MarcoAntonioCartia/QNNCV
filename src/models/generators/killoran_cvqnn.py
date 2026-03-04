"""
Killoran CV-QNN Architecture
=============================

Implements the continuous-variable quantum neural network from:
"Continuous-variable quantum neural networks" - Killoran et al. (2018)
https://arxiv.org/abs/1806.06871

Key insight: The non-Gaussian gate (Kerr) is ESSENTIAL for:
1. Nonlinearity (like activation functions in classical NNs)
2. Universality (ability to approximate any transformation)
3. Producing non-Gaussian states (multi-peaked distributions)

Layer structure (Eq. 16 in paper):
    L := Φ ◦ D ◦ U₂ ◦ S ◦ U₁

Where:
- U₁, U₂: Interferometers (for multi-mode: beamsplitters + rotations)
- S: Squeeze gates
- D: Displacement gates  
- Φ: Non-Gaussian gate (Kerr: K(κ) = exp(iκn̂²))

For single mode, interferometers reduce to rotation gates.

EXPRESSIVITY STUDY:
This version supports N-modal target distributions to test the
expressivity limits of a single qumode CV-QNN.
"""

import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import math
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
from typing import Tuple, Optional, List, Dict, Any

# Warning suppression
try:
    from src.utils.warning_suppression import enable_clean_training
    enable_clean_training()
except ImportError:
    pass


# =============================================================================
# HERMITE TRANSFORM (Fock → Position basis)
# =============================================================================

_FACTORIAL_CACHE = np.array([math.factorial(n) for n in range(50)], dtype=np.float64)


def hermite_functions(x: tf.Tensor, cutoff_dim: int, hbar: float = 2.0) -> tf.Tensor:
    """Compute Hermite functions ψ₀(x) through ψ_{n-1}(x)."""
    sigma = np.sqrt(hbar / 2.0)
    x_scaled = tf.cast(x / sigma, tf.float64)
    
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
    
    H_all = tf.cast(H.stack(), tf.float32)
    n_vals = np.arange(cutoff_dim)
    norms = 1.0 / np.sqrt((2.0 ** n_vals) * _FACTORIAL_CACHE[:cutoff_dim] * np.sqrt(np.pi) * sigma)
    norms = tf.constant(norms.reshape(-1, 1), dtype=tf.float32)
    gaussian = tf.exp(-tf.cast(x_scaled, tf.float32) ** 2 / 2.0)
    
    return norms * H_all * gaussian


def fock_to_probability(state_ket: tf.Tensor, xvec: tf.Tensor, cutoff_dim: int, hbar: float = 2.0) -> tf.Tensor:
    """Convert Fock-basis state to homodyne probability distribution."""
    psi_n = tf.cast(hermite_functions(xvec, cutoff_dim, hbar), tf.complex64)
    state_ket = tf.reshape(state_ket, [cutoff_dim, 1])
    psi_x = tf.reduce_sum(state_ket * psi_n, axis=0)
    P_x = tf.abs(psi_x) ** 2
    dx = xvec[1] - xvec[0]
    P_x = P_x / (tf.reduce_sum(P_x) * dx + 1e-10)
    return P_x


# =============================================================================
# KILLORAN CV-QNN GENERATOR
# =============================================================================

class KilloranCVQNN:
    """
    CV-QNN following Killoran et al. (2018) architecture.
    
    Each layer contains:
    1. Rotation R₁ (single-mode interferometer)
    2. Squeeze S
    3. Rotation R₂ (single-mode interferometer)  
    4. Displacement D
    5. Kerr gate K (NON-GAUSSIAN - provides nonlinearity!)
    
    The Kerr gate K(κ) = exp(iκn̂²) is the "activation function" that
    enables the network to produce non-Gaussian states.
    """
    
    def __init__(
        self,
        n_modes: int = 1,
        n_layers: int = 4,
        cutoff_dim: int = 8,  # Higher cutoff needed for Kerr
        num_bins: int = 100,
        x_min: float = -5.0,
        x_max: float = 5.0,
        hbar: float = 2.0,
        use_kerr: bool = True,  # Toggle Kerr gate
        kerr_scale: float = 0.1  # Initial scale for Kerr parameters
    ):
        """
        Args:
            n_modes: Number of quantum modes (1 for now)
            n_layers: Number of CV-QNN layers
            cutoff_dim: Fock space truncation (higher for Kerr states)
            num_bins: Output distribution resolution
            x_min, x_max: x-quadrature range
            hbar: Planck constant
            use_kerr: Whether to include Kerr gates
            kerr_scale: Initialization scale for Kerr parameters
        """
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.cutoff_dim = cutoff_dim
        self.num_bins = num_bins
        self.x_min = x_min
        self.x_max = x_max
        self.hbar = hbar
        self.use_kerr = use_kerr
        self.kerr_scale = kerr_scale
        
        # Latent dimension (displacement encoding)
        self.latent_dim = 2 * n_modes
        
        # Fixed x-grid
        self.xvec = tf.constant(np.linspace(x_min, x_max, num_bins), dtype=tf.float32)
        
        # Initialize weights
        self._init_weights()
        
        # Build SF program
        self._build_program()
    
    def _init_weights(self):
        """
        Initialize weights for the Killoran architecture.
        
        Per layer (single mode):
        - R₁: 1 rotation angle
        - S: 2 params (r, φ)
        - R₂: 1 rotation angle
        - D: 2 params (r, φ)
        - K: 1 Kerr parameter (if use_kerr)
        
        Total per layer: 7 (with Kerr) or 6 (without)
        """
        params_per_layer = 7 if self.use_kerr else 6
        
        weights = []
        for layer in range(self.n_layers):
            layer_weights = []
            
            # R₁: rotation (passive, small init)
            layer_weights.append(np.random.normal(0, 0.1))
            
            # S: squeeze (r, φ) - active parameter, small init
            layer_weights.append(np.random.normal(0, 0.1))  # r
            layer_weights.append(np.random.normal(0, 0.1))  # φ
            
            # R₂: rotation
            layer_weights.append(np.random.normal(0, 0.1))
            
            # D: displacement (r, φ) - active parameter
            layer_weights.append(np.random.normal(0, 0.1))  # r
            layer_weights.append(np.random.normal(0, 0.1))  # φ
            
            # K: Kerr parameter (if enabled)
            if self.use_kerr:
                layer_weights.append(np.random.normal(0, self.kerr_scale))
            
            weights.append(layer_weights)
        
        self.weights = tf.Variable(
            np.array(weights, dtype=np.float32),
            name="killoran_qnn_weights"
        )
    
    def _build_program(self):
        """Build the Strawberry Fields program with Killoran architecture."""
        self.prog = sf.Program(self.n_modes)
        self.eng = sf.Engine("tf", backend_options={"cutoff_dim": self.cutoff_dim})
        
        # Input encoding parameters (displacement)
        self.input_params = [
            self.prog.params("in_r"),
            self.prog.params("in_phi")
        ]
        
        # QNN parameters
        params_per_layer = 7 if self.use_kerr else 6
        total_params = self.n_layers * params_per_layer
        self.qnn_params = [self.prog.params(f"w_{i}") for i in range(total_params)]
        
        # Build circuit
        with self.prog.context as q:
            # Input encoding: displace vacuum based on latent z
            ops.Dgate(self.input_params[0], self.input_params[1]) | q[0]
            
            # Killoran layers
            idx = 0
            for layer in range(self.n_layers):
                # R₁: First rotation (single-mode interferometer)
                ops.Rgate(self.qnn_params[idx]) | q[0]
                idx += 1
                
                # S: Squeeze gate
                ops.Sgate(self.qnn_params[idx], self.qnn_params[idx + 1]) | q[0]
                idx += 2
                
                # R₂: Second rotation
                ops.Rgate(self.qnn_params[idx]) | q[0]
                idx += 1
                
                # D: Displacement
                ops.Dgate(self.qnn_params[idx], self.qnn_params[idx + 1]) | q[0]
                idx += 2
                
                # Φ: Kerr gate (NON-GAUSSIAN ACTIVATION!)
                if self.use_kerr:
                    ops.Kgate(self.qnn_params[idx]) | q[0]
                    idx += 1
    
    def _execute_circuit(self, z_single: tf.Tensor) -> tf.Tensor:
        """Execute circuit and return state ket."""
        if self.eng.run_progs:
            self.eng.reset()
        
        # Build parameter mapping
        mapping = {
            self.input_params[0].name: z_single[0],
            self.input_params[1].name: z_single[1] if len(z_single) > 1 else tf.constant(0.0)
        }
        
        flat_weights = tf.reshape(self.weights, [-1])
        for i, param in enumerate(self.qnn_params):
            mapping[param.name] = flat_weights[i]
        
        # Run circuit
        result = self.eng.run(self.prog, args=mapping)
        return result.state.ket()
    
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
            state_ket = self._execute_circuit(z[i])
            P_x = fock_to_probability(state_ket, self.xvec, self.cutoff_dim, self.hbar)
            distributions.append(P_x)
        
        return tf.stack(distributions)
    
    @property
    def trainable_variables(self):
        return [self.weights]
    
    @property
    def num_params(self) -> int:
        return int(np.prod(self.weights.shape))
    
    def get_weights_by_layer(self) -> Dict[str, np.ndarray]:
        """
        Get weights organized by layer and gate type.
        
        Returns:
            Dictionary with keys like 'layer_0_R1', 'layer_0_S', etc.
        """
        weights_np = self.weights.numpy()
        params_per_layer = 7 if self.use_kerr else 6
        
        organized = {}
        for layer in range(self.n_layers):
            base_idx = 0
            # R1: rotation
            organized[f'layer_{layer}_R1'] = weights_np[layer, base_idx]
            base_idx += 1
            # S: squeeze (r, φ)
            organized[f'layer_{layer}_S_r'] = weights_np[layer, base_idx]
            organized[f'layer_{layer}_S_phi'] = weights_np[layer, base_idx + 1]
            base_idx += 2
            # R2: rotation
            organized[f'layer_{layer}_R2'] = weights_np[layer, base_idx]
            base_idx += 1
            # D: displacement (r, φ)
            organized[f'layer_{layer}_D_r'] = weights_np[layer, base_idx]
            organized[f'layer_{layer}_D_phi'] = weights_np[layer, base_idx + 1]
            base_idx += 2
            # K: Kerr
            if self.use_kerr:
                organized[f'layer_{layer}_K'] = weights_np[layer, base_idx]
        
        return organized
    
    def get_weight_statistics(self) -> Dict[str, float]:
        """Get summary statistics of all weights."""
        w = self.weights.numpy().flatten()
        return {
            'mean': float(np.mean(w)),
            'std': float(np.std(w)),
            'min': float(np.min(w)),
            'max': float(np.max(w)),
            'norm': float(np.linalg.norm(w))
        }
    
    def get_quantum_state(self, z: tf.Tensor):
        """
        Get the full quantum state (not just distribution) for visualization.
        
        Args:
            z: Single latent vector [latent_dim]
        
        Returns:
            SF State object (for Wigner function plotting)
        """
        if self.eng.run_progs:
            self.eng.reset()
        
        mapping = {
            self.input_params[0].name: z[0],
            self.input_params[1].name: z[1] if len(z) > 1 else tf.constant(0.0)
        }
        
        flat_weights = tf.reshape(self.weights, [-1])
        for i, param in enumerate(self.qnn_params):
            mapping[param.name] = flat_weights[i]
        
        result = self.eng.run(self.prog, args=mapping)
        return result.state
    
    def get_config(self) -> dict:
        return {
            'n_modes': self.n_modes,
            'n_layers': self.n_layers,
            'cutoff_dim': self.cutoff_dim,
            'num_bins': self.num_bins,
            'x_range': (self.x_min, self.x_max),
            'latent_dim': self.latent_dim,
            'num_params': self.num_params,
            'use_kerr': self.use_kerr,
            'architecture': 'Killoran_CV-QNN'
        }


# =============================================================================
# TARGET DISTRIBUTIONS
# =============================================================================

def gaussian_distribution(xvec: tf.Tensor, mean: float, std: float) -> tf.Tensor:
    """Single Gaussian distribution."""
    P = tf.exp(-0.5 * ((xvec - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    dx = xvec[1] - xvec[0]
    return P / (tf.reduce_sum(P) * dx)


def bimodal_gaussian(xvec: tf.Tensor, 
                     mean1: float, std1: float, weight1: float,
                     mean2: float, std2: float) -> tf.Tensor:
    """
    Bimodal Gaussian mixture.
    
    P(x) = w₁ * N(μ₁, σ₁²) + w₂ * N(μ₂, σ₂²)
    """
    weight2 = 1.0 - weight1
    
    G1 = tf.exp(-0.5 * ((xvec - mean1) / std1) ** 2) / (std1 * np.sqrt(2 * np.pi))
    G2 = tf.exp(-0.5 * ((xvec - mean2) / std2) ** 2) / (std2 * np.sqrt(2 * np.pi))
    
    P = weight1 * G1 + weight2 * G2
    dx = xvec[1] - xvec[0]
    return P / (tf.reduce_sum(P) * dx)


def n_modal_gaussian(
    xvec: tf.Tensor,
    n_peaks: int,
    x_min: float = -2.0,
    x_max: float = 2.0,
    std: float = 0.3,
    weights: Optional[List[float]] = None
) -> Tuple[tf.Tensor, List[float]]:
    """
    Create N-modal Gaussian mixture with equally spaced peaks.
    
    Peaks are placed at: x_min, x_min + step, x_min + 2*step, ..., x_max
    where step = (x_max - x_min) / (n_peaks - 1) for n_peaks > 1
    
    Args:
        xvec: x-axis grid
        n_peaks: Number of Gaussian peaks (modes)
        x_min: Left boundary for peak placement
        x_max: Right boundary for peak placement
        std: Standard deviation for each peak
        weights: Optional list of weights (must sum to 1). Default: equal weights.
    
    Returns:
        Tuple of (probability distribution, list of peak positions)
    
    Examples:
        n_peaks=1: peak at center (x_min + x_max) / 2
        n_peaks=2: peaks at x_min, x_max
        n_peaks=3: peaks at x_min, (x_min+x_max)/2, x_max
        n_peaks=4: peaks at -2, -0.67, 0.67, 2 (for x_min=-2, x_max=2)
    """
    if n_peaks < 1:
        raise ValueError("n_peaks must be at least 1")
    
    # Calculate peak positions
    if n_peaks == 1:
        means = [(x_min + x_max) / 2.0]
    else:
        step = (x_max - x_min) / (n_peaks - 1)
        means = [x_min + i * step for i in range(n_peaks)]
    
    # Set weights
    if weights is None:
        weights = [1.0 / n_peaks] * n_peaks
    else:
        assert len(weights) == n_peaks, f"weights must have {n_peaks} elements"
        total = sum(weights)
        weights = [w / total for w in weights]  # Normalize
    
    # Build mixture
    P = tf.zeros_like(xvec)
    for mean, weight in zip(means, weights):
        G = tf.exp(-0.5 * ((xvec - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        P = P + weight * G
    
    # Normalize
    dx = xvec[1] - xvec[0]
    P = P / (tf.reduce_sum(P) * dx)
    
    return P, means


def get_target_description(n_peaks: int, x_min: float, x_max: float, std: float) -> str:
    """Get human-readable description of target distribution."""
    if n_peaks == 1:
        center = (x_min + x_max) / 2.0
        return f"Unimodal: N({center:.2f}, {std}²)"
    elif n_peaks == 2:
        return f"Bimodal: peaks at {x_min:.2f}, {x_max:.2f} (σ={std})"
    else:
        step = (x_max - x_min) / (n_peaks - 1)
        means = [x_min + i * step for i in range(n_peaks)]
        means_str = ", ".join([f"{m:.2f}" for m in means])
        return f"{n_peaks}-modal: peaks at [{means_str}] (σ={std})"


# =============================================================================
# DISCRIMINATOR
# =============================================================================

class DistributionDiscriminator:
    """Discriminator for probability distributions."""
    
    def __init__(self, input_dim: int = 100, hidden_dims: List[int] = [128, 64, 32]):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self._build_network()
    
    def _build_network(self):
        dims = [self.input_dim] + self.hidden_dims + [1]
        self.layers = []
        for i in range(len(dims) - 1):
            w = tf.Variable(tf.random.normal([dims[i], dims[i+1]], stddev=0.1), name=f"d_{i}_w")
            b = tf.Variable(tf.zeros([dims[i+1]]), name=f"d_{i}_b")
            self.layers.append((w, b))
    
    def discriminate(self, P_x: tf.Tensor) -> tf.Tensor:
        x = P_x
        for i, (w, b) in enumerate(self.layers):
            x = tf.matmul(x, w) + b
            if i < len(self.layers) - 1:
                x = tf.nn.leaky_relu(x, alpha=0.2)
        return x
    
    @property
    def trainable_variables(self):
        return [v for w, b in self.layers for v in [w, b]]
    
    @property
    def num_params(self) -> int:
        return sum(int(np.prod(v.shape)) for v in self.trainable_variables)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Killoran CV-QNN Architecture")
    print("=" * 60)
    
    # Test with Kerr gate
    print("\n1. Creating CV-QNN WITH Kerr gate...")
    gen_kerr = KilloranCVQNN(
        n_layers=4,
        cutoff_dim=8,
        use_kerr=True
    )
    print(f"   Config: {gen_kerr.get_config()}")
    
    # Test forward pass
    z = tf.random.normal([4, gen_kerr.latent_dim])
    print(f"\n2. Forward pass with batch size 4...")
    P_x = gen_kerr.generate(z)
    print(f"   Input shape: {z.shape}")
    print(f"   Output shape: {P_x.shape}")
    
    # Check normalization
    dx = gen_kerr.xvec[1] - gen_kerr.xvec[0]
    norms = tf.reduce_sum(P_x, axis=1) * dx
    print(f"   Norms (should be ~1.0): {norms.numpy()}")
    
    # Test gradient flow
    print(f"\n3. Testing gradient flow...")
    with tf.GradientTape() as tape:
        P_x = gen_kerr.generate(z)
        # Loss: try to maximize variance (spread the distribution)
        mean = tf.reduce_sum(P_x * gen_kerr.xvec, axis=1) * dx
        loss = -tf.reduce_mean(mean ** 2)  # Dummy loss
    
    grads = tape.gradient(loss, gen_kerr.trainable_variables)
    print(f"   Gradients computed: {grads[0] is not None}")
    if grads[0] is not None:
        print(f"   Gradient norm: {tf.norm(grads[0]).numpy():.6f}")
    
    # Compare with/without Kerr
    print(f"\n4. Comparing architectures...")
    gen_no_kerr = KilloranCVQNN(n_layers=4, cutoff_dim=8, use_kerr=False)
    print(f"   With Kerr: {gen_kerr.num_params} params")
    print(f"   Without Kerr: {gen_no_kerr.num_params} params")
    
    # Test bimodal target
    print(f"\n5. Creating bimodal target distribution...")
    xvec = gen_kerr.xvec
    P_bimodal = bimodal_gaussian(xvec, mean1=-1.5, std1=0.5, weight1=0.5, mean2=1.5, std2=0.5)
    print(f"   Target shape: {P_bimodal.shape}")
    print(f"   Target has two peaks at x ≈ -1.5 and x ≈ 1.5")
    
    print("\n" + "=" * 60)
    print("✓ Killoran CV-QNN architecture ready!")
    print("=" * 60)
