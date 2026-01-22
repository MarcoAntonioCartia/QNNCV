"""
Quantum SF Generator with Input Encoding
=========================================

Continuous-Variable Quantum Generator using Strawberry Fields.

This implementation includes proper INPUT ENCODING via displacement gates,
separating:
- ENCODING parameters (from latent vector, NOT optimized)
- QNN parameters (learned weights, ARE optimized)

Encoding Options:
- 'displacement_simple': 1 param per mode (magnitude only), latent_dim = n_modes
- 'displacement_full': 2 params per mode (magnitude + phase), latent_dim = 2 * n_modes
- 'displacement_squeezing': 4 params per mode, latent_dim = 4 * n_modes

Based on Strawberry Fields tutorial pattern for guaranteed gradient flow.
"""

# =============================================================================
# SCIPY COMPATIBILITY PATCH - MUST BE APPLIED BEFORE IMPORTING STRAWBERRYFIELDS
# scipy.integrate.simps was renamed to scipy.integrate.simpson in SciPy 1.14+
# This patch ensures backward compatibility with Strawberry Fields
# =============================================================================
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson
# =============================================================================

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops


def init_weights(n_modes: int, n_layers: int, active_sd: float = 0.0001, 
                 passive_sd: float = 0.1) -> tf.Variable:
    """
    Initialize QNN weight matrix.
    
    Structure per layer (for n_modes modes):
    - n_modes squeeze magnitudes
    - n_modes squeeze phases  
    - n_modes displacement magnitudes
    - n_modes displacement phases
    - (n_modes * (n_modes - 1)) // 2 beamsplitter angles (theta)
    - (n_modes * (n_modes - 1)) // 2 beamsplitter angles (phi)
    - n_modes rotation angles
    
    Total per layer: n_modes * 4 + n_modes * (n_modes - 1) + n_modes
                   = n_modes * 5 + n_modes * (n_modes - 1)
                   = n_modes * (5 + n_modes - 1)
                   = n_modes * (n_modes + 4)
    
    Args:
        n_modes: Number of quantum modes
        n_layers: Number of QNN layers
        active_sd: Std for active gates (squeeze, displace)
        passive_sd: Std for passive gates (beamsplitter, rotation)
        
    Returns:
        tf.Variable with shape [n_layers, params_per_layer]
    """
    # Calculate params per layer
    n_bs = (n_modes * (n_modes - 1)) // 2  # Beamsplitter pairs
    
    params_per_layer = (
        n_modes +      # Squeeze magnitudes
        n_modes +      # Squeeze phases
        n_modes +      # Displacement magnitudes
        n_modes +      # Displacement phases
        n_bs +         # BS theta
        n_bs +         # BS phi
        n_modes        # Rotation angles
    )
    
    # Initialize with appropriate scales
    weights = np.zeros((n_layers, params_per_layer))
    
    idx = 0
    for _ in range(n_layers):
        # Squeeze (active) - small initialization
        weights[:, idx:idx + n_modes] = np.random.normal(0, active_sd, (n_layers, n_modes))
        idx += n_modes
        weights[:, idx:idx + n_modes] = np.random.normal(0, passive_sd, (n_layers, n_modes))
        idx += n_modes
        
        # Displacement (active) - small initialization
        weights[:, idx:idx + n_modes] = np.random.normal(0, active_sd, (n_layers, n_modes))
        idx += n_modes
        weights[:, idx:idx + n_modes] = np.random.normal(0, passive_sd, (n_layers, n_modes))
        idx += n_modes
        
        # Beamsplitter (passive)
        weights[:, idx:idx + n_bs] = np.random.normal(0, passive_sd, (n_layers, n_bs))
        idx += n_bs
        weights[:, idx:idx + n_bs] = np.random.normal(0, passive_sd, (n_layers, n_bs))
        idx += n_bs
        
        # Rotation (passive)
        weights[:, idx:idx + n_modes] = np.random.normal(0, passive_sd, (n_layers, n_modes))
    
    return tf.Variable(
        tf.cast(weights, tf.float32),
        name="qnn_weights",
        trainable=True
    )


def layer(params, q):
    """
    Apply one layer of the CV QNN.
    
    Structure:
    1. Squeeze each mode
    2. Displacement each mode
    3. Interferometer (beamsplitters + rotations)
    
    Args:
        params: 1D array of parameters for this layer
        q: Quantum modes
    """
    n_modes = len(q)
    n_bs = (n_modes * (n_modes - 1)) // 2
    
    idx = 0
    
    # Squeeze gates
    for i in range(n_modes):
        ops.Sgate(params[idx], params[idx + n_modes]) | q[i]
    idx += 2 * n_modes
    
    # Displacement gates
    for i in range(n_modes):
        ops.Dgate(params[idx], params[idx + n_modes]) | q[i]
    idx += 2 * n_modes
    
    # Interferometer: beamsplitters
    bs_idx = 0
    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            ops.BSgate(params[idx + bs_idx], params[idx + n_bs + bs_idx]) | (q[i], q[j])
            bs_idx += 1
    idx += 2 * n_bs
    
    # Rotation gates
    for i in range(n_modes):
        ops.Rgate(params[idx + i]) | q[i]


class QuantumSFGenerator:
    """
    CV Quantum Generator with proper input encoding.
    
    Architecture:
    1. INPUT ENCODING: Displacement gates parameterized by latent vector
       - These are NOT optimized (values come from z, not trainable)
    2. QNN LAYERS: Parameterized quantum layers
       - These ARE optimized (weights are tf.Variable)
    3. MEASUREMENT: X-quadrature expectation values
    4. DECODER: Classical layer to map to output dimension
    
    The key insight is that input_params receive LATENT VALUES (data),
    while sf_params receive LEARNED WEIGHTS (trainable variables).
    
    Args:
        n_modes: Number of quantum modes
        n_layers: Number of QNN layers
        cutoff_dim: Fock space cutoff dimension
        output_dim: Output dimension (after classical decoder)
        encoding_type: 'displacement_simple', 'displacement_full', or 'displacement_squeezing'
    """
    
    def __init__(
        self,
        n_modes: int = 1,
        n_layers: int = 1,
        cutoff_dim: int = 6,
        output_dim: int = 1,
        encoding_type: str = 'displacement_full'
    ):
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.cutoff_dim = cutoff_dim
        self.output_dim = output_dim
        self.encoding_type = encoding_type
        
        # Determine latent dimension based on encoding type
        if encoding_type == 'displacement_simple':
            self.latent_dim = n_modes  # 1 param per mode
        elif encoding_type == 'displacement_full':
            self.latent_dim = 2 * n_modes  # 2 params per mode (r, phi)
        elif encoding_type == 'displacement_squeezing':
            self.latent_dim = 4 * n_modes  # 4 params per mode
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")
        
        # Initialize SF engine and program
        self.eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
        self.prog = sf.Program(n_modes)
        
        # Create INPUT ENCODING parameters (NOT optimized)
        # These will receive values from the latent vector
        self._create_input_params()
        
        # Create QNN parameters (ARE optimized)
        self.weights = init_weights(n_modes, n_layers)
        self._create_qnn_params()
        
        # Build the symbolic program ONCE
        self._build_program()
        
        # Classical decoder
        self.decoder = tf.Variable(
            tf.random.normal([n_modes, output_dim], stddev=0.1),
            name="decoder",
            trainable=True
        )
    
    def _create_input_params(self):
        """Create symbolic parameters for input encoding."""
        if self.encoding_type == 'displacement_simple':
            # 1 param per mode: Dgate(r, 0)
            self.input_params = [
                self.prog.params(f"input_{i}")
                for i in range(self.n_modes)
            ]
        elif self.encoding_type == 'displacement_full':
            # 2 params per mode: Dgate(r, phi)
            self.input_params = [
                self.prog.params(f"input_{i}")
                for i in range(2 * self.n_modes)
            ]
        elif self.encoding_type == 'displacement_squeezing':
            # 4 params per mode: Dgate(r_d, phi_d) + Sgate(r_s, phi_s)
            self.input_params = [
                self.prog.params(f"input_{i}")
                for i in range(4 * self.n_modes)
            ]
    
    def _create_qnn_params(self):
        """Create symbolic parameters for QNN layers."""
        num_qnn_params = np.prod(self.weights.shape)
        self.sf_params = np.array([
            self.prog.params(f"qnn_{i}")
            for i in range(num_qnn_params)
        ]).reshape(self.weights.shape)
    
    def _build_program(self):
        """Build the symbolic quantum program (done ONCE)."""
        with self.prog.context as q:
            # =====================================================
            # INPUT ENCODING LAYER (values from latent, NOT learned)
            # =====================================================
            if self.encoding_type == 'displacement_simple':
                # Dgate(r, 0) for each mode
                for i in range(self.n_modes):
                    ops.Dgate(self.input_params[i], 0.0) | q[i]
                    
            elif self.encoding_type == 'displacement_full':
                # Dgate(r, phi) for each mode
                for i in range(self.n_modes):
                    r_param = self.input_params[2 * i]
                    phi_param = self.input_params[2 * i + 1]
                    ops.Dgate(r_param, phi_param) | q[i]
                    
            elif self.encoding_type == 'displacement_squeezing':
                # Dgate(r_d, phi_d) + Sgate(r_s, phi_s) for each mode
                for i in range(self.n_modes):
                    r_d = self.input_params[4 * i]
                    phi_d = self.input_params[4 * i + 1]
                    r_s = self.input_params[4 * i + 2]
                    phi_s = self.input_params[4 * i + 3]
                    ops.Dgate(r_d, phi_d) | q[i]
                    ops.Sgate(r_s, phi_s) | q[i]
            
            # =====================================================
            # QNN LAYERS (values from weights, ARE learned)
            # =====================================================
            for k in range(self.n_layers):
                layer(self.sf_params[k], q)
    
    def _execute_circuit(self, z_single: tf.Tensor) -> tf.Tensor:
        """
        Execute the quantum circuit for a single latent sample.
        
        Args:
            z_single: Single latent vector [latent_dim]
            
        Returns:
            Measurements tensor [n_modes]
        """
        # Reset engine if needed
        if self.eng.run_progs:
            self.eng.reset()
        
        # =====================================================
        # CREATE PARAMETER MAPPING
        # =====================================================
        mapping = {}
        
        # Map INPUT ENCODING params (from latent data, NOT optimized)
        for i, param in enumerate(self.input_params):
            if i < len(z_single):
                mapping[param.name] = z_single[i]
            else:
                # Pad with zeros if latent is smaller than expected
                mapping[param.name] = tf.constant(0.0)
        
        # Map QNN params (from trainable weights, ARE optimized)
        for p, w in zip(self.sf_params.flatten(), tf.reshape(self.weights, [-1])):
            mapping[p.name] = w
        
        # =====================================================
        # EXECUTE CIRCUIT
        # =====================================================
        result = self.eng.run(self.prog, args=mapping)
        state = result.state
        
        # =====================================================
        # MEASUREMENTS (X-quadrature expectation values)
        # =====================================================
        # NOTE: quad_expectation returns (mean, variance) tuple
        # We only want the mean for generation
        measurements = []
        for mode in range(self.n_modes):
            quad_result = state.quad_expectation(mode, 0)  # phi=0 for X quadrature
            # quad_result is (mean, variance) - extract just the mean
            x_quad_mean = quad_result[0]
            measurements.append(x_quad_mean)
        
        return tf.stack(measurements)
    
    def generate(self, z: tf.Tensor) -> tf.Tensor:
        """
        Generate samples from latent vectors.
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
            
        Returns:
            Generated samples [batch_size, output_dim]
        """
        batch_size = tf.shape(z)[0]
        
        # Process each sample (can't batch SF circuits easily)
        outputs = []
        for i in range(batch_size):
            z_single = z[i]
            measurements = self._execute_circuit(z_single)
            outputs.append(measurements)
        
        # Stack: [batch_size, n_modes]
        batch_measurements = tf.stack(outputs)
        
        # Apply decoder: [batch_size, n_modes] @ [n_modes, output_dim] = [batch_size, output_dim]
        output = tf.matmul(batch_measurements, self.decoder)
        
        return output
    
    @property
    def trainable_variables(self):
        """
        Return trainable variables (QNN weights + decoder).
        
        NOTE: input_params are NOT included - they receive latent values,
        not learned weights.
        """
        return [self.weights, self.decoder]
    
    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return (
            np.prod(self.weights.shape) +
            np.prod(self.decoder.shape)
        )
    
    def get_config(self) -> dict:
        """Return configuration dictionary."""
        return {
            'n_modes': self.n_modes,
            'n_layers': self.n_layers,
            'cutoff_dim': self.cutoff_dim,
            'output_dim': self.output_dim,
            'encoding_type': self.encoding_type,
            'latent_dim': self.latent_dim,
            'num_params': self.num_params
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_simple_generator(
    n_modes: int = 1,
    n_layers: int = 1,
    cutoff_dim: int = 6,
    output_dim: int = 1
) -> QuantumSFGenerator:
    """
    Create a simple generator for testing.
    
    Uses displacement_full encoding: latent_dim = 2 * n_modes
    """
    return QuantumSFGenerator(
        n_modes=n_modes,
        n_layers=n_layers,
        cutoff_dim=cutoff_dim,
        output_dim=output_dim,
        encoding_type='displacement_full'
    )


def create_1mode_generator(
    n_layers: int = 1,
    cutoff_dim: int = 6,
    encoding_type: str = 'displacement_full'
) -> QuantumSFGenerator:
    """
    Create a 1-mode generator (simplest case).
    
    For 1 mode with displacement_full: latent_dim = 2
    """
    return QuantumSFGenerator(
        n_modes=1,
        n_layers=n_layers,
        cutoff_dim=cutoff_dim,
        output_dim=1,
        encoding_type=encoding_type
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing QuantumSFGenerator with Input Encoding")
    print("=" * 60)
    
    # Test 1: Simple encoding
    print("\n--- Test 1: displacement_simple encoding ---")
    gen1 = QuantumSFGenerator(n_modes=1, encoding_type='displacement_simple')
    print(f"Config: {gen1.get_config()}")
    
    z1 = tf.random.normal([4, gen1.latent_dim])
    out1 = gen1.generate(z1)
    print(f"Input shape: {z1.shape}, Output shape: {out1.shape}")
    
    # Test 2: Full encoding
    print("\n--- Test 2: displacement_full encoding ---")
    gen2 = QuantumSFGenerator(n_modes=1, encoding_type='displacement_full')
    print(f"Config: {gen2.get_config()}")
    
    z2 = tf.random.normal([4, gen2.latent_dim])
    out2 = gen2.generate(z2)
    print(f"Input shape: {z2.shape}, Output shape: {out2.shape}")
    
    # Test 3: Gradient flow
    print("\n--- Test 3: Gradient Flow ---")
    with tf.GradientTape() as tape:
        z = tf.random.normal([4, gen2.latent_dim])
        output = gen2.generate(z)
        loss = tf.reduce_mean(tf.square(output - 2.0))
    
    grads = tape.gradient(loss, gen2.trainable_variables)
    
    print(f"Loss: {loss.numpy():.4f}")
    for i, (g, v) in enumerate(zip(grads, gen2.trainable_variables)):
        if g is not None:
            print(f"  {v.name}: grad_norm = {tf.norm(g).numpy():.6f}")
        else:
            print(f"  {v.name}: grad = None")
    
    # Verify gradients exist
    if all(g is not None for g in grads):
        print("\n✓ All gradients computed successfully!")
    else:
        print("\n✗ Some gradients are None!")
    
    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)
