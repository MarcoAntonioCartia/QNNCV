"""
Proper Strawberry Fields Implementation Following Official Tutorial Pattern

This module implements quantum circuits using the exact pattern from the official
SF quantum_neural_network.py example, ensuring proper gradient flow.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


def interferometer(params: tf.Tensor, q: List):
    """
    Parameterised interferometer acting on N modes.
    Exact implementation from SF tutorial for guaranteed gradient compatibility.
    
    Args:
        params: Tensor of parameters for the interferometer
        q: List of quantum registers
    """
    N = len(q)
    theta = params[: N * (N - 1) // 2]
    phi = params[N * (N - 1) // 2 : N * (N - 1)]
    rphi = params[-N + 1 :]

    if N == 1:
        # Single mode case
        ops.Rgate(rphi[0]) | q[0]
        return

    n = 0  # parameter counter

    # Apply rectangular beamsplitter array
    for l in range(N):
        for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
            # Skip even or odd pairs depending on layer
            if (l + k) % 2 != 1:
                ops.BSgate(theta[n], phi[n]) | (q1, q2)
                n += 1

    # Apply final local phase shifts
    for i in range(max(1, N - 1)):
        ops.Rgate(rphi[i]) | q[i]


def quantum_layer(params: tf.Tensor, q: List):
    """
    CV quantum neural network layer following SF tutorial pattern.
    
    Args:
        params: Layer parameters tensor
        q: List of quantum registers
    """
    N = len(q)
    M = int(N * (N - 1)) + max(1, N - 1)

    # Parameter slicing exactly like SF tutorial
    int1 = params[:M]
    s = params[M : M + N]
    int2 = params[M + N : 2 * M + N]
    dr = params[2 * M + N : 2 * M + 2 * N]
    dp = params[2 * M + 2 * N : 2 * M + 3 * N]
    k = params[2 * M + 3 * N : 2 * M + 4 * N]

    # Layer operations
    interferometer(int1, q)

    for i in range(N):
        ops.Sgate(s[i]) | q[i]

    interferometer(int2, q)

    for i in range(N):
        ops.Dgate(dr[i], dp[i]) | q[i]
        ops.Kgate(k[i]) | q[i]


def init_sf_weights(modes: int, layers: int, active_sd: float = 0.0001, passive_sd: float = 0.1):
    """
    Initialize weights exactly like SF tutorial for gradient compatibility.
    
    Args:
        modes: Number of quantum modes
        layers: Number of layers
        active_sd: Standard deviation for active parameters
        passive_sd: Standard deviation for passive parameters
        
    Returns:
        tf.Variable: Properly structured weight matrix
    """
    # Number of interferometer parameters
    M = int(modes * (modes - 1)) + max(1, modes - 1)

    # Create weight components (exactly like SF tutorial)
    int1_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
    s_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    int2_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
    dr_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    dp_weights = tf.random.normal(shape=[layers, modes], stddev=passive_sd)
    k_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)

    # Concatenate exactly like SF tutorial
    weights = tf.concat(
        [int1_weights, s_weights, int2_weights, dr_weights, dp_weights, k_weights], 
        axis=1
    )
    
    return tf.Variable(weights, name="sf_quantum_weights")


class SFTutorialQuantumCircuit:
    """
    Quantum circuit implementation following SF tutorial pattern exactly.
    This ensures proper gradient flow by using SF's intended design.
    """
    
    def __init__(self, n_modes: int, layers: int, cutoff_dim: int = 6):
        """
        Initialize quantum circuit using SF tutorial pattern.
        
        Args:
            n_modes: Number of quantum modes
            layers: Number of quantum layers
            cutoff_dim: Fock space cutoff dimension
        """
        self.n_modes = n_modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        
        # Initialize SF components (SINGLE program and engine like SF tutorial)
        self.eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
        self.prog = sf.Program(n_modes)
        
        # Initialize weights using SF tutorial pattern
        self.weights = init_sf_weights(n_modes, layers)
        num_params = np.prod(self.weights.shape)
        
        # Create symbolic parameters exactly like SF tutorial
        sf_params = np.arange(num_params).reshape(self.weights.shape).astype(str)
        self.sf_params = np.array([self.prog.params(*i) for i in sf_params])
        
        # Build the symbolic program
        self._build_program()
        
        logger.info(f"SF Tutorial Circuit initialized: {n_modes} modes, {layers} layers")
        logger.info(f"Total parameters: {num_params}")
    
    def _build_program(self):
        """Build the symbolic SF program once (like SF tutorial)."""
        with self.prog.context as q:
            for k in range(self.layers):
                quantum_layer(self.sf_params[k], q)
    
    def execute(self, input_encoding: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Execute quantum circuit with proper gradient flow.
        SF tutorial pattern processes single samples, so we handle batching here.
        
        Args:
            input_encoding: Optional input encoding to modulate parameters
            
        Returns:
            Quantum measurements as tensor
        """
        if input_encoding is None:
            # No input encoding - use base weights directly (like SF tutorial)
            mapping = {
                p.name: w for p, w in zip(
                    self.sf_params.flatten(), 
                    tf.reshape(self.weights, [-1])
                )
            }
            
            # Execute circuit (reset if needed)
            if self.eng.run_progs:
                self.eng.reset()
                
            state = self.eng.run(self.prog, args=mapping).state
            return self._extract_measurements(state)
        
        else:
            # With input encoding - process batch (SF limitation workaround)
            batch_size = tf.shape(input_encoding)[0]
            
            # For now, use mean of batch for quantum parameters (SF tutorial limitation)
            # This preserves gradient flow while handling batching
            mean_encoding = tf.reduce_mean(input_encoding, axis=0, keepdims=True)
            
            # Create small modulation based on mean input
            modulation_weights = tf.Variable(
                tf.random.normal([tf.shape(mean_encoding)[1], tf.shape(self.weights)[1]], stddev=0.001),
                trainable=False  # Don't train these, just for encoding
            )
            
            modulation = tf.matmul(mean_encoding, modulation_weights)
            modulated_weights = self.weights + 0.01 * modulation
            
            # Create parameter mapping (SF tutorial exact pattern)
            mapping = {
                p.name: w for p, w in zip(
                    self.sf_params.flatten(), 
                    tf.reshape(modulated_weights, [-1])
                )
            }
            
            # Execute circuit (reset if needed)
            if self.eng.run_progs:
                self.eng.reset()
                
            state = self.eng.run(self.prog, args=mapping).state
            
            # Extract measurements and replicate for batch
            single_measurement = self._extract_measurements(state)
            
            # Replicate measurement for each batch sample
            # This is a limitation of SF tutorial pattern - not truly batch-native
            measurements = tf.tile([single_measurement], [batch_size, 1])
            
            return measurements
    
    def _extract_measurements(self, state) -> tf.Tensor:
        """Extract measurements from quantum state."""
        # Get state vector
        ket = state.ket()
        
        # Extract real and imaginary parts of relevant amplitudes
        real_parts = tf.math.real(ket[:self.n_modes])
        imag_parts = tf.math.imag(ket[:self.n_modes])
        
        # Combine into measurement vector
        measurements = tf.concat([real_parts, imag_parts], axis=0)
        
        return measurements
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return trainable variables."""
        return [self.weights]
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return int(np.prod(self.weights.shape))


class SFTutorialGenerator:
    """
    Quantum generator using proper SF tutorial pattern.
    Should achieve 100% gradient flow without NaN gradients.
    """
    
    def __init__(self, latent_dim: int = 6, output_dim: int = 2, n_modes: int = 4, 
                 layers: int = 2, cutoff_dim: int = 6):
        """
        Initialize quantum generator with SF tutorial pattern.
        
        Args:
            latent_dim: Latent space dimension
            output_dim: Output dimension
            n_modes: Number of quantum modes
            layers: Number of quantum layers
            cutoff_dim: Fock space cutoff
        """
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_modes = n_modes
        self.layers = layers
        
        # SF tutorial quantum circuit
        self.quantum_circuit = SFTutorialQuantumCircuit(n_modes, layers, cutoff_dim)
        
        # Classical components for input/output processing
        self.input_encoder = tf.Variable(
            tf.random.normal([latent_dim, self.quantum_circuit.get_parameter_count()], stddev=0.01),
            name="input_encoder"
        )
        
        self.output_processor = tf.Variable(
            tf.random.normal([n_modes * 2, output_dim], stddev=0.1),
            name="output_processor"
        )
        
        self.output_bias = tf.Variable(
            tf.zeros([output_dim]),
            name="output_bias"
        )
        
        logger.info(f"SF Tutorial Generator initialized: {latent_dim} → {output_dim}")
    
    def generate(self, z: tf.Tensor) -> tf.Tensor:
        """
        Generate samples using proper SF gradient flow.
        
        Args:
            z: Latent input tensor
            
        Returns:
            Generated samples
        """
        batch_size = tf.shape(z)[0]
        
        # Encode input for quantum circuit
        input_encoding = tf.matmul(z, self.input_encoder)
        
        # Execute quantum circuit (should have proper gradients!)
        quantum_output = self.quantum_circuit.execute(input_encoding)
        
        # Process quantum output to final result
        output = tf.matmul(quantum_output, self.output_processor) + self.output_bias
        
        return output
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return all trainable variables."""
        return [
            self.input_encoder,
            self.output_processor, 
            self.output_bias
        ] + self.quantum_circuit.trainable_variables


def test_sf_tutorial_gradients():
    """Test that SF tutorial pattern gives proper gradients."""
    print("🧪 Testing SF Tutorial Gradient Flow...")
    
    # Create generator
    generator = SFTutorialGenerator(latent_dim=6, output_dim=2, n_modes=4, layers=2)
    
    # Test input
    z = tf.random.normal([4, 6])
    
    # Test gradient computation
    with tf.GradientTape() as tape:
        output = generator.generate(z)
        loss = tf.reduce_mean(tf.square(output))
    
    gradients = tape.gradient(loss, generator.trainable_variables)
    
    # Check gradient health
    total_vars = len(generator.trainable_variables)
    valid_gradients = 0
    nan_gradients = 0
    
    for i, (var, grad) in enumerate(zip(generator.trainable_variables, gradients)):
        if grad is not None:
            valid_gradients += 1
            if tf.reduce_any(tf.math.is_nan(grad)):
                nan_gradients += 1
                print(f"❌ NaN gradient for {var.name}")
            else:
                grad_norm = tf.norm(grad)
                print(f"✅ Valid gradient for {var.name}: norm = {grad_norm:.6f}")
        else:
            print(f"❌ None gradient for {var.name}")
    
    print(f"\n📊 Gradient Summary:")
    print(f"   Valid gradients: {valid_gradients}/{total_vars}")
    print(f"   NaN gradients: {nan_gradients}/{valid_gradients}")
    print(f"   Success rate: {(valid_gradients - nan_gradients)/total_vars*100:.1f}%")
    
    if nan_gradients == 0 and valid_gradients == total_vars:
        print("🎉 SUCCESS: 100% valid gradients with SF tutorial pattern!")
        return True
    else:
        print("⚠️  Still have gradient issues")
        return False


if __name__ == "__main__":
    # Test the SF tutorial pattern
    test_sf_tutorial_gradients()
