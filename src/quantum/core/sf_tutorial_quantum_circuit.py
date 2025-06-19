"""
SF Tutorial Quantum Circuit - Production Integration

This module implements a quantum circuit using the exact SF tutorial pattern
that eliminates NaN gradients, while maintaining compatibility with our
modular architecture for future extensibility.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging
from typing import Optional, Dict, List, Tuple, Any, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def interferometer(params: tf.Tensor, q: List) -> None:
    """
    Interferometer implementation from SF tutorial (exact copy).
    
    Args:
        params: Interferometer parameters
        q: List of quantum modes
    """
    N = len(q)
    if N == 1:
        # Single mode case
        ops.Rgate(params[0]) | q[0]
        return

    theta = params[: N * (N - 1) // 2]
    phi = params[N * (N - 1) // 2 : N * (N - 1)]
    rphi = params[-N + 1 :]

    n = 0  # parameter counter

    # Apply rectangular beamsplitter array
    for l in range(N):
        for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
            if (l + k) % 2 != 1:
                ops.BSgate(theta[n], phi[n]) | (q1, q2)
                n += 1

    # Apply final local phase shifts
    for i in range(max(1, N - 1)):
        ops.Rgate(rphi[i]) | q[i]


def quantum_layer(params: tf.Tensor, q: List) -> None:
    """
    CV quantum neural network layer from SF tutorial (exact copy).
    
    Args:
        params: Layer parameters
        q: List of quantum modes
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


def init_sf_weights(modes: int, layers: int, active_sd: float = 0.0001, passive_sd: float = 0.1) -> tf.Variable:
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
    Production quantum circuit using SF tutorial pattern.
    
    Combines proven SF tutorial gradient flow with modular architecture
    for future extensibility in encoding/decoding.
    """
    
    def __init__(self, n_modes: int, layers: int, cutoff_dim: int = 6):
        """
        Initialize SF tutorial quantum circuit.
        
        Args:
            n_modes: Number of quantum modes
            layers: Number of quantum layers  
            cutoff_dim: Fock space cutoff dimension
        """
        self.n_modes = n_modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        
        # SF tutorial components (CRITICAL for gradients)
        self.eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
        self.prog = sf.Program(n_modes)
        
        # Initialize SF tutorial weights
        self.weights = init_sf_weights(n_modes, layers)
        num_params = int(np.prod(self.weights.shape))
        
        # Create symbolic parameters exactly like SF tutorial
        sf_params = np.arange(num_params).reshape(self.weights.shape).astype(str)
        self.sf_params = np.array([self.prog.params(*i) for i in sf_params])
        
        # Build the symbolic program
        self._build_program()
        
        # Calculate measurement dimension for modular integration
        self.measurement_dim = self._calculate_measurement_dim()
        
        logger.info(f"SF Tutorial Circuit initialized: {n_modes} modes, {layers} layers")
        logger.info(f"Total parameters: {num_params}")
        logger.info(f"Measurement dimension: {self.measurement_dim}")
    
    def _build_program(self) -> None:
        """Build the symbolic SF program once (like SF tutorial)."""
        with self.prog.context as q:
            for k in range(self.layers):
                quantum_layer(self.sf_params[k], q)
    
    def _calculate_measurement_dim(self) -> int:
        """Calculate the dimension of measurement output for modular integration."""
        # Using homodyne X and P measurements: two measurements per mode
        return 2 * self.n_modes
    
    def execute(self, input_encoding: Optional[tf.Tensor] = None) -> Any:
        """
        Execute quantum circuit with SF tutorial pattern.
        
        Args:
            input_encoding: Optional input encoding for data-dependent parameters
            
        Returns:
            Strawberry Fields quantum state
        """
        # SF tutorial pattern: start with base weights
        weights_to_use = self.weights
        
        # CRITICAL FIX: NO BATCH AVERAGING - use individual sample encoding
        if input_encoding is not None:
            # Ensure single sample processing (remove batch averaging!)
            if len(input_encoding.shape) > 1 and input_encoding.shape[0] > 1:
                # Take first sample only - this should be called per sample
                input_encoding = input_encoding[0:1]
            
            # Create static encoding weights (not trainable, just for input modulation)
            if not hasattr(self, '_encoding_weights'):
                encoding_weight_shape = [tf.shape(input_encoding)[1], tf.shape(self.weights)[1]]
                self._encoding_weights = tf.constant(
                    tf.random.normal(encoding_weight_shape, stddev=0.001),
                    name="static_encoding_weights"
                )
            
            # Apply input-dependent modulation (larger effect)
            modulation = tf.matmul(input_encoding, self._encoding_weights)
            weights_to_use = self.weights + 0.1 * modulation  # Increased from 0.01 to 0.1
        
        # Create parameter mapping (EXACT SF tutorial pattern - CRITICAL for gradients)
        mapping = {
            p.name: w for p, w in zip(
                self.sf_params.flatten(), 
                tf.reshape(weights_to_use, [-1])
            )
        }
        
        # Execute circuit (reset if needed)
        if self.eng.run_progs:
            self.eng.reset()
            
        state = self.eng.run(self.prog, args=mapping).state
        
        return state
    
    def extract_measurements(self, state: Any) -> tf.Tensor:
        """
        Extract physically realistic homodyne X and P measurements.
        
        Args:
            state: Quantum state from SF
            
        Returns:
            Measurement tensor with X and P quadrature measurements for each mode
        """
        # Perform both X and P quadrature measurements on each mode
        # This is physically realistic (standard CV measurements) and differentiable in SF
        measurements = []
        
        for mode in range(self.n_modes):
            # X quadrature measurement (position-like)
            x_val = state.quad_expectation(mode, 0)  # 0 = X quadrature
            measurements.append(x_val)
            
            # P quadrature measurement (momentum-like) 
            p_val = state.quad_expectation(mode, 1)  # 1 = P quadrature
            measurements.append(p_val)
        
        # Convert to tensor [x0, p0, x1, p1, x2, p2, x3, p3]
        measurements = tf.stack(measurements)
        
        # Ensure real-valued output (should already be real from quadrature measurements)
        measurements = tf.cast(measurements, tf.float32)
        
        return measurements
    
    def get_measurement_dim(self) -> int:
        """Get measurement dimension for modular integration."""
        return self.measurement_dim
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return trainable variables (SF tutorial weights only)."""
        return [self.weights]
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return int(np.prod(self.weights.shape))
    
    def get_circuit_info(self) -> Dict[str, Any]:
        """Get circuit information for debugging and analysis."""
        return {
            'n_modes': self.n_modes,
            'layers': self.layers,
            'cutoff_dim': self.cutoff_dim,
            'parameter_count': self.get_parameter_count(),
            'measurement_dim': self.measurement_dim,
            'weight_shape': self.weights.shape.as_list(),
            'sf_tutorial_compatible': True
        }


class SFTutorialCircuitWithEncoding(SFTutorialQuantumCircuit):
    """
    Extended SF tutorial circuit with trainable input encoding.
    
    This version includes trainable encoding weights for input-dependent
    quantum parameter modulation while preserving SF tutorial gradient flow.
    """
    
    def __init__(self, n_modes: int, layers: int, input_dim: int, cutoff_dim: int = 6):
        """
        Initialize SF tutorial circuit with encoding.
        
        Args:
            n_modes: Number of quantum modes
            layers: Number of quantum layers
            input_dim: Dimension of input encoding
            cutoff_dim: Fock space cutoff dimension
        """
        super().__init__(n_modes, layers, cutoff_dim)
        
        self.input_dim = input_dim
        
        # Trainable input encoding weights
        self.input_encoder = tf.Variable(
            tf.random.normal([input_dim, self.get_parameter_count()], stddev=0.01),
            name="input_encoder"
        )
        
        logger.info(f"SF Tutorial Circuit with encoding: input_dim={input_dim}")
    
    def execute_with_encoding(self, input_data: tf.Tensor) -> Any:
        """
        Execute circuit with trainable input encoding.
        
        Args:
            input_data: Input data tensor [batch_size, input_dim]
            
        Returns:
            Quantum state
        """
        # Encode input to parameter modulation
        batch_size = tf.shape(input_data)[0]
        
        # For now, use mean of batch for parameter modulation (SF limitation)
        mean_input = tf.reduce_mean(input_data, axis=0, keepdims=True)
        
        # Create parameter modulation
        param_modulation = tf.matmul(mean_input, self.input_encoder)
        
        # Modulate base weights
        modulated_weights = self.weights + 0.01 * param_modulation
        
        # Create parameter mapping (SF tutorial pattern)
        mapping = {
            p.name: w for p, w in zip(
                self.sf_params.flatten(), 
                tf.reshape(modulated_weights, [-1])
            )
        }
        
        # Execute circuit
        if self.eng.run_progs:
            self.eng.reset()
            
        state = self.eng.run(self.prog, args=mapping).state
        
        return state
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return trainable variables (weights + encoder)."""
        return [self.weights, self.input_encoder]


def test_sf_tutorial_circuit_gradients():
    """Test function to verify gradient flow."""
    print("🧪 Testing SF Tutorial Circuit Gradients...")
    
    # Create circuit
    circuit = SFTutorialQuantumCircuit(n_modes=4, layers=2, cutoff_dim=6)
    
    # Test gradient computation
    with tf.GradientTape() as tape:
        state = circuit.execute()
        measurements = circuit.extract_measurements(state)
        loss = tf.reduce_mean(tf.square(measurements))
    
    gradients = tape.gradient(loss, circuit.trainable_variables)
    
    # Check gradient health
    all_valid = True
    for i, (var, grad) in enumerate(zip(circuit.trainable_variables, gradients)):
        if grad is None:
            print(f"❌ No gradient for variable {i}: {var.name}")
            all_valid = False
        elif tf.reduce_any(tf.math.is_nan(grad)):
            print(f"❌ NaN gradient for variable {i}: {var.name}")
            all_valid = False
        else:
            grad_norm = tf.norm(grad)
            print(f"✅ Valid gradient for {var.name}: norm = {grad_norm:.6f}")
    
    if all_valid:
        print("🎉 SUCCESS: 100% valid gradients with SF tutorial circuit!")
        return True
    else:
        print("❌ FAILED: Gradient issues detected")
        return False


if __name__ == "__main__":
    # Test the SF tutorial circuit
    test_sf_tutorial_circuit_gradients()
